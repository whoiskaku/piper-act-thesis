"""阶段二命令行入口 —— `reports/PREREG_Q2_STAGE2.md` §5、§11。

⚠ `prepare / refs / nom-inner / nom-final / inc-inner / inc-final` 是**正式运行**：入口先查预注册 ——
状态行须以「**状态：已冻结」开头，`code_sha=` 与 `config_hash=` 须与当前一致，否则拒绝；不接受训练参数覆盖。
`timing` 只用于另行授权的短程耗时测量（1–30 步，越界在读数据之前拒绝）：不评分外层留出会话、不选步、
不保存正式训练产物（它不是运行的旁路）。

作业（外层折 = 留出的会话；内层验证会话 ∈ 该折另外两个会话）：
- `nom-inner FOLD VAL SEED`：名义 2a 用「另一个训练会话」的裸跑训到上限，在 VAL 的裸跑上给每个检查点评分；
- `nom-final FOLD SEED`：由两张内层表选步 → 用两个训练会话的裸跑训到选中步数并冻结；评分：留出裸跑（Q1 复查）、
  留出补正运行上的零响应与固定 H、ab4 P2/P3 上的零响应与固定 H（含 H 的 β 分子分母）；
- `inc-inner FORM FOLD VAL SEED`：「固定名义模型下的增量选步验证」——增量用「另一个训练会话」的补正运行训到上限，
  在 VAL 的补正运行上给每个检查点评分；名义 = 该折已冻结的 2a；
- `inc-final FORM FOLD SEED`：由两张内层表选步 → 用两个训练会话的补正运行训到选中步数（第 0 步 = 保留零响应，不训练）；
  评分：留出补正运行上的候选、ab4 P2/P3 上的候选（含 β 分子分母）；
- `refs FOLD`：G-2 同口径窗口重预测在留出会话裸跑上的分数（Q1 复查的背景）。
续跑只跳过配置指纹与输入指纹都一致的作业；终训作业还须 `.pt`、`.npz` 存在且哈希与记录一致，损坏就重新排队。
输入指纹含数据文件、标定 R/b 文件、固定 H 文件、依赖代码与库版本；截止时间与数据名单的落实同阶段一。
冻结代码一改，`code_sha` / `config_hash` 就变：须重新冻结，且旧目录的截止时间记录绑定旧指纹 ⇒ 不能在旧目录里续跑。

    python -u sim/q2/q2_run.py timing --steps 30
    python -u sim/q2/q2_run.py prepare   --out outputs/q2_stage2
    python -u sim/q2/q2_run.py list-jobs --out outputs/q2_stage2 --phase nom-inner
    python -u sim/q2/q2_run.py nom-inner --out outputs/q2_stage2 ab5 ab6 0
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q1_data as D1           # noqa: E402
import q1_run as R1            # noqa: E402
import q1_train as TR1         # noqa: E402
import q1_windows as W         # noqa: E402
import q2_data as D2           # noqa: E402
import q2_models as Mo         # noqa: E402
import q2_train as TR          # noqa: E402
import q2_windows as W2        # noqa: E402

ROOT = D1.ROOT
FOLDS = D2.FOLDS
SEEDS = TR.CONFIG["seeds"]
FORMS = Mo.FORMS
PREREG = ROOT / "reports/PREREG_Q2_STAGE2.md"
CALIB_FILES = ("outputs/step1_H/step1_H_v1.json", "outputs/p2_labels/pp2a-deploy/manifest.json")
H_FILES = ("outputs/step1_H/step1_H_v1.npz",)
TIMING_FOLD, TIMING_VAL = "ab5", "ab7"
TIMING_MAX_STEPS = 30
RUNNER_JOBS = 15
atomic_json, atomic_npz, atomic_torch = R1.atomic_json, R1.atomic_npz, R1.atomic_torch
file_sha, files_fp, fp_of = R1.file_sha, R1.files_fp, R1.fp_of


class NotFrozenError(RuntimeError):
    """预注册不存在、未冻结，或冻结时的代码/配置指纹与当前不符。"""


class DeadlineExceeded(RuntimeError):
    """本轮已超出预注册的墙钟上限：停止，并标记「未完成」。"""


class TimingRefused(ValueError):
    """测速步数越界：`timing` 只是另行授权的短程测速（1–30 步），不是运行的旁路。"""


def require_frozen(prereg=None) -> None:
    p = Path(prereg) if prereg is not None else PREREG
    if not p.exists():
        raise NotFrozenError(f"拒绝运行：{p} 不存在 —— 正式运行须先冻结预注册")
    text = p.read_text()
    if not re.search(r"^\*\*状态：已冻结", text, re.M):
        raise NotFrozenError(f"拒绝运行：{p.name} 尚未冻结（状态行须以「**状态：已冻结」开头）")
    m = re.search(r"code_sha=([0-9a-f]{16})", text)
    if not m or m.group(1) != TR.code_sha()[:16]:
        raise NotFrozenError("拒绝运行：预注册记录的 code_sha 与当前代码不符 —— 代码改动后须重新冻结")
    m = re.search(r"config_hash=([0-9a-f]{16})", text)
    if not m or m.group(1) != TR.config_hash():
        raise NotFrozenError("拒绝运行：预注册记录的 config_hash 与当前规则不符 —— 须重新冻结")


# ── 折与路径 ──────────────────────────────────────────────────────────────
def train_sessions(fold: str) -> tuple:
    if fold not in FOLDS:
        raise ValueError(f"未知的折 {fold}")
    return tuple(s for s in FOLDS if s != fold)


def inner_train(fold: str, val: str) -> str:
    """该折的内层：VAL 是内层验证会话，另一个训练会话是内层训练会话。"""
    tr = train_sessions(fold)
    if val not in tr:
        raise ValueError(f"{val} 不是折 {fold} 的训练会话")
    return next(s for s in tr if s != val)


def nom_inner_json(out, fold, val, seed) -> Path:
    return Path(out) / "nom" / "inner" / f"seed{seed}" / f"{fold}__val_{val}.json"


def nom_final_base(out, fold, seed) -> Path:
    return Path(out) / "nom" / "final" / f"seed{seed}" / fold


def inc_inner_json(out, form, fold, val, seed) -> Path:
    return Path(out) / "inc" / form / "inner" / f"seed{seed}" / f"{fold}__val_{val}.json"


def inc_final_base(out, form, fold, seed) -> Path:
    return Path(out) / "inc" / form / "final" / f"seed{seed}" / fold


def ref_json(out, fold) -> Path:
    return Path(out) / "refs" / "g2" / f"{fold}.json"


# ── 输入指纹、名单与截止时间 ───────────────────────────────────────────────
def roster_files() -> list:
    """冻结名单里全部 183 集的数据文件。"""
    return [ROOT / "outputs" / n.split("/")[0] / f"{n.split('/')[1]}.json" for v in D2.ROSTER.values() for n in v]


def inputs_file(out) -> Path:
    return Path(out) / "inputs.json"


def read_inputs(out) -> dict:
    p = inputs_file(out)
    if not p.exists():
        raise RuntimeError(f"缺 {p} —— 先跑 prepare")
    return json.loads(p.read_text())


def current_inputs() -> dict:
    dfiles, dfp = files_fp(roster_files())
    cfiles, cfp = files_fp([ROOT / f for f in CALIB_FILES])
    hfiles, hfp = files_fp([ROOT / f for f in H_FILES])
    pfiles, _ = files_fp([ROOT / f for f in R1.DEP_FILES])
    gfiles, gfp = files_fp([ROOT / f for f in R1.G2_FILES])
    env = R1.env_versions()
    return {"code_sha": TR.code_sha(), "data": dfp, "calib": cfp, "h": hfp,
            "deps": fp_of({"files": pfiles, "env": env}), "g2": gfp,
            "data_files": dfiles, "calib_files": cfiles, "h_files": hfiles, "deps_files": pfiles,
            "g2_files": gfiles, "env": env, "roster": D2.ROSTER}


def check_inputs(I: dict) -> None:
    now = current_inputs()
    bad = [k for k in ("code_sha", "data", "calib", "h", "deps", "g2") if now[k] != I.get(k)]
    if bad:
        raise RuntimeError(f"输入自 prepare 以来已变：{bad} —— 重跑 prepare（或换 --out）")


def deadline_file(out) -> Path:
    return Path(out) / "deadline.json"


def incomplete_file(out) -> Path:
    return Path(out) / "INCOMPLETE.json"


def mark_incomplete(out, reason: str) -> None:
    if not incomplete_file(out).exists():
        atomic_json(incomplete_file(out), {"status": "未完成", "reason": reason, "at": time.time()})


def enforce_deadline(out, create: bool = False) -> dict:
    """第一次 prepare 记录截止时间（续跑不重置）；超时 ⇒ 标记未完成并拒绝；否则设闹钟，到点中止本作业。"""
    p = deadline_file(out)
    fp = {"code_sha": TR.code_sha(), "config_hash": TR.config_hash()}
    if p.exists():
        d = json.loads(p.read_text())
        if {k: d.get(k) for k in fp} != fp:
            raise RuntimeError(f"{p}: 本目录属于另一版冻结的实验 —— 换一个 --out")
    elif create:
        now = time.time()
        d = {**fp, "budget_hours": TR.CONFIG["budget_hours"], "started_at": now,
             "deadline": now + TR.CONFIG["budget_hours"] * 3600}
        atomic_json(p, d)
    else:
        raise RuntimeError(f"缺 {p} —— 先跑 prepare")
    left = d["deadline"] - time.time()
    if left <= 0:
        mark_incomplete(out, f"超出预注册的墙钟上限（{TR.CONFIG['budget_hours']} 小时）")
        raise DeadlineExceeded("已超出本轮截止时间 —— 拒绝运行，本轮标记为未完成")

    def on_alarm(signum, frame):
        mark_incomplete(out, f"作业运行中到达墙钟上限（{TR.CONFIG['budget_hours']} 小时）")
        raise DeadlineExceeded("到达本轮截止时间 —— 作业中止，本轮标记为未完成")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(left) + 1)
    return d


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_done(path: Path, inputs) -> bool:
    j = _load(path) if inputs is not None else None
    return bool(j) and j.get("config_hash") == TR.config_hash() and j.get("input_fp") == fp_of(inputs)


def final_done(base: Path, inputs) -> bool:
    """终训作业是否完成：JSON 的配置与输入指纹一致，且 `.pt`、`.npz` 存在、哈希与 JSON 记录一致（损坏就重新排队）。"""
    if not is_done(base.with_suffix(".json"), inputs):
        return False
    j = _load(base.with_suffix(".json"))
    return all(base.with_suffix(ext).exists() and file_sha(base.with_suffix(ext)) == j.get(key)
               for ext, key in ((".pt", "pt_sha256"), (".npz", "npz_sha256")))


def base_inputs(I: dict) -> dict:
    return {"data": I["data"], "calib": I["calib"], "deps": I["deps"]}


def nom_inner_inputs(I, fold, val) -> dict:
    return base_inputs(I)


def nom_final_inputs(out, I, fold, seed):
    inner = {}
    for v in train_sessions(fold):
        p = nom_inner_json(out, fold, v, seed)
        if not p.exists():
            return None
        inner[v] = file_sha(p)
    return {**base_inputs(I), "h": I["h"], "inner": inner}


def nominal_pt_sha(out, fold, seed):
    p = nom_final_base(out, fold, seed).with_suffix(".pt")
    j = _load(nom_final_base(out, fold, seed).with_suffix(".json"))
    if not p.exists() or not j or j.get("pt_sha256") != file_sha(p):
        return None
    return j["pt_sha256"]


def inc_inner_inputs(out, I, form, fold, val, seed):
    nom = nominal_pt_sha(out, fold, seed)
    return None if nom is None else {**base_inputs(I), "nominal": nom}


def inc_final_inputs(out, I, form, fold, seed):
    nom = nominal_pt_sha(out, fold, seed)
    if nom is None:
        return None
    inner = {}
    for v in train_sessions(fold):
        p = inc_inner_json(out, form, fold, v, seed)
        if not p.exists():
            return None
        inner[v] = file_sha(p)
    return {**base_inputs(I), "nominal": nom, "inner": inner}


def ref_inputs(I) -> dict:
    return {**base_inputs(I), "g2": I["g2"]}


# ── 载入与评分小工具 ──────────────────────────────────────────────────────
def load_eps(folds, kinds) -> list:
    """按冻结名单核对后载入；`folds` 可以为空（只要 ab4 辅助时）。"""
    return D2.load(folds=tuple(folds), kinds=tuple(kinds), check=True)


def load_nominal(out, fold, seed):
    import torch
    base = nom_final_base(out, fold, seed)
    j = _load(base.with_suffix(".json"))
    if not j or j.get("pt_sha256") != file_sha(base.with_suffix(".pt")):
        raise RuntimeError(f"{base}: 名义模型缺失或与记录不符 —— 先跑 nom-final")
    ck = torch.load(base.with_suffix(".pt"), weights_only=False)
    model = TR1.with_weights("2a", ck["stats"], ck["state_dict"])
    model.eval()
    return model, ck["stats"]


def selection(paths, ckpts) -> tuple:
    """两张内层表 → (V {step: 平均 ρ}, step*)。表缺失或检查点不符 ⇒ RuntimeError。"""
    rho = []
    for p in paths:
        j = _load(p)
        if j is None:
            raise RuntimeError(f"缺内层表 {p}")
        if sorted(int(s) for s in j["scores"]) != list(ckpts):
            raise RuntimeError(f"{p}: 检查点与选步规则不符")
        rho.append({int(s): float(r["rho"]) for s, r in j["scores"].items()})
    V = {s: float(np.mean([r[s] for r in rho])) for s in ckpts}
    return V, TR.pick_step(V)


def stamp(rec: dict, inputs: dict) -> dict:
    return {**rec, "config_hash": TR.config_hash(), "code_sha": TR.code_sha(), "inputs": inputs,
            "input_fp": fp_of(inputs), "finished_at": time.time()}


def _guard(out, prereg, create=False) -> dict:
    require_frozen(prereg)
    if create:
        return {}
    enforce_deadline(out)
    I = read_inputs(out)
    check_inputs(I)
    return I


# ── prepare / refs ────────────────────────────────────────────────────────
def cmd_prepare(out, prereg=None) -> None:
    require_frozen(prereg)
    eps = D2.load()                                                   # 全部 183 集，核对名单与日志契约
    enforce_deadline(out, create=True)                                # 数据通过核验后才开始计时；续跑不重置
    I = current_inputs()
    atomic_json(inputs_file(out), I)
    print(f"  载入 {len(eps)} 集 · 数据 {I['data']} · 标定 {I['calib']} · H {I['h']} · 依赖 {I['deps']}", flush=True)


def cmd_refs(out, fold: str, prereg=None) -> None:
    I = _guard(out, prereg)
    inputs = ref_inputs(I)
    if is_done(ref_json(out, fold), inputs):
        print(f"  参照 {fold} 已是最新，跳过", flush=True)
        return
    import q1_g2ref as G
    eps = load_eps((fold,), ("bare",))
    R, b = D1.calibration()
    K = W.kernel()
    g2 = G.load_g2()
    rows = [TR.ep_metrics(e, G.episode_ehat(g2, e, W.chunk_windows(e), R, b), K) for e in eps]
    atomic_json(ref_json(out, fold), stamp({"kind": "g2", "fold": fold, "episodes": rows}, inputs))
    print(f"  参照 {fold} 完成", flush=True)


# ── 名义 ──────────────────────────────────────────────────────────────────
def score_nominal(model, st, eps, K) -> list:
    return [TR.ep_metrics(e, TR.nominal_episode(model, st, e), K) for e in eps]


def cmd_nom_inner(out, fold, val, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    tr_s = inner_train(fold, val)
    eps = load_eps((tr_s, val), ("bare",))
    tr, va = D2.select(eps, tr_s), D2.select(eps, val)
    K = W.kernel()
    res = TR.train_nominal(tr, seed, log=lambda m: print(m, flush=True))
    scores = {}
    for step in sorted(res["ckpt"]):
        m = TR1.with_weights("2a", res["stats"], res["ckpt"][step])
        rows = score_nominal(m, res["stats"], va, K)
        scores[str(step)] = {"rho": TR.pooled_rho(rows), "state_sha256": TR1.state_dict_sha(res["ckpt"][step])}
    atomic_json(nom_inner_json(out, fold, val, seed), stamp(
        {"kind": "nom-inner", "fold": fold, "val": val, "seed": seed, "train_sessions": [tr_s],
         "scores": scores, "sec": res["sec"]}, nom_inner_inputs(I, fold, val)))
    print(f"  名义内层 {fold} 验证 {val} seed{seed} 完成", flush=True)


def cmd_nom_final(out, fold, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    H = W2.load_h()
    inputs = nom_final_inputs(out, I, fold, seed)
    if inputs is None:
        raise RuntimeError(f"折 {fold} 种子 {seed} 的名义内层表不全 —— 先跑 nom-inner")
    V, step = selection([nom_inner_json(out, fold, v, seed) for v in train_sessions(fold)], TR.CONFIG["ckpts"])
    eps = load_eps(FOLDS, ("bare", "corr", "aux"))
    tr = [e for e in eps if e.kind == "bare" and e.fold in train_sessions(fold)]
    K = W.kernel()
    res = TR.train_nominal(tr, seed, ckpts=(step,), log=lambda m: print(m, flush=True))
    sd = res["ckpt"][step]
    model = TR1.with_weights("2a", res["stats"], sd)
    base = nom_final_base(out, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"state_dict": sd, "stats": res["stats"], "step": step})
    rec = {"kind": "nom-final", "fold": fold, "seed": seed, "train_sessions": list(train_sessions(fold)),
           "V": {str(s): V[s] for s in sorted(V)}, "step_star": step, "boundary": step == TR.CONFIG["max_steps"],
           "state_sha256": TR1.state_dict_sha(sd), "pt_sha256": file_sha(base.with_suffix(".pt")), "sec": res["sec"]}
    held_bare = D2.select(eps, fold, "bare")
    held_corr = D2.select(eps, fold, "corr")
    aux = D2.select(eps, "ab4", "aux")
    rec["q1"] = score_nominal(model, res["stats"], held_bare, K)
    preds, zero, hres, aux_zero, aux_h, aux_beta = {}, [], [], [], [], []
    for e in held_corr:
        en, h = TR.nominal_episode(model, res["stats"], e), TR.h_episode(H, e)
        zero.append(TR.ep_metrics(e, en, K))
        hres.append(TR.ep_metrics(e, en + h, K))
        preds[f"nom__{e.name.replace('/', '__')}"] = en
        preds[f"H__{e.name.replace('/', '__')}"] = en + h
    for e in aux:
        en, h = TR.nominal_episode(model, res["stats"], e), TR.h_episode(H, e)
        aux_zero.append(TR.ep_metrics(e, en, K))
        aux_h.append(TR.ep_metrics(e, en + h, K))
        num, den = TR.beta_parts(e, en, h, K)
        aux_beta.append({"name": e.name, "arm": e.arm, "num": num, "den": den})
    atomic_npz(base.with_suffix(".npz"), **preds)
    rec.update({"zero": zero, "H": hres, "aux_zero": aux_zero, "aux_H": aux_h, "aux_H_beta": aux_beta,
                "npz_sha256": file_sha(base.with_suffix(".npz"))})
    atomic_json(base.with_suffix(".json"), stamp(rec, inputs))
    note = "（最佳点落在搜索边界）" if rec["boundary"] else ""
    print(f"  名义终训 {fold} seed{seed}：step* = {step}{note}", flush=True)


# ── 增量 ──────────────────────────────────────────────────────────────────
def score_increment(form, model, r, nom_st, eps, e_nom, K) -> list:
    return [TR.ep_metrics(e, e_nom[e.name] + TR.increment_episode(form, model, r["st_inc"], r["scale"], nom_st, e), K)
            for e in eps]


def cmd_inc_inner(out, form, fold, val, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    inputs = inc_inner_inputs(out, I, form, fold, val, seed)
    if inputs is None:
        raise RuntimeError(f"折 {fold} 种子 {seed} 的名义模型缺失 —— 先跑 nom-final")
    nom_model, nom_st = load_nominal(out, fold, seed)
    tr_s = inner_train(fold, val)
    eps = load_eps((tr_s, val), ("corr",))
    tr, va = D2.select(eps, tr_s), D2.select(eps, val)
    K = W.kernel()
    res = TR.train_increment(form, (nom_model, nom_st), tr, seed, log=lambda m: print(m, flush=True))
    e_nom = {e.name: TR.nominal_episode(nom_model, nom_st, e) for e in va}
    scores = {}
    for step in sorted(res["ckpt"]):
        m = TR.with_increment(form, res["ckpt"][step])
        rows = score_increment(form, m, res, nom_st, va, e_nom, K)
        scores[str(step)] = {"rho": TR.pooled_rho(rows), "state_sha256": TR1.state_dict_sha(res["ckpt"][step])}
    atomic_json(inc_inner_json(out, form, fold, val, seed), stamp(
        {"kind": "inc-inner", "form": form, "fold": fold, "val": val, "seed": seed, "train_sessions": [tr_s],
         "scores": scores, "sec": res["sec"]}, inputs))
    print(f"  增量内层 {form} {fold} 验证 {val} seed{seed} 完成", flush=True)


def cmd_inc_final(out, form, fold, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    inputs = inc_final_inputs(out, I, form, fold, seed)
    if inputs is None:
        raise RuntimeError(f"{form} 折 {fold} 种子 {seed} 的增量内层表或名义模型不全")
    V, step = selection([inc_inner_json(out, form, fold, v, seed) for v in train_sessions(fold)], TR.CONFIG["ckpts"])
    nom_model, nom_st = load_nominal(out, fold, seed)
    eps = load_eps(FOLDS, ("corr", "aux"))
    tr = [e for e in eps if e.kind == "corr" and e.fold in train_sessions(fold)]
    held = D2.select(eps, fold, "corr")
    aux = D2.select(eps, "ab4", "aux")
    K = W.kernel()
    res = TR.train_increment(form, (nom_model, nom_st), tr, seed, ckpts=(step,), log=lambda m: print(m, flush=True))
    sd = res["ckpt"][step]                                            # step* = 0 ⇒ 初始权重 = 零响应，未训练
    model = TR.with_increment(form, sd)
    base = inc_final_base(out, form, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"form": form, "state_dict": sd, "st_inc": res["st_inc"],
                                           "scale": res["scale"], "step": step})
    e_nom = {e.name: TR.nominal_episode(nom_model, nom_st, e) for e in held + aux}
    preds, cand, aux_c, aux_beta = {}, [], [], []
    for e in held:
        d = TR.increment_episode(form, model, res["st_inc"], res["scale"], nom_st, e)
        cand.append(TR.ep_metrics(e, e_nom[e.name] + d, K))
        preds[f"cand__{e.name.replace('/', '__')}"] = e_nom[e.name] + d
    for e in aux:
        d = TR.increment_episode(form, model, res["st_inc"], res["scale"], nom_st, e)
        aux_c.append(TR.ep_metrics(e, e_nom[e.name] + d, K))
        num, den = TR.beta_parts(e, e_nom[e.name], d, K)
        aux_beta.append({"name": e.name, "arm": e.arm, "num": num, "den": den})
    atomic_npz(base.with_suffix(".npz"), **preds)
    rec = {"kind": "inc-final", "form": form, "fold": fold, "seed": seed, "train_sessions": list(train_sessions(fold)),
           "V": {str(s): V[s] for s in sorted(V)}, "step_star": step, "boundary": step == TR.CONFIG["max_steps"],
           "state_sha256": TR1.state_dict_sha(sd), "pt_sha256": file_sha(base.with_suffix(".pt")),
           "npz_sha256": file_sha(base.with_suffix(".npz")), "cand": cand, "aux_cand": aux_c,
           "aux_beta": aux_beta, "sec": res["sec"]}
    atomic_json(base.with_suffix(".json"), stamp(rec, inputs))
    note = "（选中第 0 步：保留零响应）" if step == 0 else ("（最佳点落在搜索边界）" if rec["boundary"] else "")
    print(f"  增量终训 {form} {fold} seed{seed}：step* = {step}{note}", flush=True)


# ── 待办清单 ──────────────────────────────────────────────────────────────
PHASES = ("refs", "nom-inner", "nom-final", "inc-inner", "inc-final")


def jobs(out, phase: str) -> list:
    """→ 该阶段尚未完成的作业参数（字符串列表的列表）。"""
    I = read_inputs(out)
    todo = []
    for fold in FOLDS:
        if phase == "refs":
            if not is_done(ref_json(out, fold), ref_inputs(I)):
                todo.append([fold])
            continue
        for seed in SEEDS:
            if phase == "nom-inner":
                todo += [[fold, v, str(seed)] for v in train_sessions(fold)
                         if not is_done(nom_inner_json(out, fold, v, seed), nom_inner_inputs(I, fold, v))]
            elif phase == "nom-final":
                if not final_done(nom_final_base(out, fold, seed), nom_final_inputs(out, I, fold, seed)):
                    todo.append([fold, str(seed)])
            else:
                for form in FORMS:
                    if phase == "inc-inner":
                        todo += [[form, fold, v, str(seed)] for v in train_sessions(fold)
                                 if not is_done(inc_inner_json(out, form, fold, v, seed),
                                                inc_inner_inputs(out, I, form, fold, v, seed))]
                    elif not final_done(inc_final_base(out, form, fold, seed), inc_final_inputs(out, I, form, fold, seed)):
                        todo.append([form, fold, str(seed)])
    return todo


def cmd_list_jobs(out, phase: str) -> None:
    for args in jobs(out, phase):
        print(" ".join(args))


# ── 耗时测量（另行授权的短程测速；不评分外层留出会话、不选步、不保存产物）─────────
def cmd_timing(steps: int) -> None:
    if not 1 <= steps <= TIMING_MAX_STEPS:                            # 在读数据、训练之前拒绝
        raise TimingRefused(f"拒绝测速：--steps 须在 1–{TIMING_MAX_STEPS} 之间（得到 {steps}）—— "
                            "timing 只是另行授权的短程测速，不是运行的旁路")
    tr_s = inner_train(TIMING_FOLD, TIMING_VAL)
    eps = load_eps((tr_s,), ("bare", "corr"))
    bare, corr = D2.select(eps, tr_s, "bare"), D2.select(eps, tr_s, "corr")
    K = W.kernel()
    res = TR.train_nominal(bare, 0, ckpts=(steps,))
    per = {"nominal": res["sec"] / steps}
    m = TR1.with_weights("2a", res["stats"], res["ckpt"][steps])
    t = time.time()
    score_nominal(m, res["stats"], bare[:5], K)                       # 只在训练会话上计时，不打印 ρ
    per_ep = {"nominal": (time.time() - t) / 5}
    e_nom = {e.name: TR.nominal_episode(m, res["stats"], e) for e in corr[:5]}
    for form in FORMS:
        r = TR.train_increment(form, (m, res["stats"]), corr, 0, ckpts=(steps,))
        per[form] = r["sec"] / steps
        mi = TR.with_increment(form, r["ckpt"][steps])
        t = time.time()
        score_increment(form, mi, r, res["stats"], corr[:5], e_nom, K)
        per_ep[form] = (time.time() - t) / 5
    for k in per:
        print(f"  {k}: {per[k]:.3f} s/步 · 评分 {per_ep[k]:.3f} s/集", flush=True)
    S, n_ck, load = TR.CONFIG["max_steps"], len(TR.CONFIG["ckpts"]), 90.0
    n_val = {"ab5": (40, 43), "ab6": (30, 30), "ab7": (15, 15)}                    # 各会话的（裸跑, 补正）集数
    nom = sum(len(SEEDS) * (S * per["nominal"] + n_ck * n_val[v][0] * per_ep["nominal"] + load)
              for f in FOLDS for v in train_sessions(f))
    nomf = len(FOLDS) * len(SEEDS) * (S * per["nominal"] + 120 * per_ep["nominal"] + load)
    inc = sum(len(SEEDS) * (S * per[fm] + n_ck * n_val[v][1] * per_ep[fm] + load)
              for fm in FORMS for f in FOLDS for v in train_sessions(f))
    incf = sum(len(FOLDS) * len(SEEDS) * (S * per[fm] + 60 * per_ep[fm] + load) for fm in FORMS)
    total = nom + nomf + inc + incf
    print(f"  合计 ≤ {total / 3600:.1f} 核·小时（按步数取上限计；含评分与载入）；{RUNNER_JOBS} 路并行、四个阶段依次进行，"
          f"墙钟约 {sum(x / RUNNER_JOBS for x in (nom, nomf, inc, incf)) / 3600 + 0.5:.1f} 小时（含 prepare 与参照约 0.5 小时）")


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--prereg", default=None)
    p = sub.add_parser("refs")
    p.add_argument("--out", required=True)
    p.add_argument("--prereg", default=None)
    p.add_argument("fold", choices=FOLDS)
    p = sub.add_parser("list-jobs")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", required=True, choices=PHASES)
    for name, args in (("nom-inner", ("fold", "val", "seed")), ("nom-final", ("fold", "seed")),
                       ("inc-inner", ("form", "fold", "val", "seed")), ("inc-final", ("form", "fold", "seed"))):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--prereg", default=None)
        for a in args:
            if a == "seed":
                p.add_argument(a, type=int, choices=SEEDS)
            elif a == "form":
                p.add_argument(a, choices=FORMS)
            else:
                p.add_argument(a, choices=FOLDS)
    p = sub.add_parser("timing")
    p.add_argument("--steps", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            cmd_prepare(a.out, a.prereg)
        elif a.cmd == "refs":
            cmd_refs(a.out, a.fold, a.prereg)
        elif a.cmd == "list-jobs":
            cmd_list_jobs(a.out, a.phase)
        elif a.cmd == "nom-inner":
            cmd_nom_inner(a.out, a.fold, a.val, a.seed, a.prereg)
        elif a.cmd == "nom-final":
            cmd_nom_final(a.out, a.fold, a.seed, a.prereg)
        elif a.cmd == "inc-inner":
            cmd_inc_inner(a.out, a.form, a.fold, a.val, a.seed, a.prereg)
        elif a.cmd == "inc-final":
            cmd_inc_final(a.out, a.form, a.fold, a.seed, a.prereg)
        else:
            cmd_timing(a.steps)
    except (NotFrozenError, TimingRefused) as err:
        print(err, flush=True)
        return 2
    except DeadlineExceeded as err:
        print(err, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
