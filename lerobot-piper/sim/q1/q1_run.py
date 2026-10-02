"""阶段一命令行入口：内层选步 → 终训与外层评分（两阶段）。

⚠ `prepare / refs / inner / final` 是**正式运行**：入口先查 `reports/PREREG_Q1_STAGE1.md` ——
状态行须以「**状态：已冻结」开头，且记录的 `code_sha=`、`config_hash=` 与当前代码和配置一致，
否则拒绝运行。正式入口**不接受**训练步数等覆盖参数：训练配置只来自冻结的 `q1_train.CONFIG`。
`timing` 不经过闸门：只测每步耗时与检查点评分耗时（不预测外层留出会话、不存权重）。

停止规则（用户 2026-10-02 定）：
- `inner MODEL FOLD VAL SEED`：外层折 FOLD 的训练会话去掉 VAL，用其余会话训到上限，在每个检查点给
  VAL 会话的主窗口评分（ρ = Σnum/Σden）。归一化、损失尺度、传感器层都只用这几个会话。
- `final MODEL FOLD SEED`：读 FOLD 的全部内层评分表，`V(step)` = 各验证会话 ρ 的等权平均，step* = 最小者
  （完整精度平局取最早）；在 FOLD 的全部训练会话上同种子训到 step*，只在最后给外层留出会话评分一次。
  `FOLD = all`：五会话内层留一选步、全数据训练，**只供 pp2 图**。
续跑：每个作业记录配置指纹与输入指纹（数据文件、依赖代码与环境、所用传感器层、内层评分表）；
两者都与当前一致才算已完成。`prepare` 遇到已是最新的传感器层文件不重拟合。
墙钟上限：第一次 `prepare` 记录本轮截止时间（开始时刻 + 24 小时），续跑不重置；正式入口超时即拒绝并
标记「未完成」，运行中的作业到点由闹钟中止；汇总遇到未完成或晚于截止时间的产物不作通过判决。
数据名单：载入的集必须与冻结名单（`q1_roster.json`）逐会话、逐集名一致，否则停止。

    python -u sim/q1/q1_run.py timing --steps 30
    python -u sim/q1/q1_run.py prepare   --out outputs/q1_stage1
    python -u sim/q1/q1_run.py refs      --out outputs/q1_stage1
    python -u sim/q1/q1_run.py list-jobs --out outputs/q1_stage1 --phase inner
    python -u sim/q1/q1_run.py inner     --out outputs/q1_stage1 1a ab5/A ab6/A 0
    python -u sim/q1/q1_run.py list-jobs --out outputs/q1_stage1 --phase final
    python -u sim/q1/q1_run.py final     --out outputs/q1_stage1 1a ab5/A 0
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_train as TR          # noqa: E402
import q1_windows as W         # noqa: E402

ROOT = D.ROOT
SEEDS = (0, 1, 2)
SESS = tuple(f"{s}/{a}" for s, a in D.SESSIONS)
FOLDS = SESS
ALL = "all"
REFS = ("g2", "g2_whole", "ar")
TIMING_FOLD, TIMING_VAL = "ab5/A", "ab6/A"
RUNNER_JOBS = 15                       # run_q1_stage1.sh 的默认并行数（耗时预计用）
PREREG = ROOT / "reports/PREREG_Q1_STAGE1.md"
DEP_FILES = ("sim/fit_step1_candidate.py", "lerobot/common/policies/act/step1_correction_runtime.py",
             "lerobot/common/policies/act/b_runtime.py", "sim/piper_model.xml",
             "sim/step3_canobs/ceiling_200hz.py", "sim/step2_g/ab_decompose.py", "sim/step2_gap_baseline.py",
             "sim/step4_gvnext/train_rq.py", "sim/step4_gvnext/train_rs.py", "sim/step4_gvnext/data.py",
             "sim/g2_on_ab4.py", "sim/freeze_pi_g2wall.py", "sim/step2_g/g_q_v1.py",
             "sim/step2_g/params.py", "sim/step2_g/optuna_ident_enc.py", "sim/step2_fidelity.py")
G2_FILES = ("outputs/step4_gvnext/rq/rq_fir.npz", "outputs/step4_gvnext/rq/select.json",
            "outputs/step4_gvnext/rs/rs_fir.npz")


class NotFrozenError(RuntimeError):
    """预注册不存在、未冻结，或冻结时的代码/配置指纹与当前不符。"""


class DeadlineExceeded(RuntimeError):
    """本轮已超出预注册的墙钟上限：停止，并标记「未完成」。"""


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
        raise NotFrozenError("拒绝运行：预注册记录的 config_hash 与当前选模规则不符 —— 须重新冻结")


# ── 小工具 ────────────────────────────────────────────────────────────────
def tag(s: str) -> str:
    return s.replace("/", "_")


def set_tag(sessions) -> str:
    return "+".join(sorted(tag(s) for s in sessions))


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fp_of(d) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _rel(path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def files_fp(paths) -> tuple[dict, str]:
    """→ ({相对路径: sha256}, 16 位指纹)。"""
    files = {_rel(p): file_sha(ROOT / _rel(p)) for p in paths}
    return files, fp_of(files)


def env_versions() -> dict:
    import mujoco
    import scipy
    import torch
    return {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "mujoco": mujoco.__version__}


def data_files(eps) -> list:
    return [ROOT / e.path for e in eps] + [D.CALIB_JSON]


def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def atomic_json(path: Path, obj) -> None:
    t = _tmp(path)
    t.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(t, path)


def atomic_npz(path: Path, **arrays) -> None:
    t = _tmp(path)
    with open(t, "wb") as f:
        np.savez(f, **arrays)
    os.replace(t, path)


def atomic_torch(path: Path, obj) -> None:
    import torch
    t = _tmp(path)
    torch.save(obj, t)
    os.replace(t, path)


def episode_metrics(ep, main, aux, K) -> dict:
    num, den, f3 = W.score(ep.gap, main, K)
    return {"name": ep.name, "sess": ep.sess, "num": num, "den": den, "f3": f3,
            "num_aux": W.score(ep.gap, aux, K)[0],
            "lead": W.lead_score(ep.gap, main, W.chunk_windows(ep), K)}


# ── 折与会话集合 ──────────────────────────────────────────────────────────
def train_sessions(fold: str) -> tuple:
    """外层折的训练会话；`all` = 全部 5 个会话（只供 pp2 图）。"""
    if fold == ALL:
        return SESS
    if fold not in FOLDS:
        raise ValueError(f"未知的折 {fold}")
    return tuple(s for s in SESS if s != fold)


def inner_train(fold: str, val: str) -> tuple:
    if val not in train_sessions(fold):
        raise ValueError(f"{val} 不是折 {fold} 的训练会话")
    return tuple(s for s in train_sessions(fold) if s != val)


def seeds_of(fold: str) -> tuple:
    return (0,) if fold == ALL else SEEDS


def sensor_sets() -> list:
    """所有需要传感器层的训练会话集合：外层 4 会话 × 5、内层 3 会话 × 20、全部 5 会话 × 1。"""
    sets = set()
    for f in FOLDS + (ALL,):
        sets.add(tuple(sorted(train_sessions(f))))
        for v in train_sessions(f):
            sets.add(tuple(sorted(inner_train(f, v))))
    return sorted(sets)


def is_layered(model: str) -> bool:
    return TR.MODELS[model][0] == "layered"


# ── 输入指纹与传感器层 ────────────────────────────────────────────────────
def inputs_file(out) -> Path:
    return Path(out) / "inputs.json"


def read_inputs(out) -> dict:
    p = inputs_file(out)
    if not p.exists():
        raise RuntimeError(f"缺 {p} —— 先跑 prepare")
    return json.loads(p.read_text())


def check_data(eps, I: dict) -> None:
    D.check_roster(eps)
    if files_fp(data_files(eps))[1] != I["data"]:
        raise RuntimeError("数据文件与 prepare 时不同 —— 重跑 prepare")


# ── 本轮截止时间 ──────────────────────────────────────────────────────────
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


def sensor_set_file(out, sessions) -> Path:
    return Path(out) / "sensor" / f"{set_tag(sessions)}.npz"


def fit_sensor(tr, R) -> tuple[np.ndarray, np.ndarray]:
    import q1_sensor as S
    Wrs = S.fit_rs(tr, R)
    C = S.fit_cmap(tr, R, Wrs)
    if sorted(C) != list(range(len(C))):
        raise RuntimeError("1b 的前瞻项不连续")
    return Wrs, np.stack([C[h] for h in range(len(C))])


def write_sensor(path: Path, tr, R, sessions, I: dict) -> dict:
    Wrs, C = fit_sensor(tr, R)
    man = {"train_sessions": sorted(sessions), "n_train": len(tr), "Wrs_sha256": sha(Wrs),
           "C_sha256": sha(C), "code": TR.code_sha(), "data": I["data"], "deps": I["deps"]}
    atomic_npz(path, Wrs=Wrs, C=C, manifest=json.dumps(man))
    return man


def _read_sensor(path: Path):
    if not path.exists():
        raise RuntimeError(f"缺 {path} —— 先跑 prepare")
    z = np.load(path)
    man = json.loads(str(z["manifest"]))
    if sha(z["Wrs"]) != man["Wrs_sha256"] or sha(z["C"]) != man["C_sha256"]:
        raise RuntimeError(f"{path}: 传感器层文件与清单哈希不符")
    if man["code"] != TR.code_sha():
        raise RuntimeError(f"{path}: 传感器层由另一版代码生成 —— 重跑 prepare")
    return z["Wrs"], {h: z["C"][h] for h in range(len(z["C"]))}, man


def load_sensor_set(out, sessions):
    """只用 `sessions` 拟合的传感器层（内层 = 3 个会话，外层 = 4 个，all = 5 个）；对不上即报错。"""
    Wrs, C, man = _read_sensor(sensor_set_file(out, sessions))
    if man["train_sessions"] != sorted(sessions):
        raise RuntimeError(f"传感器层的训练会话 {man['train_sessions']} ≠ {sorted(sessions)}")
    return Wrs, C, man


def sensor_is_current(path: Path, sessions, I: dict) -> bool:
    try:
        _, _, man = _read_sensor(path)
    except (RuntimeError, OSError, KeyError, ValueError):
        return False
    return (man["train_sessions"] == sorted(sessions) and man.get("data") == I["data"]
            and man.get("deps") == I["deps"])


def cmd_prepare(out, prereg=None) -> None:
    """输入指纹 + 全部训练会话集合的传感器层（`r_s` + 1b 前瞻项）；已是最新的文件不重拟合。"""
    require_frozen(prereg)
    eps = D.load_bare()
    D.check_roster(eps)
    enforce_deadline(out, create=True)             # 数据通过名单核验后才开始计时；续跑不重置
    R, _ = D.calibration()
    dfiles, dfp = files_fp(data_files(eps))
    pfiles, _ = files_fp([ROOT / f for f in DEP_FILES])
    gfiles, gfp = files_fp([ROOT / f for f in G2_FILES])
    env = env_versions()
    I = {"code_sha": TR.code_sha(), "data": dfp, "deps": fp_of({"files": pfiles, "env": env}), "g2": gfp,
         "data_files": dfiles, "deps_files": pfiles, "env": env, "g2_files": gfiles, "sensor": {},
         "roster": {s: sorted(e.name for e in eps if e.sess == s) for s in SESS}}
    for ss in sensor_sets():
        path = sensor_set_file(out, ss)
        if not sensor_is_current(path, ss, I):
            man = write_sensor(path, [e for e in eps if e.sess in ss], R, ss, I)
            print(f"  传感器层 {set_tag(ss)}：{man['n_train']} 集", flush=True)
        I["sensor"][set_tag(ss)] = file_sha(path)
    atomic_json(inputs_file(out), I)
    print(f"  输入指纹：数据 {dfp} · 依赖 {I['deps']} · G-2 {gfp} · 传感器层 {len(I['sensor'])} 份", flush=True)


# ── 作业路径与期望输入 ────────────────────────────────────────────────────
def inner_json(out, model: str, fold: str, val: str, seed: int) -> Path:
    return Path(out) / "inner" / model / f"seed{seed}" / f"{tag(fold)}__val_{tag(val)}.json"


def final_base(out, model: str, fold: str, seed: int) -> Path:
    """终训产物（无后缀）：外层折 → `jobs/…/{fold}`；`all` → `fullfit/{model}`（只供作图）。"""
    if fold == ALL:
        return Path(out) / "fullfit" / model
    return Path(out) / "jobs" / model / f"seed{seed}" / tag(fold)


def inner_inputs(I: dict, model: str, fold: str, val: str) -> dict:
    s = I["sensor"][set_tag(inner_train(fold, val))] if is_layered(model) else None
    return {"data": I["data"], "deps": I["deps"], "sensor": s}


def final_inputs(out, I: dict, model: str, fold: str, seed: int):
    """终训的期望输入；任一内层评分表缺失 ⇒ None。"""
    inner = {}
    for v in train_sessions(fold):
        p = inner_json(out, model, fold, v, seed)
        if not p.exists():
            return None
        inner[tag(v)] = file_sha(p)
    s = I["sensor"][set_tag(train_sessions(fold))] if is_layered(model) else None
    return {"data": I["data"], "deps": I["deps"], "sensor": s, "inner": inner}


def refs_inputs(I: dict) -> dict:
    return {"data": I["data"], "deps": I["deps"], "g2": I["g2"]}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_done(path: Path, inputs) -> bool:
    j = _load(path) if inputs is not None else None
    return bool(j) and j.get("config_hash") == TR.config_hash() and j.get("input_fp") == fp_of(inputs)


def cmd_list_jobs(out, phase: str) -> None:
    I = read_inputs(out)
    for model in TR.MODELS:
        for fold in FOLDS + (ALL,):
            for seed in seeds_of(fold):
                if phase == "inner":
                    for val in train_sessions(fold):
                        if not is_done(inner_json(out, model, fold, val, seed),
                                       inner_inputs(I, model, fold, val)):
                            print(f"{model} {fold} {val} {seed}")
                elif not is_done(final_base(out, model, fold, seed).with_suffix(".json"),
                                 final_inputs(out, I, model, fold, seed)):
                    print(f"{model} {fold} {seed}")


# ── 内层：训到上限，给内层验证会话的每个检查点评分 ────────────────────────
def cmd_inner(out, model: str, fold: str, val: str, seed: int, prereg=None) -> None:
    require_frozen(prereg)
    enforce_deadline(out)
    tr_s = inner_train(fold, val)
    I = read_inputs(out)
    eps = D.load_bare()
    check_data(eps, I)
    R, _ = D.calibration()
    tr = [e for e in eps if e.sess in tr_s]
    va = [e for e in eps if e.sess == val]
    Wrs = C = None
    if is_layered(model):
        Wrs, C, _ = load_sensor_set(out, tr_s)
    res = TR.train_one(model, tr, seed, log=lambda m: print(m, flush=True))
    scores = {}
    for step in sorted(res["ckpt"]):
        sd = res["ckpt"][step]
        m = TR.with_weights(model, res["stats"], sd)
        num, den = TR.session_scores(model, m, res["stats"], va, R, Wrs, C)[val]
        scores[str(step)] = {"num": num, "den": den, "rho": num / den, "state_sha256": TR.state_dict_sha(sd)}
    inputs = inner_inputs(I, model, fold, val)
    atomic_json(inner_json(out, model, fold, val, seed),
                {"model": model, "fold": fold, "val": val, "seed": seed,
                 "config_hash": TR.config_hash(), "code_sha": TR.code_sha(),
                 "inputs": inputs, "input_fp": fp_of(inputs), "train_sessions": list(tr_s),
                 "n_val": len(va), "scores": scores, "sec": res["sec"],
                 "loss_tail": float(np.mean(res["loss"][-100:])) if res["loss"] else None,
                 "finished_at": time.time()})
    print(f"  内层完成 {model} {fold} 验证 {val} seed{seed}  {res['sec']:.0f}s", flush=True)


# ── 终训：内层选步 → 训到 step* → 外层留出会话只评一次 ─────────────────────
def selection(out, model: str, fold: str, seed: int):
    """→ (V {step: 等权平均 ρ}, step*)。内层评分表缺失、过期或不完整 ⇒ RuntimeError。"""
    I = read_inputs(out)
    ck = list(TR.CONFIG["ckpts"])
    rho = {}
    for v in train_sessions(fold):
        p = inner_json(out, model, fold, v, seed)
        j = _load(p)
        if j is None:
            raise RuntimeError(f"缺内层评分表 {p} —— 先跑 inner")
        if j.get("config_hash") != TR.config_hash() or j.get("input_fp") != fp_of(inner_inputs(I, model, fold, v)):
            raise RuntimeError(f"{p}: 内层评分表的配置或输入指纹已过期 —— 重跑 inner")
        if j["train_sessions"] != list(inner_train(fold, v)) or sorted(int(s) for s in j["scores"]) != ck:
            raise RuntimeError(f"{p}: 内层评分表与选步规则不符")
        rho[v] = {int(s): float(r["rho"]) for s, r in j["scores"].items()}
    V = {s: float(np.mean([rho[v][s] for v in train_sessions(fold)])) for s in ck}
    return V, TR.pick_step(V)


def cmd_final(out, model: str, fold: str, seed: int, prereg=None) -> None:
    require_frozen(prereg)
    enforce_deadline(out)
    I = read_inputs(out)
    V, step = selection(out, model, fold, seed)
    inputs = final_inputs(out, I, model, fold, seed)
    eps = D.load_bare()
    check_data(eps, I)
    R, _ = D.calibration()
    K = W.kernel()
    vals = train_sessions(fold)
    tr = [e for e in eps if e.sess in vals]
    Wrs = C = None
    if is_layered(model):
        Wrs, C, _ = load_sensor_set(out, vals)
    res = TR.train_one(model, tr, seed, ckpts=(step,), log=lambda m: print(m, flush=True))
    sd = res["ckpt"][step]
    boundary = step == TR.CONFIG["max_steps"]
    base = final_base(out, model, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"model_id": model, "state_dict": sd, "stats": res["stats"],
                                           "norms": res["norms"], "step": step,
                                           "config_hash": TR.config_hash(), "code_sha": TR.code_sha()})
    rec = {"model": model, "fold": fold, "seed": seed, "config_hash": TR.config_hash(),
           "code_sha": TR.code_sha(), "inputs": inputs, "input_fp": fp_of(inputs),
           "train_sessions": list(vals), "V": {str(s): V[s] for s in sorted(V)}, "step_star": step,
           "boundary": boundary, "state_sha256": TR.state_dict_sha(sd),
           "pt_sha256": file_sha(base.with_suffix(".pt")), "sec": res["sec"]}
    if fold != ALL:
        m = TR.with_weights(model, res["stats"], sd)
        rows, preds = [], {}
        for e in (x for x in eps if x.sess == fold):          # 外层留出会话：只在最后评分一次
            ws, mws = W.chunk_windows(e), W.mid_windows(e)
            main = W.stitch(e.n, ws, TR.predict_windows(model, m, res["stats"], e, ws, R, Wrs, C))
            aux = W.overlay(main, mws, TR.predict_windows(model, m, res["stats"], e, mws, R, Wrs, C))
            rows.append(episode_metrics(e, main, aux, K))
            preds[e.name.replace("/", "__")] = main
        atomic_npz(base.with_suffix(".npz"), **preds)
        rec.update({"npz_sha256": file_sha(base.with_suffix(".npz")), "episodes": rows})
    rec["finished_at"] = time.time()
    atomic_json(base.with_suffix(".json"), rec)
    note = "（最佳点落在搜索边界，尚不能确认训练充分）" if boundary else ""
    print(f"  终训完成 {model} {fold} seed{seed}：step* = {step}{note}", flush=True)


def load_fullfit(out, model_id: str):
    """pp2 图用的全数据模型（`final … all 0` 的产物）：代码与配置指纹、step* 记录必须一致。"""
    import torch
    base = final_base(out, model_id, ALL, 0)
    p, jp = base.with_suffix(".pt"), base.with_suffix(".json")
    if not p.exists() or not jp.exists():
        raise RuntimeError(f"缺 {p} —— 先跑 final {model_id} all 0")
    ck, j = torch.load(p, weights_only=False), json.loads(jp.read_text())
    if (ck.get("code_sha") != TR.code_sha() or ck.get("config_hash") != TR.config_hash()
            or j.get("config_hash") != TR.config_hash() or ck["step"] != j["step_star"]
            or j["pt_sha256"] != file_sha(p)):
        raise RuntimeError(f"{p}: 与当前代码、选模规则或选步记录不符 —— 重跑")
    model = TR.with_weights(model_id, ck["stats"], ck["state_dict"])
    model.eval()
    Wrs = C = None
    if is_layered(model_id):
        Wrs, C, _ = load_sensor_set(out, train_sessions(ALL))
    return model, ck["stats"], Wrs, C


# ── 参照 ──────────────────────────────────────────────────────────────────
def _ref_json(out, name: str, fold: str) -> Path:
    return Path(out) / "refs" / name / f"{tag(fold)}.json"


def cmd_refs(out, prereg=None) -> None:
    """G-2 同口径重预测、旧口径整集 G-2（只对照）、折内 AR 参照；输入未变的折跳过。"""
    require_frozen(prereg)
    enforce_deadline(out)
    I = read_inputs(out)
    inputs = refs_inputs(I)
    fp = fp_of(inputs)

    def current(name, fold):
        j = _load(_ref_json(out, name, fold))
        return bool(j) and j.get("code_sha") == TR.code_sha() and j.get("input_fp") == fp

    todo = [f for f in FOLDS if not all(current(n, f) for n in REFS)]
    if not todo:
        print("  参照均为最新，跳过", flush=True)
        return
    import q1_ar as AR
    import q1_g2ref as G
    eps = D.load_bare()
    check_data(eps, I)
    R, b = D.calibration()
    K = W.kernel()
    g2 = G.load_g2()
    for fold in todo:
        tr, te = TR.split_fold(eps, fold)
        coef = AR.fit(tr)
        rows = {n: [] for n in REFS}
        for e in te:
            ws, mws = W.chunk_windows(e), W.mid_windows(e)
            g = G.episode_ehat(g2, e, ws, R, b)
            rows["g2"].append(episode_metrics(e, g, g, K))     # 不读实测 ⇒ 中段起点预测与主窗口逐位相同
            wh = G.whole_episode_ehat(g2, e, R, b)
            rows["g2_whole"].append(episode_metrics(e, wh, wh, K))
            a = W.stitch(e.n, ws, [AR.window_ehat(coef, e, w) for w in ws])
            rows["ar"].append(episode_metrics(
                e, a, W.overlay(a, mws, [AR.window_ehat(coef, e, w) for w in mws]), K))
        for name, r in rows.items():
            atomic_json(_ref_json(out, name, fold),
                        {"model": name, "fold": fold, "seed": "-", "config_hash": TR.code_sha()[:16],
                         "code_sha": TR.code_sha(), "inputs": inputs, "input_fp": fp,
                         "train_sessions": sorted({e.sess for e in tr}) if name == "ar" else [],
                         "episodes": r, "finished_at": time.time()})
        print(f"  参照 {fold} 完成", flush=True)


# ── 耗时测量（不经冻结闸门；不预测外层留出会话、不存权重）───────────────────
def cmd_timing(steps: int) -> None:
    eps = D.load_bare()
    R, _ = D.calibration()
    tr_s = inner_train(TIMING_FOLD, TIMING_VAL)
    tr = [e for e in eps if e.sess in tr_s]
    va = [e for e in eps if e.sess == TIMING_VAL]
    Wrs, Ca = fit_sensor(tr, R)                                   # 只为计时，不写文件
    C = {h: Ca[h] for h in range(len(Ca))}
    per_step, per_ep = {}, {}
    for model in TR.MODELS:
        res = TR.train_one(model, tr, 0, ckpts=(steps,))
        per_step[model] = res["sec"] / steps
        m = TR.with_weights(model, res["stats"], res["ckpt"][steps])
        lay = is_layered(model)
        t = time.time()
        TR.session_scores(model, m, res["stats"], va, R, Wrs if lay else None, C if lay else None)
        per_ep[model] = (time.time() - t) / len(va)
        print(f"  {model}: {per_step[model]:.3f} s/步 · 检查点评分 {per_ep[model]:.3f} s/集", flush=True)
    n_sess = {s: sum(e.sess == s for e in eps) for s in SESS}
    S_, n_ck, load = TR.CONFIG["max_steps"], len(TR.CONFIG["ckpts"]), 60.0
    n_inner = sum(len(train_sessions(f)) * len(seeds_of(f)) for f in FOLDS + (ALL,))
    n_final = sum(len(seeds_of(f)) for f in FOLDS + (ALL,))
    val_eps = sum(n_sess[v] * len(seeds_of(f)) for f in FOLDS + (ALL,) for v in train_sessions(f))
    held_eps = len(SEEDS) * sum(n_sess.values())
    total = 0.0
    for model in TR.MODELS:
        inner = n_inner * (S_ * per_step[model] + load) + n_ck * val_eps * per_ep[model]
        final = n_final * (S_ * per_step[model] + load) + 2 * held_eps * per_ep[model]
        total += inner + final
        print(f"  {model}：内层 {n_inner} 次 ≈ {inner / 3600:.1f} 核·小时，终训 {n_final} 次 ≤ "
              f"{final / 3600:.1f} 核·小时（按 step* = 上限计）", flush=True)
    longest = max(S_ * per_step[m] + n_ck * max(n_sess.values()) * per_ep[m] + load for m in TR.MODELS)
    print(f"  合计 ≤ {total / 3600:.1f} 核·小时（含检查点评分与载入）；最长单作业 ≈ {longest / 60:.0f} 分钟；"
          f"{RUNNER_JOBS} 路并行（运行脚本默认）墙钟 ≈ {total / 3600 / RUNNER_JOBS:.1f} 小时 + 参照/作图")


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "refs"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--prereg", default=None)
    p = sub.add_parser("list-jobs")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", required=True, choices=("inner", "final"))
    p = sub.add_parser("inner")
    p.add_argument("--out", required=True)
    p.add_argument("--prereg", default=None)
    p.add_argument("model", choices=list(TR.MODELS))
    p.add_argument("fold", choices=FOLDS + (ALL,))
    p.add_argument("val", choices=SESS)
    p.add_argument("seed", type=int, choices=SEEDS)
    p = sub.add_parser("final")
    p.add_argument("--out", required=True)
    p.add_argument("--prereg", default=None)
    p.add_argument("model", choices=list(TR.MODELS))
    p.add_argument("fold", choices=FOLDS + (ALL,))
    p.add_argument("seed", type=int, choices=SEEDS)
    p = sub.add_parser("timing")
    p.add_argument("--steps", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            cmd_prepare(a.out, a.prereg)
        elif a.cmd == "refs":
            cmd_refs(a.out, a.prereg)
        elif a.cmd == "list-jobs":
            cmd_list_jobs(a.out, a.phase)
        elif a.cmd == "inner":
            cmd_inner(a.out, a.model, a.fold, a.val, a.seed, a.prereg)
        elif a.cmd == "final":
            cmd_final(a.out, a.model, a.fold, a.seed, a.prereg)
        else:
            cmd_timing(a.steps)
    except NotFrozenError as err:
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
