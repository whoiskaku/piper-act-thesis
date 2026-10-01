"""阶段一命令行入口。

⚠ `prepare` / `job` / `refs` / `fullfit` 是**正式运行**：入口先查 `reports/PREREG_Q1_STAGE1.md`
—— 状态行须以「**状态：已冻结」开头，且记录的 `code_sha=` 与当前代码一致，否则拒绝运行。
`timing` 不经过这道闸门：只测每步耗时（在 ab5/A 折的训练会话上训几十步，不预测留出、不存权重），
用来填预注册的预算。

    python -u sim/q1/q1_run.py timing --steps 30
    python -u sim/q1/q1_run.py prepare   --out outputs/q1_stage1
    python -u sim/q1/q1_run.py list-jobs --out outputs/q1_stage1
    python -u sim/q1/q1_run.py job       --out outputs/q1_stage1 1a ab5/A 0
    python -u sim/q1/q1_run.py refs      --out outputs/q1_stage1
    python -u sim/q1/q1_run.py fullfit   --out outputs/q1_stage1 2b
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_train as TR          # noqa: E402
import q1_windows as W         # noqa: E402

SEEDS = (0, 1, 2)
FOLDS = tuple(f"{s}/{a}" for s, a in D.SESSIONS)
TIMING_FOLD = "ab5/A"
PREREG = D.ROOT / "reports/PREREG_Q1_STAGE1.md"


class NotFrozenError(RuntimeError):
    """预注册不存在、未冻结，或冻结时的代码指纹与当前代码不符。"""


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


def tag(fold: str) -> str:
    return fold.replace("/", "_")


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def sensor_file(out, fold: str) -> Path:
    return Path(out) / "sensor" / f"{tag(fold)}.npz"


def fit_sensor(tr, R) -> tuple[np.ndarray, np.ndarray]:
    import q1_sensor as S
    Wrs = S.fit_rs(tr, R)
    C = S.fit_cmap(tr, R, Wrs)
    if sorted(C) != list(range(len(C))):
        raise RuntimeError("1b 的前瞻项不连续")
    return Wrs, np.stack([C[h] for h in range(len(C))])


def cmd_prepare(out, prereg=None) -> None:
    """每折的传感器层（`r_s` + 1b 前瞻项），只用该折 4 个训练会话。"""
    require_frozen(prereg)
    eps = D.load_bare()
    R, _ = D.calibration()
    for fold in FOLDS:
        tr, _ = TR.split_fold(eps, fold)
        Wrs, C = fit_sensor(tr, R)
        man = {"fold": fold, "train_sessions": sorted({e.sess for e in tr}), "n_train": len(tr),
               "Wrs_sha256": sha(Wrs), "C_sha256": sha(C), "code": TR.code_sha()}
        atomic_npz(sensor_file(out, fold), Wrs=Wrs, C=C, manifest=json.dumps(man))
        print(f"  传感器层 {fold}: 训练 {man['train_sessions']} ({len(tr)} 集)", flush=True)


def load_sensor(out, fold: str):
    z = np.load(sensor_file(out, fold))
    man = json.loads(str(z["manifest"]))
    if man["fold"] != fold or fold in man["train_sessions"]:
        raise RuntimeError(f"{fold}: 留出会话进入了传感器层拟合 / 文件错配")
    if sha(z["Wrs"]) != man["Wrs_sha256"] or sha(z["C"]) != man["C_sha256"]:
        raise RuntimeError(f"{fold}: 传感器层文件与清单哈希不符")
    if man["code"] != TR.code_sha():
        raise RuntimeError(f"{fold}: 传感器层由另一版代码生成 —— 重跑 prepare")
    return z["Wrs"], {h: z["C"][h] for h in range(len(z["C"]))}, man


def job_json(out, model: str, fold: str, seed: int) -> Path:
    return Path(out) / "jobs" / model / f"seed{seed}" / f"{tag(fold)}.json"


def is_done(path: Path, chash: str) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("config_hash") == chash
    except json.JSONDecodeError:
        return False


def cmd_list_jobs(out, steps: int) -> None:
    ch = TR.config_hash(steps)
    for model in TR.MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                if not is_done(job_json(out, model, fold, seed), ch):
                    print(f"{model} {fold} {seed}")


def predict_main_aux(model_id, res, e, R, Wrs, C):
    ws, mws = W.chunk_windows(e), W.mid_windows(e)
    main = W.stitch(e.n, ws, TR.predict_windows(model_id, res["model"], res["stats"], e, ws, R, Wrs, C))
    aux = W.overlay(main, mws, TR.predict_windows(model_id, res["model"], res["stats"], e, mws, R, Wrs, C))
    return main, aux


def cmd_job(out, model: str, fold: str, seed: int, steps: int, prereg=None) -> None:
    require_frozen(prereg)
    eps = D.load_bare()
    R, _ = D.calibration()
    K = W.kernel()
    tr, te = TR.split_fold(eps, fold)
    Wrs = C = None
    if TR.MODELS[model][0] == "layered":
        Wrs, C, _ = load_sensor(out, fold)
    res = TR.train_one(model, tr, seed, steps, log=lambda m: print(m, flush=True))
    rows, preds = [], {}
    for e in te:                                     # 留出会话：只在最后预测一次
        main, aux = predict_main_aux(model, res, e, R, Wrs, C)
        rows.append(episode_metrics(e, main, aux, K))
        preds[e.name.replace("/", "__")] = main
    base = job_json(out, model, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"model_id": model, "state_dict": res["model"].state_dict(),
                                           "stats": res["stats"], "norms": res["norms"]})
    atomic_npz(base.with_suffix(".npz"), **preds)
    atomic_json(base, {"model": model, "fold": fold, "seed": seed, "steps": steps,
                       "config_hash": TR.config_hash(steps), "code_sha": TR.code_sha(),
                       "state_sha256": TR.state_sha256(res["model"]),
                       "pt_sha256": file_sha(base.with_suffix(".pt")),
                       "npz_sha256": file_sha(base.with_suffix(".npz")),
                       "train_sessions": sorted({e.sess for e in tr}), "sec": res["sec"],
                       "loss_tail": float(np.mean(res["loss"][-100:])), "episodes": rows})
    print(f"  完成 {model} {fold} seed{seed}  {res['sec']:.0f}s", flush=True)


def cmd_refs(out, prereg=None) -> None:
    """G-2 同口径重预测、旧口径整集 G-2（只对照）、折内 AR 参照。"""
    require_frozen(prereg)
    import q1_ar as AR
    import q1_g2ref as G
    eps = D.load_bare()
    R, b = D.calibration()
    K = W.kernel()
    g2 = G.load_g2()
    ch = TR.code_sha()[:16]
    for fold in FOLDS:
        tr, te = TR.split_fold(eps, fold)
        coef = AR.fit(tr)
        rows = {"g2": [], "g2_whole": [], "ar": []}
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
            atomic_json(Path(out) / "refs" / name / f"{tag(fold)}.json",
                        {"model": name, "fold": fold, "seed": "-", "config_hash": ch,
                         "code_sha": TR.code_sha(),
                         "train_sessions": sorted({e.sess for e in tr}) if name == "ar" else [],
                         "episodes": r})
        print(f"  参照 {fold} 完成", flush=True)


def cmd_timing(steps: int) -> None:
    eps = D.load_bare()
    tr, _ = TR.split_fold(eps, TIMING_FOLD)
    per = {}
    for model in TR.MODELS:
        per[model] = TR.train_one(model, tr, 0, steps)["sec"] / steps
        print(f"  {model}: {per[model]:.3f} s/步", flush=True)
    n = len(FOLDS) * len(SEEDS)
    S_ = TR.CONFIG["steps"]
    core_h = sum(per.values()) * S_ * n / 3600
    print(f"  预计 {S_} 步 × {n} 个（折×种子）× 6 模型 ≈ {core_h:.1f} 核·小时"
          f"（另加每个作业载入数据约 1 分钟）；最长单作业 ≈ {max(per.values()) * S_ / 60:.0f} 分钟")


def cmd_fullfit(out, model: str, steps: int, prereg=None) -> None:
    """**只供 pp2 波形图**：用全部 100 集训练（seed 0），不进任何判据。"""
    require_frozen(prereg)
    eps = D.load_bare()
    R, _ = D.calibration()
    fdir = Path(out) / "fullfit"
    if TR.MODELS[model][0] == "layered":
        sp = fdir / "sensor_all.npz"
        if not sp.exists():
            Wrs, C = fit_sensor(eps, R)
            atomic_npz(sp, Wrs=Wrs, C=C, code=np.asarray(TR.code_sha()))
    res = TR.train_one(model, eps, 0, steps, log=lambda m: print(m, flush=True))
    atomic_torch(fdir / f"{model}.pt", {"model_id": model, "state_dict": res["model"].state_dict(),
                                        "stats": res["stats"], "config_hash": TR.config_hash(steps)})


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "list-jobs", "refs"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=TR.CONFIG["steps"])
        p.add_argument("--prereg", default=None)
    p = sub.add_parser("job")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=TR.CONFIG["steps"])
    p.add_argument("--prereg", default=None)
    p.add_argument("model", choices=list(TR.MODELS))
    p.add_argument("fold", choices=FOLDS)
    p.add_argument("seed", type=int, choices=SEEDS)
    p = sub.add_parser("fullfit")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=TR.CONFIG["steps"])
    p.add_argument("--prereg", default=None)
    p.add_argument("model", choices=list(TR.MODELS))
    p = sub.add_parser("timing")
    p.add_argument("--steps", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            cmd_prepare(a.out, a.prereg)
        elif a.cmd == "list-jobs":
            cmd_list_jobs(a.out, a.steps)
        elif a.cmd == "job":
            cmd_job(a.out, a.model, a.fold, a.seed, a.steps, a.prereg)
        elif a.cmd == "refs":
            cmd_refs(a.out, a.prereg)
        elif a.cmd == "fullfit":
            cmd_fullfit(a.out, a.model, a.steps, a.prereg)
        else:
            cmd_timing(a.steps)
    except NotFrozenError as err:
        print(err, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
