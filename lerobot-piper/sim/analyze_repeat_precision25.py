#!/usr/bin/env python3
"""重复精度封顶验证（25 次）的分析与判定 —— `reports/PLAN_REPEAT_PRECISION_CAPPED.md` 第 2–6 节，实施计划第 4 节。

- 先核验：代码指纹与 release 一致（不符即拒绝运行）；25 条全部完成、记录哈希与账本一致、逐条有效性复核。
  后两项任一不满足 ⇒ 「无法判定」，只报告执行现状，不计算主指标。
- 主分析：`precision_chain.frame_locked` → `P(·)`（A0 不参与）；σ_c、σ_bare；分半复现度 R_c（两半各用各的裸基线）。
- 判定：纯函数 `verdict(complete, R_c)`，只看四个 R_c；阈值 0.6 / 0.3 是工程阈值，不是显著性。
  「支持」只表示本套分半设计下的复现度达到工程阈值，不表示总体测量精度或 1/√n 次数公式已经验证。
- 辅助报告在判定之后计算，不能流入判定。

    $PY -u sim/analyze_repeat_precision25.py      # → outputs/repeat_precision25_analysis/analysis.json（打印 sha16）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# 本机 OpenBLAS 默认 20 线程：负载下一次 182×182 求解（filtfilt 的初值）要约 3.5 s，并空转约 11 个核。
# 在 numpy 载入之前钉住单线程（只影响本进程）。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import run_repeat_precision25 as R              # noqa: E402

OUT_DIR = R.ROOT / "outputs/repeat_precision25_analysis"
SCHEMA = "repeat-precision25-analysis-v1"
SPLITS = (((1, 3), (2, 4)), ((1, 4), (2, 3)))   # 主分析：两种交错分法
FRONT_BACK = ((1, 2), (3, 4))                    # 前半对后半：只作漂移的辅助报告
CONDITIONS = ("P+", "P-", "Q+", "Q-")
THRESH_SUPPORT, THRESH_STOP = 0.6, 0.3
LABELS = {"support": "支持另行设计正式采集", "stop": "停止推进", "indeterminate": "无法判定"}


# ── 分半与指标 ────────────────────────────────────────────────────────────────
def halves(runs, blocks) -> dict:
    """只由给定区组构成的一半：每个条件的复放与本半自己的裸复放。"""
    sel = [r for r in runs if r["block"] in blocks]
    h = {c: [r["run_id"] for r in sel if r["condition"] == c] for c in CONDITIONS}
    h["bare"] = [r["run_id"] for r in sel if r["condition"] == "bare"]
    if any(len(h[c]) != len(blocks) for c in CONDITIONS) or len(h["bare"]) != 2 * len(blocks):
        raise ValueError("分半的构成与区组表不符")
    return h


def split_pair(runs, split) -> tuple:
    a, b = halves(runs, split[0]), halves(runs, split[1])
    ra = {x for v in a.values() for x in v}
    rb = {x for v in b.values() for x in v}
    if not ra.isdisjoint(rb):
        raise ValueError("两半共享了复放")
    return a, b


def r_half(X, h, cond) -> np.ndarray:
    """一半的响应估计 = 该半同条件复放的均值 − 该半裸复放的均值。"""
    return np.mean([X[r] for r in h[cond]], 0) - np.mean([X[r] for r in h["bare"]], 0)


def split_R(ra, rb) -> float:
    den = float(np.sum(np.square(ra)) + np.sum(np.square(rb)))
    if den == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.square(ra - rb))) / den


def _pooled(xs) -> float:
    S = np.stack([np.asarray(x, np.float64) for x in xs])
    d = S - S.mean(0)
    return float(np.sqrt(np.square(d).sum() / ((len(S) - 1) * d[0].size)))


def main_metrics(X, runs) -> dict:
    by = {c: [r["run_id"] for r in runs if r["condition"] == c] for c in (*CONDITIONS, "bare")}
    sig = {c: _pooled([X[k] for k in by[c]]) for c in CONDITIONS}
    out = {"sigma_c": sig, "df_c": {c: len(by[c]) - 1 for c in CONDITIONS},
           "sigma_bare": _pooled([X[k] for k in by["bare"]]), "df_bare": len(by["bare"]) - 1,
           "sigma_pool": float(np.sqrt(np.mean([sig[c] ** 2 for c in CONDITIONS]))),
           "df_pool": sum(len(by[c]) - 1 for c in CONDITIONS), "R": {}, "R_c": {}}
    for c in CONDITIONS:
        vals = []
        for split in SPLITS:
            a, b = split_pair(runs, split)
            vals.append(split_R(r_half(X, a, c), r_half(X, b, c)))
        out["R"][c] = {"splits": vals, "split_blocks": [list(map(list, s)) for s in SPLITS]}
        out["R_c"][c] = float(np.mean(vals))
    return out


def verdict(complete, R_c) -> dict:
    """三种出口；只接收「执行是否完整」和四个 R_c。"""
    def out(v, reason, **kw):
        return dict(verdict=v, label=LABELS[v], reason=reason, **kw)
    if not complete or R_c is None:
        return out("indeterminate", "执行不完整或有记录无效")
    vals = [R_c[c] for c in CONDITIONS]
    if not all(math.isfinite(v) for v in vals):
        return out("indeterminate", "有条件的 R_c 非有限")
    mean = math.fsum(vals) / len(vals)
    if all(v >= THRESH_SUPPORT for v in vals):
        return out("support", "四个条件的 R_c 都 ≥ 0.6", R_c_mean=mean)
    if mean < THRESH_STOP:
        return out("stop", "四个条件 R_c 的平均 < 0.3", R_c_mean=mean)
    return out("indeterminate", "既不是四个都 ≥ 0.6，平均也不 < 0.3", R_c_mean=mean)


# ── 核验与处理 ────────────────────────────────────────────────────────────────
def verify(root, release) -> dict:
    """代码指纹不符 ⇒ 抛 ValueError（拒绝运行）；其余问题都记为原因（⇒ 无法判定）。"""
    root, release = Path(root), Path(release)
    R.OUT, R.RELEASE = root, release                                    # 入口模块的核验函数按这两个路径工作
    rel = json.loads(release.read_text())
    if rel.get("code_sha256") != R.code_fingerprints():
        raise ValueError("代码指纹与 release 锁定的不符：拒绝运行分析")
    try:
        _, m = R.load_frozen(require_motion=False)
    except (ValueError, KeyError, OSError) as e:
        return dict(complete=False, reasons=[f"冻结输入或 manifest 核验失败: {e}"], attempts=[])
    sp = root / "session.json"
    if not sp.exists():
        return dict(complete=False, reasons=["没有会话账本"], attempts=[], manifest=m)
    s = json.loads(sp.read_text())
    reasons = []
    if (s.get("run_set_id") != R.RUN_SET_ID or s.get("release_sha256") != R.sha(release)
            or s.get("manifest_sha256") != R.sha(root / "manifest.json")):
        reasons.append("账本与 release / manifest 的绑定不符")
    att = s.get("attempts", [])
    if s.get("halted"):
        reasons.append("会话已锁停")
    if len(att) != len(m["runs"]):
        reasons.append(f"只完成 {len(att)} / {len(m['runs'])} 条尝试")
    checks = {}
    for a, run in zip(att, m["runs"]):
        rid = run["run_id"]
        if a.get("run_id") != rid:
            reasons.append(f"{rid}: 账本顺序不符")
            continue
        if a.get("status") != "complete":
            reasons.append(f"{rid}: 状态 {a.get('status')}")
            continue
        p = root / "records" / f"replay_{rid}.json"
        if not p.exists() or R.sha(p) != a.get("record_sha256"):
            reasons.append(f"{rid}: 记录哈希与账本不符")
            continue
        try:
            checks[rid] = R.check_record(run, m)
        except (ValueError, KeyError) as e:
            reasons.append(f"{rid}: 记录无效: {e}")
    summary = [dict(run_id=a.get("run_id"), status=a.get("status"), exit_code=a.get("exit_code"),
                    exception=a.get("exception")) for a in att]
    return dict(complete=not reasons, reasons=reasons, attempts=summary, record_checks=checks, manifest=m, session=s)


def process(recs, runs) -> dict:
    import precision_chain as C
    import q1_windows as W
    K = W.kernel()
    return {r["run_id"]: W.P(C.frame_locked(recs[r["run_id"]]), K) for r in runs if r["condition"] != "A0"}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def analyze(root=None, release=None) -> dict:
    import precision_chain as C
    root, release = Path(root or R.OUT), Path(release or R.RELEASE)
    v = verify(root, release)
    out = dict(schema=SCHEMA, run_set_id=R.RUN_SET_ID, release_sha256=R.sha(release),
               chain=dict(fir_sha16=C.FIR_SHA16, offset_s=C.OFFSET_S, numtaps=C.NUMTAPS, padlen=C.PADLEN),
               execution=dict(complete=v["complete"], reasons=v["reasons"], attempts=v["attempts"],
                              record_checks=v.get("record_checks", {})))
    if not v["complete"]:
        out["verdict"] = verdict(False, None)
        return _jsonable(out)
    runs = v["manifest"]["runs"]
    recs = {r["run_id"]: json.loads((root / "records" / f"replay_{r['run_id']}.json").read_text()) for r in runs}
    X = process(recs, runs)
    mm = main_metrics(X, runs)
    out["main"] = mm
    out["verdict"] = verdict(True, mm["R_c"])
    return _jsonable(out)


def main(argv=None) -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=R.OUT)
    ap.add_argument("--release", type=Path, default=R.RELEASE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    try:
        res = analyze(a.root, a.release)
    except (OSError, ValueError, KeyError) as e:
        print(f"拒绝：{type(e).__name__}: {e}", file=sys.stderr)
        return 4
    a.out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(res, ensure_ascii=False, indent=1, allow_nan=False, sort_keys=True) + "\n"
    (a.out / "analysis.json").write_text(payload)
    print(f"判定：{res['verdict']['label']}（{res['verdict']['reason']}）")
    print(f"analysis.json sha16 {hashlib.sha256(payload.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
