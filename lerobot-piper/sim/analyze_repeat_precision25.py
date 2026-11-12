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
PILOT_ROOT = R.ROOT / "outputs/pilot_response14"                 # 跨日对比：只读
PILOT_RUNS = {"P+": ("BP_p_r1", "BP_p_r2"), "P-": ("BP_m_r1", "BP_m_r2"),
              "Q+": ("BQ_p_r1", "BQ_p_r2"), "Q-": ("BQ_m_r1", "BQ_m_r2")}
PILOT_BARE = ("A1", "A2", "A3", "A4", "A5")
OFFSETS_MS = (0, 10, 20, 30, 40)                                    # 帧锁定偏移（辅助）
SPIKE_SIGMAS = 4.0                                                  # 尖峰 / 大偏差帧阈值 = 4 σ_pool（主分析口径）


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


# ── 辅助报告（方案第 6 节）：在判定之后计算，不能流入判定 ──────────────────────
def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _corr(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _full_response(X, runs, cond) -> np.ndarray:
    cr = [r["run_id"] for r in runs if r["condition"] == cond]
    br = [r["run_id"] for r in runs if r["condition"] == "bare"]
    return np.mean([X[k] for k in cr], 0) - np.mean([X[k] for k in br], 0)


def kappa(X, runs, mm) -> dict:
    """两半之差的实测功率 / 按 σ_c、σ_bare 的预测。忽略裸噪声时按构造 ≤ 1.5（方案第三节）——只描述。"""
    meas, pred = {}, {}
    for c in CONDITIONS:
        ms, pr = [], []
        for split in SPLITS:
            a, b = split_pair(runs, split)
            ms.append(float(np.mean(np.square(r_half(X, a, c) - r_half(X, b, c)))))
            pr.append(mm["sigma_c"][c] ** 2 * (1 / len(a[c]) + 1 / len(b[c]))
                      + mm["sigma_bare"] ** 2 * (1 / len(a["bare"]) + 1 / len(b["bare"])))
        meas[c], pred[c] = float(np.mean(ms)), float(np.mean(pr))
    kap = {c: (meas[c] / pred[c] if pred[c] > 0 else float("nan")) for c in CONDITIONS}
    tp = sum(pred.values())
    return {"kappa": kap, "kappa_pool": (sum(meas.values()) / tp if tp > 0 else float("nan")),
            "measured": meas, "predicted": pred,
            "note": "代数耦合：忽略裸基线噪声时按构造 ≤ 1.5；κ≈1 不能证明独立噪声模型，也不能保证 1/√n 推算"}


def plus_minus(X, runs) -> dict:
    """r₊ 与 −r₋ 的相关和功率比；只描述，不要求互为相反。"""
    out = {}
    for g in "PQ":
        rp, rm = _full_response(X, runs, g + "+"), _full_response(X, runs, g + "-")
        den = float(np.sum(np.square(rm)))
        out[g] = {"corr_plus_vs_neg_minus": _corr(rp, -rm),
                  "power_ratio_plus_over_minus": (float(np.sum(np.square(rp))) / den if den > 0 else float("nan"))}
    return out


def spike_match_halves(X, runs, sigma_pool) -> dict:
    """pilot 收口诊断的延续：按主分析的分半，两半各用各的裸基线。"""
    import diag_pilot_response14_spikes as S
    thr = SPIKE_SIGMAS * sigma_pool
    out = {}
    for c in CONDITIONS:
        out[c] = []
        for split in SPLITS:
            a, b = split_pair(runs, split)
            m = S.match_spikes(r_half(X, a, c), r_half(X, b, c), thr=thr)
            out[c].append(dict(m, thr=thr, split_blocks=[list(split[0]), list(split[1])]))
    return out


def count_steps(counts) -> np.ndarray:
    """计数台阶 = 某关节上一帧计数不变（平台）、本帧改变的原始帧号。"""
    ch = np.diff(np.asarray(counts), axis=0) != 0                       # ch[i-1]：帧 i-1 → i 是否改变
    st = np.zeros(len(counts), bool)
    st[2:] = (ch[1:] & ~ch[:-1]).any(1)
    return np.flatnonzero(st)


def colocation(X, runs, sigma_pool, features, edge) -> dict:
    """大偏差帧（|r_c| 任一轴 ≥ 4 σ_pool）落在门控边缘或计数台阶 ±1 帧内的比例，与全部打分帧的比例对照。"""
    out = {}
    for c, (gate, counts) in features.items():
        r = _full_response(X, runs, c)
        n = len(r)
        feat = set(np.flatnonzero(np.diff(np.asarray(gate).astype(int)) != 0) + 1) | set(count_steps(counts).tolist())
        near_raw = {f + k for f in feat for k in (-1, 0, 1)}
        near = np.array([(j + edge) in near_raw for j in range(n)])
        big = np.abs(r).max(1) >= SPIKE_SIGMAS * sigma_pool
        out[c] = {"n_big_frames": int(big.sum()), "n_features": len(feat),
                  "frac_big_near_feature": (float(near[big].mean()) if big.any() else None),
                  "frac_all_near_feature": float(near.mean())}
    return out


def drift(X, runs, times) -> dict:
    """裸复放按实际时刻：三段（3 / 2 / 3 条）比较与线性斜率；各条件（及每一半）与对应裸复放的实际平均时刻差。"""
    bare = sorted([r["run_id"] for r in runs if r["condition"] == "bare"], key=lambda k: times[k])
    t = np.array([times[k] for k in bare]) / 60.0
    S = np.stack([X[k] for k in bare])
    tc = t - t.mean()
    slope = np.tensordot(tc, S, axes=(0, 0)) / float(np.sum(tc ** 2))
    groups = (bare[:3], bare[3:5], bare[5:])
    gm = [np.mean([X[k] for k in g], 0) for g in groups]
    gaps = {}
    for c in CONDITIONS:
        cr = [r["run_id"] for r in runs if r["condition"] == c]
        g = {"all": float(np.mean([times[k] for k in cr]) - np.mean([times[k] for k in bare]))}
        for split in (*SPLITS, FRONT_BACK):
            for blocks in split:
                h = halves(runs, blocks)
                g["blocks_" + "".join(map(str, blocks))] = float(np.mean([times[k] for k in h[c]])
                                                                 - np.mean([times[k] for k in h["bare"]]))
        gaps[c] = g
    return {"slope_rms_per_min": _rms(slope), "group_mean_time_min": [float(np.mean([times[k] for k in g]) / 60) for g in groups],
            "group_diff_rms": {"mid-early": _rms(gm[1] - gm[0]), "late-mid": _rms(gm[2] - gm[1]),
                               "late-early": _rms(gm[2] - gm[0])},
            "noise_factor_x_sigma_bare": {"mid-early": float(np.sqrt(1 / 3 + 1 / 2)), "late-mid": float(np.sqrt(1 / 2 + 1 / 3)),
                                          "late-early": float(np.sqrt(1 / 3 + 1 / 3))},
            "time_gap_s": gaps}


def front_back(X, runs) -> dict:
    a, b = split_pair(runs, FRONT_BACK)
    return {c: {"R": split_R(r_half(X, a, c), r_half(X, b, c)), "diff_rms": _rms(r_half(X, a, c) - r_half(X, b, c))}
            for c in CONDITIONS}


def _summary(mm) -> dict:
    return {"sigma_pool": mm["sigma_pool"], "sigma_bare": mm["sigma_bare"], "R_c": mm["R_c"]}


def cross_day(X, runs, pilot_root) -> dict:
    """pilot 的 100 Hz 原始流按同一处理链复算，与本次的完整响应对比（只读；记录哈希须与 pilot 账本一致）。"""
    import precision_chain as C
    import q1_windows as W
    root = Path(pilot_root)
    led = {a["run_id"]: a for a in json.loads((root / "session.json").read_text())["attempts"]
           if a.get("status") == "complete"}
    K = W.kernel()

    def load(rid):
        raw = (root / "records" / f"replay_{rid}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != led[rid]["record_sha256"]:
            raise ValueError(f"pilot 记录哈希与账本不符: {rid}")
        return W.P(C.frame_locked(json.loads(raw)), K)
    pb = np.mean([load(k) for k in PILOT_BARE], 0)
    out = {}
    for c in CONDITIONS:
        rp = np.mean([load(k) for k in PILOT_RUNS[c]], 0) - pb
        rn = _full_response(X, runs, c)
        den = float(np.sum(np.square(rp)))
        out[c] = {"corr": _corr(rn, rp), "power_ratio_now_over_pilot": (float(np.sum(np.square(rn))) / den if den > 0 else None)}
    return out


def aux_report(X, recs, runs, mm, session, root, pilot_root) -> dict:
    """每一项单独计算，出错只记在该项；整个结果只放进 aux 节。"""
    import precision_chain as C
    import q1_data as D
    import q1_windows as W
    K = W.kernel()
    out = {}

    def clean(msg):                                                     # 产物不含绝对路径（报错信息也一样）
        for p, tag in ((Path(pilot_root), "<pilot_root>"), (Path(root), "<root>")):
            msg = msg.replace(str(p.resolve()), tag).replace(str(p), tag)
        return msg

    def item(name, fn):
        try:
            out[name] = fn()
        except Exception as e:                                          # noqa: BLE001
            out[name] = {"error": clean(f"{type(e).__name__}: {e}")}

    def features():
        f = {}
        for c in CONDITIONS:
            seq = next(r["sequence"] for r in runs if r["condition"] == c)
            with np.load(Path(root) / seq, allow_pickle=False) as z:
                f[c] = (z["gate"].copy(), z["planned_counts"].copy())
        return f

    def offsets():
        res = {}
        for o in OFFSETS_MS:
            try:
                Xo = {k: W.P(C.frame_locked(recs[k], offset=o / 1000.0), K) for k in X}
                res[str(o)] = _summary(main_metrics(Xo, runs))
            except Exception as e:                                      # noqa: BLE001
                res[str(o)] = {"error": clean(f"{type(e).__name__}: {e}")}
        return res
    times = {k: float(np.mean(recs[k]["t_send"])) - float(session["started_monotonic"]) for k in recs}
    item("kappa", lambda: kappa(X, runs, mm))
    item("plus_minus", lambda: plus_minus(X, runs))
    item("spike_match_halves", lambda: spike_match_halves(X, runs, mm["sigma_pool"]))
    item("colocation", lambda: colocation(X, runs, mm["sigma_pool"], features(), D.EDGE))
    item("drift", lambda: drift(X, runs, times))
    item("front_back", lambda: front_back(X, runs))
    item("naive_20hz", lambda: _summary(main_metrics(
        {k: W.P(np.asarray(recs[k]["imu"], float)[:, 3:6], K) for k in X}, runs)))
    item("frame_locked_offsets", offsets)
    item("cross_day_pilot", lambda: cross_day(X, runs, pilot_root))
    return out


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


def analyze(root=None, release=None, pilot_root=None) -> dict:
    import copy
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
    try:                                                                # 判定已定；辅助报告拿副本，出错不影响判定
        out["aux"] = aux_report(X, recs, runs, copy.deepcopy(mm), v["session"], root, pilot_root or PILOT_ROOT)
    except Exception as e:                                              # noqa: BLE001
        out["aux"] = {"error": f"{type(e).__name__}: {e}"}
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
