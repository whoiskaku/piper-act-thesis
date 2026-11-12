#!/usr/bin/env python3
"""14 次响应 pilot 的零机时收口诊断（用户 2026-10-05 决定；只用已有记录）。

问题：同一探针、同一符号重复时，那些大尖峰是否在相同位置、以相近方向和幅度出现？
同时核对请求序列及控制 / IMU 时间戳，避免把时间错位当成响应变化。
**不删尖峰、不换滤波（带通仍是 pilot 整理用的 0.5–9 Hz `P(·)`）、不比较模型、不判定响应是否存在。**

- 偏差：探针运行 d = X − ē（ē = A1…A5 均值）。裸复放对照用相邻两遍 (A1,A2)、(A4,A5)，
  以另外三次裸复放的均值为基线（不让这一对自己参与定义基线）。
- 尖峰：|d| ≥ 阈值的「帧 × 轴」。阈值 = 4·σ_adj（σ_adj 取 pilot 整理的 0.0353 rad/s ⇒ 0.141 rad/s），事前固定，不调。
- 匹配：第一遍的每个尖峰，看第二遍同轴 ±1 帧内有没有同号且过阈值的点；同时记录同一帧的符号是否一致、幅度比（另一遍 / 这一遍）。
  两个方向合并汇总。偶然匹配率：把另一遍整体循环错开 20 帧以上后，按同样规则得到的平均匹配率。
- 时间戳：控制间隔（t_send）、读数延迟（t − t_send）、20 Hz 样本所取的那个 100 Hz 样本的「采样相位」（t_send − 该样本的主机时刻）、
  100 Hz 设备时间戳是否丢样；同一条件两遍的逐帧相位差，与尖峰帧处两遍之差的关系。
- 请求序列：记录的发送指令 = 计划 cmd7 + δa（探针）/ cmd7（裸），换算成计数后与 planned_counts 比较；同一条件两遍的发送指令逐帧比较。
- 相位一致的对照（**只用于判断时间错位能否解释差异，不替代 pilot 的整理口径**）：用 100 Hz 流（按设备毫秒拟合出的主机时刻），
  在每帧 t_send + 固定偏移处线性插值取样（偏移 = 记录样本相对 t_send 的中位位置），再做同样的带通、偏差与匹配。

    python -u sim/diag_pilot_response14_spikes.py
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOT = HERE.parent
PILOT = ROOT / "outputs/pilot_response14"
OUT = ROOT / "outputs/pilot_response14_analysis"
SIGMA_ADJ = 0.0353                                   # pilot 整理的 σ_adj（rad/s）
COUNTS_PER_RAD = 57324.840764                         # audit.json 的 planned_counts_semantics
THR = 4 * SIGMA_ADJ
COND = {"P+": ("BP_p_r1", "BP_p_r2"), "P−": ("BP_m_r1", "BP_m_r2"),
        "Q+": ("BQ_p_r1", "BQ_p_r2"), "Q−": ("BQ_m_r1", "BQ_m_r2")}
BARE_PAIRS = {"A1/A2": (("A1", "A2"), ("A3", "A4", "A5")), "A4/A5": (("A4", "A5"), ("A1", "A2", "A3"))}
SEQ = {"A": "A.npz", "BP_p": "P_plus.npz", "BP_m": "P_minus.npz", "BQ_p": "Q_plus.npz", "BQ_m": "Q_minus.npz"}


# ── 尖峰匹配 ──────────────────────────────────────────────────────────────────
def _one_way(a, b, thr: float, tol: int):
    idx = np.argwhere(np.abs(a) >= thr)
    rows = []
    for f, ax in idx:
        v1 = float(a[f, ax])
        w = b[max(0, f - tol):min(len(b), f + tol + 1), ax] * np.sign(v1)
        v2 = float(b[f, ax])
        rows.append((bool((w >= thr).any()), bool(np.sign(v2) == np.sign(v1)), abs(v2) / abs(v1), v1, v2))
    return rows


def match_spikes(d1, d2, thr: float = THR, tol: int = 1, min_shift: int = 20, n_shift: int = 25) -> dict:
    """两遍偏差 (T, 3) → 尖峰是否同位、同号、幅度相近（两个方向合并），以及循环错开后的偶然匹配率。"""
    d1, d2 = np.asarray(d1, np.float64), np.asarray(d2, np.float64)
    rows = _one_way(d1, d2, thr, tol) + _one_way(d2, d1, thr, tol)
    T = len(d1)
    shifts = np.linspace(min_shift, T - min_shift, n_shift).astype(int)
    chance = []
    for s in shifts:
        r = _one_way(d1, np.roll(d2, s, axis=0), thr, tol) + _one_way(d2, np.roll(d1, s, axis=0), thr, tol)
        if r:
            chance.append(np.mean([x[0] for x in r]))
    if not rows:
        return {"n_spikes": 0, "match_rate": None, "sign_agree": None, "amp_ratio_median": None, "corr": None,
                "chance_rate": float(np.mean(chance)) if chance else None}
    v1, v2 = np.array([x[3] for x in rows]), np.array([x[4] for x in rows])
    return {"n_spikes": len(rows), "match_rate": float(np.mean([x[0] for x in rows])),
            "sign_agree": float(np.mean([x[1] for x in rows])),
            "amp_ratio_median": float(np.median([x[2] for x in rows])),
            "corr": float(np.corrcoef(v1, v2)[0, 1]) if len(rows) > 2 else None,
            "chance_rate": float(np.mean(chance)) if chance else None,
            "n_spike_frames_each": [int((np.abs(d1) >= thr).sum()), int((np.abs(d2) >= thr).sum())]}


# ── 时间戳 ────────────────────────────────────────────────────────────────────
def _arr(rec, k):
    return np.asarray(rec[k], np.float64)


def sampling_phase(rec) -> np.ndarray:
    """每帧：t_send − 该帧 20 Hz 陀螺所取的那个 100 Hz 样本的主机时刻（秒）。"""
    t, ts = _arr(rec, "t"), _arr(rec, "t_send")
    g20, i100 = _arr(rec, "imu")[:, 3:6], _arr(rec, "imu100")
    host, g100 = i100[:, 0], i100[:, 5:8]
    out = np.empty(len(t))
    for k in range(len(t)):
        j0 = int(np.searchsorted(host, t[k])) - 1
        cand = np.arange(max(0, j0 - 3), min(len(host), j0 + 4))
        err = np.abs(g100[cand] - g20[k]).max(1)
        j = cand[int(np.argmin(err))]
        out[k] = ts[k] - host[j]
    return out


def resample_consistent(rec, offset: float) -> np.ndarray:
    """在每帧 t_send + offset 处，从 100 Hz 流线性插值取陀螺 → (T, 3)。主机时刻按设备毫秒线性拟合（去掉接收抖动）。"""
    i100, ts = _arr(rec, "imu100"), _arr(rec, "t_send")
    b, a = np.polyfit(i100[:, 1], i100[:, 0], 1)
    th = a + b * i100[:, 1]
    q = ts + offset
    return np.stack([np.interp(q, th, i100[:, 5 + j]) for j in range(3)], 1)


def timing_audit(rec) -> dict:
    ts, t = _arr(rec, "t_send"), _arr(rec, "t")
    d = np.diff(ts) * 1000
    i100 = _arr(rec, "imu100")
    dms = np.diff(i100[:, 1])
    out = {"send_interval_mean_ms": float(d.mean()), "send_interval_sd_ms": float(d.std()),
           "send_interval_min_ms": float(d.min()), "send_interval_max_ms": float(d.max()),
           "late_frames_gt60ms": int((d > 60).sum()),
           "read_delay_median_ms": float(np.median((t - ts) * 1000)), "read_delay_max_ms": float(((t - ts) * 1000).max()),
           "imu100_gaps": int((dms > 15).sum()), "imu100_n": int(len(i100))}
    if "t_send_span" in rec:
        sp = _arr(rec, "t_send_span") * 1000
        out.update(send_span_median_ms=float(np.nanmedian(sp)), send_span_max_ms=float(np.nanmax(sp)))
    return out


def command_check(rec_cmd, cmd7, da, use_probe: bool) -> dict:
    exp = np.asarray(cmd7, np.float64)[:, :6] + (np.asarray(da, np.float64) if use_probe else 0.0)
    return {"max_abs_rad": float(np.abs(np.asarray(rec_cmd, np.float64)[:, :6] - exp).max())}


# ── 诊断 ────────────────────────────────────────────────────────────────────
def diagnose(root=PILOT) -> dict:
    import analyze_pilot_response14 as A
    import q1_windows as W
    root = Path(root)
    att = json.loads((root / "session.json").read_text())["attempts"]
    A.load_runs(root, att)                                              # 文件哈希与账本一致
    recs = {a["run_id"]: json.loads((root / "records" / f"replay_{a['run_id']}.json").read_text())
            for a in att if a["run_id"] != "A0"}
    K = W.kernel()
    phase = {k: sampling_phase(r) for k, r in recs.items()}
    offset = -float(np.median(np.concatenate(list(phase.values()))))
    sigs = {"logged": {k: W.P(_arr(r, "imu")[:, 3:6], K) for k, r in recs.items()},
            "consistent": {k: W.P(resample_consistent(r, offset), K) for k, r in recs.items()}}
    res = {"threshold_rad_s": THR, "offset_s": offset, "spikes": {}, "sigma_adj": {}, "plus_minus_corr": {}}
    for name, X in sigs.items():
        ebar = np.mean([X[k] for k in ("A1", "A2", "A3", "A4", "A5")], 0)
        sp = {}
        for c, (r1, r2) in COND.items():
            sp[c] = match_spikes(X[r1] - ebar, X[r2] - ebar)
        for c, ((r1, r2), base) in BARE_PAIRS.items():
            eb = np.mean([X[k] for k in base], 0)
            sp[c] = match_spikes(X[r1] - eb, X[r2] - eb)
        res["spikes"][name] = sp
        pairs = [("A1", "A2"), ("A4", "A5")] + list(COND.values())
        res["sigma_adj"][name] = float(np.sqrt(np.mean([A.pair_sigma(X[a], X[b]) ** 2 for a, b in pairs])))
        for g, (p, m) in (("P", ("P+", "P−")), ("Q", ("Q+", "Q−"))):
            rp = np.mean([X[k] for k in COND[p]], 0) - ebar
            rm = np.mean([X[k] for k in COND[m]], 0) - ebar
            res["plus_minus_corr"].setdefault(name, {})[g] = A.corr(rp, -rm)
    # 相位差与尖峰处两遍之差：尖峰帧（记录的 20 Hz 信号，任一遍过阈值）上 |Δ相位| 与 |d1 − d2| 的关系
    X = sigs["logged"]
    ebar = np.mean([X[k] for k in ("A1", "A2", "A3", "A4", "A5")], 0)
    edge = W.D.EDGE if hasattr(W, "D") else 20
    rel = {}
    for c, (r1, r2) in COND.items():
        d1, d2 = X[r1] - ebar, X[r2] - ebar
        f = np.unique(np.argwhere((np.abs(d1) >= THR) | (np.abs(d2) >= THR))[:, 0])
        dph = np.abs(phase[r1] - phase[r2])[edge:edge + len(d1)][f] * 1000
        dd = np.abs(d1 - d2)[f].max(1)
        big = dph > 5.0
        rel[c] = {"n_frames": int(len(f)), "dphase_ms_median": float(np.median(dph)) if len(f) else None,
                  "mean_abs_diff_dphase_gt5ms": float(dd[big].mean()) if big.any() else None,
                  "mean_abs_diff_dphase_le5ms": float(dd[~big].mean()) if (~big).any() else None,
                  "spearman": _spearman(dph, dd) if len(f) > 3 else None}
    res["phase_vs_spike_diff"] = rel
    res["timing"] = {k: {**timing_audit(r), "phase_median_ms": float(np.median(phase[k]) * 1000),
                         "phase_p5_ms": float(np.percentile(phase[k], 5) * 1000),
                         "phase_p95_ms": float(np.percentile(phase[k], 95) * 1000)} for k, r in recs.items()}
    cmd = {}
    for k, r in recs.items():
        z = np.load(root / SEQ[k.rsplit("_r", 1)[0] if k.startswith("B") else "A"], allow_pickle=False)
        c = command_check(r["cmd"], z["cmd7"], z["da"], bool(r.get("use_probe")))
        # 计数按审计文件写明的语义：round(x * 57324.840764)（= 180000 / 3.14，SDK 的换算），不是精确的 π
        counts = np.round(np.asarray(r["cmd"], np.float32).astype(np.float64)[:, :6] * COUNTS_PER_RAD)
        c["max_abs_counts_vs_planned"] = float(np.abs(counts - z["planned_counts"]).max())
        cmd[k] = c
    for c, (r1, r2) in COND.items():
        cmd[f"{c} 两遍"] = {"max_abs_rad": float(np.abs(_arr(recs[r1], "cmd") - _arr(recs[r2], "cmd")).max())}
    res["commands"] = cmd
    res["_sigs"] = sigs
    return res


def _spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def figure(res: dict, path: Path) -> None:
    """每个条件：尖峰帧上第一遍偏差对第二遍偏差的散点（记录的 20 Hz / 相位一致的对照）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 4, figsize=(13.33, 6.6), squeeze=False)
    for i, name in enumerate(("logged", "consistent")):
        X = res["_sigs"][name]
        ebar = np.mean([X[k] for k in ("A1", "A2", "A3", "A4", "A5")], 0)
        for j, (c, (r1, r2)) in enumerate(COND.items()):
            d1, d2 = X[r1] - ebar, X[r2] - ebar
            m = (np.abs(d1) >= THR) | (np.abs(d2) >= THR)
            a = ax[i][j]
            a.scatter(d1[m], d2[m], s=10, color="#1f6fb2" if i == 0 else "#e07a1f", alpha=0.7)
            lim = max(0.5, float(np.abs(np.concatenate([d1[m], d2[m]])).max()) if m.any() else 0.5)
            a.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8)
            a.axhline(0, color="0.85", lw=0.6)
            a.axvline(0, color="0.85", lw=0.6)
            s = res["spikes"][name][c]
            a.set_title(f"{c} · {'記録 20 Hz' if i == 0 else '位相一致'}\n一致 {s['match_rate']:.2f}（偶然 {s['chance_rate']:.2f}）"
                        f" · 同符号 {s['sign_agree']:.2f}", fontsize=9.5)
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            a.set_xlabel("1 回目の偏差 (rad/s)", fontsize=8.5)
            if j == 0:
                a.set_ylabel("2 回目の偏差 (rad/s)", fontsize=8.5)
            a.grid(alpha=0.25)
    fig.suptitle("同一探針・同符号の 2 回：大きなスパイク（|偏差| ≥ 4σ_adj）の位置・符号・振幅の一致", fontsize=13)
    fig.text(0.5, 0.01, "※ 零機時の収口診断。スパイクは削除していない。帯域 0.5–9 Hz は pilot 整理と同じ。"
             "位相一致は時間ずれの検証用の対照で、整理口径の置き換えではない。", ha="center", fontsize=9, color="#c1121f")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PILOT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = diagnose(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    (a.out / "diag_spikes.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False))
    figure(res, a.out / "fig_spike_repeat.png")
    print(json.dumps(clean, indent=1, ensure_ascii=False))
    print(f"→ {a.out / 'diag_spikes.json'} · {a.out / 'fig_spike_repeat.png'}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
