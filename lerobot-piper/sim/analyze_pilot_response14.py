#!/usr/bin/env python3
"""14 次响应 pilot 的整理 —— 冻结方案 `reports/PLAN_PILOT_RESPONSE_14.md` §2–§6：只估重复性、漂移与响应的不确定性。

不判定响应是否存在、不比较模型、不设门槛（方案 §0、§6）；结果只供用户决定正式采集是否值得投入。
另附三项装置状态统计（只报告；不据此换轨迹、改幅值或剔除记录）。

口径（冻结方案没有逐项写死的地方用以下默认，报告里注明）：
- 信号：复放记录的 20 Hz 陀螺 `imu[:, 3:6]`（传感器系，同 ③a）。三项估计都是实机对实机的差，不需要对齐（旋转不改变能量）。
- 带通 0.5–9 Hz：`P(·)` = 去均值 → bandpass41 → 丢首尾 20 帧（同阶段一 / 二的计分口径），逐次复放各自处理后再比较。
  另报未带通的相邻两遍散布，供与 ③a 的历史锚点（σ = 0.03911 rad/s，未带通）对表。
- 裸基线 ē = A1…A5 的均值（A0 只用于对齐，不进基线；方案 §1）。
- 响应：r̄_{g,s} = mean(同符号两遍) − ē（g ∈ {P, Q}，s ∈ {+, −}），正负分开报告（§3.3）；奇 / 偶分量 (r̄₊ ∓ r̄₋)/2 只作辅助。
- 重复性：σ_bare = 5 次裸复放逐帧逐轴样本方差（÷(M−1)）的池化开方（复放级 df = 4；含漂移）；
  σ_adj = 相邻两遍之差（A1/A2、A4/A5 与四个同符号对）的池化散布（df = 6；基本不含慢漂移），用于不确定性。
- 漂移：(A1,A2)、A3、(A4,A5) 三点（§2），按实际时间戳；各差的 RMS 与「只有复放噪声时」的预期并列，只描述、不建立上界。
  线性漂移：逐帧逐轴对实际时刻回归 A1…A5 得斜率波形；它对某组响应估计的偏置 ≈ 斜率 × (该组平均时刻 − 基线平均时刻)。
- 不确定性（§4）：Var(r̄) = σ²(1/n + 1/M)；共享基线 ⇒ Cov(r̄ᵢ, r̄ₖ) = σ²/M、相关 = (1/M)/(1/n + 1/M)；
  四个响应估计取均值时实际方差与「误作独立」之比；σ 的名义卡方 95% 区间（iid 正态，复放级自由度）。
  纯噪声的响应估计，其逐帧 RMS 的期望 ≈ 标准误；带通后帧间强相关，RMS 本身的有效自由度很少（只作描述）。
- ABBA（§2.1）：按实际时间戳核对每组正负两侧的平均采集时刻。

    python -u sim/analyze_pilot_response14.py
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOT = HERE.parent
PILOT = ROOT / "outputs/pilot_response14"
OUT = ROOT / "outputs/pilot_response14_analysis"
NOMINAL = ROOT / "outputs/ab5/A_ep0.json"
ERA = ROOT / "outputs/q1_diag_era/diag_q1_era.json"
BARE = ("A1", "A2", "A3", "A4", "A5")
PROBES = {"P": {"plus": ("BP_p_r1", "BP_p_r2"), "minus": ("BP_m_r1", "BP_m_r2")},
          "Q": {"plus": ("BQ_p_r1", "BQ_p_r2"), "minus": ("BQ_m_r1", "BQ_m_r2")}}
ADJ_PAIRS = (("A1", "A2"), ("A4", "A5"), ("BP_p_r1", "BP_p_r2"), ("BP_m_r1", "BP_m_r2"),
             ("BQ_p_r1", "BQ_p_r2"), ("BQ_m_r1", "BQ_m_r2"))


# ── 统计小工具 ────────────────────────────────────────────────────────────────
def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64)))))


def pooled_sigma(xs) -> tuple:
    """M 次复放 → (单次复放散布 σ，复放级 df = M − 1)：逐帧逐轴样本方差的池化。"""
    X = np.stack([np.asarray(x, np.float64) for x in xs])
    r = X - X.mean(0)
    return float(np.sqrt((r ** 2).sum() / ((len(X) - 1) * r[0].size))), len(X) - 1


def pair_sigma(a, b) -> float:
    """同条件两遍之差 → 单次复放散布（df = 1）。"""
    return rms(np.asarray(a, np.float64) - np.asarray(b, np.float64)) / np.sqrt(2)


def corr(a, b) -> float:
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def responses(plus, minus, bare_mean) -> dict:
    """§3.1：先平均同符号重复的波形，再减共享裸基线；正负分开，奇 / 偶分量只作辅助。"""
    rp = np.mean(plus, 0) - bare_mean
    rm = np.mean(minus, 0) - bare_mean
    return {"plus": rp, "minus": rm, "odd": 0.5 * (rp - rm), "even": 0.5 * (rp + rm)}


def uncertainty(sigma: float, n: int = 2, M: int = 5, k: int = 4) -> dict:
    """§4：共享裸基线下各响应估计的方差、协方差与相关；k 个估计取均值时的方差比。"""
    s2 = sigma ** 2
    var_r, cov = s2 * (1 / n + 1 / M), s2 / M
    indep, actual = var_r / k, (k * var_r + k * (k - 1) * cov) / k ** 2
    var_odd, var_even = s2 / (2 * n), s2 / (2 * n) + s2 / M
    return {"var_r": var_r, "cov": cov, "corr": cov / var_r, "var_ratio_mean4": actual / indep,
            "se_ratio_mean4": float(np.sqrt(actual / indep)), "var_odd": var_odd, "var_even": var_even,
            "se_r": float(np.sqrt(var_r)), "se_odd": float(np.sqrt(var_odd)), "se_even": float(np.sqrt(var_even))}


def chi2_ci(df: int) -> tuple:
    """单次复放散布 σ 的名义 95% 区间系数（iid 正态）：σ ∈ [lo, hi] × σ̂。"""
    from scipy.stats import chi2
    return float(np.sqrt(df / chi2.ppf(0.975, df))), float(np.sqrt(df / chi2.ppf(0.025, df)))


def drift(bare, times) -> dict:
    """裸复放（按 A1…A5 顺序）与实际时刻（分钟）→ 三点组差、线性斜率波形（每分钟）。"""
    X = np.stack([np.asarray(x, np.float64) for x in bare])
    t = np.asarray(times, np.float64)
    tc = t - t.mean()
    slope = np.tensordot(tc, X, axes=(0, 0)) / (tc ** 2).sum()
    g = {"A12": X[:2].mean(0), "A3": X[2], "A45": X[3:].mean(0)}
    diffs = {"A3-A12": g["A3"] - g["A12"], "A45-A3": g["A45"] - g["A3"], "A45-A12": g["A45"] - g["A12"]}
    # 只有复放噪声时各差的标准差 / σ（组内平均的次数：2、1、2）
    factor = {"A3-A12": float(np.sqrt(1 + 1 / 2)), "A45-A3": float(np.sqrt(1 / 2 + 1)),
              "A45-A12": float(np.sqrt(1 / 2 + 1 / 2))}
    return {"slope": slope, "slope_rms": rms(slope), "t_mean": float(t.mean()),
            "groups": {"A12": float(t[:2].mean()), "A3": float(t[2]), "A45": float(t[3:].mean())},
            "diff_rms": {k: rms(v) for k, v in diffs.items()}, "noise_factor": factor}


def drift_bias(d: dict, t: float):
    """线性漂移给「平均采集时刻为 t 的一组」带来的偏置（相对基线平均时刻）。"""
    return d["slope"] * (t - d["t_mean"])


# ── 记录、时间戳与装置状态 ────────────────────────────────────────────────────
def _ts(u: str) -> float:
    return datetime.fromisoformat(u.replace("Z", "+00:00")).timestamp()


def mid_time(a: dict) -> float:
    return 0.5 * (_ts(a["started_utc"]) + _ts(a["ended_utc"]))


def abba(attempts) -> dict:
    """§2.1：每组正负两侧按实际时间戳的平均采集时刻与差（秒）。"""
    by = {a["run_id"]: mid_time(a) for a in attempts if a.get("ended_utc")}
    out = {}
    for g, spec in PROBES.items():
        tp, tm = [by[r] for r in spec["plus"] if r in by], [by[r] for r in spec["minus"] if r in by]
        if tp and tm:
            out[g] = {"plus_mean_unix": float(np.mean(tp)), "minus_mean_unix": float(np.mean(tm)),
                      "plus_minus_gap_s": float(np.mean(tp) - np.mean(tm)), "n_plus": len(tp), "n_minus": len(tm)}
    return out


def load_runs(root, attempts) -> dict:
    """账本里已完成的每一次 → 20 Hz 陀螺 (T, 3)；文件哈希必须与账本记录一致。"""
    out = {}
    for a in attempts:
        if a.get("status") != "complete":
            continue
        p = Path(root) / "records" / f"replay_{a['run_id']}.json"
        b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() != a.get("record_sha256"):
            raise ValueError(f"{p.name}: 文件哈希与账本记录不符 —— 记录被改动或不是这次采集的")
        out[a["run_id"]] = np.asarray(json.loads(b)["imu"], np.float64)[:, 3:6]
    return out


def replay_episode(e0, w):
    """名义轨迹的阶段一数据对象（理想 IMU、指令不变），换成某次复放的陀螺，并按同一定义重建 gap。"""
    import q1_data as D
    w = np.asarray(w, np.float64)
    gap = w[D.ALIGN_L:D.ALIGN_L + e0.n] - e0.ideal[:e0.n]
    return replace(e0, w=w, w_valid=np.isfinite(w).all(1), gap=gap)


def setup_stats(bare_raw: dict) -> dict:
    """三项装置状态统计（只报告）：沿用采集时期诊断（diag_q1_era）的同一方法。"""
    import diag_q1_era as G
    import q1_data as D
    import q1_windows as W
    e0 = D.load_episode(NOMINAL, "ab5/A")
    for k, w in bare_raw.items():
        if len(w) != e0.T:
            raise ValueError(f"{k}: 复放帧数 {len(w)} ≠ 名义轨迹 {e0.T}")
    eps = [replay_episode(e0, w) for w in bare_raw.values()]
    K = W.kernel()
    out = {}
    for name, es in (("pilot_bare_A1_A5", eps), ("ab5_A_ep0_original", [e0])):
        ang, gain, _, n = G.rotation_calib(es)
        gs = G.gap_stats(es, K)
        out[name] = {"rotation_deg": ang, "rotation_gain": gain, "rotation_n": n,
                     **{k: v for k, v in gs.items() if k.startswith("gap_share")}}
    era = json.loads(ERA.read_text())
    out["era_sessions"] = {s: {"rotation_deg": era["rotation"][s]["angle_deg"],
                               **{k: v for k, v in era["gap"][s].items() if k.startswith("gap_share")}}
                           for s in era["rotation"]}
    out["wrist_operating_point"] = "复放的是 ab5/A_ep0 的固定指令：腕旋工作点按构造与 ab5/A_ep0 相同，不反映装置状态"
    return out


# ── 整理 ────────────────────────────────────────────────────────────────────
def analyze(root=PILOT) -> dict:
    import q1_windows as W
    root = Path(root)
    s = json.loads((root / "session.json").read_text())
    att = s["attempts"]
    runs = load_runs(root, att)
    need = list(BARE) + [r for spec in PROBES.values() for v in spec.values() for r in v]
    missing = [r for r in need if r not in runs]
    if missing:
        raise ValueError(f"缺少已完成的复放：{missing}（按实际执行情况另行整理）")
    K = W.kernel()
    X = {k: W.P(v, K) for k, v in runs.items()}
    by = {a["run_id"]: a for a in att}
    t0 = mid_time(by["A1"])
    tmin = {k: (mid_time(by[k]) - t0) / 60.0 for k in runs}

    bare = [X[r] for r in BARE]
    ebar = np.mean(bare, 0)
    s_bare, df_bare = pooled_sigma(bare)
    pair_var = {f"{a}/{b}": pair_sigma(X[a], X[b]) ** 2 for a, b in ADJ_PAIRS}
    s_adj = float(np.sqrt(np.mean(list(pair_var.values()))))
    s_bare_axes = [pooled_sigma([x[:, j:j + 1] for x in bare])[0] for j in range(3)]
    raw_adj = {f"{a}/{b}": pair_sigma(runs[a], runs[b]) for a, b in (("A1", "A2"), ("A4", "A5"))}
    corrs = {f"{a}/{b}": corr(X[a], X[b]) for i, a in enumerate(BARE) for b in BARE[i + 1:]}

    d = drift(bare, [tmin[r] for r in BARE])
    unc = uncertainty(s_adj, n=2, M=len(BARE))
    resp = {}
    for g, spec in PROBES.items():
        est = responses([X[r] for r in spec["plus"]], [X[r] for r in spec["minus"]], ebar)
        tp = float(np.mean([tmin[r] for r in spec["plus"]]))
        tm = float(np.mean([tmin[r] for r in spec["minus"]]))
        bias = {"plus": drift_bias(d, tp), "minus": drift_bias(d, tm)}
        bias["odd"], bias["even"] = 0.5 * (bias["plus"] - bias["minus"]), 0.5 * (bias["plus"] + bias["minus"])
        se = {"plus": unc["se_r"], "minus": unc["se_r"], "odd": unc["se_odd"], "even": unc["se_even"]}
        resp[g] = {k: {"rms": rms(est[k]), "se": se[k], "rms_over_se": rms(est[k]) / se[k],
                       "drift_bias_rms": rms(bias[k]),
                       "rms_axes": [rms(est[k][:, j]) for j in range(3)]} for k in est}
        resp[g]["plus_minus_corr"] = corr(est["plus"], -est["minus"])
        resp[g]["t_plus_min"], resp[g]["t_minus_min"] = tp, tm
        resp[g]["_waves"] = est
    return {
        "session": {"pilot_id": s.get("pilot_id"), "started_utc": s.get("started_utc"), "halted": s.get("halted"),
                    "attempts": [{k: a.get(k) for k in ("run_id", "status", "exit_code", "started_utc", "ended_utc",
                                                        "record_sha256")} for a in att],
                    "on_site_minutes": (max(_ts(a["ended_utc"]) for a in att if a.get("ended_utc"))
                                        - _ts(s["started_utc"])) / 60.0},
        "frames_after_bandpass": int(ebar.shape[0]),
        "repeatability": {"sigma_bare": s_bare, "df_bare": df_bare, "sigma_bare_axes": s_bare_axes,
                          "sigma_adj": s_adj, "df_adj": len(ADJ_PAIRS),
                          "pair_sigma": {k: float(np.sqrt(v)) for k, v in pair_var.items()},
                          "raw_unfiltered_pair_sigma": raw_adj, "bare_corr": corrs,
                          "ci_df4": chi2_ci(4), "ci_df6": chi2_ci(6), "ci_df1": chi2_ci(1)},
        "drift": {k: v for k, v in d.items() if k != "slope"}
                 | {"expected_noise_rms": {k: s_adj * f for k, f in d["noise_factor"].items()},
                    "slope_x_span_rms": d["slope_rms"] * (tmin["A5"] - tmin["A1"]),
                    "times_min": {r: tmin[r] for r in BARE}},
        "uncertainty": unc,
        "abba": abba(att),
        "responses": {g: {k: v for k, v in r.items() if k != "_waves"} for g, r in resp.items()},
        "_waves": {g: r["_waves"] for g, r in resp.items()},
        "_slope": d["slope"],
    }


def figure(res: dict, path: Path) -> None:
    """展示响应（§3.1）：每组探针 r(+) 与 −r(−) 叠画，灰带 = ±2 个标准误（只有复放噪声时）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    se = res["uncertainty"]["se_r"]
    fig, ax = plt.subplots(2, 3, figsize=(13.33, 6.6), sharex=True, squeeze=False)
    for i, g in enumerate(("P", "Q")):
        w = res["_waves"][g]
        t = np.arange(len(w["plus"])) / 20.0
        for j in range(3):
            a = ax[i][j]
            a.fill_between(t, -2 * se, 2 * se, color="0.85", lw=0, label="±2 SE（噪声のみ）")
            a.plot(t, w["plus"][:, j], color="#1f6fb2", lw=1.2, label="r(+)")
            a.plot(t, -w["minus"][:, j], color="#d1495b", lw=1.0, ls="--", label="−r(−)")
            a.set_title(f"探針 {g} · 軸 ω{'xyz'[j]}   RMS r(+) {res['responses'][g]['plus']['rms_axes'][j]:.4f}"
                        f" · r(−) {res['responses'][g]['minus']['rms_axes'][j]:.4f}", fontsize=9.5, loc="left")
            a.grid(alpha=0.25)
            if i == 1:
                a.set_xlabel("時間 (s)")
            if j == 0:
                a.set_ylabel("角速度 (rad/s)\n0.5–9 Hz")
            if i == 0 and j == 0:
                a.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    fig.suptitle("14 回 pilot：応答推定（同符号 2 回の平均 − 裸 5 回の平均；正負は別々に表示）", fontsize=13)
    fig.text(0.5, 0.01, "※ pilot は応答の有無を判定しない・モデル比較なし。SE は共有裸基線込み（σ_adj·√(1/2+1/5)）。"
             "帯域通過後はフレーム間の相関が強い。", ha="center", fontsize=9, color="#c1121f")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PILOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-setup", action="store_true", help="不算装置状态统计（不渲染理想 IMU）")
    a = ap.parse_args(argv)
    res = analyze(a.root)
    if not a.no_setup:
        att = json.loads((a.root / "session.json").read_text())["attempts"]
        res["setup_state"] = setup_stats({r: v for r, v in load_runs(a.root, att).items() if r in BARE})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "analysis.json").write_text(json.dumps(_jsonable(res), indent=1, ensure_ascii=False))
    figure(res, a.out / "fig_responses.png")
    print(json.dumps(_jsonable(res), indent=1, ensure_ascii=False))
    print(f"→ {a.out / 'analysis.json'} · {a.out / 'fig_responses.png'}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
