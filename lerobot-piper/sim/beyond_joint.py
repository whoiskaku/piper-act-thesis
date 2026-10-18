#!/usr/bin/env python3
"""编码器解释不了、IMU 看得到的运动偏差（方案与判据见 reports/PLAN_beyond_joint.md v1.4）。

实测编码器关节角 → 刚体运动学（MuJoCo 安装点）→ 按标定得到预测 IMU，与实测 IMU 在同一时间、坐标、频带下比：
  大小（离线）：编码器 30 Hz 零相位低通 + 样条；各频带残差、线性相干分解、分段；控制条件（静止噪声底、标定回放的慢速
        单关节运动）；常数延迟、固定旋转的敏感性。
  可复现度（离线）：可重复性回放（A、B 开 TE 各 8 遍）按指令帧对齐，留一法 R²；主频带 3–15 Hz。
  可预测度（因果）：预测分支全程只用过去的数据（200 Hz 上先因果抗混叠低通再差分、残差晚 10 ms 才用、无样条），在各遍自己的
        IMU 采样点上看不可复现部分能否由它过去 0.5 s 预测 100 ms 后（固定岭回归、留一遍外推、对照「保持现值」）。

用法：
    python -u sim/beyond_joint.py        # → outputs/beyond_joint/{summary.json, metrics.csv, spectra.png, waveforms.png}
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import cmd_tracking as T  # noqa: E402  读取、标定后的 IMU 渲染、频带、分段
import imu_calib as C  # noqa: E402  Kabsch、旋转角
import repeat_reversal as RR  # noqa: E402  回放读取、接触标记、留一法、分项判定
from step4_gvnext import data as D  # noqa: E402  低通、采样率、CAN 关节分组
from step4_gvnext.train_rq import fit as ridge_fit  # noqa: E402  固定相对岭 λ = c·tr(XᵀX)/p

OUT = HERE.parent / "outputs/beyond_joint"
FS = D.FS_I
LP_Q = 30.0                      # Hz：编码器求导前的低通（离线分支零相位；因果分支单向，同时抗混叠）；比较带最高 15 Hz
MAIN_BAND, SUB_BAND = T.BAND, T.LP_VEL   # 主频带 3–15 Hz；<15 Hz 另列
LAGS = np.arange(50)             # 历史窗口 0.5 s（100 Hz 下 50 个样本）
D_AVAIL = 1                      # 残差晚一个 IMU 样本（10 ms）才可用：网格插值、取值插值各用到之后约 5 ms 的编码器样本
RIDGE_REL = 1e-2                 # 固定岭强度，同 sim/q1/q1_ar.py（用户 2026-10-02 定，不调参）
HS = (1, 5, 10, 15)              # 前瞻步数：10、50、100、150 ms（h = 0 时历史含当前样本、R² 恒为 1，改报一步）
H_JUDGE, R2_FB = 10, 0.3         # 可预测度判据：100 ms 时 R² ≥ 0.3 且优于「保持现值」
NPERSEG = 256                    # 相干分解的 Welch 段长（2.56 s）
DELAYS = np.arange(-30, 31) / 1000.0   # 敏感性：在标定延迟 ±30 ms 内找最佳常数延迟


# ── 离线预测与滤波（大小、频谱、可复现度）─────────────────────────────────
def predict_on_grid(ep, calib, model, sid):
    """实测编码器关节角 → 预测 IMU（加计、陀螺；IMU 坐标、含标定），在编码器网格 tq 上。
    零相位低通 + 样条，会用到前后的数据：只用于离线描述，不用于可预测度。"""
    from scipy.interpolate import CubicSpline
    sp = CubicSpline(ep["tq"], np.column_stack([D.lp(ep["q"], LP_Q, D.FS_Q), np.zeros(len(ep["tq"]))]))
    return T.ideal_imu(sp, ep["tq"], calib, model, sid)


def at(ep, x, delay):
    """编码器网格上的预测量，取在「IMU 时刻 − delay」（IMU 记下的是 delay 秒前的运动，同标定约定）。"""
    return np.stack([np.interp(ep["ti"] - delay, ep["tq"], x[:, k]) for k in range(x.shape[1])], 1)


def lt15(x):
    """<15 Hz，零相位；逐条去均值（零偏漂移不算运动）。"""
    return D.lp(x - x.mean(0), T.LP_VEL, FS)


def nrms(x):
    """3 轴量的 RMS：sqrt(各样本三轴平方和的均值)。"""
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.sum(x ** 2, axis=-1)))) if x.size else float("nan")


def spectra(meas, pred):
    """3 轴各自 Welch 再按轴相加。实测与预测的相干 γ² 把功率分成：实测中与预测线性相干的部分、实测侧的线性不相干功率、
    预测侧的线性不相干功率。不相干功率不直接等于「某个传感器独有的真实运动」（非线性、时变的关系也会落在这里）。"""
    from scipy.signal import csd, welch
    f, smm = welch(meas, FS, nperseg=NPERSEG, axis=0)
    _, spp = welch(pred, FS, nperseg=NPERSEG, axis=0)
    _, see = welch(meas - pred, FS, nperseg=NPERSEG, axis=0)
    _, smp = csd(meas, pred, FS, nperseg=NPERSEG, axis=0)
    g2 = np.abs(smp) ** 2 / (smm * spp)
    return {"f": f, "meas": smm.sum(1), "pred": spp.sum(1), "resid": see.sum(1), "coherent": (g2 * smm).sum(1),
            "meas_incoh": ((1 - g2) * smm).sum(1), "pred_incoh": ((1 - g2) * spp).sum(1)}


def band_rms(s, key, band=T.BAND):
    """功率谱在频带内积分，折成 RMS。"""
    m = (s["f"] >= band[0]) & (s["f"] <= band[1])
    return float(np.sqrt(np.sum(s[key][m]) * (s["f"][1] - s["f"][0])))


# ── 因果预测分支（可预测度）─────────────────────────────────────────────
class Held:
    """按 `sp(t, nu)` 接口取值：t 之前（含）最近的样本（零阶保持，只用过去）；加速度给 0（这里只用陀螺）。"""

    def __init__(self, t, q, v):
        self.t, self.q, self.v = t, q, v

    def __call__(self, t, nu=0):
        k = np.clip(np.searchsorted(self.t, np.atleast_1d(t), side="right") - 1, 0, len(self.t) - 1)
        y = (self.q, self.v, np.zeros_like(self.q))[nu][k]
        return np.column_stack([y, np.zeros(len(y))])


def causal_filter(x, band, fs=FS):
    """四阶 Butterworth 单向（因果）滤波：band = 标量（低通）或 (低, 高)（带通）。"""
    from scipy.signal import butter, sosfilt
    sos = butter(4, np.asarray(band, float) / (fs / 2), btype="low" if np.isscalar(band) else "band", output="sos")
    return sosfilt(sos, x, axis=0)


def causal_state(tq, q, t):
    """编码器 200 Hz 网格 → 时刻 t 的关节角、角速度：先过 LP_Q 单向低通（抗混叠：之后要取到 100 Hz 的 IMU 时刻，不先低通，
    50 Hz 以上的成分会折叠进低频——v1.2、v1.3 的伪影），再后向差分，最后在网格间线性插值（用到 t 之后至多一个网格点）。"""
    qf = causal_filter(q - q[0], LP_Q, D.FS_Q) + q[0]
    vf = np.vstack([np.zeros((1, q.shape[1])), np.diff(qf, axis=0) * D.FS_Q])
    return tuple(np.stack([np.interp(t, tq, x[:, k]) for k in range(x.shape[1])], 1) for x in (qf, vf))


def causal_parts(ep, calib, model, sid):
    """因果分支的 (实测陀螺, 编码器预测)，都在 IMU 采样点上。预测：causal_state 在「IMU 时刻 − 延迟」处的关节角、角速度 → 刚体运动学
    （同 ideal_imu_traj，陀螺只依赖关节角和角速度）→ 按标定转到 IMU 坐标；实测陀螺过同样设计的 LP_Q 单向低通（100 Hz 下），
    抵消编码器侧低通的滞后。网格插值和取值插值各用到之后约 5 ms，所以第 k 个残差到 IMU 时刻 + 10 ms 才可用（D_AVAIL）。"""
    import mujoco
    from ideal_imu_kinematic import ideal_imu_traj
    t = ep["ti"] - calib["delay_s"]
    q, v = causal_state(ep["tq"], ep["q"], t)
    w = ideal_imu_traj(model, mujoco.MjData(model), Held(t, q, v), t, sid)[1][:, 3:]
    g = ep["imu"][:, 3:]
    return causal_filter(g - g[0], LP_Q) + g[0], w @ np.asarray(calib["R"]).T + np.asarray(calib["gyro_bias"])


def causal_nominal(ep):
    """IMU 时刻 → 名义时间：上一个已发指令的帧号 × DT + 自那以后经过的时间（只用过去的 t_send）。"""
    k = np.clip(np.searchsorted(ep["t_send"], ep["ti"], side="right") - 1, 0, len(ep["t_send"]) - 1)
    return RR.DT * k + (ep["ti"] - ep["t_send"][k])


# ── 大小：基线逐条 ───────────────────────────────────────────────────────
def episode(ep, calib, model, sid, masks):
    """一条的残差大小、线性相干分解、分段（masks：segment_masks 的 IMU 时刻掩码）、常数延迟与固定旋转的敏感性。"""
    acc_q, gyr_q = predict_on_grid(ep, calib, model, sid)
    g, a = ep["imu"][:, 3:], ep["imu"][:, :3]
    p, pa = at(ep, gyr_q, calib["delay_s"]), at(ep, acc_q, calib["delay_s"])
    e = g - p
    el, eh = lt15(e), T.band(e, FS)
    s = spectra(g - g.mean(0), p - p.mean(0))
    rec = {"lt15": nrms(el), "hf": nrms(eh), "meas_lt15": nrms(lt15(g)), "meas_hf": nrms(T.band(g, FS)),
           "pred_lt15": nrms(lt15(p)), "pred_hf": nrms(T.band(p, FS)),
           "coherent_hf": band_rms(s, "coherent"), "meas_incoh_hf": band_rms(s, "meas_incoh"), "pred_incoh_hf": band_rms(s, "pred_incoh"),
           "acc_hf": nrms(T.band(a - pa, FS)), "acc_meas_hf": nrms(T.band(a, FS))}
    for seg in ("contact", "noncontact", "boundary", "nonboundary"):
        m = masks[seg]
        rec[f"lt15_{seg}"], rec[f"hf_{seg}"] = (nrms(el[m]), nrms(eh[m])) if m.any() else (float("nan"), float("nan"))
    hf_d = [nrms(T.band(g - at(ep, gyr_q, calib["delay_s"] + d), FS)) for d in DELAYS]
    rec["hf_best_delay"], rec["best_delay_ms"] = float(min(hf_d)), float(DELAYS[int(np.argmin(hf_d))] * 1e3)
    bg = np.asarray(calib["gyro_bias"])
    dR = C._kabsch(lt15(p), lt15(g))                 # 逐条再找一个最佳固定旋转：使 Σ|g − ΔR·p|² 最小
    rec["hf_best_rot"], rec["best_rot_deg"] = nrms(T.band(g - ((p - bg) @ dR.T + bg), FS)), C._deg(dR)
    return rec, s


def controls(calib, model, sid):
    """控制条件：标定回放的静止姿态（噪声底）和单关节慢速来回（平滑运动下的残差，只作量级参照）。
    <15 Hz 偏乐观：标定本身就是在这段数据的 0–3 Hz 上拟合的；3–15 Hz 不受影响。"""
    h = json.loads((HERE.parent / "outputs/imu_calib/replay_calib.json").read_text())
    ep = T.from_record(h, np.asarray(h["cmd"], float))
    segs = json.loads(str(np.load(HERE.parent / "outputs/imu_calib/calib_seq.npz")["segs"]))
    e = ep["imu"][:, 3:] - at(ep, predict_on_grid(ep, calib, model, sid)[1], calib["delay_s"])
    el, eh, ts = lt15(e), T.band(e, FS), ep["t_send"]

    def win(kind, start):
        return np.any([(ep["ti"] >= ts[s[start]]) & (ep["ti"] <= ts[s["end"] - 1]) for s in segs if s["kind"] == kind], axis=0)

    st, mv = win("static", "fit_start"), win("move", "start")
    return {"calib_static": {"lt15": nrms(el[st]), "hf": nrms(eh[st])}, "calib_move": {"lt15": nrms(el[mv]), "hf": nrms(eh[mv])}}


# ── 可复现度与可预测度：可重复性回放 ─────────────────────────────────────
def to_nominal(ep, x, grid):
    """（离线）一遍的 IMU 时刻按本遍 t_send 换算成名义时间（第 k 帧 = k·DT，帧内线性），再插到公共网格 grid 上。"""
    tau = np.interp(ep["ti"], ep["t_send"], RR.DT * np.arange(len(ep["t_send"])))
    return np.stack([np.interp(grid, tau, x[:, k]) for k in range(x.shape[1])], 1)


def replays(calib, model, sid):
    """离线：各遍零相位残差（3–15 Hz、<15 Hz）对齐到公共名义网格（保持段之后）+ 非接触段掩码 + 保持段噪声底；
    因果：各遍在自己的 IMU 采样点上的因果残差（主频带、<15 Hz）、名义时间（因果的用于查询，帧间线性的用作其余各遍的模板坐标）、有效掩码。"""
    center = T.target_center("pickcube_raw", model)
    out = {}
    for name in RR.SEQS:
        cmd7, _, src, reps = RR.load_reps(name)
        marked = RR.contact_frames(cmd7, model, center)
        grid = np.arange(RR.HOLD * RR.DT, (len(cmd7) - 2) * RR.DT, 0.01)
        free = ~marked[np.floor(grid / RR.DT + 1e-9).astype(int)]
        Z, Zh, hold, series = [], [], [], []
        for r, ep, _ in reps:
            if ep is None:
                continue
            e = ep["imu"][:, 3:] - at(ep, predict_on_grid(ep, calib, model, sid)[1], calib["delay_s"])
            Z.append(to_nominal(ep, lt15(e), grid))
            Zh.append(to_nominal(ep, T.band(e, FS), grid))
            hold.append(nrms(T.band(e, FS)[(ep["ti"] >= ep["t_send"][0] + 0.4) & (ep["ti"] < ep["t_send"][RR.HOLD] - 0.1)]))
            still = ep["ti"] < ep["t_send"][RR.HOLD]                              # 去零偏只用开头 1 s 保持段
            g, p = causal_parts(ep, calib, model, sid)
            tau = causal_nominal(ep)
            ok = ((ep["ti"] >= ep["t_send"][RR.HOLD]) & (tau <= (len(cmd7) - 2) * RR.DT)
                  & ~marked[np.clip(np.floor(tau / RR.DT + 1e-9).astype(int), 0, len(cmd7) - 1)])
            series.append({"tau": tau, "tau_lin": np.interp(ep["ti"], ep["t_send"], RR.DT * np.arange(len(ep["t_send"]))), "ok": ok,
                           **{k: causal_filter(x - x[still].mean(0), b) for k, x, b in
                              (("main", g - p, MAIN_BAND), ("sub", g - p, SUB_BAND),
                               ("meas", g, MAIN_BAND), ("pred", p, MAIN_BAND))}})   # 后两项只作诊断：实测侧、预测侧各自
        out[name] = {"src": src, "n": len(Z), "free": free, "Z": np.array(Z), "Zh": np.array(Zh), "hold_hf": hold, "series": series}
    return out


def reproducibility(Z, free):
    """留一法 R²（主值 = 非接触段；接触段、全部另列），及非接触段的可复现 / 不可复现部分 RMS。"""
    m = Z.mean(0)
    res = {k: RR.loo_r2(Z[:, sel]) if sel.any() else None
           for k, sel in (("free", free), ("contact", ~free), ("all", np.ones_like(free)))}
    res["size_free"] = [nrms(m[free]), nrms((Z - m)[:, free])]
    return res


def predictability(series, key="main", hs=HS, lags=LAGS):
    """不可复现部分能否由它自己过去 0.5 s 预测 h 步之后；都在各遍自己的 IMU 采样点上（不重采样，不用未来样本）。
    第 r 遍留出：d_r = e_r − 其余各遍在同一名义时间的平均；训练遍 s 的 d_s = e_s − 除 s、r 外各遍的平均（留出那一遍不进训练）。
    其余各遍是事先已有的重复数据，用作模板。固定岭回归（λ = 1e-2·tr(XᵀX)/p，train_rq.fit；无截距，同 2026-10-02 的 AR 约定）；
    决策时刻 i 只用 i − D_AVAIL 及更早的残差（可用时延）；前瞻 h 从 i 算起；「保持现值」= 最近一个可用残差 d(i − D_AVAIL)。
    R² 以预测零为基线，模型与「保持现值」在同一批有效样本上比；有效样本：i − D_AVAIL − 0.49 s 到 i + max(h) 都在有效段内。
    → {h: [R², R²_保持现值]}。"""
    n = len(series)
    first = D_AVAIL + lags[-1]

    def d_of(s, excl):
        others = [u for u in range(n) if u != s and u not in excl]
        tmpl = np.mean([np.stack([np.interp(series[s]["tau"], series[u]["tau_lin"], series[u][key][:, k]) for k in range(3)], 1)
                        for u in others], axis=0)
        return series[s][key] - tmpl

    def rows(d, ok):
        cs = np.r_[0, np.cumsum(~ok)]
        i = np.arange(first, len(d) - max(hs))
        i = i[cs[i + max(hs) + 1] - cs[i - first] == 0]
        return i, np.concatenate([d[i - D_AVAIL - lag] for lag in lags], 1)

    sse, ss0 = {h: np.zeros(2) for h in hs}, {h: 0.0 for h in hs}
    for r in range(n):
        dr = d_of(r, ())
        ir, Xr = rows(dr, series[r]["ok"])
        tr = [(d, *rows(d, series[s]["ok"])) for s in range(n) if s != r for d in [d_of(s, (r,))]]
        for h in hs:
            B = ridge_fit([X for _, _, X in tr], [d[i + h] for d, i, _ in tr], RIDGE_REL)
            y = dr[ir + h]
            sse[h] += [np.sum((y - Xr @ B) ** 2), np.sum((y - dr[ir - D_AVAIL]) ** 2)]
            ss0[h] += float(np.sum(y ** 2))
    return {h: [float(1 - sse[h][0] / ss0[h]), float(1 - sse[h][1] / ss0[h])] for h in hs}


def verdict_feedback(pred):
    """可预测度判定：A、B 各自在 100 ms 时都 R² ≥ 0.3 且优于「保持现值」。pred = {序列名: {h: [R², R²_保持]} 或 None}。"""
    v = [pred.get(s) for s in RR.JUDGED]
    if any(x is None for x in v):
        return "无法判定"
    if all(x[H_JUDGE][0] >= R2_FB and x[H_JUDGE][0] > x[H_JUDGE][1] for x in v):
        return "支持把反馈作为候选"
    return "未通过：本轮方法没有提供推进依据"


# ── 图 ───────────────────────────────────────────────────────────────────
def plot_spectra(spec):
    """各任务开 TE 的逐条功率谱取中位：实测、编码器预测、两侧的线性不相干功率；灰底 = 3–15 Hz。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    fig, ax = plt.subplots(1, len(spec), figsize=(12, 4), squeeze=False)
    for A, (task, ss) in zip(ax[0], spec.items()):
        f = ss[0]["f"]
        for key, color, ls, label in (("meas", T.BLUE, "-", "实测 IMU"), ("pred", T.ORANGE, "-", "编码器预测"),
                                      ("meas_incoh", T.INK, "-", "实测侧线性不相干"), ("pred_incoh", T.MUTED, "--", "预测侧线性不相干")):
            A.semilogy(f, np.median([s[key] for s in ss], axis=0), color=color, ls=ls, lw=1.2, label=label)
        A.axvspan(*T.BAND, color=T.SHADE, lw=0)
        A.set_xlim(0.5, 40)
        A.set_title(f"{task} 开 TE：陀螺功率谱（三轴相加，{len(ss)} 条中位）", loc="left", color=T.INK)
        A.set_xlabel("频率（Hz）")
        for side in ("top", "right"):
            A.spines[side].set_visible(False)
    ax[0, 0].set_ylabel("(rad/s)² / Hz")
    ax[0, -1].legend(frameon=False, labelcolor=T.INK)
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def plot_waveforms(picks, calib, model, sid):
    """picks = [(任务, 集号, 标签)]：整条（残差最大的陀螺轴，<15 Hz）和残差最集中的 2 s（加残差）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    fig, ax = plt.subplots(len(picks), 2, figsize=(15, 2.6 * len(picks)), squeeze=False, gridspec_kw={"width_ratios": [3, 1.3]})
    for r, (task, i, tag) in enumerate(picks):
        ep = T.load(T.TASKS[task][1], i)
        g = lt15(ep["imu"][:, 3:])
        p = lt15(at(ep, predict_on_grid(ep, calib, model, sid)[1], calib["delay_s"]))
        k = int(np.argmax(np.mean((g - p) ** 2, axis=0)))
        s0 = int(np.argmax(np.convolve(np.sum((g - p) ** 2, axis=1), np.ones(200), "valid")))
        t = ep["ti"] - ep["t_send"][0]
        for c, sl in ((0, slice(None)), (1, slice(s0, s0 + 200))):
            A = ax[r, c]
            A.plot(t[sl], g[sl, k], color=T.BLUE, lw=0.9)
            A.plot(t[sl], p[sl, k], color=T.ORANGE, lw=0.9)
            if c == 1:
                A.plot(t[sl], g[sl, k] - p[sl, k], color=T.INK, lw=0.9)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        ax[r, 0].axvspan(t[s0], t[min(s0 + 199, len(t) - 1)], fill=False, edgecolor=T.INK, lw=1.0)
        ax[r, 0].set_title(f"{task} 开 TE · {tag}（第 {i} 条）：陀螺 {'xyz'[k]} 轴 <15 Hz（rad/s）", loc="left", color=T.INK)
        ax[r, 1].set_title("残差最集中的 2 s", loc="left", color=T.INK)
    for c in range(2):
        ax[-1, c].set_xlabel("时间（s，从第一帧起）")
    handles = [plt.Line2D([], [], color=T.BLUE, lw=2, label="实测 IMU"), plt.Line2D([], [], color=T.ORANGE, lw=2, label="编码器预测"),
               plt.Line2D([], [], color=T.INK, lw=2, label="残差（实测 − 预测）")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, labelcolor=T.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "waveforms.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


# ── 汇总 ─────────────────────────────────────────────────────────────────
def main() -> int:
    gc.disable()                     # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    import mujoco
    import pandas as pd
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    calib = json.loads((HERE.parent / "outputs/imu_calib/imu_calib.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    rows, spec = [], {}
    for task, (no_te, te, demo) in T.TASKS.items():
        center = T.target_center(demo, model)
        for cond, repo in (("TE", te), ("noTE", no_te)):
            for i in range(json.loads((T.LR / repo / "meta/info.json").read_text())["total_episodes"]):
                ep = T.load(repo, i)
                rec, s = episode(ep, calib, model, sid, T.segment_masks(ep, center, cond == "TE", model)[0]["i"])
                rows.append({"task": task, "cond": cond, "episode": i, **rec})
                if cond == "TE":
                    spec.setdefault(task, []).append(s)
        print(f"{task}: 基线 40 条完成", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False)
    ctrl = controls(calib, model, sid)
    rp = replays(calib, model, sid)
    ctrl["replay_hold_hf"] = float(np.median([v for x in rp.values() for v in x["hold_hf"]]))
    repro = {s: {"hf": reproducibility(x["Zh"], x["free"]), "lt15": reproducibility(x["Z"], x["free"])}
             for s, x in rp.items() if x["n"] >= 2}
    pred = {s: {k: predictability(x["series"], k) for k in ("main", "sub", "meas", "pred")}
            for s, x in rp.items() if x["n"] >= 3}
    ok = {s: rp[s]["n"] >= RR.MIN_OK for s in RR.SEQS}
    v_repro = RR.verdict({s: repro[s]["hf"]["free"] if ok[s] and s in repro else None for s in RR.SEQS})
    v_fb = verdict_feedback({s: pred[s]["main"] if ok[s] and s in pred else None for s in RR.SEQS})

    q = {f"{t}_{c}": {k: [float(v) for v in np.nanpercentile(g[k], [25, 50, 75])] if g[k].notna().any() else None
                      for k in df.columns if k not in ("task", "cond", "episode")}          # 开 TE 没有 chunk 边界，那几列为空
         for (t, c), g in df.groupby(["task", "cond"])}
    (OUT / "summary.json").write_text(json.dumps({"plan": "reports/PLAN_beyond_joint.md v1.4", "baseline": q, "controls": ctrl,
                                                  "reproducibility": repro, "predictability": pred,
                                                  "n_replays": {s: x["n"] for s, x in rp.items()},
                                                  "verdicts": {"reproducibility": v_repro, "feedback": v_fb}},
                                                 ensure_ascii=False, indent=1))
    print("\n=== 大小（离线；中位 [四分位]，rad/s）")
    for key, x in q.items():
        def m(k):
            return f"{x[k][1]:.3f} [{x[k][0]:.3f}, {x[k][2]:.3f}]"
        print(f"  {key}：残差 3–15 Hz {m('hf')}、<15 Hz {m('lt15')}；实测本身 {x['meas_hf'][1]:.3f} / {x['meas_lt15'][1]:.3f}，"
              f"预测本身 {x['pred_hf'][1]:.3f} / {x['pred_lt15'][1]:.3f}")
        print(f"      3–15 Hz 线性相干分解：相干 {x['coherent_hf'][1]:.3f}、实测侧不相干 {x['meas_incoh_hf'][1]:.3f}、"
              f"预测侧不相干 {x['pred_incoh_hf'][1]:.3f}；加计 3–15 Hz 残差 {x['acc_hf'][1]:.3f}（实测本身 {x['acc_meas_hf'][1]:.3f} m/s²，只报告）")
        print(f"      分段 3–15 Hz：疑似接触 / 其余 {x['hf_contact'][1]:.3f} / {x['hf_noncontact'][1]:.3f}"
              + (f"；边界 / 非边界 {x['hf_boundary'][1]:.3f} / {x['hf_nonboundary'][1]:.3f}" if key.endswith("noTE") else ""))
        print(f"      敏感性（本次搜索的常数延迟、固定旋转）：最佳常数延迟 → 3–15 Hz 残差 {x['hf_best_delay'][1]:.3f}"
              f"（延迟中位 {x['best_delay_ms'][1]:+.0f} ms）；最佳固定旋转 → {x['hf_best_rot'][1]:.3f}（旋转中位 {x['best_rot_deg'][1]:.2f}°）")
    print(f"\n=== 控制条件（rad/s）：噪声底 3–15 Hz 回放保持段 {ctrl['replay_hold_hf']:.4f}、标定静止姿态 {ctrl['calib_static']['hf']:.4f}；"
          f"标定单关节慢转 3–15 Hz {ctrl['calib_move']['hf']:.4f}、<15 Hz {ctrl['calib_move']['lt15']:.4f}（偏乐观）")
    for task in T.TASKS:
        x = q[f"{task}_TE"]
        print(f"  量级参照（只作参照，不据此断言或否定额外的机械运动）：{task} 开 TE 残差 / 慢转 = "
              f"3–15 Hz {x['hf'][1] / ctrl['calib_move']['hf']:.1f} 倍、<15 Hz {x['lt15'][1] / ctrl['calib_move']['lt15']:.1f} 倍")
    print("\n=== 可复现度（离线；回放；主值 = 非接触段、3–15 Hz）")
    for s, x in repro.items():
        a, b = x["hf"], x["lt15"]
        print(f"  {s}（{rp[s]['src']['repo']} 第 {rp[s]['src']['episode']} 条，{rp[s]['n']} 遍）：R² 非接触 {a['free']:.2f}（<15 Hz {b['free']:.2f}）、"
              f"接触 {a['contact']:.2f}、全部 {a['all']:.2f}；非接触段可复现 / 不可复现 RMS {a['size_free'][0]:.3f} / {a['size_free'][1]:.3f}")
    print(f"  判定（A、B 分别 0.6 / 0.3）：{v_repro}（「暂不推进」只说明本轮方法没有提供推进依据）")
    print("\n=== 可预测度（因果，残差晚 10 ms 才用；回放；不可复现部分、非接触段；过去 0.5 s、固定岭；R² 以预测零为基线）")
    for s, x in pred.items():
        for band, label in (("main", "残差 3–15 Hz"), ("sub", "残差 <15 Hz")):
            print(f"  {s} {label}：" + "，".join(f"{h * 10} ms {v[0]:.2f}（保持现值 {v[1]:.2f}）" for h, v in x[band].items()))
        print(f"  {s} 诊断（只报告，3–15 Hz，100 ms）：只看实测陀螺 {x['meas'][H_JUDGE][0]:.2f}，只看编码器预测 {x['pred'][H_JUDGE][0]:.2f}"
              "（预测侧明显高于实测侧 = 可预测的结构来自预测分支，不在实际运动里）")
    print(f"  判定（主频带；A、B 在 100 ms 都 R² ≥ {R2_FB} 且优于保持现值）：{v_fb}")
    plot_spectra(spec)
    picks = []
    for task in T.TASKS:
        g = df[(df.task == task) & (df.cond == "TE")].sort_values("hf")
        picks += [(task, int(g.iloc[len(g) // 2].episode), "中位"), (task, int(g.iloc[-1].episode), "最难")]
    plot_waveforms(picks, calib, model, sid)
    print(f"\n→ {OUT}/summary.json、metrics.csv、spectra.png、waveforms.png")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
