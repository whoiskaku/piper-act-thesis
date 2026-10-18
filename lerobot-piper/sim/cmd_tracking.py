#!/usr/bin/env python3
"""实际运动 vs 指令：基线数据相对理想渲染的偏离（方案与口径见 reports/PLAN_actual_vs_command.md v2）。

理想渲染 = 每帧实发指令（数据集 action）在写总线时刻 t_send 交给加速度受限规划器
（a_max = 5 rad/s²，本轮参考假设；算法同 sim/step2_g/optuna_ident_enc.accel_limited_dt），不加延迟、不滤波。
实际 = 编码器 can_q200（200 Hz）与 IMU imu100（100 Hz，按 outputs/imu_calib/imu_calib.json 的标定比较）。
命名按测量口径：相对参考的最佳对齐延迟、钝化增益、过冲、高频跟踪残差（3–15 Hz）、疑似接触段、chunk 边界段。
比较频段（用户 2026-10-18 选 A）：参考在加速度层面是 ±a_max 的 bang-bang 切换（每秒约 50 次），
所以位置不滤波，速度层（陀螺、编码器速度）两边同用 <15 Hz，加速度层（加计、编码器加速度）两边同用 <3 Hz；
每个残差旁并列实测、参考各自在该频段的量。

用法：
    python -u sim/cmd_tracking.py        # 4 组基线 → outputs/cmd_tracking/{metrics.csv, summary.json, overlay_*.png}
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
import imu_calib as C  # noqa: E402  标定约定、_skew、_kabsch、predict_site
from step4_gvnext import data as D  # noqa: E402  低通、CAN 关节分组与单位

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = HERE.parent / "outputs/cmd_tracking"
A_MAX = 5.0                      # rad/s²：参考假设（固件寄存器 500，单位与是否强制均未证实）
SUB = 20                         # 规划器每帧细分步数（同 accel_limited_dt）
LP_VEL, LP_ACC = 15.0, 3.0       # Hz：速度层 / 加速度层比较时两边同用的零相位低通（选项 A）
BAND = (3.0, 15.0)               # Hz：陀螺的高频跟踪残差
TAUS = np.arange(0, 301, 5) / 1000.0
CONTACT_S, ZONE_M, BOUNDARY_S, CHUNK, GRIP_OPEN = 0.5, 0.03, 0.25, 100, 0.04
ACTIVE = [0, 1, 2, 4, 5]         # j4 两个任务都从不指令运动，不进关节平均（逐关节结果照常输出）
TASKS = {"pickcube": ("eval_pickcube_base_noTE", "eval_pickcube_base_TE", "pickcube_raw"),
         "insert": ("eval_insert_base_noTE", "eval_insert_base_TE", "insert39")}
SEGS = ("all", "contact", "noncontact", "boundary", "nonboundary")


# ── 理想渲染 ─────────────────────────────────────────────────────────────
def plan_substeps(cmd, dts, a_max, sub=SUB, q0=None):
    """`accel_limited_dt` 的细分步版：同样的算术，返回每个细分步末的 (q, v, a)。帧末的 q 与它逐位相同（测试）。
    q0：起始位置（缺省 = 第一帧指令，同参考实现；评测时传实测起始位置，因为 ACT 第一帧指令不一定等于当前姿态）。"""
    q, v = float(cmd[0] if q0 is None else q0), 0.0
    out = np.empty((len(cmd) * sub, 3))
    k = 0
    for tgt, T in zip(cmd, dts):
        dt = float(T) / sub
        for _ in range(sub):
            e = float(tgt) - q
            d_stop = v * abs(v) / (2.0 * a_max)
            s = e - d_stop
            if abs(s) < a_max * dt * dt and abs(v) < a_max * dt:
                q, v, a = float(tgt), 0.0, 0.0
            else:
                a = float(np.clip(a_max * np.sign(s), -a_max, a_max))
                v += a * dt
                q += v * dt
            out[k] = q, v, a
            k += 1
    return out[:, 0], out[:, 1], out[:, 2]


class PlanTrajectory:
    """细分步结果，按 `ideal_imu_traj` 要的「样条」接口取值：sp(t) 位置、sp(t, 1) 速度、sp(t, 2) 加速度（7 列，夹爪恒 0）。"""

    def __init__(self, cmd, t_send, a_max=A_MAX, q0=None):
        dts = np.r_[np.diff(t_send), np.median(np.diff(t_send))]
        self.t = (t_send[:, None] + dts[:, None] * np.arange(1, SUB + 1) / SUB).ravel()
        cols = [plan_substeps(cmd[:, j], dts, a_max, q0=None if q0 is None else q0[j]) for j in range(cmd.shape[1])]
        self.Q, self.V, self.A = (np.stack([c[k] for c in cols], 1) for k in range(3))

    def __call__(self, t, nu=0):
        t = np.atleast_1d(t)
        if nu == 2:                  # 加速度在细分步内恒定：取 t 所在细分步的值
            y = self.A[np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1)]
        else:
            Y = self.Q if nu == 0 else self.V
            y = np.stack([np.interp(t, self.t, Y[:, j]) for j in range(Y.shape[1])], 1)
        return np.column_stack([y, np.zeros(len(t))])


def ideal_imu(traj, t, calib, model, sid):
    """理想轨迹在 IMU 安装点的读数，按标定换到 IMU 坐标系：acc = R(f + M·dp) + b，gyro = Rω + 陀螺零偏。"""
    import mujoco
    from ideal_imu_kinematic import ideal_imu_traj
    imu = ideal_imu_traj(model, mujoco.MjData(model), traj, t, sid)[1]
    f, w = imu[:, :3], imu[:, 3:]
    M = C._skew(np.gradient(w, t, axis=0)) + C._skew(w) @ C._skew(w)
    R, b, bg, dp = (np.asarray(calib[k], float) for k in ("R", "b", "gyro_bias", "dp"))
    return (f + M @ dp) @ R.T + b, w @ R.T + bg


# ── 数据 ─────────────────────────────────────────────────────────────────
def load(repo, i):
    """一条评测 → 指令、写总线时刻、200 Hz 编码器（网格 tq）、100 Hz IMU（设备时钟映射，同 data.load）。"""
    import pandas as pd
    root = LR / repo
    h = json.loads((root / "high_rate" / f"episode_{i:06d}.json").read_text())
    act = np.stack(pd.read_parquet(root / f"data/chunk-000/episode_{i:06d}.parquet")["action"].to_numpy())
    t_send = np.asarray(h["t_send"], float)
    n_nan = int(np.isnan(t_send).sum())
    if n_nan:                        # 取不到写总线时刻的帧：按帧号插值补上，并记入质量检查
        k = np.arange(len(t_send)); ok = ~np.isnan(t_send)
        t_send = np.interp(k, k[ok], t_send[ok])
    off = h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    Q = np.asarray(h["can_q200"], float)
    tq = np.arange(t_send[0], t_send[-1], 1.0 / D.FS_Q)
    q = np.zeros((len(tq), 6))
    for g, cols in D.Q_GROUPS:
        m = Q[:, 1] == g
        for c, col in enumerate(cols):
            q[:, col] = np.interp(tq, Q[m, 0] - off, Q[m, 2 + c]) * D.D2R
    I = np.asarray(h["imu100"], float)
    a, c = np.linalg.lstsq(np.stack([I[:, 1], np.ones(len(I))], 1), I[:, 0], rcond=None)[0]
    ti = a * I[:, 1] + c
    keep = (ti >= t_send[0]) & (ti <= t_send[-1])
    return {"t_send": t_send, "cmd": act[:, :6].astype(float), "grip": act[:, 6], "tq": tq, "q": q,
            "ti": ti[keep], "imu": I[keep, 2:8], "n_nan_send": n_nan}


def tcp(q, model):
    """编码器关节角 → 两指中点的位置（m）。"""
    import mujoco
    d = mujoco.MjData(model)
    b7, b8 = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("link7", "link8"))
    out = np.empty((len(q), 3))
    for k, qk in enumerate(q):
        d.qpos[:] = 0; d.qpos[:6] = qk
        mujoco.mj_kinematics(model, d)
        out[k] = 0.5 * (d.xpos[b7] + d.xpos[b8])
    return out


def grip_events(grip):
    """夹爪指令越过张开阈值的帧号（张开、合拢都算）。"""
    hi = grip > GRIP_OPEN
    return np.flatnonzero(hi[1:] != hi[:-1]) + 1


def target_center(repo, model):
    """示教里每条第一次合拢后第一次张开时的末端位置的中位数 = 盒子中心 / 六角孔（疑似接触的目标区）。"""
    import pandas as pd
    root = LR / repo
    n = json.loads((root / "meta/info.json").read_text())["total_episodes"]
    pts = []
    for i in range(n):
        df = pd.read_parquet(root / f"data/chunk-000/episode_{i:06d}.parquet")
        a, s = np.stack(df["action"].to_numpy()), np.stack(df["observation.state"].to_numpy())
        hi = a[:, 6] > GRIP_OPEN
        dn = np.flatnonzero(hi[:-1] & ~hi[1:]) + 1
        up = np.flatnonzero(~hi[:-1] & hi[1:]) + 1
        rel = next((u for u in up if len(dn) and u > dn[0]), None)
        if rel is not None:
            pts.append(tcp(np.deg2rad(s[rel:rel + 1, :6] / 1000.0), model)[0])
    return np.median(pts, axis=0)


def near(t, events, half):
    """t 中离任一事件时刻不超过 half 秒的样本。"""
    if len(events) == 0:
        return np.zeros(len(t), bool)
    idx = np.clip(np.searchsorted(events, t), 1, len(events) - 1)
    return np.minimum(np.abs(t - events[idx - 1]), np.abs(t - events[idx])) <= half


# ── 指标 ─────────────────────────────────────────────────────────────────
def rms(x, m=None):
    x = np.asarray(x)[m] if m is not None else np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def band(x, fs):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(4, BAND, btype="bandpass", fs=fs, output="sos"), x, axis=0)


def best_delay(t, q_act, ideal_fn, taus=TAUS):
    """相对参考的最佳对齐延迟：使 RMS(q_act(t) − ideal(t − τ)) 最小的 τ。→ (τ*, 不对齐 RMS, 对齐后 RMS)。"""
    ok = t - taus[-1] >= t[0]
    errs = [rms(q_act[ok] - ideal_fn(t[ok] - tau)) for tau in taus]
    k = int(np.argmin(errs))
    return float(taus[k]), rms(q_act[ok] - ideal_fn(t[ok])), errs[k]


def speed_gain(v_act, v_ideal, moving=0.05):
    """钝化增益：实际速度对理想速度的过原点最小二乘斜率（只用理想在动的样本；< 1 为钝化）。"""
    m = np.abs(v_ideal) > moving
    return float(np.sum(v_act[m] * v_ideal[m]) / np.sum(v_ideal[m] ** 2)) if m.any() else float("nan")


def overshoot(t, q_act, q_ideal, v_ideal, min_peak=0.1, before=0.1, after=0.3):
    """过冲：理想速度换向处，实际位置越过理想换向点的量（rad，正 = 越过）。→ 各换向点的列表。"""
    s = np.sign(v_ideal)
    rev = np.flatnonzero((s[:-1] != 0) & (s[1:] != s[:-1]))
    out, last = [], 0
    for r in rev:
        if np.abs(v_ideal[last:r + 1]).max(initial=0) >= min_peak:
            w = (t >= t[r] - before) & (t <= t[r] + after)
            ext = q_act[w].max() if s[r] > 0 else q_act[w].min()
            out.append(float(s[r] * (ext - q_ideal[r])))
        last = r + 1
    return out


def segment_masks(ep, center, te, model):
    """疑似接触段 / chunk 边界段，编码器网格（"q"）与 IMU 时刻（"i"）各一份。整体之外单列用，不从整体删除。"""
    tq, ti = ep["tq"], ep["ti"]
    events = ep["t_send"][grip_events(ep["grip"])]
    zone_q = np.linalg.norm(tcp(ep["q"], model) - center, axis=1) <= ZONE_M
    zone_i = np.interp(ti, tq, zone_q.astype(float)) > 0.5
    bnd = ep["t_send"][CHUNK::CHUNK] if not te else np.array([])
    masks = {}
    for name, t, zone in (("q", tq, zone_q), ("i", ti, zone_i)):
        contact = near(t, events, CONTACT_S) | zone
        boundary = near(t, bnd, BOUNDARY_S)
        masks[name] = {"all": np.ones(len(t), bool), "contact": contact, "noncontact": ~contact,
                       "boundary": boundary, "nonboundary": ~boundary}
    return masks, bnd


def lp_deriv(x, fc, order, fs=D.FS_Q):
    """零相位低通后求 order 阶导数（实测与参考两边同用）。"""
    y = D.lp(x, fc, fs)
    for _ in range(order):
        y = np.gradient(y, 1.0 / fs, axis=0)
    return y


def episode_metrics(ep, traj, calib, model, sid, center, te):
    tq, q, ti = ep["tq"], ep["q"], ep["ti"]
    rec = {"n_frames": len(ep["t_send"]), "n_nan_send": ep["n_nan_send"]}
    masks, _ = segment_masks(ep, center, te, model)
    deg = 180 / np.pi
    # 编码器层：位置不滤波；速度两边 <15 Hz；加速度两边 <3 Hz。旁列实测、参考各自的量
    qi = traj(tq)[:, :6]
    va, vi = lp_deriv(q, LP_VEL, 1), lp_deriv(qi, LP_VEL, 1)
    aa, ai = lp_deriv(q, LP_ACC, 2), lp_deriv(qi, LP_ACC, 2)
    for key, act, ref in (("enc_pos", q, qi), ("enc_vel_lt15", va, vi), ("enc_acc_lt3", aa, ai)):
        for seg in SEGS:
            m = masks["q"][seg]
            rec[f"{key}_{seg}"] = rms((act - ref)[m][:, ACTIVE]) * deg if m.any() else float("nan")
        if key != "enc_pos":
            rec[f"{key}_act_all"], rec[f"{key}_ref_all"] = rms(act[:, ACTIVE]) * deg, rms(ref[:, ACTIVE]) * deg
    # 逐关节：相对参考的最佳对齐延迟、对齐后残差、钝化增益、过冲
    for j in range(6):
        tau, e0, e1 = best_delay(tq, q[:, j], lambda t, j=j: traj(t)[:, j])
        qis = traj(tq - tau)[:, j]
        vis = lp_deriv(qis, LP_VEL, 1)                               # 与实际速度同一处理，用于钝化增益
        ov = overshoot(tq, q[:, j], qis, traj(tq - tau, 1)[:, j])   # 换向/停下用规划器自己的速度判断（停下时正好为 0）
        rec.update({f"tau_j{j+1}": tau * 1e3, f"pos_aligned_j{j+1}": e1 * 180 / np.pi,
                    f"gain_j{j+1}": speed_gain(va[:, j], vis),
                    f"overshoot_med_j{j+1}": float(np.median(ov)) * 180 / np.pi if ov else float("nan"),
                    f"overshoot_max_j{j+1}": float(np.max(ov)) * 180 / np.pi if ov else float("nan")})
    rec["tau_med"] = float(np.nanmedian([rec[f"tau_j{j+1}"] for j in ACTIVE]))
    rec["gain_med"] = float(np.nanmedian([rec[f"gain_j{j+1}"] for j in ACTIVE]))
    rec["overshoot_med"] = float(np.nanmedian([rec[f"overshoot_med_j{j+1}"] for j in ACTIVE]))
    # IMU 层：陀螺 <15 Hz、陀螺 3–15 Hz（高频跟踪残差）、加计 <3 Hz；旁列实测、参考各自在该频段的量（去均值）
    acc_i, gyr_i = ideal_imu(traj, ti - calib["delay_s"], calib, model, sid)
    gm, am = ep["imu"][:, 3:], ep["imu"][:, :3]
    for key, filt, meas, ref in (("imu_gyro_lt15", lambda x: D.lp(x, LP_VEL, D.FS_I), gm, gyr_i),
                                 ("imu_gyro_hf", lambda x: band(x, D.FS_I), gm, gyr_i),
                                 ("imu_acc_lt3", lambda x: D.lp(x, LP_ACC, D.FS_I), am, acc_i)):
        e = filt(meas - ref)
        for seg in SEGS:
            m = masks["i"][seg]
            rec[f"{key}_{seg}"] = rms(np.linalg.norm(e[m], axis=1)) if m.any() else float("nan")
        rec[f"{key}_act_all"] = rms(np.linalg.norm(filt(meas - meas.mean(0)), axis=1))
        rec[f"{key}_ref_all"] = rms(np.linalg.norm(filt(ref - ref.mean(0)), axis=1))
    # 质量检查：逐条用陀螺对编码器运动学的朝向，与标定 R 的夹角
    f, w, _ = (D.lp(x, 3.0, D.FS_Q) for x in C.predict_site(tq, q))
    g = D.lp(ep["imu"][:, 3:], 3.0, D.FS_I) - np.asarray(calib["gyro_bias"])
    wi = np.stack([np.interp(ti - calib["delay_s"], tq, w[:, k]) for k in range(3)], 1)
    mv = np.linalg.norm(g, axis=1) > C.GYRO_MIN
    rec["qc_rot_vs_calib_deg"] = C._deg(C._kabsch(wi[mv], g[mv]) @ np.asarray(calib["R"]).T) if mv.sum() > 10 else float("nan")
    return rec


# ── 汇总 ─────────────────────────────────────────────────────────────────
MAIN = ["imu_gyro_lt15_all", "imu_gyro_hf_all", "imu_acc_lt3_all",
        "enc_pos_all", "enc_vel_lt15_all", "enc_acc_lt3_all", "tau_med", "gain_med", "overshoot_med"]
UNITS = {"imu_gyro_lt15": "rad/s", "imu_gyro_hf": "rad/s", "imu_acc_lt3": "m/s²", "enc_pos": "°",
         "enc_vel_lt15": "°/s", "enc_acc_lt3": "°/s²", "tau_med": "ms", "gain_med": "", "overshoot_med": "°"}


def paired(a, b, rng, n_boot=10000):
    """配对差 b − a（开 TE − 不开 TE）：中位差、bootstrap 95% 置信区间、Wilcoxon 符号秩 p。"""
    from scipy.stats import wilcoxon
    d = np.asarray(b) - np.asarray(a)
    d = d[np.isfinite(d)]
    boot = np.median(rng.choice(d, (n_boot, len(d))), axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"median_diff": float(np.median(d)), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "wilcoxon_p": p, "n": int(len(d))}


def summarize(df):
    rng = np.random.default_rng(0)
    out = {}
    for task in TASKS:
        cur = {}
        for cond in ("noTE", "TE"):
            sub = df[(df.task == task) & (df.cond == cond)]
            cur[cond] = {c: [float(x) for x in np.nanpercentile(sub[c], [25, 50, 75])]
                         for c in sub.columns if c not in ("task", "cond", "episode")}
        a = df[(df.task == task) & (df.cond == "noTE")].sort_values("episode")
        b = df[(df.task == task) & (df.cond == "TE")].sort_values("episode")
        cur["TE_minus_noTE"] = {c: paired(a[c].to_numpy(), b[c].to_numpy(), rng) for c in MAIN}
        out[task] = cur
    return out


BLUE, ORANGE, INK, MUTED, SHADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdbd6"   # 实测 / 参考：分类色前两位


def plot_overlays(df, model, sid, calib, centers):
    """每个任务一张：4 行（不开 TE 中位/最难、开 TE 中位/最难，按陀螺 <15 Hz 残差挑）× 3 列
    （残差最大的陀螺轴整条、残差最集中的 2 s、误差最大的关节的位置误差）。灰底 = 疑似接触段，虚线 = chunk 边界。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    for task, (no_te, te, _) in TASKS.items():
        rows = []
        for cond, repo in (("noTE", no_te), ("TE", te)):
            sub = df[(df.task == task) & (df.cond == cond)].sort_values("imu_gyro_lt15_all")
            rows += [(cond, repo, int(sub.iloc[len(sub) // 2].episode), "中位"), (cond, repo, int(sub.iloc[-1].episode), "最难")]
        data = []
        for cond, repo, i, tag in rows:
            ep = load(repo, i)
            traj = PlanTrajectory(ep["cmd"], ep["t_send"], q0=ep["q"][0])
            masks, bnd = segment_masks(ep, centers[task], cond == "TE", model)
            gm = D.lp(ep["imu"][:, 3:], LP_VEL, D.FS_I)
            gr = D.lp(ideal_imu(traj, ep["ti"] - calib["delay_s"], calib, model, sid)[1], LP_VEL, D.FS_I)
            k = int(np.argmax(np.sqrt(np.mean((gm - gr) ** 2, axis=0))))
            s0 = int(np.argmax(np.convolve(np.sum((gm - gr) ** 2, axis=1), np.ones(200), "valid")))   # 2 s 窗
            eq = (ep["q"] - traj(ep["tq"])[:, :6]) * 180 / np.pi
            j = ACTIVE[int(np.argmax(np.sqrt(np.mean(eq[:, ACTIVE] ** 2, axis=0))))]
            t0 = ep["t_send"][0]
            data.append(dict(cond=cond, i=i, tag=tag, t=ep["ti"] - t0, tq=ep["tq"] - t0, gm=gm[:, k], gr=gr[:, k], k=k,
                             win=(ep["ti"][s0] - t0, ep["ti"][min(s0 + 199, len(ep["ti"]) - 1)] - t0), eq=eq[:, j], j=j,
                             ci=masks["i"]["contact"], cq=masks["q"]["contact"], bnd=bnd - t0))
        glim = 1.05 * max(max(np.abs(d["gm"]).max(), np.abs(d["gr"]).max()) for d in data)
        qlim = 1.05 * max(np.abs(d["eq"]).max() for d in data)
        fig, ax = plt.subplots(4, 3, figsize=(15, 11), gridspec_kw={"width_ratios": [3, 1.3, 3]})
        for r, d in enumerate(data):
            for c in range(3):
                A = ax[r, c]
                t, shade = (d["tq"], d["cq"]) if c == 2 else (d["t"], d["ci"])
                A.fill_between(t, 0, 1, where=shade, transform=A.get_xaxis_transform(), color=SHADE, lw=0)
                for b in d["bnd"]:
                    A.axvline(b, color=MUTED, ls="--", lw=0.8)
                for side in ("top", "right"):
                    A.spines[side].set_visible(False)
                if c < 2:
                    A.plot(d["t"], d["gm"], color=BLUE, lw=1.0)
                    A.plot(d["t"], d["gr"], color=ORANGE, lw=1.0)
                    A.set_ylim(-glim, glim)
                else:
                    A.plot(d["tq"], d["eq"], color=BLUE, lw=1.0)
                    A.axhline(0, color=ORANGE, lw=1.0)
                    A.set_ylim(-qlim, qlim)
            ax[r, 0].axvspan(*d["win"], fill=False, edgecolor=INK, lw=1.0)
            ax[r, 1].set_xlim(*d["win"])
            ax[r, 0].set_xlim(0, d["t"][-1]); ax[r, 2].set_xlim(0, d["tq"][-1])
            ax[r, 0].set_title(f"{'不开 TE' if d['cond'] == 'noTE' else '开 TE'} · {d['tag']}（第 {d['i']} 条）："
                               f"陀螺 {'xyz'[d['k']]} 轴（rad/s）", loc="left", color=INK)
            ax[r, 1].set_title("残差最集中的 2 s", loc="left", color=INK)
            ax[r, 2].set_title(f"j{d['j'] + 1} 位置误差（实测 − 参考，°）", loc="left", color=INK)
        for c in range(3):
            ax[-1, c].set_xlabel("时间（s，从第一帧起）")
        handles = [plt.Line2D([], [], color=BLUE, lw=2, label="实测"), plt.Line2D([], [], color=ORANGE, lw=2, label="参考（理想渲染）"),
                   plt.Rectangle((0, 0), 1, 1, color=SHADE, label="疑似接触段"),
                   plt.Line2D([], [], color=MUTED, ls="--", lw=0.8, label="chunk 边界"),
                   plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, label="放大窗口")]
        fig.legend(handles=handles, loc="upper right", ncol=5, frameon=False, labelcolor=INK)
        fig.suptitle(f"{task}：实测 vs 理想渲染（a_max = 5 rad/s² 参考假设、按 t_send、不含延迟；陀螺两边同用 <15 Hz 低通）",
                     x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT / f"overlay_{task}.png", dpi=120, facecolor="#fcfcfb")
        plt.close(fig)


def main() -> int:
    gc.disable()                     # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    import mujoco
    import pandas as pd
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    calib = json.loads((HERE.parent / "outputs/imu_calib/imu_calib.json").read_text())
    rows, centers = [], {}
    for task, (no_te, te, demo) in TASKS.items():
        centers[task] = target_center(demo, model)
        print(f"{task}: 目标区中心（示教放下点中位）{np.round(centers[task] * 100, 1)} cm", flush=True)
        for cond, repo in (("noTE", no_te), ("TE", te)):
            n = json.loads((LR / repo / "meta/info.json").read_text())["total_episodes"]
            for i in range(n):
                ep = load(repo, i)
                rec = episode_metrics(ep, PlanTrajectory(ep["cmd"], ep["t_send"], q0=ep["q"][0]), calib, model, sid,
                                      centers[task], cond == "TE")
                rows.append({"task": task, "cond": cond, "episode": i, **rec})
            print(f"  {repo}: {n} 条完成", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "metrics.csv", index=False)
    summary = summarize(df)
    (OUT / "summary.json").write_text(json.dumps({"plan": "reports/PLAN_actual_vs_command.md v2", "a_max": A_MAX,
                                                  "band_hz": BAND, "summary": summary}, ensure_ascii=False, indent=1))
    for task, cur in summary.items():
        nt, te_ = cur["noTE"], cur["TE"]
        print(f"\n=== {task}（中位数 [四分位]；配对差 = 开 TE − 不开 TE）")
        for c in MAIN:
            p, base = cur["TE_minus_noTE"][c], c.removesuffix("_all")
            print(f"  {base:14s}{UNITS[base]:>6s}  不开TE {nt[c][1]:8.3f} [{nt[c][0]:.3f}, {nt[c][2]:.3f}]  "
                  f"开TE {te_[c][1]:8.3f} [{te_[c][0]:.3f}, {te_[c][2]:.3f}]  "
                  f"差 {p['median_diff']:+.3f} CI[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] p={p['wilcoxon_p']:.3g}")
            if f"{base}_act_all" in nt:
                print(f"  {'':20s}  实测/参考自身该频段量  不开TE {nt[base + '_act_all'][1]:.3f}/{nt[base + '_ref_all'][1]:.3f}"
                      f"  开TE {te_[base + '_act_all'][1]:.3f}/{te_[base + '_ref_all'][1]:.3f}")
        for base in ("imu_gyro_lt15", "imu_gyro_hf", "enc_pos"):
            print(f"  分段 {base}：疑似接触/其余  不开TE {nt[base + '_contact'][1]:.3f}/{nt[base + '_noncontact'][1]:.3f}"
                  f"  开TE {te_[base + '_contact'][1]:.3f}/{te_[base + '_noncontact'][1]:.3f}；"
                  f"边界/非边界（不开TE）{nt[base + '_boundary'][1]:.3f}/{nt[base + '_nonboundary'][1]:.3f}")
    print(f"\n标定质量检查：逐条朝向与标定 R 的夹角 中位 {np.nanmedian(df.qc_rot_vs_calib_deg):.2f}°，"
          f"最大 {np.nanmax(df.qc_rot_vs_calib_deg):.2f}°（>1° 的条：{int((df.qc_rot_vs_calib_deg > 1).sum())}）；"
          f"t_send 缺失帧共 {int(df.n_nan_send.sum())}")
    plot_overlays(df, model, sid, calib, centers)
    print(f"→ {OUT}/metrics.csv、summary.json、overlay_pickcube.png、overlay_insert.png")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
