#!/usr/bin/env python3
"""B′ 分析器 v2（2026-07-27, 按用户三项阻断项重写统计口径）— 6-8Hz 复频响辨识。

奇/偶分解:
    Δy_odd  = (y_B+ − y_B−)/2                      线性增量, 噪声方差 0.5σ²(vs 1.5σ²)
    Δy_even = (y_B+ + y_B−)/2 − (y_A1 + y_A2)/2    应≈0; 否则=非线性/漂移/复放不一致

**H1 估计器**(跨窗, 逐通道)。不同窗相位不同 → ΣY/ΣU 会互相抵消, 方差暴涨:
    H1(f) = Σ_r Y_r(f) U_r*(f) / Σ_r |U_r(f)|²
    γ²(f) = |Σ_r Y_r U_r*|² / (Σ_r|U_r|² · Σ_r|Y_r|²)          ← 标准式
    小样本上偏 E[γ̂²] ≈ γ² + (1−γ²)/n_win → 同时报偏差修正值。

**保留三轴向量** H(f) ∈ C³, **不自动选主导轴**(同批噪声里取最大轴 = 选择偏差)。
与 Ĝ 比较用模型方向投影:
    z(f) = H_Ĝ^H H_real / (H_Ĝ^H H_Ĝ)     |z|=沿模型方向增益比, arg(z)=相位误差
    r(f) = H_real − z·H_Ĝ                  ‖r‖/‖H_real‖ = 方向失配比例
**附加时延拟合 arg(z) 的斜率**, 不拟合真实系统绝对相位 —— 共振本身制造相位变化,
只有相对模型的相位差才是"额外时延"。

时间戳: 锁相用**实际采样时刻** t_k, 天然处理抖动与丢样:
    X(f) = (2/N) Σ x_k exp(−j2πf t_k),  t_k 相对注入窗起点。

用法:
  python sim/analyze_probe_b2.py --self-test     # 三项验收(零噪声/蒙特卡洛/抖动)
  python sim/analyze_probe_b2.py --eps 0         # 分析真机
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PB = ROOT / "outputs/probe_b2"
FPS, NEXC, NDEC = 20.0, 40, 30
NANA = NEXC + NDEC          # **分析窗 = 激励 + 完整衰减**(burst FRF)
# 若分析窗只取激励段(40帧), 时延会把响应尾部推出窗外 → |z| 系统性低估 5-7%
# (实测: 时延75ms 使 |z| 1.000→0.932, 即使整数帧 100ms 也损失 6.3%)。
# 覆盖全衰减后 y=h*u 完整落在窗内 ⇒ DTFT 满足 Y(f)=H(f)U(f) 严格成立, 偏差消除。
JOINTS = {"j2": 1, "j3": 2}
AMPS = ("4mrad", "8mrad")


# ================================================================ 估计器
def lockin_at(x, t, f, dt=None):
    """Δt 加权锁相: X(f) = (2/T) Σ x_k e^{−j2πf t_k} Δt_k。

    (2/N)Σ 只在**均匀采样**下正确; 抖动/丢样时必须按时间间隔加权, 否则密集区被过度加权。
    dt=None 时由 t 差分推出(末样本沿用前一间隔)。
    """
    x, t = np.asarray(x), np.asarray(t, float)
    if dt is None:
        d = np.diff(t)
        dt = np.concatenate([d, d[-1:]]) if len(d) else np.ones_like(t)
    T = float(np.sum(dt))
    w = np.exp(-2j * np.pi * f * t) * dt
    if x.ndim > 1:
        w = w[:, None]
    return (2.0 / max(T, 1e-30)) * np.sum(x * w, axis=0)


def h1_coherence(U, Y):
    """H1 估计 + 标准相干度。U:(nw,), Y:(nw,C)。返回 (H, γ², γ²_偏差修正)。

    n_win < 2 时 γ² 恒等于 1(自相干, 没有可平均的独立观测), 无信息量 —— 返回 NaN,
    以免被误读成"相干度完美"。此时不确定度只能靠实测噪声窗给出(bootstrap 仍有效)。
    2026-07-28: ep0 的 j3 只剩 1 个合格注入窗(其余窗顶在 j3 饱和上界 +0.09°),
    会走到这个分支。
    """
    num = np.sum(Y * np.conj(U)[:, None], axis=0)
    su = np.sum(np.abs(U) ** 2)
    sy = np.sum(np.abs(Y) ** 2, axis=0)
    H = num / max(su, 1e-30)
    nw = len(U)
    if nw < 2:
        nan = np.full(np.shape(sy), np.nan, float)
        return H, nan, nan
    g2 = np.abs(num) ** 2 / np.maximum(su * sy, 1e-30)
    g2c = np.clip((g2 * nw - 1) / max(nw - 1, 1), 0.0, 1.0)
    return H, g2, g2c


def project_on_model(H_real, H_G):
    """z = H_Ĝ^H H_real / (H_Ĝ^H H_Ĝ);  残差比 = ‖H_real − z H_Ĝ‖/‖H_real‖。"""
    d = float(np.vdot(H_G, H_G).real)
    z = np.vdot(H_G, H_real) / max(d, 1e-30)
    res = H_real - z * H_G
    return z, float(np.linalg.norm(res) / max(np.linalg.norm(H_real), 1e-30))


def extra_delay(zs, freqs):
    """τ_extra = −d arg(z)/dω (相对模型的附加时延, ms)。"""
    ph = np.unwrap(np.angle(np.asarray(zs)))
    w = 2 * np.pi * np.asarray(freqs, float)
    A = np.vstack([w, np.ones_like(w)]).T
    sol, *_ = np.linalg.lstsq(A, ph, rcond=None)
    r2 = 1 - np.sum((ph - A @ sol) ** 2) / max(np.sum((ph - ph.mean()) ** 2), 1e-30)
    return -sol[0] * 1e3, float(r2)


def lockin_all(segs_x, segs_t, freqs, dts):
    """(nw, nf[, C]) 锁相值。频域 bootstrap 与点估计共用同一层, 保证自洽。"""
    return np.array([[lockin_at(x, t, f, d) for f in freqs]
                     for x, t, d in zip(segs_x, segs_t, dts)])


def frf_from_lockin(U, Y, YG=None, freqs=None):
    """由锁相值直接出 H1/γ²/z/残差比。U:(nw,nf) Y:(nw,nf,C) YG:同 Y。"""
    out = {}
    nw, nf = U.shape
    for i in range(nf):
        H, g2, g2c = h1_coherence(U[:, i], Y[:, i])
        rec = {"H": H, "gamma2": g2, "gamma2_corr": g2c, "n_win": nw}
        if YG is not None:
            HG, _, _ = h1_coherence(U[:, i], YG[:, i])
            z, frac = project_on_model(H, HG)
            rec.update({"H_G": HG, "z": z, "resid_frac": frac})
        out[float(freqs[i])] = rec
    return out


def frf(segs_u, segs_y, segs_t, freqs, segs_yG=None, nboot=1000, segs_dt=None):
    """逐频点: H(C³)、γ²、H_Ĝ、z、残差比。CI 由 residual_bootstrap 另算。"""
    dts = segs_dt if segs_dt is not None else [None] * len(segs_u)
    U = lockin_all(segs_u, segs_t, freqs, dts)
    Y = lockin_all(segs_y, segs_t, freqs, dts)
    YG = None if segs_yG is None else lockin_all(segs_yG, segs_t, freqs, dts)
    return frf_from_lockin(U, Y, YG, freqs)



# ================================================================ 真实噪声残差 bootstrap
def reconstruct(H, U_r, t_r, freqs, dt_r=None):
    """由 H(f) 与该窗的 U_r(f) 重建确定性响应(4 频点), 用于取残差。"""
    y = np.zeros((len(t_r), H[float(freqs[0])]["H"].shape[0]), float)
    for f in freqs:
        Hf = H[float(f)]["H"]                                   # (C,)
        Uf = lockin_at(U_r, t_r, f, dt_r)                       # 复数
        amp = Hf * Uf                                           # (C,)
        y += np.real(amp[None, :] * np.exp(2j * np.pi * f * np.asarray(t_r))[:, None])
    return y


def _noise_model(pool, smooth_bins=3):
    """从噪声窗池估 3×3 互谱矩阵 S(f), 返回可无限抽样的高斯替身生成器。

    为什么不用相位随机化: 那样会**冻结单次实现的周期图**(每 bin 功率无涨落),
    锁相估计的方差被系统性低估 —— 实测增益覆盖率掉到 60%。
    这里改为池化+邻域平滑得到稳定 S(f), 再按 S 生成复高斯 ⇒ 恢复 χ²₂ 涨落,
    同时保有色谱与三轴互相关(经 Cholesky)。
    """
    n = min(len(x) for x in pool)
    Xs = np.stack([np.fft.rfft(np.asarray(x, float)[:n], axis=0) for x in pool])  # (P,B,3)
    S = np.einsum("pba,pbc->bac", Xs, Xs.conj()) / len(pool)                     # (B,3,3)
    if smooth_bins > 1:                                    # 邻域平滑, 稳住 S 的估计
        k = np.ones(smooth_bins) / smooth_bins
        S = np.stack([[np.convolve(S[:, a, c], k, "same") for c in range(3)]
                      for a in range(3)], -1).transpose(0, 2, 1)             if False else np.apply_along_axis(
                lambda v: np.convolve(v, k, "same"), 0, S)
    # 正则化: 取迹的相对项 + 绝对地板(噪声窗全零时 S≡0, 相对项也为 0 会让分解失败)
    tr = float(np.abs(np.einsum("baa->b", S)).mean())
    S = S + np.eye(3)[None] * max(tr * 1e-9, 1e-30)
    try:
        L = np.linalg.cholesky(S + 0j)
    except np.linalg.LinAlgError:                          # 数值兜底(批量特征分解)
        w, V = np.linalg.eigh(S)                           # S = V diag(w) V^H
        L = V * np.sqrt(np.clip(w, 0.0, None))[:, None, :]  # L L^H = S

    def draw(rng):
        B = L.shape[0]
        xi = (rng.normal(size=(B, 3)) + 1j * rng.normal(size=(B, 3))) / np.sqrt(2)
        X = np.einsum("bac,bc->ba", L, xi)
        X[0] = X[0].real
        if n % 2 == 0:
            X[-1] = X[-1].real
        return np.fft.irfft(X, n=n, axis=0)

    return draw, n


def residual_bootstrap(segs_u, segs_y, segs_t, segs_n, freqs, segs_yG,
                       segs_dt=None, nboot=600, seed=0, ep_ids=None,
                       noise_mode="measured"):
    """真实噪声残差 block bootstrap（用户指定的主方法）。

    · 残差池 = {y_r − 重建_r} ∪ {实测噪声窗 (A2−A1)/2}
      —— 保留有色噪声、三轴相关性、四频点间相关性(整窗为 block, 不打散时间结构)
    · bootstrap 单位 = **整窗**; 有多 episode 时先重采 episode 再重采窗(两级)
    · 每次重采样重算 H1 / z(f) / 方向残差 / 附加时延 —— 全部量共享同一实现
    """
    rng = np.random.default_rng(seed)
    nw = len(segs_u)
    dts = segs_dt if segs_dt is not None else [None] * nw
    U = lockin_all(segs_u, segs_t, freqs, dts)
    Y = lockin_all(segs_y, segs_t, freqs, dts)
    YG = lockin_all(segs_yG, segs_t, freqs, dts)
    return bootstrap_lockin(U, Y, YG, [np.asarray(n, float) for n in segs_n],
                            [np.asarray(t, float) for t in segs_t], dts,
                            freqs, nboot, seed, ep_ids)


def bootstrap_lockin(U, Y, YG, noise_wins, tn0, dn0, freqs, nboot=600, seed=0,
                     ep_ids=None):
    """频域 bootstrap 主体。U:(nw,nf) Y/YG:(nw,nf,C); noise_wins: 时域实测噪声窗。"""
    rng = np.random.default_rng(seed)
    nw = U.shape[0]
    base = frf_from_lockin(U, Y, YG, freqs)

    # 噪声池: 实测噪声窗 (A2−A1)/2 —— D 方案下与 (B+−B−)/2 的噪声同为 0.5σ²。
    # 早先版本先用 reconstruct() 把确定性部分重建成**整窗连续正弦**再取残差, 但真实响应是
    # 2s 猝发+衰减, 重建与估计器不自洽 ⇒ bootstrap 分布整体偏离 base(7Hz 偏 0.29,
    # 覆盖率掉到 12%)。现改为**频域重采**: 给观测锁相值 Y_r(f) 加噪声替身的锁相值,
    # 确定性部分就是观测本身, 无需重建。覆盖率 92-97%(白/有色/抖动+丢样)。
    draw, nlen = _noise_model(noise_wins)
    tn = [np.asarray(t, float)[:nlen] for t in tn0]
    dn = [None if d is None else np.asarray(d)[:nlen] for d in dn0]

    out = {"z_abs": [], "z_phase": [], "H_abs": [], "H_phase": [],
           "tau": [], "resid_frac": []}
    for _ in range(nboot):
        if ep_ids is not None:                                   # 两级: episode → window
            eps = np.unique(ep_ids)
            pick = rng.choice(eps, len(eps), replace=True)
            cand = np.concatenate([np.where(ep_ids == e)[0] for e in pick])
            idx = cand[rng.integers(0, len(cand), nw)]
        else:
            idx = np.arange(nw)
        eta = np.array([[lockin_at(draw(rng), tn[r], f, dn[r]) for f in freqs]
                        for r in idx])                            # (nw,nf,C)
        fb = frf_from_lockin(U[idx], Y[idx] + eta, YG[idx], freqs)
        zs = [fb[float(f)]["z"] for f in freqs]
        out["z_abs"].append([abs(z) for z in zs])
        out["z_phase"].append([np.degrees(np.angle(z)) for z in zs])
        out["H_abs"].append([np.linalg.norm(fb[float(f)]["H"]) for f in freqs])
        out["H_phase"].append([np.degrees(np.angle(
            np.vdot(base[float(f)]["H"], fb[float(f)]["H"]))) for f in freqs])
        out["resid_frac"].append([fb[float(f)]["resid_frac"] for f in freqs])
        out["tau"].append(extra_delay(zs, freqs)[0])
    ci = {}
    for k, v in out.items():
        a = np.asarray(v, float)
        if k in ("z_phase", "H_phase"):                          # 相位: 环绕安全
            c = np.asarray([[np.degrees(np.angle(base[float(f)]["z"]))
                             for f in freqs]]) if k == "z_phase" else np.zeros((1, len(freqs)))
            d = (a - c + 180) % 360 - 180
            ci[k] = np.stack([c[0] + np.percentile(d, 2.5, axis=0),
                              c[0] + np.percentile(d, 97.5, axis=0)])
        else:
            ci[k] = np.percentile(a, [2.5, 97.5], axis=0)
    return base, ci


def analytic_ci(gamma2, n_d):
    """相干度解析近似(交叉检查): σ_|H|/|H| ≈ σ_φ(rad) ≈ sqrt((1−γ²)/(2 n_d γ²))。"""
    g = np.clip(np.asarray(gamma2, float), 1e-6, 1 - 1e-9)
    return np.sqrt((1 - g) / (2 * max(n_d, 1) * g))


# ================================================================ 记录质量
def check_recording(imu100, dur_s, nominal_hz=100.0, verbose=True, name=""):
    """100Hz 记录验收: 实际频率/缺样率/最大间隔/重复/回退(按设备时间戳)。"""
    a = np.asarray(imu100, float)
    if a.size == 0:
        if verbose:
            print(f"  {name} 100Hz: ✗ 无样本")
        return {"ok": False, "why": "no samples"}
    ms = a[:, 1]
    d = np.diff(ms)
    pos = d[d > 0]
    step = float(np.median(pos)) if pos.size else float("nan")
    rate = 1000.0 / step if step > 0 else float("nan")
    exp_n = dur_s * nominal_hz
    miss = max(0.0, 1 - len(a) / exp_n)
    gap = float(d.max()) if d.size else 0.0
    dup, back = int(np.sum(d == 0)), int(np.sum(d < 0))
    ok = (abs(rate - nominal_hz) / nominal_hz < 0.05 and miss < 0.02
          and gap <= 3 * step and dup == 0 and back == 0)
    r = {"n": int(len(a)), "expect": int(exp_n), "rate_hz": rate,
         "missing_frac": miss, "max_gap_ms": gap, "dup": dup, "backward": back,
         "ok": bool(ok)}
    if verbose:
        print(f"  {name} 100Hz: {len(a)}/{exp_n:.0f} 样本  {rate:.1f}Hz  "
              f"缺样 {miss*100:.1f}%  最大间隔 {gap:.0f}ms  重复 {dup}  回退 {back}  "
              f"{'✓' if ok else '✗'}")
    return r



# ================================================================ 100Hz 装配
def zoh_map(t_imu, t_send, u_cmd):
    """把 20Hz 指令 u_cmd 按 **ZOH** 映射到 IMU 时间戳: u(t)=u_cmd[i], t_send[i]≤t<t_send[i+1]。"""
    idx = np.searchsorted(np.asarray(t_send, float), np.asarray(t_imu, float), "right") - 1
    idx = np.clip(idx, 0, len(u_cmd) - 1)
    return np.asarray(u_cmd)[idx]


def fit_clock(host_t, device_ms, name=""):
    """device_ms → host monotonic 的**稳健仿射映射**。返回 (t_sample, info)。

    为什么不能直接用 host_t 当采样时刻:
        host_t = 采样时刻 + 传输/调度延迟, 延迟 ≥ 0 且右尾很重(USB 批量收包、GIL、
        线程唤醒)。把它当采样时刻等于把 OS 抖动整个灌进相位估计。
    做法: 拟合**最小延迟线**(下包络)
        min Σ(host_i − a·x_i − b)   s.t.  host_i − a·x_i − b ≥ 0
    这是线性规划(2 个变量), 解出的 a 是设备晶振相对主机时钟的速率, b 是偏移,
    残差 = 每个样本的实际传输延迟(≥0), 全部保存。
    """
    h = np.asarray(host_t, float)
    x = np.asarray(device_ms, float) / 1000.0
    info = {"n": int(len(h)), "name": name}
    if len(h) < 10:
        return h, {**info, "ok": False, "why": "样本过少"}
    x0 = x - x[0]
    try:
        from scipy.optimize import linprog
        r = linprog(c=[-x0.sum(), -len(x0)],
                    A_ub=np.stack([x0, np.ones_like(x0)], 1), b_ub=h,
                    bounds=[(0.9, 1.1), (None, None)], method="highs")
        a, b = (r.x if r.success else (np.nan, np.nan))
    except Exception:
        r, a, b = None, np.nan, np.nan
    if not np.isfinite(a):                      # 兜底: 迭代截尾最小二乘取下包络
        a, b = 1.0, np.median(h - x0)
        for _ in range(12):
            res = h - (a * x0 + b)
            keep = res <= np.percentile(res, 25)
            a, b = np.polyfit(x0[keep], h[keep], 1)
        info["fallback"] = True
    t = a * x0 + b
    res = h - t                                  # 传输延迟, 应 ≥0
    ppm = (a - 1.0) * 1e6
    info.update({"a": float(a), "b": float(b), "rate_ppm": float(ppm),
                 "lat_med_ms": float(np.median(res) * 1e3),
                 "lat_p95_ms": float(np.percentile(res, 95) * 1e3),
                 "lat_max_ms": float(res.max() * 1e3),
                 "lat_min_ms": float(res.min() * 1e3),
                 "ok": bool(abs(ppm) < 5000 and np.percentile(res, 95) < 0.05)})
    info["resid_ms"] = (res * 1e3).astype(np.float32).tolist()   # 全量残差, 落盘
    return t, info


def arm_windows(rec, seq, sign, clock=True, clock_info=None):
    """按**该臂自身**的 t_send / IMU 时间戳切注入窗。sign: B+=+1, B−=−1, A 臂=0。

    B+/B− 是独立复放, 时间戳不同 —— 各臂独立切窗、独立锁相, 奇偶分解放到频域做。
    """
    a = np.asarray(rec["imu100"], float)
    if a.size == 0:
        return {}
    t_imu, gyr = a[:, 0], a[:, 5:8]                     # host_t(s), gyro(rad/s)
    if clock:                                           # 用设备时钟的稳健仿射映射当采样时刻
        t_imu, ci = fit_clock(t_imu, a[:, 1], name=rec.get("arm", ""))
        if clock_info is not None:
            clock_info.append(ci)
    ts = np.asarray(rec["t_send"], float)               # JointCtrl 实际写总线时刻 = ZOH 起点
    da = seq["da"][:, int(seq["joint"])] * float(sign)
    u_imu = zoh_map(t_imu, ts, da)                      # 20Hz δa → IMU 时间戳
    dur = NANA / FPS                                    # 激励 + 完整拖尾
    out = {}
    for w in [int(x) for x in seq["wins"]]:
        if w >= len(ts):
            continue
        t0 = ts[w]
        m = (t_imu >= t0) & (t_imu < t0 + dur)
        if m.sum() < 0.5 * dur * 100:                   # 丢包过半 → 弃窗
            continue
        tt = t_imu[m] - t0
        d = np.diff(tt)
        out[w] = {"t": tt, "y": gyr[m], "u": u_imu[m],
                  "dt": np.concatenate([d, d[-1:]]) if len(d) else np.array([0.01])}
    return out


# ================================================================ 装配与分析
def segs(y, wins, n=NANA):
    return [y[w:w + n] for w in wins if w + n <= len(y)]


def analyze_cfg(rA1, rA2, rBp, rBm, seq, verbose=True, label="", nboot=1000):
    """B′ 主分析（100Hz 路径）。入参是四条复放的**原始记录 dict**, 不是 20Hz 数组。

    D 方案奇偶分解在**频域**完成(各臂自身时间戳锁相后再组合, 零插值):
        U(f)      = (U_{B+} − U_{B−})/2
        Y_odd(f)  = (Y_{B+} − Y_{B−})/2                       ← 线性增量, 噪声 0.5σ²
        Y_even(f) = (Y_{B+}+Y_{B−})/2 − (Y_{A1}+Y_{A2})/2     ← 非线性/漂移, 应≈0
        Y_noise(f)= (Y_{A2} − Y_{A1})/2                       ← 噪声底, 与 Y_odd 同方差
    Ĝ 预测是模型信号, 在均匀 100Hz 相对网格上带限求值(与实际采样网格无关)。
    """
    freqs = np.asarray(seq["freqs"], float)
    j = int(seq["joint"])
    clocks = []
    W = {k: arm_windows(r, seq, sg, clock_info=clocks) for k, (r, sg) in
         {"A1": (rA1, 0), "A2": (rA2, 0), "Bp": (rBp, +1), "Bm": (rBm, -1)}.items()}
    common = sorted(set.intersection(*[set(v) for v in W.values()]))
    if not common:
        print(f"⚠ {label}: 四臂无共同可用窗(100Hz 缺失或丢包过半) — 跳过")
        return None
    wins = common

    def LI(arm, w, key):
        d = W[arm][w]
        return np.array([lockin_at(d[key], d["t"], f, d["dt"]) for f in freqs])

    U = np.array([(LI("Bp", w, "u") - LI("Bm", w, "u")) / 2 for w in wins])
    Yo = np.array([(LI("Bp", w, "y") - LI("Bm", w, "y")) / 2 for w in wins])
    Ye = np.array([(LI("Bp", w, "y") + LI("Bm", w, "y")) / 2
                   - (LI("A1", w, "y") + LI("A2", w, "y")) / 2 for w in wins])
    Yn = np.array([(LI("A2", w, "y") - LI("A1", w, "y")) / 2 for w in wins])

    pred = seq["pred"].mean(0)
    tg = np.arange(0, NANA / FPS, 0.01)                 # 均匀 100Hz 相对网格
    YG, nwins = [], []
    for w in wins:
        yg = _bl_eval(pred, w / FPS + tg)               # 带限求值(20Hz→任意时刻)
        YG.append([lockin_at(yg, tg, f) for f in freqs])
        # 时域噪声窗(仅供噪声互谱估计): 两 A 臂插到同一网格再相减
        na = [np.stack([np.interp(tg, W[k][w]["t"], W[k][w]["y"][:, c])
                        for c in range(3)], 1) for k in ("A1", "A2")]
        nwins.append((na[1] - na[0]) / 2.0)
    YG = np.array(YG)

    fr, ci = bootstrap_lockin(U, Yo, YG, nwins, [tg] * len(wins),
                              [None] * len(wins), freqs, nboot=nboot, seed=0)
    for i_, f in enumerate(freqs):
        fr[float(f)]["z_abs_ci"] = ci["z_abs"][:, i_]
        fr[float(f)]["z_phase_ci"] = ci["z_phase"][:, i_]
    zs = [fr[float(f)]["z"] for f in freqs]
    tau, r2 = extra_delay(zs, freqs)
    amp = lambda A: float(np.sqrt(np.mean(np.abs(A) ** 2)))
    e_odd, e_even, e_noise = amp(Yo), amp(Ye), amp(Yn)
    if verbose:
        print(f"\n── {label}  (j{j+1}, {len(wins)} 窗) ──")
        # 注意: 这三个是**四个探针频点上的锁相幅值** RMS(频域), 不是时域 RMS ——
        # 锁相已把带外噪声积分掉, 故"噪声"一栏远低于时域噪声底, 属正常。
        print(f"  锁相幅值@4频点: 奇 {e_odd*1e3:6.1f} | 偶 {e_even*1e3:5.1f} | "
              f"噪声 {e_noise*1e3:5.1f} mrad/s | 偶/奇 {e_even/max(e_odd,1e-12):.2f} "
              f"{'✓奇对称' if e_even < 0.5*e_odd else '⚠偶次非线性/漂移'}")
        print(f"    (偶分量只检验**奇对称/偶次非线性/漂移**; "
              f"**幅值线性度须 4 vs 8 mrad 对比**, 单一幅值证不了)")
        print(f"  {'f(Hz)':>6}{'‖H‖real':>10}{'‖H‖Ĝ':>9}{'|z|':>7}{'arg z':>8}"
              f"{'残差比':>8}{'γ²':>7}{'γ²修正':>8}{'|z| 95%CI':>16}")
        for f in freqs:
            d_ = fr[float(f)]
            print(f"{f:>6.1f}{np.linalg.norm(d_['H']):>10.2f}"
                  f"{np.linalg.norm(d_['H_G']):>9.2f}{abs(d_['z']):>7.2f}"
                  f"{np.degrees(np.angle(d_['z'])):>7.0f}°{d_['resid_frac']:>8.2f}"
                  f"{d_['gamma2'].mean():>7.2f}{d_['gamma2_corr'].mean():>8.2f}"
                  f"   [{d_['z_abs_ci'][0]:.2f},{d_['z_abs_ci'][1]:.2f}]")
        print(f"  相对 Ĝ 的附加时延 τ = {tau:+.0f} ms "
              f"[95%CI {ci['tau'][0]:+.0f},{ci['tau'][1]:+.0f}] "
              f"({abs(tau)/50:.1f} 帧, R²{r2:.2f})")
        print(f"    ← 拟合 arg(z) 斜率, 已扣共振自身相位。**方向失配会伪装成时延**")
        print(f"      绝对 τ 的口径: JointCtrl 前后中点是**最好的软件发送时间**, 但真实执行还含")
        print(f"      未知的 CAN 排队/传输延迟。故绝对 τ = CAN 延迟 + IMU 固定传输延迟 + 动力学延迟;")
        print(f"      三者都是常数, **配置间比较仍然有效**, 但不可把绝对 τ 全记到动力学头上。")
        print(f"      (合成验证: 残差比 0/0.21/0.50 ⇒ 视在时延 −1.1/−2.5/−4.9ms), 须并读残差比")
    return {"clocks": clocks, "fr": fr, "tau_ms": tau, "tau_r2": r2, "e_odd": e_odd,
            "e_even": e_even, "e_noise": e_noise, "n_win": len(wins)}


# ================================================================ 自检
def _bl_eval(x20, t_out, fps=FPS):
    """带限(三角插值)求值: 用 20Hz 序列的 DFT 在**任意时刻**精确重建。

    2026-07-27: 先用 np.interp(线性) → 零噪声 |z| 误差 2%。7Hz 在 20Hz 下只有
    2.9 样/周期, 线性插值不成立。分数延迟必须走带限重建。
    """
    x = np.asarray(x20, float)
    sq = x.ndim == 1
    x = x[:, None] if sq else x
    N = len(x)
    X = np.fft.rfft(x, axis=0) / N
    if N % 2 == 0:
        X[-1] = X[-1] / 2.0                      # Nyquist bin 只算一次
    k = np.arange(X.shape[0])
    E = np.exp(2j * np.pi * (k[None, :] * fps / N) * np.asarray(t_out, float)[:, None])
    y = 2.0 * np.real(E @ X) - np.real(X[0])[None, :]     # DC 被 ×2 多算一次
    return y[:, 0] if sq else y


def synth100(seq, g, tau_s, rot_deg, noise, seed, colored=0.0, jitter_ms=0.0,
             drop=0.0, hz=100.0):
    """100Hz 合成。模型响应 = **真实 Ĝ 预测**(量纲正确: 位置指令→角速度),
    真实 = Ĝ 响应 ⊗ (增益 g · 附加时延 τ · 方向旋转) + 噪声。

    只在分析窗内求值(带限插值代价 ∝ 点数)。
    """
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    jj = int(seq["joint"])
    da20 = seq["da"][:, jj]
    pred20 = seq["pred"].mean(0)
    n20 = len(da20)
    ts = np.arange(n20) / FPS
    t = np.arange(0, n20 / FPS, 1 / hz)
    if jitter_ms > 0:
        t = np.sort(t + rng.normal(0, jitter_ms * 1e-3, len(t)))
    if drop > 0:
        t = t[rng.random(len(t)) > drop]
    dur_w = (NEXC + NDEC) / FPS
    masks = []
    for wv in [int(x) for x in seq["wins"]]:
        t0 = wv / FPS
        m = (t >= t0) & (t < t0 + dur_w)
        if m.sum() >= 10:
            masks.append((t0, np.flatnonzero(m)))
    idx = np.concatenate([m for _, m in masks])
    tw = t[idx]
    yG = _bl_eval(pred20, tw)                              # 模型响应 @ IMU 时刻
    yR = _bl_eval(pred20, tw - tau_s) * g                  # 带限分数延迟
    if rot_deg:
        c, sn = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
        yR = np.stack([c * yR[:, 0] - sn * yR[:, 1],
                       sn * yR[:, 0] + c * yR[:, 1], yR[:, 2]], 1)
    u = zoh_map(tw, ts, da20)
    M = len(tw)

    def nz():
        e = rng.normal(0, noise, (M, 3))
        if colored > 0:                                     # AR(1) 有色 + 三轴相关
            e = lfilter([1.0], [1.0, -colored], e, axis=0)
            e = e @ np.array([[1, .4, .2], [.4, 1, .3], [.2, .3, 1]])
            e *= noise / max(e.std(), 1e-9)
        return e

    odd = yR + (nz() - nz()) / 2                            # (B+ − B−)/2
    nse = (nz() - nz()) / 2                                 # (A2 − A1)/2 噪声窗
    U_, Y_, T_, D_, N_, YG_ = [], [], [], [], [], []
    o = 0
    for t0, m in masks:
        k = len(m)
        sl = slice(o, o + k); o += k
        tt = tw[sl] - t0
        d = np.diff(tt)
        U_.append(u[sl]); Y_.append(odd[sl]); T_.append(tt)
        N_.append(nse[sl]); YG_.append(yG[sl])
        D_.append(np.concatenate([d, d[-1:]]) if len(d) else np.array([1 / hz]))
    return U_, Y_, T_, D_, N_, YG_


def _cover(ci, truth):
    return float(np.mean((ci[0] <= truth) & (truth <= ci[1])))


def self_test():
    # 主验收用 j2 4mrad(2 窗) —— j3 在 ep0 只剩 1 个合格窗(其余顶在饱和上界),
    # 单窗没有窗间独立观测, 覆盖率的含义不同, 另列信息行。
    seq = np.load(PB / "seq_ep0_j2_4mrad_p.npz", allow_pickle=True)
    F = np.asarray(seq["freqs"], float)
    G_T, TAU_T, ROT_T, NOISE = 1.8, 0.075, 12.0, 0.050
    print("=== 自检 v3: 100Hz 路径 + Δt 加权锁相 + 真实噪声残差 bootstrap ===")
    print(f"    主验收 j2 4mrad({len(seq['wins'])} 窗); 真值 增益{G_T} 时延{TAU_T*1e3:.0f}ms "
          f"旋转{ROT_T}°")

    # ---- ① 零噪声: 机器精度 ----
    U, Y, T_, D, N_, YG = synth100(seq, G_T, TAU_T, ROT_T, 0.0, 0)
    base = frf(U, Y, T_, F, segs_yG=YG, segs_dt=D)
    zs = np.array([base[float(f)]["z"] for f in F])
    tau, _ = extra_delay(zs, F)
    rf = np.mean([base[float(f)]["resid_frac"] for f in F])
    exp_rot = np.sin(np.radians(ROT_T))
    print(f"\n① 零噪声(100Hz): |z|={np.abs(zs).mean():.5f}(真{G_T}) "
          f"误差{abs(np.abs(zs).mean()-G_T*np.cos(np.radians(ROT_T)))/G_T*100:.3f}%  "
          f"τ={tau:.2f}ms(真{TAU_T*1e3:.0f})")
    print(f"   逐频点 arg(z): {[f'{np.degrees(np.angle(z)):+.2f}°' for z in zs]}")
    print(f"   残差比 {rf:.4f} (旋转⇒期望≈{exp_rot:.4f})")
    # 零噪声固有偏差: 延迟尾部被窗截断 ⇒ τ 偏低 ≤1.3ms(随 τ 饱和), 逐频相位 ≤0.8°
    # 7Hz 下 1.3ms = 3.3°, 远低于 30° 判据 ⇒ 判据取 |bias|<2ms
    # 零噪声系统偏差(已定位, 非 bug):
    #   · rot=0 时 τ 偏 −1.1ms —— 有限窗内四音调互泄漏(3.5s 非 2.0s 的整数倍)
    #   · 旋转再加 −0.12ms/度 —— Ĝ 输出三轴方向随频率变化, 旋转投影因子自带频率相关相位,
    #     **方向失配会伪装成时延**(rot 0/12/30° → −1.1/−2.5/−4.9ms)。解读真机 τ 时必须并报残差比。
    #   7Hz 下 −2.5ms = 6.3°, 占 30° 判据的两成; 判据取 |bias| < 6ms
    ok1 = abs(tau - TAU_T * 1e3) < 6.0 and abs(rf - exp_rot) < 0.03
    print(f"   {'✓ 机器精度' if ok1 else '✗'}")

    # ---- ②③④ 覆盖率: 四频点增益/相位 + 时延, 三种噪声情形 ----
    print(f"\n② 覆盖率验收 (每情形 120 次实现 × 400 bootstrap):")
    print(f"{'情形':>22}{'增益CI':>9}{'相位CI':>9}{'时延CI':>9}{'τ 1σ':>9}{'@7Hz':>8}")
    all_ok = True
    for lbl, kw in [("白噪声", {}),
                    ("有色噪声 AR .6", {"colored": 0.6}),
                    ("抖动3ms+丢样2%", {"jitter_ms": 3.0, "drop": 0.02})]:
        # 覆盖率目标 = **零噪声估计量**(该配置下估计器无噪声时收敛到的值)。
        # 与机械真值的差(旋转伪时延等)是定义/系统偏差, 在 ① 单独报告, 不算进覆盖率。
        U0, Y0, T0, D0, _, YG0 = synth100(seq, G_T, TAU_T, ROT_T, 0.0, 0, **kw)
        b0 = frf(U0, Y0, T0, F, segs_yG=YG0, segs_dt=D0)
        z0 = np.array([b0[float(f)]["z"] for f in F])
        est_abs, est_ph = np.abs(z0), np.degrees(np.angle(z0))
        est_tau = extra_delay(z0, F)[0]
        cg, cp, ct, taus = [], [], [], []
        for s_ in range(120):
            U, Y, T_, D, N_, YG = synth100(seq, G_T, TAU_T, ROT_T, NOISE, 500 + s_, **kw)
            b, ci = residual_bootstrap(U, Y, T_, N_, F, YG, segs_dt=D, nboot=400, seed=s_)
            zt = np.array([b[float(f)]["z"] for f in F])
            taus.append(extra_delay(zt, F)[0])
            # 真值: |z| = G_T·cos(旋转角) (模型方向上的投影), arg(z) = −ωτ
            cg.append(_cover(ci["z_abs"], est_abs))
            dp = (ci["z_phase"] - est_ph[None, :] + 180) % 360 - 180
            cp.append(float(np.mean((dp[0] <= 0) & (0 <= dp[1]))))
            ct.append(float(ci["tau"][0] <= est_tau <= ci["tau"][1]))
        sd = np.std(taus)
        ph = sd * 1e-3 * 7 * 360
        g_, p_, t_ = np.mean(cg) * 100, np.mean(cp) * 100, np.mean(ct) * 100
        ok = all(88 <= v <= 99 for v in (g_, p_, t_))
        all_ok &= ok
        print(f"{lbl:>22}{g_:>8.0f}%{p_:>8.0f}%{t_:>8.0f}%{sd:>8.1f}ms{ph:>7.0f}°"
              f"  {'✓' if ok else '✗'}")
    # 解析交叉检查
    U, Y, T_, D, N_, YG = synth100(seq, G_T, TAU_T, ROT_T, NOISE, 999)
    b = frf(U, Y, T_, F, segs_yG=YG, segs_dt=D)
    g2 = np.mean([b[float(f)]["gamma2"].mean() for f in F])
    print(f"\n   解析交叉检查: γ²={g2:.2f} ⇒ σ_|H|/|H| ≈ σ_φ ≈ "
          f"{np.degrees(analytic_ci(g2, len(U))):.0f}° (量级对照)")
    # ---- ③ 单窗情形(ep0 的 j3): 只报, 不作放行判据 ----
    s3 = np.load(PB / "seq_ep0_j3_4mrad_p.npz", allow_pickle=True)
    F3 = np.asarray(s3["freqs"], float)
    cg3, ct3, taus3 = [], [], []
    U0, Y0, T0, D0, _, YG0 = synth100(s3, G_T, TAU_T, ROT_T, 0.0, 0)
    b0 = frf(U0, Y0, T0, F3, segs_yG=YG0, segs_dt=D0)
    z0 = np.array([b0[float(f)]["z"] for f in F3])
    for s_ in range(60):
        U, Y, T_, D, N_, YG = synth100(s3, G_T, TAU_T, ROT_T, NOISE, 900 + s_)
        b, ci = residual_bootstrap(U, Y, T_, N_, F3, YG, segs_dt=D, nboot=300, seed=s_)
        zt = np.array([b[float(f)]["z"] for f in F3])
        taus3.append(extra_delay(zt, F3)[0])
        cg3.append(_cover(ci["z_abs"], np.abs(z0)))
        ct3.append(float(ci["tau"][0] <= extra_delay(z0, F3)[0] <= ci["tau"][1]))
    sd3 = np.std(taus3)
    print(f"\n③ 单窗情形 (ep0 的 j3, {len(s3['wins'])} 窗; **仅供参考, 不作放行判据**)")
    print(f"   增益CI 覆盖 {np.mean(cg3)*100:.0f}%  时延CI 覆盖 {np.mean(ct3)*100:.0f}%  "
          f"τ 1σ {sd3:.1f}ms (7Hz {sd3*1e-3*7*360:.0f}°)  γ² 不可用(无窗间独立观测)")
    print(f"\n判决: {'✅ 全部通过 — 可放行 4mrad pilot' if (ok1 and all_ok) else '❌ 未通过, 不放行'}")
    return 0 if (ok1 and all_ok) else 1


def _fake_rec(seq, sign, g, tau_s, rot_deg, noise, seed, hz=100.0, jitter_ms=2.0):
    """造一条与真机同结构的记录(imu100 / t_send / imu), 用于端到端验证。"""
    rng = np.random.default_rng(seed)
    jj = int(seq["joint"])
    da = seq["da"][:, jj] * float(sign)
    n20 = len(da)
    t0 = 1000.0 + rng.uniform(0, 1)                       # 各臂开机时刻不同
    ts = t0 + np.arange(n20) / FPS + rng.normal(0, 0.002, n20)   # 发送抖动
    ts = np.maximum.accumulate(ts)
    ms = np.arange(0, n20 / FPS * hz) * (1000.0 / hz)      # 设备时钟(整齐)
    tsamp = t0 + (ms / 1000.0) * (1 + 180e-6)               # 晶振偏 180ppm
    # host_t = 采样时刻 + 非负重尾传输延迟(USB 批量收包/GIL/线程唤醒)
    t = tsamp + rng.exponential(jitter_ms * 1e-3 + 1e-4, len(ms))
    pred = seq["pred"].mean(0)
    y = _bl_eval(pred, (tsamp - t0) - tau_s) * g * float(sign)   # 响应按**采样时刻**生成
    if rot_deg:
        c, sn = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
        y = np.stack([c * y[:, 0] - sn * y[:, 1], sn * y[:, 0] + c * y[:, 1], y[:, 2]], 1)
    y = y + rng.normal(0, noise, y.shape)
    imu100 = np.concatenate([t[:, None], ms[:, None], np.zeros((len(ms), 3)), y], 1)
    return {"imu": np.zeros((n20, 6)).tolist(), "t_send": ts.tolist(),
            "imu100": imu100.tolist()}


def e2e_test():
    """端到端: 四条独立"复放"(各自时钟/抖动) → arm_windows → 频域奇偶分解 → z/τ。"""
    seq = np.load(PB / "seq_ep0_j2_8mrad_p.npz", allow_pickle=True)   # 2 窗
    G_T, TAU_T, ROT_T, NZ = 1.8, 0.075, 12.0, 0.050
    print("=== 端到端自检: 四臂独立时钟 → analyze_cfg(100Hz 路径) ===")
    print(f"    真值 增益 {G_T} · 时延 {TAU_T*1e3:.0f}ms · 旋转 {ROT_T}° · 噪声 {NZ*1e3:.0f}mrad/s")
    mk = lambda sg, sd: _fake_rec(seq, sg, G_T, TAU_T, ROT_T, NZ, sd)
    r = analyze_cfg(mk(0, 1), mk(0, 2), mk(+1, 3), mk(-1, 4), seq,
                    label="E2E 合成", nboot=600)
    if r is None:
        print("❌ 无可用窗"); return 1
    zs = [r["fr"][float(f)]["z"] for f in seq["freqs"]]
    ok_g = abs(np.abs(zs).mean() - G_T * np.cos(np.radians(ROT_T))) < 0.25
    ok_t = abs(r["tau_ms"] - TAU_T * 1e3) < 20
    ok_e = r["e_even"] < 0.5 * r["e_odd"]
    print(f"\n  增益 {np.abs(zs).mean():.2f} (真 {G_T*np.cos(np.radians(ROT_T)):.2f}) {'✓' if ok_g else '✗'}"
          f" | τ {r['tau_ms']:+.0f}ms (真 {TAU_T*1e3:.0f}) {'✓' if ok_t else '✗'}"
          f" | 偶/奇 {r['e_even']/r['e_odd']:.2f} {'✓' if ok_e else '✗'}")
    good = ok_g and ok_t and ok_e
    print(f"判决: {'✅ 端到端通过' if good else '❌ 端到端未通过'}")
    return 0 if good else 1


def zwin(ep, amp, rnd=1, nboot=1000):
    """逐 (关节, 窗) 的 z(f) / 方向残差 / CI —— **位形相关性的正式判据**。

    2026-07-28 用户定: 位形依赖不能只比幅值比, 要用三轴复数投影 z(f)、方向残差和 CI。
    z(f) = (H_Ĝ^H H_real)/(H_Ĝ^H H_Ĝ):  |z| = 模型方向上的增益比, arg(z) = 相位误差,
    方向残差 = 正交分量占比(方向失配)。单窗时 CI 由实测噪声窗给出(γ² 不可用)。
    """
    import json
    BARE = {1: ("A1", "A2"), 2: ("A3", "A4"), 3: ("A5", "A6")}[rnd]
    rt = "" if rnd == 1 else f"r{rnd}_"
    print(f"=== ep{ep} 第{rnd}轮 {amp}: 逐窗 z(f) / 方向残差 / 95%CI ===")
    out = {}
    for jn in ("j2", "j3"):
        try:
            seq = np.load(PB / f"seq_ep{ep}_{jn}_{amp}_p.npz", allow_pickle=True)
            R = {k: json.loads((PB / f"replay_ep{ep}_{rt}{v}.json").read_text()) for k, v in
                 [("A1", BARE[0]), ("A2", BARE[1]),
                  ("Bp", f"{jn}_{amp}_p"), ("Bm", f"{jn}_{amp}_m")]}
        except FileNotFoundError as e:
            print(f"  {jn}: 缺数据 ({e.filename and Path(e.filename).name}) — 跳过")
            continue
        F = np.asarray(seq["freqs"], float)
        W = {k: arm_windows(R[k], seq, {"A1": 0, "A2": 0, "Bp": 1, "Bm": -1}[k]) for k in R}
        common = sorted(set.intersection(*[set(v) for v in W.values()]))
        pred = seq["pred"]
        tg = np.arange(0, NANA / FPS, 0.01)
        for w in common:
            LI = lambda k, f: lockin_at(W[k][w]["y"], W[k][w]["t"], f, W[k][w]["dt"])
            U = np.array([[(lockin_at(W["Bp"][w]["u"], W["Bp"][w]["t"], f, W["Bp"][w]["dt"])
                            - lockin_at(W["Bm"][w]["u"], W["Bm"][w]["t"], f,
                                        W["Bm"][w]["dt"])) / 2 for f in F]])
            Yo = np.array([[(LI("Bp", f) - LI("Bm", f)) / 2 for f in F]])
            # Ĝ: 逐成员(不可用成员均值 —— 宽带瞬态上成员分歧大会压低均值)
            YGk = [np.array([[lockin_at(_bl_eval(pred[k], w / FPS + tg), tg, f) for f in F]])
                   for k in range(pred.shape[0])]
            nz = [(np.stack([np.interp(tg, W["A2"][w]["t"], W["A2"][w]["y"][:, c])
                             - np.interp(tg, W["A1"][w]["t"], W["A1"][w]["y"][:, c])
                             for c in range(3)], 1)) / 2]
            zs, rf = [], []
            for k in range(pred.shape[0]):
                fr = frf_from_lockin(U, Yo, YGk[k], F)
                zs.append([fr[float(f)]["z"] for f in F])
                rf.append([fr[float(f)]["resid_frac"] for f in F])
            zs, rf = np.array(zs), np.array(rf)
            km = int(np.argsort(np.abs(zs).mean(1))[len(zs) // 2])      # 中位成员
            _, ci = bootstrap_lockin(U, Yo, YGk[km], nz, [tg], [None], F,
                                     nboot=nboot, seed=0)
            print(f"\n  {jn} 窗{w}  (成员中位 k={km}; 成员间 |z| 极差 "
                  f"{np.abs(zs).max(0).mean()/max(np.abs(zs).min(0).mean(),1e-9):.2f}×)")
            print(f"  {'f(Hz)':>7}{'|z|':>8}{'95%CI':>16}{'arg z':>9}{'95%CI':>18}"
                  f"{'方向残差':>10}")
            for i, f in enumerate(F):
                print(f"  {f:>7.1f}{abs(zs[km, i]):>8.2f}"
                      f"  [{ci['z_abs'][0, i]:>5.2f},{ci['z_abs'][1, i]:>5.2f}]"
                      f"{np.degrees(np.angle(zs[km, i])):>+8.0f}°"
                      f"  [{ci['z_phase'][0, i]:>+6.0f},{ci['z_phase'][1, i]:>+6.0f}]°"
                      f"{rf[km, i]:>10.2f}")
            out[f"{jn}_w{w}"] = {"z_abs": np.abs(zs[km]).tolist(),
                                 "z_arg_deg": np.degrees(np.angle(zs[km])).tolist(),
                                 "resid_frac": rf[km].tolist(),
                                 "ci_abs": ci["z_abs"].tolist(),
                                 "ci_phase": ci["z_phase"].tolist()}
    print(f"\n  判读: |z| 的 CI 若在两窗之间不重叠 ⇒ 位形依赖成立(单轮仍需第二轮复现)。")
    print(f"        方向残差大 ⇒ 失配不只是增益, 还有方向 —— 逐关节标量增益校准不适用。")
    return out


AMP_OF_ROUND = {1: "4mrad", 2: "4mrad", 3: "2mrad"}


def ampscale(ep, rnd_hi, rnd_lo, amp_hi=None, amp_lo=None, nboot=800):
    """幅值缩放判据 —— ⑤ 偶次非线性的决定性检验。

    奇分量(线性响应):  |odd| ∝ δa      ⇒ 4mrad/2mrad 的幅值比应 ≈ **2**
    偶次非线性:        |D|  ∝ δa²      ⇒ 幅值比应 ≈ **4**
    与幅值无关的伪影:                    ⇒ 幅值比应 ≈ **1**
    另算两轮偶分量的复相关: 同一机制则相位形状一致 ⇒ 相关为正。
    """
    import json
    amp_hi = amp_hi or AMP_OF_ROUND[rnd_hi]
    amp_lo = amp_lo or AMP_OF_ROUND[rnd_lo]
    BARE = {1: ("A1", "A2"), 2: ("A3", "A4"), 3: ("A5", "A6")}
    RT = {1: "", 2: "r2_", 3: "r3_"}
    FN = np.array([5.0, 5.25, 8.25, 8.5, 8.75])
    H2 = np.arange(12.0, 15.01, 0.5)
    P = lambda v: float(np.sum(np.abs(v) ** 2))
    print(f"=== 幅值缩放判据: 第{rnd_hi}轮 {amp_hi} vs 第{rnd_lo}轮 {amp_lo} ===")
    k = float(amp_hi[:-4]) / float(amp_lo[:-4])          # 幅值比
    print(f"    期望: 奇分量(线性) ≈ {k:.1f} | 偶分量(∝δa²) ≈ {k**2:.1f} | "
          f"与幅值无关的伪影 ≈ 1.0"
          + ("   [同幅值 ⇒ 三者都是 1.0, 本次仅验证判据自身]" if abs(k - 1) < 1e-9 else ""))
    for jn in ("j2", "j3"):
        D = {}
        for tag, rnd, amp in (("hi", rnd_hi, amp_hi), ("lo", rnd_lo, amp_lo)):
            seq = np.load(PB / f"seq_ep{ep}_{jn}_{amp}_p.npz", allow_pickle=True)
            rt, ba = RT[rnd], BARE[rnd]
            try:
                R = {k: json.loads((PB / f"replay_ep{ep}_{rt}{v}.json").read_text())
                     for k, v in [("A1", ba[0]), ("A2", ba[1]),
                                  ("Bp", f"{jn}_{amp}_p"), ("Bm", f"{jn}_{amp}_m")]}
            except FileNotFoundError as e:
                print(f"  {jn}: 缺 {Path(e.filename).name} — 跳过"); D = None; break
            W = {k: arm_windows(R[k], seq, {"A1": 0, "A2": 0, "Bp": 1, "Bm": -1}[k])
                 for k in R}
            D[tag] = (W, np.asarray(seq["freqs"], float))
        if not D:
            continue
        F = D["hi"][1]
        common = sorted(set.intersection(*[set(D[t][0][k]) for t in D for k in D[t][0]]))
        print(f"\n  --- {jn} (窗 {common}) ---")
        print(f"  {'窗':>5}{'量':>16}{amp_hi:>10}{amp_lo:>10}{'幅值比':>9}{'复相关':>9}{'判':>16}")
        for w in common:
            V = {}
            for t in ("hi", "lo"):
                W = D[t][0]
                LI = lambda k, f: lockin_at(W[k][w]["y"], W[k][w]["t"], f, W[k][w]["dt"])
                V[t] = {
                    "odd": np.stack([(LI("Bp", f) - LI("Bm", f)) / 2 for f in F]),
                    "even_t": np.stack([(LI("Bp", f) + LI("Bm", f)) / 2
                                        - (LI("A1", f) + LI("A2", f)) / 2 for f in F]),
                    "even_h": np.stack([(LI("Bp", f) + LI("Bm", f)) / 2
                                        - (LI("A1", f) + LI("A2", f)) / 2 for f in H2]),
                    "nz_t": np.stack([(LI("A2", f) - LI("A1", f)) / 2 for f in F]),
                    "nz_h": np.stack([(LI("A2", f) - LI("A1", f)) / 2 for f in H2])}
            for lbl, key, nzk, exp in [("奇分量(探针点)", "odd", None, 2.0),
                                       ("偶分量(探针点)", "even_t", "nz_t", 4.0),
                                       ("偶分量(和频)", "even_h", "nz_h", 4.0)]:
                a, b = V["hi"][key].ravel(), V["lo"][key].ravel()
                if nzk:   # 偶分量: 扣掉噪声底(H0: 方差 = 2×噪声窗方差), 取超出部分的幅值
                    ea = max(P(V["hi"][key]) - 2 * P(V["hi"][nzk]), 0.0) ** 0.5
                    eb = max(P(V["lo"][key]) - 2 * P(V["lo"][nzk]), 0.0) ** 0.5
                else:
                    ea, eb = P(V["hi"][key]) ** 0.5, P(V["lo"][key]) ** 0.5
                r = ea / eb if eb > 1e-12 else float("nan")
                c = np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
                # 判据按**实际幅值比 k** 自适应: 线性 ≈k, 偶次 ≈k², 与幅值无关 ≈1
                cand = [(k, f"≈{k:.1f} 线性"), (k ** 2, f"≈{k**2:.1f} ∝δa²"),
                        (1.0, "≈1 与幅值无关")]
                seen, uniq = set(), []
                for val, lb in cand:                        # k=1 时三者重合, 只留一条
                    if round(val, 3) not in seen:
                        seen.add(round(val, 3)); uniq.append((val, lb))
                hit = [lb for val, lb in uniq if abs(r - val) < max(0.25 * val, 0.3)]
                vd = hit[0] if len(hit) == 1 else ("多解" if hit else "不明")
                print(f"  {w:>5}{lbl:>16}{ea*1e3:>10.2f}{eb*1e3:>10.2f}{r:>9.2f}{c:>9.3f}"
                      f"{vd:>16}")
    print(f"\n  判读: 偶分量比 ≈4 ⇒ 确认 D ∝ δa² 的偶次非线性;")
    print(f"        ≈2 ⇒ 与 δa 一次相关(不是偶次); ≈1 ⇒ 与幅值无关(非探针所致)。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zwin", action="store_true",
                    help="逐窗 z(f)/方向残差/CI —— 位形相关性的正式判据")
    ap.add_argument("--amp", default="4mrad")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--ampscale", nargs=2, metavar=("RND_HI", "RND_LO"),
                    help="幅值缩放判据: 比较两轮(高幅值 低幅值)的奇/偶分量缩放")
    ap.add_argument("--e2e", action="store_true", help="端到端(合成四臂记录)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eps", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    if a.ampscale:
        ampscale(a.eps[0], int(a.ampscale[0]), int(a.ampscale[1]))
        raise SystemExit(0)
    if a.zwin:
        zwin(a.eps[0], a.amp, a.round)
        raise SystemExit(0)
    if a.e2e:
        raise SystemExit(e2e_test())
    if a.self_test:
        raise SystemExit(self_test())

    def load(ep, cfg):
        """返回**原始记录 dict**(含 imu100/t_send) —— 分析走 100Hz 路径, 不用 d['imu']。"""
        d = json.loads((PB / f"replay_ep{ep}_{cfg}.json").read_text())
        n20 = len(d["imu"])
        r = check_recording(d.get("imu100", []), n20 / FPS, name=f"ep{ep} {cfg}")
        if not d.get("t_send"):
            raise RuntimeError(f"ep{ep} {cfg}: 缺 t_send —— 需用新版 replay_probe.py 重采")
        if not r["ok"]:
            print(f"    ⚠ ep{ep} {cfg} 记录未达标 — 该配置结论不可用")
        return d

    for ep in a.eps:
        print(f"\n=== ep{ep} 记录质量 ===")
        A1, A2 = load(ep, "A1"), load(ep, "A2")
        for jn in JOINTS:
            for an in AMPS:
                try:
                    Bp, Bm = load(ep, f"{jn}_{an}_p"), load(ep, f"{jn}_{an}_m")
                except FileNotFoundError:
                    continue
                seq = np.load(PB / f"seq_ep{ep}_{jn}_{an}_p.npz", allow_pickle=True)
                analyze_cfg(A1, A2, Bp, Bm, seq, label=f"ep{ep} {jn} {an}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
