"""G-vNext 的共用数据层 —— 把一条复放记录读成**修正口径**下的统一网格。

口径（`PREREG_G_VNEXT.md` §7，每处引用都必须带）
------------------------------------------------
- 真机 IMU：`imu100` 的 **设备 ms** 建均匀 100 Hz 网格（步长恒 10、零丢样）；
  **不用落盘的 20 Hz `imu` 字段**（控制环 p99 有 55–65 ms 的尾巴）。
- 关节角：`can_q200` 的**内核时间戳**（REALTIME）减去 `can_hs_meta` 里的偏移
  换到 MONOTONIC，再插到均匀 200 Hz 网格。
- 抗混叠：要降到 20 Hz 时先 9 Hz 低通再抽，与仿真侧 `g_q_v1.py:231` 同型。

⚠ 设备 ms → MONOTONIC 用最小二乘拟合**斜率**；常数项由各自的位移扫描吸收
（到达延迟恒为正，严格做法是下包络，但常数项在评分时本来就被扫掉）。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FS_I, FS_Q, FPS = 100.0, 200.0, 20.0
D2R = np.pi / 180 / 1000.0          # can_q200 的 0.001 deg → rad
Q_GROUPS = ((1, (0, 1)), (3, (2, 3)), (5, (4, 5)))


def lp(x, fc, fs):
    from scipy.signal import butter, filtfilt
    return filtfilt(*butter(4, fc / (fs / 2)), np.asarray(x, float), axis=0)


def bp(x, lo, hi, fs=FPS, ntap=41):
    from scipy.signal import firwin
    k = firwin(ntap, [lo, hi], fs=fs, pass_zero=False)
    x = np.atleast_2d(np.asarray(x, float).T).T
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)


def load(path: str | Path) -> dict:
    """→ dict(gyro100, q200, tg_i, tg_q, cmd, t_send, t_read, name)。

    `gyro100` (N,3) rad/s 在均匀 100 Hz 网格；`q200` (M,6) rad 在均匀 200 Hz 网格；
    两个网格同起止，`tg_i`/`tg_q` 是它们的 MONOTONIC 时刻。
    """
    p = Path(path)
    d = json.loads(p.read_text())
    off = d["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    I = np.asarray(d["imu100"], float)
    ms, ht, gy = I[:, 1], I[:, 0], I[:, 5:8]
    a, b = np.linalg.lstsq(np.stack([ms, np.ones_like(ms)], 1), ht, rcond=None)[0]
    t_i = a * ms + b
    Q = np.asarray(d["can_q200"], float)
    tq = [Q[Q[:, 1] == g, 0] - off for g, _ in Q_GROUPS]
    qv = [Q[Q[:, 1] == g, 2:4] for g, _ in Q_GROUPS]
    t0 = max(t_i[0], *[t[0] for t in tq])
    t1 = min(t_i[-1], *[t[-1] for t in tq])
    tg_q = np.arange(t0, t1, 1.0 / FS_Q)
    tg_i = np.arange(t0, t1, 1.0 / FS_I)
    q200 = np.zeros((len(tg_q), 6))
    for (g, cols), t_, v_ in zip(Q_GROUPS, tq, qv):
        for c, col in enumerate(cols):
            q200[:, col] = np.interp(tg_q, t_, v_[:, c]) * D2R
    g100 = np.stack([np.interp(tg_i, t_i, gy[:, k]) for k in range(3)], 1)
    return {"name": p.stem[len("replay_"):], "gyro100": g100, "q200": q200,
            "stopped": d.get("stopped"), "return_blocked": d.get("return_blocked"),
            "tg_i": tg_i, "tg_q": tg_q,
            "cmd": np.asarray(d["cmd"], float),
            "t_send": np.asarray(d["t_send"], float),
            "t_read": np.asarray(d["t"], float)}


def to20(x100: np.ndarray) -> np.ndarray:
    """100 Hz → **抗混叠** 20 Hz（9 Hz 低通后 5× 抽取）。"""
    return lp(x100, 9.0, FS_I)[::5]


def q_at(dd: dict, t: np.ndarray) -> np.ndarray:
    """真关节角（先 9 Hz 低通再采到给定时刻）→ (len(t), 6) rad。"""
    qf = lp(dd["q200"], 9.0, FS_Q)
    return np.stack([np.interp(t, dd["tg_q"], qf[:, j]) for j in range(6)], 1)


def score(A: np.ndarray, S: np.ndarray) -> tuple[float, float, float]:
    """→ (残差/实测, ρ², 最优增益)。两者需已带通到同一带。"""
    aa = float((A * A).sum()); ss = float((S * S).sum()); as_ = float((A * S).sum())
    return ((aa + ss - 2 * as_) / max(aa, 1e-30),
            as_ ** 2 / max(aa * ss, 1e-30), as_ / max(ss, 1e-30))


def best_fit(A0: np.ndarray, S0: np.ndarray, lo: float, hi: float,
             maxL: int = 3, rotate: bool = True) -> tuple[int, float, np.ndarray]:
    """逐系列最优位移(+可选 Procrustes 装着旋转)下的 ρ²。→ (L, ρ², 对齐后的 S)。

    **必须逐系列各自扫位移**：20 Hz 下 1 样本 = 50 ms，在 3–9 Hz 是 54°–162° 相移；
    用共同 L 会把某些系列压到近零（2026-09-01 实测天井 L=0→0.66 / L=1→0.008）。
    """
    from scipy.linalg import orthogonal_procrustes
    best = (0, -1.0, S0)
    for L in range(maxL + 1):
        n = min(len(A0) - L, len(S0))
        if n < 64:
            continue
        A, S = A0[L:L + n], S0[:n]
        if rotate and S.shape[1] == 3:
            S = S @ orthogonal_procrustes(S, A)[0]
        r2 = score(bp(A, lo, hi), bp(S, lo, hi))[1]
        if r2 > best[1]:
            best = (L, r2, S)
    return best


def valid_files(d: str | Path, pattern: str = "replay_*.json") -> list[Path]:
    """目录下**有效**的记录文件。

    排除两类，并**大声报出**（静默过滤会让数据问题变成模型问题）：
      · `.orphan_*` —— 上次中断留下的、已被 collect.py 作废移开的；
      · `stopped` 或 `return_blocked` 非空 —— `replay_probe.py` 标记的执行异常。

    ⚠ `collect.py` 按**帧数**判「录满了」，那个判据管的是「有没有录到」，
    不管「有没有执行对」。有效性标志在文件里，必须单独查。
    2026-09-02 实测: `s1/pp2_ep001` 带 `return_blocked=True`，
    集内 j2 指令 −1.44° 而实测 97.49°、j1 只走了指令行程的 38%
    ⇒ 关节残差 RMS 620 mrad(35°)，是正常集(3–7 mrad)的 ~100 倍，
    一条就占训练集 99.9% 的残差能量。
    """
    d = Path(d)
    out, dropped = [], []
    for f in sorted(d.glob(pattern)):
        if ".orphan_" in f.name:
            continue
        j = json.loads(f.read_text())
        if j.get("stopped") or j.get("return_blocked"):
            dropped.append((f.stem[len("replay_"):], j.get("stopped"),
                            j.get("return_blocked")))
            continue
        out.append(f)
    if dropped:
        print(f"  ⚠ {d.name}: 剔除 {len(dropped)} 条执行异常 —— " + "; ".join(
            f"{n}(stopped={s}, return_blocked={r})" for n, s, r in dropped))
    return out
