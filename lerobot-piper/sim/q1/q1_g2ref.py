"""G-2（冻结）的同口径重预测 —— `DESIGN_G_PI_V2.md` §2.5。

**冻结权重，按与候选相同的起点、可见指令和末端钳位重新预测，再拼接评分。**
G-2 只读指令（`rs_features` 只用指令与预测的 q̂，不读实测），故起点只通过「可见指令」起作用。
⚠ `rq_X`/`rs_features`/`kin_gyro_fs` 用中心差分 ⇒ 钳位会改变 `t1` 附近的帧，不只尾部 3 帧。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

PAD = 2                       # 让 t1+3 处的中心差分落在序列内部


def _dd(cmd, name: str) -> dict:
    """G-2 前向所需的 `dd`（均匀 20 Hz 时基；`gyro100` 只为满足接口，不进前向）。"""
    N = len(cmd)
    t = np.arange(N) / D.FS
    return {"name": name, "cmd": np.asarray(cmd, np.float64), "t_send": t, "t_read": t,
            "tg_i": np.arange(N * 5) / (D.FS * 5), "gyro100": np.zeros((N * 5, 3))}


def load_g2():
    from freeze_pi_g2wall import G2v0
    return G2v0()


def omega_sensor(g2, cmd, name: str, R, b) -> np.ndarray:
    """指令 → G-2 陀螺并旋到传感器系（与 `train_fabs_g0.g2_score` 同变换）。"""
    import fit_step1_candidate as S1
    om = np.asarray(g2.omega(_dd(cmd, name)), np.float64)
    w6 = np.zeros((len(om), 6), np.float32)
    w6[:, 3:] = om
    return S1.calibrate_ideal_imu(w6, np.asarray(R, np.float32),
                                  np.asarray(b, np.float32))[:, 3:].astype(np.float64)


def window_ehat(g2, ep, w: W.Window, R, b) -> np.ndarray:
    """一个窗口的 `ê_t, t ∈ [t0, s1]`：只用 `u_{0…t1}`，其后钳位到 `u_{t1}`。"""
    ws = omega_sensor(g2, W.visible(ep.cmd, w.t1, w.t1 + D.ALIGN_L + PAD), ep.name, R, b)
    ts = np.arange(w.t0, w.s1 + 1)
    return ws[ts + D.ALIGN_L] - ep.ideal[ts]


def episode_ehat(g2, ep, windows, R, b) -> np.ndarray:
    return W.stitch(ep.n, windows, [window_ehat(g2, ep, w, R, b) for w in windows])


def whole_episode_ehat(g2, ep, R, b) -> np.ndarray:
    """**旧口径**（整集前向，窗口尾部用到下一 chunk 的指令）—— 只作对照，**不作判据**。"""
    ws = omega_sensor(g2, ep.cmd[:ep.T], ep.name, R, b)
    return ws[D.ALIGN_L:D.ALIGN_L + ep.n] - ep.ideal[:ep.n]
