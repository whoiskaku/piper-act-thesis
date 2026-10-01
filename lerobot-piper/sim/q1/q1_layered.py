"""分层伺服响应网络的关节层（1a / 1b）—— `DESIGN_G_PI_V2.md` §2.4。

执行器网络（Hwangbo 2019）改编到运动学层：每个关节一套 MLP（3×32 softsign，权重不共享），
输入 = 20 帧跟踪误差 + 20 帧速度 + 20 帧有效位 + 下一帧指令误差（共 61 维），
输出下一帧关节增量：`Δq̂ = g_j·(u_{k+1} − q̂_k) + s_v,j·MLP_j(x)`。
头部零初始化 ⇒ 起点是一阶滞后伺服（`g = G_INIT`）。
- 1a：**不读任何实测**；起点状态取自从静止（`q̂_0 = u_0`）起、只用 `u_{≤t0}` 的开环滚动。
- 1b：起点读实测编码器 `q_{t0−20 … t0}`（无效 = `u_0`、掩码 0），之后全用自身预测。
窗口内指令超出 `t1` 钳位到 `u_{t1}`。传感器层见 `q1_sensor.py`。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

HID = 32
G_INIT = 0.3                  # 一阶滞后先验（≈ 3 帧滞后），默认值
ROLL_PAD = 2                  # 多滚 2 帧，让 t1+3 的中心差分落在内部
SD_FLOOR = 1e-6


def joint_stats(eps) -> dict:
    """关节层归一化尺度（只用传入的训练折、有效实测帧）。"""
    E, V = [], []
    for e in eps:
        k = np.where(e.q_valid)[0]
        E.append(e.cmd[k, :6] - e.q[k])
        kk = k[1:][np.diff(k) == 1]
        V.append(e.q[kk] - e.q[kk - 1])
    return {"s_eps": np.maximum(np.concatenate(E).std(0), SD_FLOOR),
            "s_v": np.maximum(np.concatenate(V).std(0), SD_FLOOR)}


def make_servo(stats: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ServoNet(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [3 * D.HIST + 1, HID, HID, HID, 1]
            self.W = nn.ParameterList()
            self.b = nn.ParameterList()
            for i in range(4):
                w = torch.empty(6, dims[i], dims[i + 1])
                for j in range(6):
                    nn.init.kaiming_uniform_(w[j].T, a=5 ** 0.5)
                if i == 3:
                    w.zero_()                                    # 头部零初始化
                self.W.append(nn.Parameter(w))
                self.b.append(nn.Parameter(torch.zeros(6, dims[i + 1])))
            self.g = nn.Parameter(torch.full((6,), G_INIT))
            self.register_buffer("s_eps", torch.tensor(stats["s_eps"], dtype=torch.float32))
            self.register_buffer("s_v", torch.tensor(stats["s_v"], dtype=torch.float32))

        def step(self, qh, uh, mh, u_next):
            """qh/uh (B,6,H+1) = 帧 k−H…k；mh (B,H+1)；u_next (B,6) → q̂_{k+1} (B,6)。"""
            m = mh[:, None, 1:]
            eps = (uh[..., 1:] - qh[..., 1:]) / self.s_eps[None, :, None] * m
            v = (qh[..., 1:] - qh[..., :-1]) / self.s_v[None, :, None] * m * mh[:, None, :-1]
            e_next = ((u_next - qh[..., -1]) / self.s_eps)[..., None]
            h = torch.cat([eps, v, m.expand_as(eps), e_next], -1)          # (B,6,61)
            for i in range(4):
                h = torch.einsum("bji,jio->bjo", h, self.W[i]) + self.b[i]
                if i < 3:
                    h = F.softsign(h)
            return qh[..., -1] + self.g * (u_next - qh[..., -1]) + self.s_v * h[..., 0]

        def rollout(self, qh, uh, mh, u_future):
            """从历史出发逐帧喂 `u_future` (B,K,6) → q̂ (B,K,6)。"""
            out = []
            for k in range(u_future.shape[1]):
                un = u_future[:, k]
                qn = self.step(qh, uh, mh, un)
                out.append(qn)
                qh = torch.cat([qh[..., 1:], qn[..., None]], -1)
                uh = torch.cat([uh[..., 1:], un[..., None]], -1)
                mh = torch.cat([mh[:, 1:], torch.ones_like(mh[:, :1])], -1)
            return torch.stack(out, 1)

    return ServoNet()


def _t(net, x):
    import torch
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=net.s_eps.device)


def history_measured(ep, t0: int):
    """1b：帧 `t0−H … t0` 的实测编码器（无效 = u_0）、对应指令、有效位 → ((6,H+1),(6,H+1),(H+1,))。"""
    lo = t0 - D.HIST
    k = np.arange(lo, t0 + 1)
    u = W.span(ep.cmd[:, :6], lo, t0, ep.cmd[0, :6])
    q = np.repeat(ep.cmd[0:1, :6], len(k), 0).astype(np.float64)
    ok = k >= 0
    ok[ok] = ep.q_valid[k[ok]]
    q[ok] = ep.q[k[ok]]
    return q.T, u.T, ok.astype(np.float64)


def history_from_traj(ep, traj, t0: int):
    """1a：帧 `t0−H … t0` 取自开环轨迹 `traj`（帧 0 = u_0）；`< 0` 帧填 u_0、掩码 0。"""
    lo = t0 - D.HIST
    k = np.arange(lo, t0 + 1)
    u = W.span(ep.cmd[:, :6], lo, t0, ep.cmd[0, :6])
    q = W.span(traj, lo, t0, ep.cmd[0, :6])
    return q.T, u.T, (k >= 0).astype(np.float64)


def open_loop_traj(net, ep, upto: int) -> np.ndarray:
    """1a：从静止（q̂_0 = u_0）起、只用真实指令 `u_1 … u_upto` 的开环滚动 → (upto+1,6)。无梯度。"""
    import torch
    u0 = ep.cmd[0, :6]
    if upto <= 0:
        return u0[None].astype(np.float64)
    with torch.no_grad():
        qh = _t(net, np.repeat(u0[:, None], D.HIST + 1, 1)[None])
        mh = torch.zeros(1, D.HIST + 1, device=qh.device)
        mh[:, -1] = 1.0
        q = net.rollout(qh, qh.clone(), mh, _t(net, ep.cmd[1:upto + 1, :6][None]))[0]
    return np.concatenate([u0[None], q.cpu().numpy().astype(np.float64)], 0)


def future_commands(ep, w: W.Window, K: int) -> np.ndarray:
    """窗口内第 `t0+1 … t0+K` 帧的可见指令（超出 t1 钳位）→ (K,6)。"""
    return W.visible(ep.cmd[:, :6], w.t1, w.t0 + K)[w.t0 + 1:]


def window_inputs(net, ep, windows, mode: str, traj=None):
    """批量窗口的 (历史 q, 历史 u, 有效位, 未来指令, K)。mode 'a' 用开环轨迹，'b' 用实测。"""
    K = max(w.t1 - w.t0 for w in windows) + D.ALIGN_L + ROLL_PAD
    Q, U, M, F = [], [], [], []
    for w in windows:
        q, u, m = history_from_traj(ep, traj, w.t0) if mode == "a" else history_measured(ep, w.t0)
        Q.append(q); U.append(u); M.append(m); F.append(future_commands(ep, w, K))
    return (_t(net, np.stack(Q)), _t(net, np.stack(U)), _t(net, np.stack(M)),
            _t(net, np.stack(F)), K)
