"""拼接后的训练损失（torch）。与评分口径一致：逐集去均值 → bandpass41 → 丢首尾 20。"""
from __future__ import annotations
import numpy as np
import q1_data as D
import q1_windows as W

W_BAND = 0.5                  # 关节损失的带内权重（同 G-2 `r_q`），默认值
Q_BAND = (3.0, 9.0)


def q_kernel() -> np.ndarray:
    from scipy.signal import firwin
    return firwin(41, list(Q_BAND), fs=D.FS, pass_zero=False).astype(np.float32)


def loss_norms(eps) -> dict:
    """损失归一化（只用训练会话）：ω 项除以零预测误差的均值；关节项按自身尺度。"""
    K = W.kernel(); Kq = q_kernel()
    e0 = [W.score(e.gap, np.zeros_like(e.gap), K) for e in eps]
    c_w = float(np.mean([a / c for a, _, c in e0]))
    E, B = [], []
    for e in eps:
        k = np.where(e.q_valid)[0]
        E.append(e.cmd[k, :6] - e.q[k])
        qv = e.q_valid[D.ALIGN_L:]
        qb = np.stack([np.convolve(e.q[D.ALIGN_L:, j] * qv, Kq, mode="same") for j in range(6)], 1)
        B.append(qb[W.band_clean(qv)])                  # 只用 ±20 帧支撑全有效的帧
    return {"c_w": c_w, "s_eps": np.maximum(np.concatenate(E).std(0), 1e-6),
            "s_qb": np.maximum(np.concatenate(B).std(0), 1e-6)}


def _bp(x, K):
    """(L,C) → 'same' 卷积后丢首尾 EDGE。"""
    import torch
    Kt = torch.as_tensor(K, dtype=x.dtype, device=x.device).view(1, 1, -1).repeat(x.shape[1], 1, 1)
    y = torch.nn.functional.conv1d(x.T[None], Kt, padding=len(K) // 2, groups=x.shape[1])[0].T
    return y[D.EDGE:-D.EDGE]


def omega_loss(ep, ehat, norms):
    """`ehat` (n,3) torch（已拼接）→ ‖P(gap−ê)‖²/(3T_e)/c_w。"""
    import torch
    g = torch.as_tensor(ep.gap, dtype=ehat.dtype, device=ehat.device)
    r = g - ehat
    r = r - r.mean(0, keepdim=True)
    v = _bp(r, W.kernel())
    return (v ** 2).sum() / v.numel() / norms["c_w"]


def joint_loss(ep, qhat, norms):
    """`qhat` (n,6) torch = 帧 3 … T−1（已拼接）→ 0.5·宽带 + 0.5·带内（3–9 Hz），各按自身尺度。

    无效帧不进宽带项；带内项只取 ±20 帧支撑全有效的帧（无效帧的值不经带通污染邻帧）。
    """
    import torch
    q = torch.as_tensor(ep.q[D.ALIGN_L:D.ALIGN_L + ep.n], dtype=qhat.dtype, device=qhat.device)
    qv = ep.q_valid[D.ALIGN_L:D.ALIGN_L + ep.n]
    m = torch.as_tensor(qv, dtype=qhat.dtype, device=qhat.device)
    cm = torch.as_tensor(W.band_clean(qv)[D.EDGE:-D.EDGE], dtype=qhat.dtype, device=qhat.device)
    d = qhat - q
    s_e = torch.as_tensor(norms["s_eps"], dtype=qhat.dtype, device=qhat.device)
    s_b = torch.as_tensor(norms["s_qb"], dtype=qhat.dtype, device=qhat.device)
    broad = (((d / s_e) ** 2) * m[:, None]).sum() / (m.sum() * 6)
    bd = _bp(d * m[:, None], q_kernel()) / s_b
    band = ((bd ** 2) * cm[:, None]).sum() / (cm.sum() * 6)
    return (1 - W_BAND) * broad + W_BAND * band
