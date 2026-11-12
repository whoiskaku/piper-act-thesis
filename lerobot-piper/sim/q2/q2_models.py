"""阶段二的两种响应形式 —— 预注册 §3.2–3.4。

两者都输出关节 6 维 + 误差 3 维的**增量**（归一化单位；乘以冻结名义模型的 s_ε、s_gap 才是物理量）。
- 非线性差分：`N(x, δa) − N(x, 0)`。N 与 2a 同构，输入 = 2a 的 13 个通道 + 6 个 δa 通道；输出头零初始化 ⇒ 起点 = 零响应，
  δa = 0 时增量严格为 0。
- 条件线性：`Δy_k = Σ_{r=1}^{4} a_r(k) · Σ_{τ=0}^{19} B_r[τ] · δa_{k−τ}`（控制周期索引）。a_r 由与 2a 同构的指令网络给出（不看 δa），
  **可正可负的普通输出、不用 softmax**，输出头零初始化 ⇒ 起点 = 零响应；基核 B_r 每个元素取 N(0, 0.05²)，由训练种子决定。
  对 δa 严格线性；20 帧限制的是 δa 的记忆，指令网络仍有 61 帧感受野。
δa 归一化保留零点：每个关节只除以训练数据上 δa 的均方根（不减均值），下限 1e-6 rad。
"""
from __future__ import annotations
import numpy as np

import q1_data as D1
import q1_e2e as E
import q2_windows as W2

FORMS = ("nonlinear", "condlinear")
N_BASIS = 4
N_LAG = 20
BASIS_SD = 0.05
SCALE_FLOOR = 1e-6


def da_scale(eps) -> np.ndarray:
    """训练数据上逐关节 δa 的均方根（不减均值），下限 SCALE_FLOOR → (6,)。"""
    X = np.concatenate([np.asarray(e.da, np.float64)[:e.T] for e in eps])
    return np.maximum(np.sqrt((X ** 2).mean(0)), SCALE_FLOOR)


def window_inputs(ep, windows, st: dict, scale) -> tuple:
    """→ (X 指令侧 (B, L_SEQ, 13) float32, Dd 归一化 δa (B, L_SEQ, 6) float32)。序列帧 = t0−19 … t0+102。"""
    X = np.stack([E.window_input(ep, w, st, False) for w in windows])
    Dd = np.stack([W2.da_span(ep, w, w.t0 - D1.HIST + 1, w.t0 + E.K_OUT) / np.asarray(scale) for w in windows])
    return X, Dd.astype(np.float32)


def make_nonlinear():
    return E.make_tcn(E.CH_A + 6)


def make_condlinear():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CondLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = E.make_tcn(E.CH_A).body                       # 与 2a 同构的指令网络主干（61 帧感受野）
            self.mix = nn.Conv1d(64, N_BASIS, 1)                      # 混合权重：普通输出，不用 softmax
            nn.init.zeros_(self.mix.weight)
            nn.init.zeros_(self.mix.bias)
            self.bases = nn.Parameter(torch.randn(N_BASIS, N_LAG, 9, 6) * BASIS_SD)

        def forward(self, x, d):                                      # x (B,L,13)、d (B,L,6) → (B,L,9)
            a = self.mix(self.body(x.transpose(1, 2)))                # (B,R,L)
            w = self.bases.flip(1).permute(0, 2, 3, 1)                # (R,9,6,LAG)：w[r,o,j,m] = B[r, LAG−1−m, o, j]
            dt = F.pad(d.transpose(1, 2), (N_LAG - 1, 0))             # 因果：只看本帧及之前的 δa
            resp = torch.stack([F.conv1d(dt, w[r]) for r in range(N_BASIS)], 1)    # (B,R,9,L)
            return (a.unsqueeze(2) * resp).sum(1).transpose(1, 2)

    return CondLinear()


def make(form: str):
    if form not in FORMS:
        raise ValueError(f"未知的响应形式 {form}")
    return make_nonlinear() if form == "nonlinear" else make_condlinear()


def increment(model, form: str, X, Dd):
    """归一化单位的增量 (B, L, 9)。非线性差分减去 δa = 0 的输出 ⇒ δa = 0 时严格为 0。"""
    import torch
    p = next(model.parameters())
    x = torch.as_tensor(X, dtype=p.dtype)
    d = torch.as_tensor(Dd, dtype=p.dtype)
    if form == "nonlinear":
        return model(torch.cat([x, d], -1)) - model(torch.cat([x, torch.zeros_like(d)], -1))
    return model(x, d)


def decode_increment(out, windows, st_nom: dict):
    """(B, L_SEQ, 9) → 每窗 (Δê (n_scored, 3), Δq̂ (n_scored, 6))；位置同 `q1_e2e.decode`（帧 t+3），
    尺度沿用冻结名义模型的 s_gap、s_ε。"""
    import torch
    s_e = torch.as_tensor(st_nom["s_gap"], dtype=out.dtype)
    s_q = torch.as_tensor(st_nom["s_eps"], dtype=out.dtype)
    dE, dQ = [], []
    for b, w in enumerate(windows):
        i = np.arange(w.n_scored) + 2 + D1.HIST
        dE.append(out[b, i, 6:9] * s_e)
        dQ.append(out[b, i, :6] * s_q)
    return dE, dQ
