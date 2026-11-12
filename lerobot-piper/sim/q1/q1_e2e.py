"""端到端直接多步模型（2a/2b TCN、3a/3b Transformer）—— `DESIGN_G_PI_V2.md` §2.4。

在起点一次输出窗口内全部帧，窗口内不递推。序列 = 过去段 20 帧（`t0−19 … t0`）+ 未来段
`K_OUT` 帧（`t0+1 … t0+K_OUT`）。未来段只含当前 chunk 的指令与理想陀螺（超出 t1 钳位）；
b 版过去段另含实测编码器/陀螺（只到 t0）。两种网络都**因果**（只看到本帧及之前的输入）。
输出头零初始化：`q̂_k = u_k + s_ε·o_q`，`ê_{k−3} = s_e·o_e` ⇒ 起点 = 零预测（G0 分母）。
"""
from __future__ import annotations
import numpy as np
import q1_data as D
import q1_windows as W

K_OUT = D.CHUNK_MAX + D.ALIGN_L - 1          # 102：覆盖 t0+1 … t1+3
L_SEQ = D.HIST + K_OUT
CH_A = 13                                     # u7 ideal3 seg inchunk m_cmd
CH_B = CH_A + 11                              # + q6 w3 m_q m_w
SD_FLOOR = 1e-6


def norm_stats(eps) -> dict:
    """输入归一化（只用训练会话）：u、ideal、q（有效帧）、w（有效帧）；以及输出尺度。"""
    U = np.concatenate([e.cmd[:e.T] for e in eps]); I = np.concatenate([e.ideal[:e.T] for e in eps])
    Q = np.concatenate([e.q[e.q_valid] for e in eps]); Wm = np.concatenate([e.w[e.w_valid] for e in eps])
    G = np.concatenate([e.gap for e in eps])
    E = np.concatenate([e.cmd[np.where(e.q_valid)[0], :6] - e.q[e.q_valid] for e in eps])
    f = lambda x: (x.mean(0), np.maximum(x.std(0), SD_FLOOR))
    st = {k: f(v) for k, v in (("u", U), ("ideal", I), ("q", Q), ("w", Wm))}
    st["s_eps"] = np.maximum(E.std(0), SD_FLOOR)
    st["s_gap"] = np.maximum(G.std(0), SD_FLOOR)
    return st


def window_input(ep, w: W.Window, st: dict, measured: bool) -> np.ndarray:
    """→ (L_SEQ, CH_A|CH_B) float32。"""
    nz = lambda x, k: (x - st[k][0]) / st[k][1]
    pb = W.past_block(ep, w.t0, measured)
    fu = W.visible(ep.cmd, w.t1, w.t0 + K_OUT)[w.t0 + 1:]
    fi = W.visible(ep.ideal, w.t1, w.t0 + K_OUT)[w.t0 + 1:]
    kk = np.arange(w.t0 + 1, w.t0 + K_OUT + 1)
    past = [nz(pb["u"], "u"), nz(pb["ideal"], "ideal"), np.zeros((D.HIST, 1)),
            np.zeros((D.HIST, 1)), pb["m_cmd"][:, None]]
    fut = [nz(fu, "u"), nz(fi, "ideal"), np.ones((K_OUT, 1)),
           (kk <= w.t1).astype(np.float64)[:, None], np.ones((K_OUT, 1))]
    if measured:
        past += [nz(pb["q"], "q") * pb["m_q"][:, None], nz(pb["w"], "w") * pb["m_w"][:, None],
                 pb["m_q"][:, None], pb["m_w"][:, None]]
        fut += [np.zeros((K_OUT, 11))]
    return np.concatenate([np.concatenate(past, 1), np.concatenate(fut, 1)], 0).astype(np.float32)


def make_tcn(in_ch: int):
    import torch.nn as nn

    class TCN(nn.Module):
        """因果空洞卷积 4 层 k=5 d=1/2/4/8 64ch（与 `F_abs` 主干同构）→ 1×1 → 9。"""

        def __init__(self):
            super().__init__()
            layers, c = [], in_ch
            for d in (1, 2, 4, 8):
                layers += [nn.ConstantPad1d(((5 - 1) * d, 0), 0.0), nn.Conv1d(c, 64, 5, dilation=d), nn.GELU()]
                c = 64
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(64, 9, 1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, x):                          # (B,L,C) → (B,L,9)
            return self.head(self.body(x.transpose(1, 2))).transpose(1, 2)

    return TCN()


def make_transformer(in_ch: int):
    import torch, torch.nn as nn

    class TinyTransformer(nn.Module):
        """因果自注意力 2 层 d=64 4 头 FF128（规模与 TCN 相近）→ 9。"""

        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(in_ch, 64)
            self.pos = nn.Parameter(torch.zeros(1, L_SEQ, 64)); nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(64)
            self.head = nn.Linear(64, 9)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(L_SEQ))

        def forward(self, x):
            L = x.shape[1]
            h = self.inp(x) + self.pos[:, :L]
            h = self.enc(h, mask=self.causal[:L, :L], is_causal=True)
            return self.head(self.norm(h))

    return TinyTransformer()


def decode(out, ep, windows, st: dict):
    """网络输出 (B,L_SEQ,9) → 每窗 (ê (n_scored,3) torch, q̂ 帧 t0+3…s1+3 (n_scored,6) torch)。

    q̂ 的基线是**钳位后的可见指令**（帧 > t1 用 u_{t1}）—— q̂ 进联合损失，同样不得读下一 chunk。
    """
    import torch
    s_e = torch.tensor(st["s_gap"], dtype=out.dtype, device=out.device)
    s_q = torch.tensor(st["s_eps"], dtype=out.dtype, device=out.device)
    E, Q = [], []
    for b, w in enumerate(windows):
        i = np.arange(w.n_scored) + 2 + D.HIST          # 帧 t+3 在序列中的位置
        E.append(out[b, i, 6:9] * s_e)
        uv = W.visible(ep.cmd[:, :6], w.t1, w.s1 + D.ALIGN_L)[np.arange(w.t0, w.s1 + 1) + D.ALIGN_L]
        u = torch.tensor(uv, dtype=out.dtype, device=out.device)
        Q.append(u + out[b, i, :6] * s_q)
    return E, Q
