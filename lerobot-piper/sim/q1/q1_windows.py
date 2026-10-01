"""阶段一的时间轴规则 —— `DESIGN_G_PI_V2.md` §2.3（**防泄漏规则只在这里实现**）。

- 起点 `t0` = chunk 开头；实测可用到 `t0`（含）；指令/理想陀螺可用到 `t1`（当前 chunk 末帧），
  超出 `t1` 一律钳位到第 `t1` 帧（与 `build_command_features` 同规）。
- 计分 `ê_t, t ∈ [t0, s1]`，`s1 = min(t1, n−1)`；前瞻 = `t − t0 + 3`。
- 逐 chunk 拼接成整集，再按 `P(·)` 计分：逐集去均值 → bandpass41（0.5–9 Hz）→ 丢首尾 20。
- 历史不足或实测无效：指令填 `u_0`、理想陀螺填 0、编码器填 `u_0[:6]`、陀螺填 0，并给有效位掩码。
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import q1_data as D

MID_OFFSET = 50               # 辅助报告的中段起点（默认值，可改）
# 3–6 = chunk 头 4 帧；97–102 = 每个完整 chunk 的末 6 个计分帧（末端钳位的影响集中处，只报告）
LEAD_BINS = ((3, 6), (7, 12), (13, 27), (28, 52), (53, 96), (97, 102))


@dataclass(frozen=True)
class Window:
    t0: int                   # 起点：首个计分帧；实测可见到此帧（含）
    t1: int                   # 当前 chunk 末帧：指令可见到此帧（含）
    s1: int                   # 末个计分帧 = min(t1, n−1)

    @property
    def n_scored(self) -> int:
        return self.s1 - self.t0 + 1


def chunk_windows(ep) -> list[Window]:
    """主分析：每个 chunk 一个窗口，恰好覆盖 `[0, n−1]` 一次。"""
    out, t0 = [], 0
    while t0 < ep.n:
        t1 = int(ep.caps[t0])
        out.append(Window(t0, t1, min(t1, ep.n - 1)))
        t0 = t1 + 1
    return out


def mid_windows(ep, offset: int = MID_OFFSET) -> list[Window]:
    """辅助报告：chunk 中段起点 `t0 + offset`。**不进主分数、不用于挑模型。**"""
    return [Window(w.t0 + offset, w.t1, w.s1) for w in chunk_windows(ep)
            if w.t0 + offset <= w.s1]


def visible(arr, t1: int, upto: int) -> np.ndarray:
    """第 `0 … upto` 行；`> t1` 的行钳位到 `arr[t1]`（下一 chunk 不可见）。"""
    a = np.asarray(arr)
    head = a[:min(upto, t1) + 1]
    if upto <= t1:
        return np.array(head, copy=True)
    return np.concatenate([head, np.repeat(a[t1:t1 + 1], upto - t1, axis=0)], 0)


def span(arr, lo: int, hi: int, fill) -> np.ndarray:
    """第 `lo … hi` 行（含）；`< 0` 的行填 `fill`（历史不足，不借未来）。"""
    a = np.asarray(arr)
    k = np.arange(lo, hi + 1)
    out = np.empty((len(k),) + a.shape[1:], dtype=np.float64)
    neg = k < 0
    out[neg] = fill
    out[~neg] = a[k[~neg]]
    return out


def past_block(ep, t0: int, measured: bool) -> dict:
    """起点前 `HIST` 帧（`t0−19 … t0`）的输入与有效位。"""
    lo = t0 - D.HIST + 1
    k = np.arange(lo, t0 + 1)
    out = {"u": span(ep.cmd, lo, t0, ep.cmd[0]),
           "ideal": span(ep.ideal, lo, t0, 0.0),
           "m_cmd": (k >= 0).astype(np.float64)}
    if measured:
        ok = k >= 0
        qv = np.zeros(len(k), bool)
        wv = np.zeros(len(k), bool)
        qv[ok] = ep.q_valid[k[ok]]
        wv[ok] = ep.w_valid[k[ok]]
        q = np.repeat(ep.cmd[0:1, :6], len(k), 0).astype(np.float64)
        w = np.zeros((len(k), 3))
        q[qv] = ep.q[k[qv]]
        w[wv] = ep.w[k[wv]]
        out.update({"q": q, "w": w, "m_q": qv.astype(np.float64), "m_w": wv.astype(np.float64)})
    return out


def stitch(n: int, windows, preds) -> np.ndarray:
    """逐窗 `ê`（各 `(n_scored,3)`）拼成整集 `(n,3)`；要求恰好覆盖每帧一次。"""
    out = np.full((n, 3), np.nan)
    seen = np.zeros(n, int)
    for w, p in zip(windows, preds):
        p = np.asarray(p, np.float64)
        if p.shape != (w.n_scored, 3):
            raise ValueError(f"窗口 {w} 的预测形状 {p.shape}")
        out[w.t0:w.s1 + 1] = p
        seen[w.t0:w.s1 + 1] += 1
    if not (seen == 1).all():
        raise ValueError("窗口没有恰好覆盖每帧一次")
    return out


def overlay(base, windows, preds) -> np.ndarray:
    """辅助报告：在主拼接序列上，用中段起点的预测覆盖 `[t0', s1]`。"""
    out = np.array(base, dtype=np.float64, copy=True)
    for w, p in zip(windows, preds):
        out[w.t0:w.s1 + 1] = p
    return out


def band_clean(ok) -> np.ndarray:
    """带通（41 tap 居中）后第 k 帧「干净」⇔ 原始第 k−20 … k+20 帧全部有效；首尾 20 帧一律否。"""
    ok = np.asarray(ok, bool)
    bad = np.convolve((~ok).astype(np.float64), np.ones(2 * D.EDGE + 1), mode="same") > 0
    out = ~bad
    out[:D.EDGE] = False
    out[len(out) - D.EDGE:] = False
    return out


def kernel() -> np.ndarray:
    """与 `train_fabs_g0.bp_kernel` 逐位相同（0.5–9 Hz、41 tap、float32）。"""
    from scipy.signal import firwin
    return firwin(41, [0.5, 9.0], fs=D.FS, pass_zero=False).astype(np.float32)


def P(v, K) -> np.ndarray:
    r = np.asarray(v, np.float64)
    r = r - r.mean(0, keepdims=True)
    f = np.stack([np.convolve(r[:, j], K, mode="same") for j in range(r.shape[1])], 1)
    return f[D.EDGE:-D.EDGE]


def score(gap, ehat, K) -> tuple[float, float, int]:
    """→ (num = ‖P(gap − ê)‖², den = ‖P(gap)‖², 3·T_e)。与 `train_fabs_g0.err_energy` 同口径。"""
    a = P(np.asarray(gap) - np.asarray(ehat), K)
    d = P(gap, K)
    return float((a ** 2).sum()), float((d ** 2).sum()), int(a.size)


def lead_of(n: int, windows) -> np.ndarray:
    lead = np.full(n, -1, int)
    for w in windows:
        lead[w.t0:w.s1 + 1] = np.arange(w.n_scored) + D.ALIGN_L
    return lead


def lead_score(gap, ehat, windows, K, bins=LEAD_BINS) -> dict:
    """带通后的逐帧残差按前瞻归组。⚠ 带通混合相邻帧 ⇒ 分层结果是近似。"""
    n = len(gap)
    L = lead_of(n, windows)[D.EDGE:n - D.EDGE]
    a = P(np.asarray(gap) - np.asarray(ehat), K)
    d = P(gap, K)
    out = {}
    for lo, hi in bins:
        m = (L >= lo) & (L <= hi)
        out[f"{lo}-{hi}"] = [float((a[m] ** 2).sum()), float((d[m] ** 2).sum())]
    return out
