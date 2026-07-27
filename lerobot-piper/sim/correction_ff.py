#!/usr/bin/env python3
"""P2 机制层1 — 确定性前馈指令整形器 (plus-on, 模型无关)。

拦截点: ACT 每 chunk 输出 A=[a_1..a_100] → 本模块变换 → 发真机。三个开关:
  FF1 lag-preview   : 组内指令前移 L̂ 帧(实测滞后 demo2/deploy3), 尾部保持 — 补录制/伺服滞后
  FF2 boundary-blend: 新 chunk 前 M 帧与旧 chunk 尾外推做余弦交叉淡入 — 消边界跳变(画像D: 指令8×)
  FF3 intent-shaping: chunk 内零相位低通(fc, 用上一 chunk 尾作左上下文避免接缝) —
                      削 4.5-10Hz 指令毛刺(画像A: 该毛刺被真机复现并激励结构振动)
  夹爪列(第7维)永远原样直通(开合必须陡)。GATE: 静止段(指令近恒)不动作。
离线驱动: apply_to_episode(actions) 逐 chunk 模拟部署路径, 供已录数据验证。
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

ARM = slice(0, 6)          # 整形只作用机械臂 6 关节
CHUNK = 100                # no-TE: 每 100 步重规划


def _lowpass_with_context(chunk_arm: np.ndarray, left_ctx: np.ndarray | None,
                          fc: float, fps: float) -> np.ndarray:
    """零相位低通; 用左上下文(上一chunk已整形尾)避免 filtfilt 起始瞬态接缝。"""
    ba = butter(2, fc / (fps / 2))
    if left_ctx is not None and len(left_ctx) > 0:
        x = np.concatenate([left_ctx, chunk_arm])
        return filtfilt(*ba, x, axis=0)[len(left_ctx):]
    return filtfilt(*ba, chunk_arm, axis=0)


def transform_chunk(chunk: np.ndarray, prev_tail: np.ndarray | None,
                    lag: int = 0, blend: int = 4, fc: float | None = 3.0,
                    fps: float = 20.0) -> np.ndarray:
    """单 chunk 变换(部署路径). chunk (100,7); prev_tail=上一chunk已发送的最后几帧(≥2)或None。"""
    out = chunk.copy()

    # FF2 边界衔接(先做, 在"原料"上把跳变缝起来): 新 chunk 前 blend 帧
    # 与旧尾线性外推做余弦淡入 → 得到连续原料流
    if blend and prev_tail is not None and len(prev_tail) >= 2:
        ext_base = prev_tail[-1, ARM]
        ext_vel = prev_tail[-1, ARM] - prev_tail[-2, ARM]
        for k in range(min(blend, len(out))):
            alpha = 0.5 * (1 - np.cos(np.pi * (k + 1) / (blend + 1)))  # 0→1
            ext = ext_base + ext_vel * (k + 1)
            out[k, ARM] = alpha * out[k, ARM] + (1 - alpha) * ext

    # FF3 意图整形(后做, 用上一chunk已整形尾作左上下文 → 接缝被一并平滑)
    if fc:
        ctx = prev_tail[:, ARM] if prev_tail is not None and len(prev_tail) >= 8 else None
        # GATE: 该 chunk 指令几乎不动(峰峰值<2mrad)则跳过, 不对噪声做功
        if np.ptp(out[:, ARM], axis=0).max() > 2e-3:
            out[:, ARM] = _lowpass_with_context(out[:, ARM], ctx, fc, fps)

    # FF1 滞后预览: 组内前移 lag 帧, 尾部保持末值
    if lag > 0:
        out[:, ARM] = np.vstack([out[lag:, ARM], np.repeat(out[-1:, ARM], lag, axis=0)])

    return out


def apply_to_episode(actions: np.ndarray, lag: int = 0, blend: int = 4,
                     fc: float | None = 3.0, fps: float = 20.0) -> np.ndarray:
    """离线: 把整条已录指令流按部署路径逐 chunk 变换(模拟真部署的信息可用性)。"""
    T = len(actions)
    out = np.empty_like(actions)
    prev_tail = None
    for s in range(0, T, CHUNK):
        ck = actions[s:s + CHUNK]
        tf = transform_chunk(ck, prev_tail, lag=lag, blend=blend, fc=fc, fps=fps)
        out[s:s + len(tf)] = tf
        prev_tail = tf[-8:].copy()
    return out
