#!/usr/bin/env python3
"""C 候选 π 的**训练与部署共用**特征构造器 —— 单一实现，两边不可能分叉。

为什么必须共用一份
------------------
2026-09-14 我在特征里把 `observation.state` 直接当 rad 用，而它其实是 **mdeg**
（实测比值 57268 ≈ 1000/(π/180)）。因为 `corr(state, cmd) = 0.997–0.999`，
真正的跟踪误差是**两个大数之差**，单位错就被完全淹没 —— 差点报成"编码器无用"的假阴性。
修正后编码器反馈从 4.37% 变成 5.05%。

⇒ 换算常数与特征装配**只此一处**，训练与部署都调它。

运行时安全（用户 2026-09-14 指定）
----------------------------------
- **启动时**可以断言（`assert_units`）。
- **运行中**若 `state` 缺失/异常，**不得**用断言中断控制 ——
  `encoder_block()` 返回 `(零特征, 原因)`，调用方据此把补正输出归零并记录。
"""
from __future__ import annotations
import numpy as np

# `observation.state` 的单位是 **mdeg**；命令是 **rad**。
MDEG_TO_RAD = np.pi / 180000.0          # mdeg × π/180000 → rad
JLAG = 8                                 # 反馈历史长度（帧）
ALIGN_L = 3                              # gap[t] = real[t+3] − ideal[t]
STATE_ABS_MAX_RAD = 3.2                  # |q| 合理上界（≈183°），超出即视为异常


def mdeg_to_rad(state_mdeg) -> np.ndarray:
    """`observation.state`（mdeg）→ rad。**唯一**的换算入口。"""
    return np.asarray(state_mdeg, float)[..., :6] * MDEG_TO_RAD


def state_ok(q_rad) -> tuple[bool, str]:
    """运行中的状态健康检查。→ (是否可用, 原因)。**不抛异常。**"""
    q = np.asarray(q_rad, float)
    if q.size == 0 or q.shape[-1] != 6:
        return False, "state_shape"
    if not np.isfinite(q).all():
        return False, "state_nan"
    if np.abs(q).max() > STATE_ABS_MAX_RAD:
        return False, "state_out_of_range"
    return True, ""


def assert_units(state_mdeg, cmd_rad) -> None:
    """**启动时**的一致性断言：换算后 state 与命令必须同量级且高度相关。"""
    q = mdeg_to_rad(state_mdeg)
    c = np.asarray(cmd_rad, float)[..., :6]
    n = min(len(q), len(c))
    q, c = q[:n], c[:n]
    rq, rc = np.sqrt((q ** 2).mean()), np.sqrt((c ** 2).mean())
    assert 0.2 < rq / max(rc, 1e-12) < 5.0, (
        f"单位不匹配: 换算后 state RMS {rq:.4f} vs 命令 RMS {rc:.4f} "
        f"(比值 {rq/max(rc,1e-12):.1f})。检查 MDEG_TO_RAD。")
    mov = [j for j in range(6) if c[:, j].std() > 1e-6 and q[:, j].std() > 1e-6]
    assert mov, "没有任何关节在动，无法校验"
    cr = float(np.mean([np.corrcoef(q[:, j], c[:, j])[0, 1] for j in mov]))
    assert cr > 0.9, f"换算后 state 与命令的相关只有 {cr:.3f}，单位或对齐有误"


def hist(src: np.ndarray, t: int, n: int = JLAG, j0: int = ALIGN_L) -> np.ndarray:
    """严格因果的历史切片：只取 `s ≤ t − j0`，越界补零。"""
    idx = [t - j0 - k for k in range(n + 1)]
    d = src.shape[1]
    return np.concatenate([src[i] if i >= 0 else np.zeros(d) for i in idx])


def encoder_block(state_mdeg, cmd_rad) -> tuple[np.ndarray, str]:
    """编码器通道：`[跟踪误差(6), 实测速度(6)]`。→ (特征, 原因)。

    原因非空 ⇒ **状态不可用**，调用方须把补正输出归零并记录，**不要抛异常**。
    """
    q = mdeg_to_rad(state_mdeg)
    ok, why = state_ok(q)
    c = np.asarray(cmd_rad, float)[..., :6]
    n = min(len(q), len(c))
    if not ok:
        return np.zeros((max(n, 1), 12)), why
    return np.concatenate([q[:n] - c[:n], np.gradient(q[:n], axis=0)], 1), ""


def build_features(x36: np.ndarray, r_imu: np.ndarray, enc: np.ndarray,
                   t: int) -> np.ndarray:
    """单帧特征：指令(36) + IMU 残差历史 + 编码器历史。**训练与部署同一份。**"""
    return np.concatenate([x36[t], hist(r_imu, t), hist(enc, t)])


FEAT_DIM = 36 + 3 * (JLAG + 1) + 12 * (JLAG + 1)      # = 36 + 27 + 108 = 171
