"""测试用合成集（不读真实数据）：100 帧一个 chunk、编码器前 2 帧无效、一阶滞后伺服。"""
from __future__ import annotations
import numpy as np
import q1_data as D


def synthetic_episode(seed: int = 0, T: int = 587, sess: str = "syn/A",
                      chunk: int = 100) -> D.Q1Episode:
    rng = np.random.default_rng(seed)
    Tc = -(-T // chunk) * chunk
    base = np.array([0.0, 0.8, -0.8, 0.0, 0.4, 0.0, 0.02])
    u = base + np.cumsum(rng.normal(0, 0.004, (Tc, 7)), 0)
    u[:, 3] = 0.0
    u[:, 6] = np.clip(u[:, 6], 0.0, 0.035)
    q = np.empty((T, 6))
    q[0] = u[0, :6]
    for k in range(1, T):
        q[k] = q[k - 1] + 0.3 * (u[k - 1, :6] - q[k - 1])
    q_valid = np.arange(T) >= D.Q_VALID_FROM
    q[~q_valid] = 0.0
    ideal = rng.normal(0, 0.05, (Tc, 3))
    w = rng.normal(0, 0.05, (T, 3))
    caps = np.repeat(np.arange(chunk - 1, Tc, chunk), chunk)
    gap = w[D.ALIGN_L:] - ideal[:T - D.ALIGN_L]
    return D.Q1Episode(name=f"{sess.split('/')[0]}/ep{seed}", sess=sess, path=f"synthetic/{seed}",
                       T=T, cmd=u, ideal=ideal, caps=caps, q=q, q_valid=q_valid,
                       w=w, w_valid=np.ones(T, bool), gap=gap)
