"""测试用合成集（不读真实数据）：阶段一的合成集 + δa（补正运行 / 探针为小随机序列，裸跑为 0）。"""
from __future__ import annotations
import numpy as np
import q1_testing as T1
import q2_data as D2


def synthetic(seed: int = 0, fold: str = "ab5", arm: str = "C", kind: str = "corr", T: int = 587) -> D2.Q2Episode:
    e1 = T1.synthetic_episode(seed, T=T, sess=f"{fold}/{arm}")
    e1.name = f"{fold}/{arm}_ep{seed}"
    rng = np.random.default_rng(1000 + seed)
    da = np.zeros((e1.T, 6)) if kind == "bare" else rng.normal(0.0, 1e-3, (e1.T, 6))
    if kind != "bare":
        da[:, 3] = 0.0                                          # 同真实数据：j4 不补正
    return D2.from_q1(e1, fold, arm, kind, da)


def synthetic_set(n_per: int = 2) -> list:
    """3 个近期会话 × {裸跑, 补正} × n_per 集 + ab4 P2/P3 各 n_per 集。"""
    out, s = [], 0
    for fold in D2.FOLDS:
        for arm in D2.BARE[fold]:
            for _ in range(n_per):
                out.append(synthetic(s, fold, arm, "bare")); s += 1
        for arm in D2.CORR[fold][:1]:
            for _ in range(n_per):
                out.append(synthetic(s, fold, arm, "corr")); s += 1
    for arm in D2.AUX["ab4"]:
        for _ in range(n_per):
            out.append(synthetic(s, "ab4", arm, "aux")); s += 1
    return out


def roster_of(eps) -> dict:
    return {k: sorted(v) for k, v in D2.groups(eps).items()}
