#!/usr/bin/env python3
"""Step 2 的参数集 —— **只含可辨识的量**。

两处合并不是取舍, 是结构性事实, 已用数值实验证实到机器精度:

① **`b` 与 `kv` 精确简并**（`sim/step2_g/test_degeneracy.py` 验证）。
   MuJoCo `<position>` 执行器 biastype=affine:
       f = kp·ctrl + biasprm[1]·q + biasprm[2]·q̇ = kp·(ctrl − q) − kv·q̇
   注意 `q̇` 是**执行器速度本身**, 不是速度误差 —— 模型里根本没有 `q̇_des`。
   叠加关节阻尼后:
       τ = kp·(ctrl − q) − (kv + b)·q̇
   ⇒ 二者只以**和**进入方程, 是 FIM 的精确零空间。恒速段/快瞬态**都打不破**它。
   ⇒ 合并为 `c = b + kv`, 全部记入 `dof_damping`, `biasprm[2]` 恒置 0。
   **若任何分析报告 b 与 kv 可分, 那是实现有误。**

② **j5/j6 的 `Ia` 与 `I_link` 结构性混淆**（`step2_identifiability.py`）。
   绕 j 轴的等效惯量只取决于**比 j 更远端**的关节姿态; j6 无远端关节 ⇒
   `I_link@j6` 恒定(实测极差 0.0%), j5 仅 1.8%。
   ⇒ 这两个关节拟合**有效惯量** `I_eff = I_link + Ia`, 并**明确声明未分离**。
   要真分离, 只能外加**已知末端负载**改变 `I_link`。
"""
from __future__ import annotations

import numpy as np

N_J = 6
BLOCKS = ("kp", "c", "fc", "Ia")          # 每块 6 个 ⇒ 24 参数
NAMES = [f"{b}{j+1}" for b in BLOCKS for j in range(N_J)]
N_P = len(NAMES)

# Ia 的解读: j1/j2/j4 可用远端位形检验 I_link 假设; j3 弱; j5/j6 = I_eff(未分离)
IA_SEPARABLE = {0: "可检验", 1: "可检验", 2: "弱", 3: "可检验",
                4: "I_eff(未分离)", 5: "I_eff(未分离)"}


def unpack(theta: np.ndarray) -> dict[str, np.ndarray]:
    t = np.asarray(theta, float).reshape(len(BLOCKS), N_J)
    return {b: t[i] for i, b in enumerate(BLOCKS)}


def apply(m, theta: np.ndarray) -> None:
    """把参数写进 MuJoCo 模型。**`kv` 恒置 0** —— 见模块 docstring ①。"""
    p = unpack(theta)
    for j in range(N_J):
        m.actuator_gainprm[j, 0] = p["kp"][j]
        m.actuator_biasprm[j, 1] = -p["kp"][j]
        m.actuator_biasprm[j, 2] = 0.0            # kv 已并入 c
        m.dof_damping[j] = p["c"][j]
        m.dof_frictionloss[j] = p["fc"][j]
        m.dof_armature[j] = p["Ia"][j]


def baseline(m) -> np.ndarray:
    """现模型的手调值作为标定起点（j1 的 kp 沿用 dyn_replay_imu 的 5000 覆盖）。"""
    kp = m.actuator_gainprm[:N_J, 0].astype(float).copy()
    kp[0] = 5000.0
    c = m.dof_damping[:N_J].astype(float).copy() - m.actuator_biasprm[:N_J, 2]
    return np.concatenate([kp, c, np.zeros(N_J), np.zeros(N_J)])


def perturb_scale(theta: np.ndarray) -> np.ndarray:
    """各参数做灵敏度分析时的物理合理扰动尺度。"""
    p = unpack(theta)
    return np.concatenate([
        np.maximum(p["kp"] * 0.15, 1.0),          # kp: 15%
        np.maximum(p["c"] * 0.15, 0.5),           # c : 15%（现值可能为 0）
        np.full(N_J, 0.05),                       # fc: 0.05 N·m
        np.array([0.02, 0.02, 0.02, 0.002, 0.002, 0.0005]),   # Ia: 按各关节惯量量级
    ])
