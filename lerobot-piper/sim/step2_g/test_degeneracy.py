#!/usr/bin/env python3
"""不变式: `b` 与 `kv` 在 MuJoCo position 伺服里**精确简并**。

这不是一个「测数据够不够」的检验, 而是模型方程的性质。它被写成断言, 是因为
如果哪天某个分析报告二者可分, 那一定是**实现有误**（例如误改成了速度误差
控制器、或 gear≠1、或把 kv 写进了别的通道）, 而不是发现了新信息。

用法: python sim/step2_g/test_degeneracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "sim/piper_model.xml"
TOL = 1e-9


def _roll(db: float, dkv: float, n: int = 400) -> np.ndarray:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.dof_damping[:6] += db
    m.actuator_biasprm[:6, 2] -= dkv          # kv 以 −kv 存于 biasprm[2]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    w = 2 * np.pi * np.array([1.0, 1.7, 2.3, 3.1, 4.3, 5.7])
    out = np.zeros((n, 6))
    for k in range(n):
        d.ctrl[:6] = 0.3 * np.sin(w * k * m.opt.timestep)
        mujoco.mj_step(m, d)
        out[k] = np.concatenate([d.sensordata[3:6], d.qvel[:3]])
    return out


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    assert np.allclose(m.actuator_gear[:6, 0], 1.0), \
        "gear≠1 —— 执行器速度不等于关节速度, 简并关系需重新推导"
    assert set(m.actuator_biastype[:6]) == {1}, "biastype 非 affine —— 前提不成立"

    base = _roll(0.0, 0.0)
    da = _roll(2.0, 0.0) - base               # 只加 b
    dk = _roll(0.0, 2.0) - base               # 只加 kv
    dz = _roll(2.0, -2.0) - base              # 和不变

    rel = float(np.linalg.norm(da - dk) / np.linalg.norm(da))
    cos = float(da.ravel() @ dk.ravel() / np.linalg.norm(da) / np.linalg.norm(dk))
    null = float(np.linalg.norm(dz) / np.linalg.norm(da))
    print(f"‖Δ_b − Δ_kv‖/‖Δ_b‖ = {rel:.3e}   cos = {cos:.12f}")
    print(f"b+=2 且 kv-=2（和不变）⇒ ‖Δ‖/‖Δ_b‖ = {null:.3e}")
    assert rel < TOL, f"b 与 kv 的效应不同({rel:.2e}) —— 简并前提被破坏, 查实现"
    assert null < TOL, f"补偿扰动未归零({null:.2e}) —— 零空间不成立, 查实现"
    print("✅ 精确简并成立 ⇒ 只能拟合 c = b + kv（params.py 已按此合并）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
