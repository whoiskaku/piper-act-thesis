#!/usr/bin/env python3
"""真机标定契约 —— 把 `m0_synthetic` 的方法学纪律接到真实参数集上。

## 与 M0 的关系

`m0_synthetic.py` 是**方法学框架**：用一个已知的二阶合成对象，在采任何真机
数据之前演练数据角色、gauge 声明、确定性标定、产物溯源、角色分离的 holdout。
它自己写明「not a MuJoCo implementation, not the final G model, and not
evidence about real Piper dynamics」。

**可以搬的是纪律，不是求解器。**

| M0 的做法 | 能否搬到真机 |
|---|---|
| 数据角色强制（`calibrate` 拒绝 role='holdout'） | ✅ 照搬 |
| gauge 显式声明并拒绝被改动 | ✅ 照搬（我们有三个 gauge，见下） |
| 拟合**前**的可辨识性闸门（rank / condition） | ✅ 照搬思想，换成灵敏度条件数 |
| 参数越界即报错，不静默 clip 之外 | ✅ 照搬 |
| 产物溯源（数据 sha / seed / schema 一并落盘） | ✅ 照搬 |
| **线性最小二乘求解** | ❌ **不能** —— 合成对象对参数线性；MuJoCo 里 `fc` 带 `sign()`、`Ia` 进质量矩阵，**非线性** |

M0 的 gauge 理由与我们独立推出的完全一致：
> "A free command scale and the plant gain occur only as a product.
>  They cannot both be estimated."

## 本契约的三个 gauge（都不是取舍，是结构性事实）

1. **`c = b + kv`** —— MuJoCo `<position>` 的 bias 用**执行器速度本身**而非速度误差：
   `τ = kp·(ctrl−q) − (kv+b)·q̇`，二者只以和进入方程。数值验证到机器精度
   （`step2_g/test_degeneracy.py`：`cos = 1.000000000000`）。
2. **`kp₆` 固定为 gauge** —— j6 的 `(kp,c)` 平面全网格 |cos| 最低仅 0.677；
   加大幅度或降低频率只提高信噪比，造不出第二个独立动力学特征。只拟合
   `τ₆ = c₆/kp₆`，报告时称「j6 有效闭环时间常数」，**不得**解释为两个独立参数。
3. **j5/j6 的 `Ia` = `I_eff`（未分离）** —— 绕 j 轴的等效惯量只取决于**比 j 更
   远端**的关节姿态；j6 无远端关节 ⇒ `I_link` 恒定（实测极差 0.0%），j5 仅 1.8%。
   要真分离只能外加**已知末端负载**。

⇒ 24 个参数中 **23 个自由**。

## 数据角色

| 角色 | 数据 | 说明 |
|---|---|---|
| `calibration` | 90 条激励套件（`build_suite`） | 专为辨识设计 |
| `holdout` | 17 集 pp2 裸 ACT | **真实任务**上的表现 |

沿用 M0 的措辞：holdout 是 **role separation only; not sealed, secret,
access-controlled, or confirmatory** —— 它回答「标定出的模型在实际用途上准不准」，
不是一次确证检验。

用法: python sim/step2_g/schema.py        # 打印契约并自检
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.step2_g import params as PA                                   # noqa: E402
from sim.step2_g.m0_synthetic import (                                 # noqa: E402
    HOLDOUT_SEAL, _canonical_json, _hash_arrays, _sha256_bytes,
)

LABEL = ("Piper j1–j6 MuJoCo 伺服参数标定契约; 纪律沿用 M0, 求解器不沿用"
         "(M0 的对象对参数线性, 本问题非线性)")

# 物理合理的搜索界。定界的依据写在注释里 —— 不写依据的界等于没有界。
BOUNDS: dict[str, tuple[float, float]] = {
    # 位置伺服增益。现 XML 值 50–200(j1 另被覆盖为 5000)。下界取 1(极软),
    # 上界 3000 —— 再高则 ωn 远超可复现带(9–12 Hz), 拟合无信息支撑。
    "kp": (1.0, 3000.0),
    # c = b + kv。现 XML 的 500(j1) 几乎肯定不物理(1 rad/s 需 500 N·m,
    # 对负载 1.5kg 的臂太离谱); 上界仍留 600 以免把真值排除在外。
    "c": (0.0, 600.0),
    # 库仑摩擦力矩。0 = 无摩擦(现状), 上界 5 N·m。
    "fc": (0.0, 5.0),
    # 转子/折算惯量。0 = 现状; 上界按各关节 I_link 量级放宽, 见 IA_BOUNDS。
    "Ia": (0.0, 0.5),
}
# Ia 逐关节的上界（I_link 量级差 3 个数量级，用统一上界等于对腕关节没有约束）
IA_BOUNDS = ((0.0, 0.5), (0.0, 0.5), (0.0, 0.3),
             (0.0, 0.05), (0.0, 0.05), (0.0, 0.01))

# 拟合**前**的可辨识性闸门（沿用 M0 的 max_design_condition 思想）。
# 阈值来自 `build_suite --analyze` 在**无噪声仿真**上的初筛结果(条件数 29.6);
# 留 3.4 倍余量给真机噪声。**这不是「真机上已可辨识」的证明。**
MAX_DESIGN_CONDITION = 100.0
MIN_SENSITIVITY_PCT = 0.5           # 低于此视为该参数不可辨识


@dataclass(frozen=True)
class Gauge:
    """一个被固定住的自由度，以及**为什么**它必须被固定。"""

    name: str
    reason: str
    evidence: str


GAUGES = (
    Gauge("c = b + kv（合并）",
          "MuJoCo <position> 的 bias 用执行器速度本身而非速度误差 ⇒ "
          "τ = kp·(ctrl−q) − (kv+b)·q̇，二者只以和进入方程",
          "sim/step2_g/test_degeneracy.py: cos = 1.000000000000, "
          "补偿扰动 ‖Δ‖/‖Δ_b‖ = 1.2e-15"),
    Gauge("kp₆ 固定 = 50",
          "j6 的 (kp,c) 平面全网格 |cos| 最低仅 0.677；加大幅度或降低频率只提高"
          "信噪比，造不出第二个独立动力学特征。自由量是 τ₆ = c₆/kp₆",
          "sim/step2_g/map_separability.py"),
    Gauge("Ia₅ / Ia₆ = I_eff（未分离）",
          "绕 j 轴的等效惯量只取决于比 j 更远端的关节姿态；j6 无远端关节 ⇒ "
          "I_link 恒定。要真分离只能外加已知末端负载",
          "sim/step2_g/../step2_identifiability.py: j6 极差 0.0%, j5 1.8%"),
)


@dataclass(frozen=True)
class DataRole:
    """带强制角色标签的一批真机数据。"""

    role: Literal["calibration", "holdout"]
    source: str
    sequence_shas: tuple[str, ...]
    n_episodes: int

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json(
            {"role": self.role, "source": self.source,
             "shas": list(self.sequence_shas), "n": self.n_episodes}))


@dataclass(frozen=True)
class PiperCalibrationSchema:
    """真机标定契约。`validate()` 会拒绝任何试图改动 gauge 的 schema。"""

    n_joints: int = PA.N_J
    kp6_gauge: float = PA.KP6_GAUGE
    max_design_condition: float = MAX_DESIGN_CONDITION
    min_sensitivity_pct: float = MIN_SENSITIVITY_PCT
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(BOUNDS))
    ia_bounds: tuple[tuple[float, float], ...] = IA_BOUNDS

    def validate(self) -> None:
        if self.n_joints != PA.N_J:
            raise ValueError(f"n_joints 必须为 {PA.N_J}")
        if self.kp6_gauge != PA.KP6_GAUGE:
            raise ValueError(
                f"kp₆ 是 gauge，必须固定为 {PA.KP6_GAUGE}；"
                f"改动它会静默改变模型 gauge（同 M0 对 command_scale_gauge 的处理）")
        if not (np.isfinite(self.max_design_condition) and self.max_design_condition > 1):
            raise ValueError("max_design_condition 必须有限且 > 1")
        for name, (lo, hi) in self.bounds.items():
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"{name} 的界非法: {(lo, hi)}")
        if len(self.ia_bounds) != PA.N_J:
            raise ValueError("ia_bounds 必须逐关节给出")

    @property
    def n_free(self) -> int:
        return PA.N_FREE

    def check_in_bounds(self, theta: np.ndarray) -> list[str]:
        """→ 越界项清单（空 = 全部在界内）。**越界报错，不静默 clip。**"""
        p = PA.unpack(theta)
        bad = []
        for blk in ("kp", "c", "fc"):
            lo, hi = self.bounds[blk]
            for j, v in enumerate(p[blk]):
                if v < lo - 1e-12 or v > hi + 1e-12:
                    bad.append(f"{blk}{j+1}={v:.4g} 越界 [{lo}, {hi}]")
        for j, v in enumerate(p["Ia"]):
            lo, hi = self.ia_bounds[j]
            if v < lo - 1e-12 or v > hi + 1e-12:
                bad.append(f"Ia{j+1}={v:.4g} 越界 [{lo}, {hi}]")
        return bad

    def start_point(self, theta: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """把**起点**投影进界内。→ (投影后的 θ, 被投影的项)。

        起点与拟合结果的处理**必须不同**:
          · **起点**越界 ⇒ 投影进界并**明确报出**（否则优化器从非法点出发）
          · **拟合结果**越界 ⇒ **报错**（`check_in_bounds`），不静默 clip
        实测现 XML 基线的 `kp1 = 5000` 越界: 那是 `dyn_replay_imu.py` 里手动
        覆盖的值(`--j-kp 1:5000`), 为硬凑 2 帧滞后而设。按 ωn = √(kp/I),
        j1 在低惯量位形(I=0.004)下 ωn = **178 Hz**, 比可复现带(9–12 Hz)高一个
        数量级 —— 该值没有数据支撑, 越界是**基线不物理**而非界定得太紧。
        """
        out = np.asarray(theta, float).copy()
        moved = []
        p_ = PA.unpack(out)
        for bi, blk in enumerate(PA.BLOCKS):
            for j in range(PA.N_J):
                lo, hi = (self.ia_bounds[j] if blk == "Ia" else self.bounds[blk])
                v = p_[blk][j]
                if v < lo or v > hi:
                    out[bi * PA.N_J + j] = float(np.clip(v, lo, hi))
                    moved.append(f"{blk}{j+1}: {v:.4g} → {out[bi*PA.N_J+j]:.4g}")
        return out, moved

    def gate_identifiability(self, condition: float,
                             sensitivity_pct: dict[str, float]) -> list[str]:
        """拟合**前**的闸门（沿用 M0：设计不可辨识就拒绝拟合，而不是硬拟合）。"""
        bad = []
        if not np.isfinite(condition) or condition > self.max_design_condition:
            bad.append(f"设计条件数 {condition:.1f} > {self.max_design_condition}")
        dead = [k for k, v in sensitivity_pct.items() if v < self.min_sensitivity_pct]
        if dead:
            bad.append(f"灵敏度 < {self.min_sensitivity_pct}% 的参数: {dead}")
        return bad


def describe() -> dict:
    s = PiperCalibrationSchema()
    s.validate()
    return {
        "label": LABEL,
        "n_parameters": PA.N_P, "n_free": s.n_free,
        "blocks": list(PA.BLOCKS),
        "gauges": [{"name": g.name, "reason": g.reason, "evidence": g.evidence}
                   for g in GAUGES],
        "bounds": s.bounds, "ia_bounds": [list(b) for b in s.ia_bounds],
        "identifiability_gate": {
            "max_design_condition": s.max_design_condition,
            "min_sensitivity_pct": s.min_sensitivity_pct,
            "provenance": "阈值源自 build_suite --analyze 的**无噪声仿真初筛**"
                          "(条件数 29.6), 留 3.4 倍余量; **不是**「真机上已可辨识」的证明",
        },
        "data_roles": {
            "calibration": "90 条激励套件 (sim/step2_g/build_suite.py)",
            "holdout": "17 集 pp2 裸 ACT —— 真实任务上的表现",
            "seal": HOLDOUT_SEAL,
        },
        "not_transferable_from_m0": [
            "线性最小二乘求解 —— M0 的对象对参数线性; 本问题 fc 带 sign()、"
            "Ia 进质量矩阵, **非线性**",
        ],
    }


def main() -> int:
    d = describe()
    print(f"{d['label']}\n")
    print(f"参数 {d['n_parameters']} 个，其中**自由 {d['n_free']} 个**"
          f"（块: {', '.join(d['blocks'])}）\n")
    print("Gauge（被固定的自由度，及其结构性理由）:")
    for g in d["gauges"]:
        print(f"  · {g['name']}")
        print(f"      理由: {g['reason']}")
        print(f"      证据: {g['evidence']}")
    print(f"\n搜索界:")
    for k, (lo, hi) in d["bounds"].items():
        print(f"  {k:<4} [{lo}, {hi}]" + ("   （Ia 另按关节逐一给界）" if k == "Ia" else ""))
    print(f"  Ia 逐关节: {d['ia_bounds']}")
    g = d["identifiability_gate"]
    print(f"\n拟合前的可辨识性闸门: 条件数 ≤ {g['max_design_condition']}, "
          f"逐参数灵敏度 ≥ {g['min_sensitivity_pct']}%")
    print(f"  {g['provenance']}")
    print(f"\n数据角色:")
    for k in ("calibration", "holdout"):
        print(f"  {k:<12} {d['data_roles'][k]}")
    print(f"  holdout 措辞: {d['data_roles']['seal']}")
    print(f"\n**不从 M0 沿用**: {d['not_transferable_from_m0'][0]}")

    # 自检
    s = PiperCalibrationSchema()
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/piper_model.xml"))
    th = PA.baseline(m)
    bad = s.check_in_bounds(th)
    print(f"\n自检: 现 XML 基线在界内? {'✅' if not bad else '否 —— ' + str(bad)}")
    th0, moved = s.start_point(th)
    if moved:
        print(f"        ⇒ **起点**投影进界: {moved}")
        print(f"        （越界是**基线不物理**: kp1=5000 是手调覆盖值，"
              f"在 j1 低惯量位形下 ωn≈178 Hz，远超可复现带 9–12 Hz）")
        print(f"        拟合**结果**若越界则报错，不投影 —— 两者处理不同。")
        assert not s.check_in_bounds(th0), "投影后仍越界"
        print(f"        投影后在界内 ✅")
    try:
        PiperCalibrationSchema(kp6_gauge=999.0).validate()
        print("自检: gauge 保护 ❌ 未生效")
    except ValueError as e:
        print(f"自检: gauge 保护 ✅（{str(e)[:40]}…）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
