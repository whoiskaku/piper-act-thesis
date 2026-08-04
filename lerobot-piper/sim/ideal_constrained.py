#!/usr/bin/env python3
"""约束感知的 ideal 渲染：把穿进桌面的指令投影到最近的可行位形（2026-08-04）。

**动机**：pp2 抓取瞬间 ACT 指令要求末端走到桌面以下（实测指令最低 z −9.62 mm，
真机可达 −5.14 mm，压深 **4.48 mm**）。原 `ideal` 是纯运动学渲染，会照单全收地
"渲染"那段穿进桌面的运动 —— 那部分是虚假的，真机不可能执行。

**做法**：渲染前先把指令投影到可行集。对违反 `z ≥ z_floor` 的位形，
用阻尼最小二乘沿位置雅可比的 z 行做最小范数修正：

    δq = Jz·(z_floor − z) / (Jz·Jzᵀ + λ)      迭代 3 次

只动 6 个臂关节，夹爪不变。投影只用**指令 + 一个标量地板**，
不读真机编码器 —— 否则 ideal 就不再是"指令蕴含的目标"，gap 会被人为抹平。

---

**这个模型有多准（必须一起报）**：

| 量 | 值 |
|---|---|
| 要修的压深 | 4.48 mm |
| 单标量地板的精度（30 集实际最低 z 的 sd） | **±2.60 mm** |

信噪比只有 **1.7**。地板浮动的来源是腕部姿态 —— 约束的真实作用点是**指尖**，
而我们约束的是 `ee_imu_site`，两者的高度差随腕部朝向变化。
模型里的 `deco_table`（顶面 −1.00 mm）是**装饰几何体**，未经标定，不能用作真值。

**已知它修不了什么**：压住帧上带通(0.5–9 Hz)分量实测
`|bp(real)| = 0.2647` vs `|bp(ideal)| = 0.1809 rad/s`（相关系数 −0.106）——
**真机的接触振动比理想还大 68%**。那是真实的执行偏离，运动学约束修不掉；
本模块只能去掉 ideal 那一侧的虚假运动。

用法:
    python sim/ideal_constrained.py --self-test     # 无数据自检
    python sim/ideal_constrained.py                 # 在 pp2 三臂上评估效果
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
Z_FLOOR = -0.00514        # m，30 集抓取窗内实际最低末端 z 的中位数（sd 2.60 mm）
DAMP = 1e-6
N_ITER = 3


class Projector:
    """把指令位形投影到 `ee_imu_site` 高度 ≥ z_floor 的可行集。"""

    def __init__(self, z_floor: float = Z_FLOOR):
        import mujoco
        from ideal_imu_kinematic import MODEL_PATH
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
        self.z_floor = float(z_floor)

    def _full(self, q7):
        from ideal_imu_kinematic import full_qpos
        return full_qpos(q7)

    def z_of(self, q7) -> float:
        self.data.qpos[:] = self._full(q7)
        self.mj.mj_fwdPosition(self.model, self.data)
        return float(self.data.site_xpos[self.sid][2])

    def project_one(self, q7):
        """→ (投影后的位形, 原始违反量 m)。不违反则原样返回。"""
        z0 = self.z_of(q7)
        if z0 >= self.z_floor:
            return q7, 0.0
        out = np.array(q7, dtype=np.float64).copy()
        for _ in range(N_ITER):
            z = self.z_of(out)
            if z >= self.z_floor - 1e-6:
                break
            jacp = np.zeros((3, self.model.nv))
            self.mj.mj_jacSite(self.model, self.data, jacp, None, self.sid)
            Jz = jacp[2, :6]
            denom = float(Jz @ Jz) + DAMP
            out[:6] += Jz * (self.z_floor - z) / denom
        return out, self.z_floor - z0

    def project_stream(self, cmd):
        """(T,7) 指令流 → (投影后的流, 被投影的帧数, 最大违反量 mm)。"""
        out = np.array(cmd, dtype=np.float64).copy()
        n_hit, worst = 0, 0.0
        for i in range(len(out)):
            q, v = self.project_one(out[i])
            if v > 0:
                out[i] = q; n_hit += 1; worst = max(worst, v)
        return out.astype(np.float32), n_hit, worst * 1000


def render_constrained(cmd, prev_cmd_tail, renderer, projector):
    """与 IdealChunkRenderer.render 同签名，但先把 chunk 投影到可行集。"""
    q, n_hit, worst = projector.project_stream(cmd)
    return renderer.render(q, prev_cmd_tail), n_hit, worst


# ---------------------------------------------------------------- 自检
def self_test():
    P = Projector()
    rng = np.random.default_rng(0)
    print("自检：投影后的位形应满足 z ≥ z_floor，且修正量最小")
    ok = True
    tested = 0
    for _ in range(400):
        q = np.zeros(7)
        q[:6] = rng.normal(0, 0.6, 6)
        z0 = P.z_of(q)
        if z0 >= P.z_floor:            # 只测真正违反的样本
            continue
        tested += 1
        qp, v = P.project_one(q)
        z1 = P.z_of(qp)
        moved = np.linalg.norm(qp[:6] - q[:6])
        if z1 < P.z_floor - 1e-4:
            print(f"  ✗ 投影后仍违反: z {z0*1000:.2f} → {z1*1000:.2f} mm"); ok = False
        if moved > 2.0:
            print(f"  ✗ 修正量过大 {moved:.2f} rad"); ok = False
        if abs(qp[6] - q[6]) > 1e-12:
            print("  ✗ 夹爪被改动了"); ok = False
    print(f"  测了 {tested} 个违反样本")
    # 不违反的样本必须**原样**返回
    q = np.zeros(7); q[1] = -0.8
    if P.z_of(q) > P.z_floor:
        qp, v = P.project_one(q)
        if not np.array_equal(qp, q) or v != 0.0:
            print("  ✗ 未违反的位形被改动了"); ok = False
    print(f"  {'✅ 自检通过' if ok else '❌ 自检失败'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- 评估
def episodes(arm, root="outputs/ab2", min_frames=500):
    fs = sorted((ROOT / root).glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return [f for f in fs
            if json.loads(f.read_text())["summary"]["frames"] > min_frames][:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arms", default="A,B3,B3b13")
    ap.add_argument("--z-floor", type=float, default=Z_FLOOR)
    a_ = ap.parse_args()
    if a_.self_test:
        return self_test()

    from analyze_ab import _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from scipy import stats
    bp = _bp()
    rend = IdealChunkRenderer()
    proj = Projector(a_.z_floor)
    R = np.array(json.loads((ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                             ).read_text())["R"], np.float32)
    arms = [s.strip() for s in a_.arms.split(",")]

    print(f"约束感知 ideal 评估   z_floor = {a_.z_floor*1000:.2f} mm  "
          f"(单标量模型精度 ±2.60 mm)")
    E = {}
    for arm in arms:
        g0, g1, hits, worsts = [], [], [], []
        for f in episodes(arm):
            d = json.loads(f.read_text())
            imu = np.array([x["imu"] for x in d["frames"] if x["imu"] is not None],
                           np.float32)
            o0, o1, prev0, prev1 = [], [], None, None
            nh, wr = 0, 0.0
            for c in d["chunks"]:
                raw = np.array(c["raw"], np.float32)
                o0.append(rend.render(raw, prev0)); prev0 = raw
                q, n, w = proj.project_stream(raw)
                o1.append(rend.render(q, prev1)); prev1 = q
                nh += n; wr = max(wr, w)
            i0 = np.concatenate(o0)[:len(imu)]
            i1 = np.concatenate(o1)[:len(imu)]
            n = min(len(i0), len(i1))
            real = imu[:n, 3:6]
            g0.append(float((bp(real - i0[:n, 3:6] @ R.T) ** 2).sum() / n))
            g1.append(float((bp(real - i1[:n, 3:6] @ R.T) ** 2).sum() / n))
            hits.append(nh / n); worsts.append(wr)
        E[arm] = (np.array(g0), np.array(g1))
        print(f"  {arm:>7}: 原 ideal {np.mean(g0):.5f} → 约束 ideal {np.mean(g1):.5f}"
              f"  ({np.mean(g1)/np.mean(g0)*100-100:+.2f}%)"
              f"   被投影帧 {np.mean(hits)*100:.2f}%  最大违反 {np.mean(worsts):.2f} mm")

    print(f"\n  {'比较':<22}{'原 ideal':>12}{'约束 ideal':>12}")
    A0, A1 = E[arms[0]]
    for arm in arms[1:]:
        x0, x1 = E[arm]
        p0 = stats.ttest_rel(x0, A0)[1]; p1 = stats.ttest_rel(x1, A1)[1]
        print(f"  {arm+' vs A':<22}{(x0.mean()/A0.mean()-1)*100:>11.1f}%"
              f"{(x1.mean()/A1.mean()-1)*100:>11.1f}%")
        print(f"  {'  p':<22}{p0:>12.4f}{p1:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
