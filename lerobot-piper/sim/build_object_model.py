#!/usr/bin/env python3
"""Stage 4.0 — 由 config + demo 抓取点反推，构建 object-aware MuJoCo 模型（B+ 方案）。

object = 一卷胶带（薄圆环 annulus，平放）。B+ 设计：
  - 碰撞：N 段 box 拼成的圆环（简化 proxy，但保留环孔 → 指能入孔、桌面接触正确、
    rim-pinch 成立）。
  - 质量分布（正确）：body 原点置于环心，`<inertial>` 显式设为解析 annulus
    质量 + 惯量 → COM 在环心（距抓取点 ~mean_radius），重力力矩 / lift / release 真实。
    （不靠 geom 自动推惯量，避免离散化误差。）
  - 摆放：环心 = 指垫汇聚点 + mean_radius 沿夹爪水平前向（图片确认侧前接近）。
    环平放（轴竖直），静置于桌面：ring_center_z = table_top + H/2。

提供：load_config / detect_grasp_frames / fk_grasp / build_object_xml /
      write_object_model / derive_placement
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
BASE_XML = SIM_DIR / "piper_model.xml"
CONFIG = SIM_DIR / "object_sim_config.json"

# link6 原点 → 指垫抓取中心 的距离（沿 link6 z 轴）。link7/8 基座在 +0.1358，
# 指垫再往前 ~0.035 → ~0.17m。近似值，4.2 标定阶段再校。
FINGER_DIST_M = 0.17


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def object_mass_kg(cfg: dict) -> float:
    m = cfg["object"].get("mass_g")
    if m is None:
        m = cfg["modeling"]["placeholder_mass_g"]
    return float(m) / 1000.0


def annulus_inertia(mass, r_out, r_in, h):
    """薄圆环（hollow cylinder）绕质心的对角惯量。轴=z。"""
    iz = 0.5 * mass * (r_out ** 2 + r_in ** 2)
    ixy = (1.0 / 12.0) * mass * (3.0 * (r_out ** 2 + r_in ** 2) + h ** 2)
    return ixy, ixy, iz


def detect_grasp_frames(demo: dict, grip_channel: int = 6) -> list[int]:
    grip = demo["action"][:, grip_channel]
    dg = np.abs(np.diff(grip))
    if dg.max() <= 0:
        return []
    evs = np.where(dg > dg.max() * 0.3)[0]
    out = []
    for e in evs:
        if not out or e - out[-1] > 15:
            out.append(int(e))
    return out


def fk_grasp(base_xml: Path, qpos8: np.ndarray):
    """FK：返回 (pinch_center_xyz, fwd_unit) —— 指垫抓取中心 + 夹爪前向(link6 z, 世界)。"""
    model = mujoco.MjModel.from_xml_path(str(base_xml))
    data = mujoco.MjData(model)
    data.qpos[:len(qpos8)] = qpos8
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    p6 = data.xpos[bid].copy()
    fwd = data.xmat[bid].reshape(3, 3)[:, 2].copy()      # link6 z 轴(世界) = 夹爪前向
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    pinch = p6 + FINGER_DIST_M * fwd
    return pinch, fwd


def _s(v):  return " ".join(f"{x:g}" for x in v)


def build_object_xml(cfg: dict, ring_center, table_top_z: float) -> str:
    base = BASE_XML.read_text()
    o = cfg["object"]
    r_out = o["d_outer_mm"] / 2000.0
    r_in = o["d_inner_mm"] / 2000.0
    h = o["height_mm"] / 1000.0
    r_mean = (r_out + r_in) / 2.0
    wall = r_out - r_in
    mass = object_mass_kg(cfg)
    N = int(cfg["modeling"]["n_segments"])
    ixy, _, iz = annulus_inertia(mass, r_out, r_in, h)

    c = cfg["contact"]
    phys = (f'friction="{_s(c["friction"])}" solref="{_s(c["solref"])}" '
            f'solimp="{_s(c["solimp"])}" condim="{c["condim"]}"')

    # 环段：每段 box，local x=径向(wall), y=切向(seg_len), z=竖直(h)；绕 z 旋转 θ_i。
    seg_len = 1.15 * (2 * math.pi * r_mean / N)            # 略重叠避免缝隙
    segs = []
    for i in range(N):
        th = 2 * math.pi * i / N
        cx, cy = r_mean * math.cos(th), r_mean * math.sin(th)
        qz = math.sin(th / 2.0)
        segs.append(
            f'            <geom name="tape_seg{i}" type="box" pos="{cx:.5f} {cy:.5f} 0" '
            f'quat="{math.cos(th/2.0):.6f} 0 0 {qz:.6f}" '
            f'size="{wall/2:g} {seg_len/2:g} {h/2:g}" rgba="0.4 0.7 0.4 1" {phys}/>')
    segs_xml = "\n".join(segs)

    he = cfg["table"]["half_extent_m"]; th_t = cfg["table"]["thickness_m"]
    table_cz = table_top_z - th_t / 2.0
    cx, cy, _ = [float(x) for x in ring_center]
    cz = table_top_z + h / 2.0                              # 环心：平放静置于桌面

    snippet = f"""
        <!-- ===== Stage 4.0 object-aware (tape annulus, B+) auto-generated ===== -->
        <body name="table" pos="0 0 {table_cz:.5f}">
            <geom name="table_geom" type="box" size="{he:g} {he:g} {th_t/2:g}"
                  rgba="0.55 0.4 0.25 1" {phys}/>
        </body>
        <body name="tape" pos="{cx:.5f} {cy:.5f} {cz:.5f}">
            <freejoint name="tape_free"/>
            <inertial pos="0 0 0" mass="{mass:g}" diaginertia="{ixy:.6g} {ixy:.6g} {iz:.6g}"/>
{segs_xml}
        </body>
"""
    if "</worldbody>" not in base:
        raise RuntimeError("base xml 缺少 </worldbody>")
    return base.replace("</worldbody>", snippet + "    </worldbody>")


def write_object_model(cfg: dict, ring_center, table_top_z: float, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(build_object_xml(cfg, ring_center, table_top_z))
    return out_path


def derive_placement(base_xml: Path, demo: dict, cfg: dict):
    """环心 = 指垫汇聚点 + mean_radius·(夹爪水平前向)；table_top = pinch_z − H/2。"""
    from sim.replay_demo_validation import state_mdeg_to_rad
    o = cfg["object"]
    H = o["height_mm"] / 1000.0
    r_mean = (o["d_outer_mm"] + o["d_inner_mm"]) / 4000.0

    gf = detect_grasp_frames(demo)
    gframe = gf[0] if gf else len(demo["action"]) // 2
    qpos8 = state_mdeg_to_rad(demo["state_mdeg"][gframe])
    pinch, fwd = fk_grasp(base_xml, qpos8)

    fwd_h = np.array([fwd[0], fwd[1], 0.0])                 # 水平前向
    fwd_h = fwd_h / (np.linalg.norm(fwd_h) + 1e-9)
    ring_xy = pinch[:2] + r_mean * fwd_h[:2]
    if cfg["table"].get("override_top_z") is not None:
        table_top_z = float(cfg["table"]["override_top_z"])
    else:
        table_top_z = float(pinch[2] - H / 2.0)
    ring_center = np.array([ring_xy[0], ring_xy[1], table_top_z + H / 2.0])
    return ring_center, table_top_z


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SIM_DIR.parent))
    from sim.replay_demo_validation import load_real_demo
    cfg = load_config()
    demo = load_real_demo(0)
    ring_center, table_top_z = derive_placement(BASE_XML, demo, cfg)
    print(f"[derive] ring_center={ring_center}  table_top_z={table_top_z:.4f}  "
          f"mass={object_mass_kg(cfg)*1000:.0f}g  N={cfg['modeling']['n_segments']}")
    out = SIM_DIR / "_obj_smoke.xml"
    write_object_model(cfg, ring_center, table_top_z, out)
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"[load ok] nq={m.nq} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    out.unlink()
    print("[smoke ok] tape annulus model builds + loads")
