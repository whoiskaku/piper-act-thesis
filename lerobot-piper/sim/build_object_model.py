#!/usr/bin/env python3
"""Stage 4.0 — 由 config + demo 抓取点反推，构建 object-aware MuJoCo 模型。

把 base piper_model.xml 注入一个静态 table 和一个带 freejoint 的 cube（参数全部
来自 object_sim_config.json）。cube 平面位置默认由 demo 抓取帧的 gripper FK 反推，
table 顶面 = grasp_z − cube_H/2，使 cube 静置于桌面且中心≈抓取点。

不做任何 friction 过调；占位参数仅用于跑通 transfer audit 管线。

提供：
  load_config()
  detect_grasp_frames(demo)            — gripper 动作跳变 → 抓取/释放帧
  gripper_world_pos(base_xml, qpos8)   — 对给定关节角做 FK，返回 ee_imu_site 世界 xyz
  build_object_xml(cfg, cube_xyz, table_top_z) -> xml 字符串
  write_object_model(cfg, cube_xyz, table_top_z, out_path)
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
BASE_XML = SIM_DIR / "piper_model.xml"
CONFIG = SIM_DIR / "object_sim_config.json"


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def box_inertia(mass: float, full_size_m) -> tuple[float, float, float]:
    """Diagonal inertia of a solid box. full_size_m = (W, D, H) in meters."""
    w, d, h = full_size_m
    ix = mass / 12.0 * (d * d + h * h)
    iy = mass / 12.0 * (w * w + h * h)
    iz = mass / 12.0 * (w * w + d * d)
    return ix, iy, iz


def detect_grasp_frames(demo: dict, grip_channel: int = 6) -> list[int]:
    """gripper 指令跳变 → 抓取/释放事件帧（聚类去抖）。"""
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


def gripper_world_pos(base_xml: Path, qpos8: np.ndarray,
                      site: str = "ee_imu_site") -> np.ndarray:
    """对 8-dim 关节角做 forward kinematics，返回 IMU site 的世界 xyz。"""
    model = mujoco.MjModel.from_xml_path(str(base_xml))
    data = mujoco.MjData(model)
    data.qpos[:len(qpos8)] = qpos8
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    return data.site_xpos[sid].copy()


def _fric_str(f):  return " ".join(f"{x:g}" for x in f)


def build_object_xml(cfg: dict, cube_xyz, table_top_z: float) -> str:
    """返回注入了 table+cube 的完整 MJCF 字符串。"""
    base = BASE_XML.read_text()

    size_m = [s / 1000.0 for s in cfg["cube"]["size_mm"]]      # W,D,H meters
    mass = cfg["cube"]["mass_g"] / 1000.0
    hx, hy, hz = [s / 2.0 for s in size_m]
    ix, iy, iz = box_inertia(mass, size_m)

    c = cfg["contact"]
    fr, solref, solimp, condim = (c["friction"], c["solref"],
                                  c["solimp"], c["condim"])
    geom_phys = (f'friction="{_fric_str(fr)}" solref="{_fric_str(solref)}" '
                 f'solimp="{_fric_str(solimp)}" condim="{condim}"')

    he = cfg["table"]["half_extent_m"]
    th = cfg["table"]["thickness_m"]
    table_cz = table_top_z - th / 2.0                          # table 几何中心 z
    cx, cy = float(cube_xyz[0]), float(cube_xyz[1])
    cz = table_top_z + hz                                      # cube 中心静置于桌面

    snippet = f"""
        <!-- ===== Stage 4.0 object-aware additions (auto-generated) ===== -->
        <body name="table" pos="0 0 {table_cz:.5f}">
            <geom name="table_geom" type="box" size="{he:g} {he:g} {th/2:g}"
                  rgba="0.55 0.4 0.25 1" {geom_phys}/>
        </body>
        <body name="cube" pos="{cx:.5f} {cy:.5f} {cz:.5f}">
            <freejoint name="cube_free"/>
            <inertial pos="0 0 0" mass="{mass:g}" diaginertia="{ix:.6g} {iy:.6g} {iz:.6g}"/>
            <geom name="cube_geom" type="box" size="{hx:g} {hy:g} {hz:g}"
                  rgba="0.85 0.2 0.2 1" {geom_phys}/>
        </body>
"""
    if "</worldbody>" not in base:
        raise RuntimeError("base xml 缺少 </worldbody>")
    return base.replace("</worldbody>", snippet + "    </worldbody>")


def write_object_model(cfg: dict, cube_xyz, table_top_z: float,
                       out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(build_object_xml(cfg, cube_xyz, table_top_z))
    return out_path


def derive_placement(base_xml: Path, demo: dict, cfg: dict):
    """由 demo 第一个抓取帧反推 cube_xyz 与 table_top_z（含 config override）。"""
    H = cfg["cube"]["size_mm"][2] / 1000.0
    pl = cfg["placement"]
    if not pl.get("derive_from_demo", True) and pl.get("override_cube_xyz"):
        cube_xyz = np.asarray(pl["override_cube_xyz"], dtype=float)
    else:
        from sim.replay_demo_validation import state_mdeg_to_rad
        gf = detect_grasp_frames(demo)
        gframe = gf[0] if gf else len(demo["action"]) // 2
        qpos8 = state_mdeg_to_rad(demo["state_mdeg"][gframe])
        cube_xyz = gripper_world_pos(base_xml, qpos8)
    if cfg["table"].get("override_top_z") is not None:
        table_top_z = float(cfg["table"]["override_top_z"])
    else:
        table_top_z = float(cube_xyz[2] - H / 2.0)
    return cube_xyz, table_top_z


if __name__ == "__main__":
    # smoke test: build with placeholder cfg + demo ep0 placement, load in mujoco.
    import sys
    sys.path.insert(0, str(SIM_DIR.parent))
    from sim.replay_demo_validation import load_real_demo
    cfg = load_config()
    demo = load_real_demo(0)
    cube_xyz, table_top_z = derive_placement(BASE_XML, demo, cfg)
    print(f"[derive] cube_xyz={cube_xyz}  table_top_z={table_top_z:.4f}")
    out = SIM_DIR / "_object_model_smoke.xml"
    write_object_model(cfg, cube_xyz, table_top_z, out)
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"[load ok] nq={m.nq} nu={m.nu} nbody={m.nbody} "
          f"(base nq=8 → +7 cube freejoint = {m.nq})")
    print(f"  cube geom id={mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')}, "
          f"table geom id={mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table_geom')}")
    out.unlink()
    print("[smoke ok] object model builds + loads")
