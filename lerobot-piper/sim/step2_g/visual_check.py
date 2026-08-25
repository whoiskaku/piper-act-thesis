#!/usr/bin/env python3
"""激励序列的批量可视回放 —— 上机前的人工签字材料。

           ██  COLLISION DISABLED — VISUAL ONLY  ██

`piper_model.xml` **关闭了全部碰撞**（`contype/conaffinity` 非零 0/17），
且执行器**没有力矩上限**。因此本工具的回放：

  ✓ 能发现：明显穿模、方向搞反、指令跳变、危险包络（伸到桌面以下/身后等）
  ✗ **不能证明碰撞安全**，也不能证明力矩/热在真机上可行

每条序列渲染**完整的上机时间线**（预移动 → 稳定 → 激励 → 回停），而不只是激励段
—— 预移动和回停同样会扫过空间，同样需要人看。

渲染的是**动力学仿真**的位形（不是指令），因为真机会过冲；当前参数下仿真过冲
约 1.5–1.9×，对安全审查而言偏保守，可用。指令与仿真的极值并列报告。

用法:
    python sim/step2_g/visual_check.py --max 6          # 先看几条
    python sim/step2_g/visual_check.py --only S_j4      # 只看 j4 的阶跃
    python sim/step2_g/visual_check.py                  # 全部 108 条
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.step2_g import params as PA                                   # noqa: E402
from sim.step2_g.build_suite import MODEL, build                       # noqa: E402
from sim.step2_g.sequences import (                                    # noqa: E402
    FPS, HOME_DEG, TABLE_Z, Z_CLEAR, lowest_geom_z,
)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "outputs/step2_g/visual"
BANNER = "COLLISION DISABLED — VISUAL ONLY"
PREMOVE_S, SETTLE_S, RETURN_S = 2.0, 1.0, 2.0
VIEWS = (("正面", 0.0, -20.0), ("侧面", 90.0, -20.0), ("俯视", 0.0, -85.0))
N_SHOT = 6


def timeline(cmd: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """完整上机时间线：HOME →(预移动)→ 起点 →(稳定)→ 激励 →(回停)→ HOME。"""
    home = np.zeros(7)
    home[:6] = np.deg2rad(HOME_DEG)
    npre, nset, nret = (int(x * FPS) for x in (PREMOVE_S, SETTLE_S, RETURN_S))
    ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, npre)))[:, None]
    pre = home[None, :] + (cmd[0] - home)[None, :] * ramp
    setl = np.repeat(cmd[0][None, :], nset, axis=0)
    ramp2 = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, nret)))[:, None]
    ret = cmd[-1][None, :] + (home - cmd[-1])[None, :] * ramp2
    tail = np.repeat(home[None, :], int(0.5 * FPS), axis=0)
    full = np.concatenate([pre, setl, cmd, ret, tail])
    k = 0
    spans = []
    for nm, n in (("预移动", npre), ("稳定", nset), ("激励", len(cmd)),
                  ("回停", nret), ("停放", len(tail))):
        spans.append((k, k + n, nm))
        k += n
    return full, spans


def roll(m, full: np.ndarray) -> np.ndarray:
    """按时间线跑动力学，返回逐帧 qpos[:6]（真机会过冲，故看仿真而非指令）。"""
    d = mujoco.MjData(m)
    d.qpos[:6] = full[0, :6]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = full[0, :6]
    sub = int(round(1 / FPS / m.opt.timestep))
    q = np.zeros((len(full), 6))
    for t in range(len(full)):
        d.ctrl[:6] = full[t, :6]
        for _ in range(sub):
            mujoco.mj_step(m, d)
        q[t] = d.qpos[:6]
    return q


def lowest_z(m, q: np.ndarray) -> float:
    """手臂 geom 的保守最低点（**几何邻近**参考，不是碰撞判定）。

    与 `sequences.lowest_geom_z` 用同一判据（包围球下界、排除 world 上的装饰件），
    以免生成端与审查端两套标准。
    """
    d = mujoco.MjData(m)
    return float(min(lowest_geom_z(m, d, qq) for qq in q[::5]))


def sheet(m, s, q: np.ndarray, full: np.ndarray, spans, path: Path) -> dict:
    r = mujoco.Renderer(m, 300, 400)
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.15]
    cam.distance = 1.1
    idx = np.linspace(0, len(q) - 1, N_SHOT).astype(int)
    # 末行放角度轨迹 —— 小幅激励(如 j4 的 ±4°)在图像上肉眼看不出, 审查者需要它
    fig = plt.figure(figsize=(N_SHOT * 2.3, len(VIEWS) * 1.9 + 2.0))
    gs = fig.add_gridspec(len(VIEWS) + 1, N_SHOT,
                          height_ratios=[1] * len(VIEWS) + [0.85], hspace=0.28)
    ax = np.array([[fig.add_subplot(gs[i, k]) for k in range(N_SHOT)]
                   for i in range(len(VIEWS))])
    for vi, (vn, az, el) in enumerate(VIEWS):
        cam.azimuth, cam.elevation = az, el
        for ci, t in enumerate(idx):
            d.qpos[:6] = q[t]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            a = ax[vi, ci]
            a.imshow(r.render())
            a.set_xticks([]); a.set_yticks([])
            if ci == 0:
                a.set_ylabel(vn, fontsize=10)
            if vi == 0:
                lbl = next(nm for k0, k1, nm in spans if k0 <= t < k1)
                a.set_title(f"{t/FPS:.1f}s · {lbl}", fontsize=8.5)
    j = s.target[0]
    tr = fig.add_subplot(gs[len(VIEWS), :])
    tt = np.arange(len(q)) / FPS
    for k0, k1, nm in spans:
        if nm in ("预移动", "回停"):
            tr.axvspan(k0 / FPS, k1 / FPS, color="#f2efe6", zorder=0)
        elif nm == "激励":
            tr.axvspan(k0 / FPS, k1 / FPS, color="#e8f0f8", zorder=0)
            tr.annotate("激励", ((k0 + k1) / 2 / FPS, 0.96), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8.5, color="#2a78d6")
    tr.plot(tt, np.rad2deg(full[:, j]), color="#eda100", lw=2.0, label="指令")
    tr.plot(tt, np.rad2deg(q[:, j]), color="#2a78d6", lw=1.4, label="仿真")
    for t_ in idx:
        tr.axvline(t_ / FPS, color="0.6", ls=":", lw=0.8)
    tr.set_xlim(0, tt[-1]); tr.grid(alpha=0.3)
    tr.set_xlabel("时间 (s)", fontsize=9)
    tr.set_ylabel(f"j{j+1} (度)", fontsize=9)
    tr.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    tr.tick_params(labelsize=8)

    mdl_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:12]
    fig.suptitle(
        f"{BANNER}\n{s.name}   序列 {s.sha}   模型 {mdl_sha}   "
        f"目标 j{s.target[0]+1}   时长 {len(q)/FPS:.1f}s",
        fontsize=11, color="#c0392b", fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.045, right=0.99)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=95)
    plt.close(fig)
    r.close()
    return {"model_sha": mdl_sha}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只处理名字含该串的序列")
    ap.add_argument("--max", type=int)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    assert not ((m.geom_contype != 0) & (m.geom_conaffinity != 0)).any(), \
        "模型碰撞已开启 —— 本工具的免责声明与实际不符, 请先确认意图"
    PA.apply(m, PA.baseline(m))
    seqs = build(m)
    if a.only:
        seqs = [s for s in seqs if a.only in s.name]
    if a.max:
        seqs = seqs[: a.max]

    print("█" * 62)
    print(f"  {BANNER}")
    print("  碰撞在模型中**全部关闭**，执行器**无力矩上限**。")
    print("  本回放可发现: 穿模 / 方向反 / 指令跳变 / 危险包络")
    print("  本回放**不能**证明: 碰撞安全、力矩可行、热可行")
    print("█" * 62)
    print(f"\n{len(seqs)} 条序列 → {OUT.relative_to(ROOT)}/\n")
    print(f"{'序列':<22}{'指令范围(度)':>16}{'仿真范围(度)':>16}"
          f"{'过冲':>7}{'最低z(m)':>10}{'序列hash':>13}")
    worst = []
    for s in seqs:
        full, spans = timeline(s.cmd)
        q = roll(m, full)
        j = s.target[0]
        # 过冲只在**激励段**上算 —— 含预移动的全程范围会把它稀释掉(实测全成 1.00×)
        k0, k1 = next((a_, b_) for a_, b_, nm in spans if nm == "激励")
        cr = (np.rad2deg(full[k0:k1, j].min()), np.rad2deg(full[k0:k1, j].max()))
        sr = (np.rad2deg(q[k0:k1, j].min()), np.rad2deg(q[k0:k1, j].max()))
        ov = (sr[1] - sr[0]) / max(cr[1] - cr[0], 1e-9)
        lz = lowest_z(m, q)
        sheet(m, s, q, full, spans, OUT / f"{s.name}.png")
        flag = "  ⚠余量不足" if lz < TABLE_Z + Z_CLEAR else ""
        if lz < TABLE_Z + Z_CLEAR:
            worst.append(s.name)
        print(f"{s.name:<22}{f'{cr[0]:7.1f}…{cr[1]:6.1f}':>16}"
              f"{f'{sr[0]:7.1f}…{sr[1]:6.1f}':>16}{ov:>6.2f}×{lz:>10.3f}{s.sha:>13}{flag}")
    print(f"\n几何邻近（**非碰撞判定**，包围球下界，排除装饰件）: "
          f"要求 z ≥ {TABLE_Z + Z_CLEAR:.3f} m；"
          f"{'余量不足的序列: ' + str(worst) if worst else '全部序列满足'}")
    print(f"\n人工签字前请逐张核对 {OUT.relative_to(ROOT)}/*.png —— "
          f"每张都带 {BANNER} 与序列/模型 hash。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
