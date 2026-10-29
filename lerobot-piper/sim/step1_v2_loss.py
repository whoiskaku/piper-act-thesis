#!/usr/bin/env python3
"""Step 1 v2 的损失 —— **只用实测 IMU**, 不涉及 G / H / 任何响应模型。

    L = mean_t ‖ bandpass(real − ideal)(t) ‖²      陀螺三轴, 0.5–9 Hz, 41 抽头 FIR
    gap(t) = real[t + 3] − ideal_cal[t]            L=3 帧对齐(与 H1 主指标同口径)

`ideal` 是**指令的运动学渲染**(MuJoCo 正运动学 + 虚拟 IMU), 即指導教員白板上
「SimulationとIMU値の差」里的 Simulation 那一侧 —— 它是**目标的定义**, 不是被控对象
模型, 与 v1 的 H 性质完全不同。

用法:
    python sim/step1_v2_loss.py outputs/spsa_v2/R03p_ep0.json
    python sim/step1_v2_loss.py outputs/spsa_v2/*.json --csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FPS = 20.0
ALIGN_L = 3
BAND = (0.5, 9.0)
NTAPS = 41
MIN_FRAMES = 550


def bandpass41(x: np.ndarray) -> np.ndarray:
    """41 抽头居中 FIR。**离线专用**(需要 20 帧未来), 与 H1 主指标同一实现。"""
    from scipy.signal import firwin
    k = firwin(NTAPS, list(BAND), fs=FPS, pass_zero=False)
    x = np.asarray(x, dtype=np.float64)
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], axis=1)


def render_ideal(chunks: list[dict], n_frames: int) -> np.ndarray:
    """逐 chunk 运动学渲染 ideal IMU (sim 系)。与训练/部署同一渲染器。"""
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    if not chunks:
        raise ValueError("日志缺 chunks[].raw —— 无法渲染 ideal")
    chunks = sorted(chunks, key=lambda c: int(c["start"]))
    rend = IdealChunkRenderer()
    out, prev = [], None
    for c in chunks:
        cmd = np.asarray(c["raw"], dtype=np.float32)
        out.append(rend.render(cmd, prev))
        prev = np.concatenate([np.asarray(x["raw"], np.float32) for x in
                               chunks[:chunks.index(c) + 1]])
    ideal = np.concatenate(out)
    return ideal[:n_frames]


def episode_loss(path: str | Path, R: np.ndarray, b: np.ndarray,
                 min_frames: int = MIN_FRAMES) -> dict:
    """单集的 L 与配套诊断。**只读实测 IMU + 已记录指令。**"""
    payload = json.loads(Path(path).read_text())
    frames = payload.get("frames", [])
    summary = payload.get("summary", {})
    if len(frames) < min_frames:
        raise ValueError(f"{path}: 帧数 {len(frames)} < {min_frames}")
    if any(f.get("imu") is None for f in frames):
        raise ValueError(f"{path}: 有帧缺 IMU")
    real = np.asarray([f["imu"] for f in frames], dtype=np.float64)
    if not np.isfinite(real).all():
        raise ValueError(f"{path}: IMU 含非有限值")

    ideal_sim = render_ideal(payload.get("chunks", []), len(frames))
    n = min(len(frames) - ALIGN_L, len(ideal_sim))
    if n <= NTAPS:
        raise ValueError(f"{path}: 对齐后过短 ({n} 帧)")

    # 标定到真机 IMU 系: 陀螺只用旋转 R(不加偏置 b) —— 与 build_dagger_labels 同口径
    ideal_gyro = ideal_sim[:n, 3:6] @ np.asarray(R, np.float64).T
    gap = real[ALIGN_L:ALIGN_L + n, 3:6] - ideal_gyro
    bp = bandpass41(gap)
    L = float(np.square(bp).mean(axis=1).mean() * 3.0)   # mean_t ‖·‖²

    return {"path": str(path), "L": L, "log_L": float(np.log(L)),
            "frames": len(frames), "n_aligned": int(n),
            "sat_pct": float(summary.get("sat_pct", 0.0) or 0.0),
            "stale": int(summary.get("stale", 0) or 0),
            "nan": int(summary.get("nan", 0) or 0),
            "theta": summary.get("theta"), "mode": summary.get("mode")}


def load_calib(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    m = json.loads(Path(path).read_text())
    return (np.asarray(m["R"], np.float64),
            np.asarray(m.get("b", [0, 0, 0]), np.float64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--calib", default=str(ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"))
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    a = ap.parse_args()
    R, b = load_calib(a.calib)
    rows = []
    for p in a.logs:
        try:
            rows.append(episode_loss(p, R, b, a.min_frames))
        except Exception as exc:                                    # noqa: BLE001
            print(f"  ✗ {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not rows:
        return 1
    if a.csv:
        print("path,L,log_L,frames,sat_pct,stale,nan")
        for r in rows:
            print(f"{r['path']},{r['L']:.8f},{r['log_L']:.6f},{r['frames']},"
                  f"{r['sat_pct']:.3f},{r['stale']},{r['nan']}")
    else:
        print(f"  {'集':<44}{'L':>12}{'log L':>10}{'帧':>7}{'sat%':>7}")
        for r in rows:
            print(f"  {Path(r['path']).name:<44}{r['L']:12.6f}{r['log_L']:10.4f}"
                  f"{r['frames']:7d}{r['sat_pct']:7.2f}")
        Ls = np.array([r["L"] for r in rows])
        if len(Ls) > 1:
            print(f"\n  n={len(Ls)}  中位 {np.median(Ls):.6f}  均值 {Ls.mean():.6f}  "
                  f"SD {Ls.std(ddof=1):.6f}  CV {Ls.std(ddof=1)/Ls.mean()*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
