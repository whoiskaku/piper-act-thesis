#!/usr/bin/env python3
"""读 tsnd151_reader 存的 npz，打印合格判据需要的三组数字。"""

import argparse
from pathlib import Path

import numpy as np


def inspect(path: Path):
    d = np.load(path)
    host_t = d["host_t"]
    device_ms = d["device_ms"].astype(np.float64)
    accel = d["accel"]
    gyro = d["gyro"]
    interval_ms = int(d["interval_ms"]) if "interval_ms" in d.files else -1

    n = len(device_ms)
    if n > 1:
        dt_ms = np.diff(device_ms)
        dt_ms = dt_ms[dt_ms > 0]
        hz_dev = 1000.0 / dt_ms.mean() if len(dt_ms) else float("nan")
        hz_host = (n - 1) / (host_t[-1] - host_t[0]) if host_t[-1] > host_t[0] else float("nan")
    else:
        hz_dev = hz_host = float("nan")

    print(f"\n=== {path.name} ===")
    print(f"samples = {n}   device interval set = {interval_ms} ms")
    print(f"actual rate: device={hz_dev:.2f} Hz   host={hz_host:.2f} Hz")
    print(f"accel (m/s^2):")
    print(f"  x: mean={accel[:,0].mean():+.3f}  std={accel[:,0].std():.3f}  "
          f"min={accel[:,0].min():+.3f}  max={accel[:,0].max():+.3f}")
    print(f"  y: mean={accel[:,1].mean():+.3f}  std={accel[:,1].std():.3f}  "
          f"min={accel[:,1].min():+.3f}  max={accel[:,1].max():+.3f}")
    print(f"  z: mean={accel[:,2].mean():+.3f}  std={accel[:,2].std():.3f}  "
          f"min={accel[:,2].min():+.3f}  max={accel[:,2].max():+.3f}")
    g_norm = np.linalg.norm(accel, axis=1)
    print(f"  |accel|: mean={g_norm.mean():.3f}  std={g_norm.std():.3f}  "
          f"(静置时应 ≈ 9.81)")
    print(f"gyro (rad/s):")
    print(f"  x: mean={gyro[:,0].mean():+.4f}  std={gyro[:,0].std():.4f}  "
          f"min={gyro[:,0].min():+.3f}  max={gyro[:,0].max():+.3f}")
    print(f"  y: mean={gyro[:,1].mean():+.4f}  std={gyro[:,1].std():.4f}  "
          f"min={gyro[:,1].min():+.3f}  max={gyro[:,1].max():+.3f}")
    print(f"  z: mean={gyro[:,2].mean():+.4f}  std={gyro[:,2].std():.4f}  "
          f"min={gyro[:,2].min():+.3f}  max={gyro[:,2].max():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="npz files to inspect")
    args = ap.parse_args()
    for p in args.paths:
        inspect(Path(p).expanduser().resolve())


if __name__ == "__main__":
    main()
