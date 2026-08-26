#!/usr/bin/env python3
"""pilot 重复对的可重复性 —— j4 的噪声底 = 任何模型在 j4 上的保真度上限。

每一级 pilot 都跑两次同一序列（hash 相同）。两次之差是**不可复现**的部分，
谁也预测不了；标定做得再好也超不过它。故本脚本给出的是**上限**，不是成绩。

口径与 `step2_ceiling.py` 一致（100 Hz 原生 IMU、逐频段），以便与全臂天花板
（0.5–9 Hz 约 93–97%）直接比较。

用法: python sim/step2_g/pilot_repeat.py P01_... P02_...
      python sim/step2_g/pilot_repeat.py            # 自动配对目录里所有重复对
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/step2_g/pilot_j4"
FS = 100.0
BANDS = ((0.5, 1), (1, 2), (2, 3.5), (3.5, 6), (6, 9), (9, 12), (12, 18), (18, 25))
J = 3


def bp(x, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def load(name: str) -> dict:
    p = OUT / (name if name.startswith("replay_") else f"replay_{name}.json")
    if not p.exists():
        raise SystemExit(f"没有 {p.name}")
    return json.loads(p.read_text())


def compare(a: str, b: str) -> None:
    A, B = load(a), load(b)
    ca = np.round(np.asarray(A["cmd"], float), 9)
    cb = np.round(np.asarray(B["cmd"], float), 9)
    n = min(len(ca), len(cb))
    same = bool(np.array_equal(ca[:n], cb[:n]))
    ga = np.asarray(A["imu100"], float)[:, 5:8]
    gb = np.asarray(B["imu100"], float)[:, 5:8]
    m = min(len(ga), len(gb))
    ga, gb = ga[:m], gb[:m]
    sa = np.asarray(A["state"], float)[:n, J] / 1000.0
    sb = np.asarray(B["state"], float)[:n, J] / 1000.0

    print(f"\n{a}  vs  {b}")
    print(f"  指令逐字节相同? {'✅' if same else '❌ —— 不是重复对, 不能当噪声底'}")
    print(f"  100Hz 样本 {len(ga)} / {len(gb)}   编码器差 RMS "
          f"{np.sqrt(((sa - sb) ** 2).mean()):.4f}°  峰 {np.abs(sa - sb).max():.4f}°")
    print(f"  {'频段(Hz)':<12}{'占能量':>8}{'可重复性':>10}")
    for lo, hi in BANDS:
        Ba, Bb = bp(ga, lo, hi), bp(gb, lo, hi)
        e = float(np.square(Bb).sum())
        if e <= 0:
            continue
        r = 1 - float(np.square(Ba - Bb).sum()) / e
        tot = float(np.square(gb - gb.mean(0)).sum())
        print(f"  {f'{lo:g}–{hi:g}':<12}{e/tot*100:>7.2f}%{r*100:>9.1f}%")
    Ba, Bb = bp(ga, 0.5, 9.0), bp(gb, 0.5, 9.0)
    r = 1 - float(np.square(Ba - Bb).sum()) / float(np.square(Bb).sum())
    print(f"  **评价带 0.5–9 Hz 可重复性 = {r*100:.1f}%**"
          f"   （全臂天花板 93–97%，见 step2_ceiling.py）")


def main() -> int:
    if len(sys.argv) >= 3:
        compare(sys.argv[1], sys.argv[2])
        return 0
    files = sorted(p.stem[len("replay_"):] for p in OUT.glob("replay_P*.json"))
    seen: dict[str, list[str]] = {}
    for f in files:
        seen.setdefault(re.sub(r"_r\d+$", "", f[4:]), []).append(f)
    pairs = [(v[0], v[1]) for v in seen.values() if len(v) >= 2]
    if not pairs:
        print(f"目录里还没有成对的重复集（现有 {len(files)} 集）")
        return 0
    for a, b in pairs:
        compare(a, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
