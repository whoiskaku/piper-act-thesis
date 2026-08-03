#!/usr/bin/env python3
"""FF 的收益从哪来 —— 按 chunk 内位置分解 + 残余滞后 + 时间趋势（2026-08-04）。

起因：insert 三臂实测把预注册的 H3 **证伪**了。
  预测：blend 的边际贡献按接缝比例缩小 —— insert 接缝 2.2× vs pp2 5.9×
        ⇒ insert 的 blend 增量应 ≈ 10.5 pp × (2.24/5.89) ≈ 4.0 pp
  实测：insert 的 blend 增量 = 20.6 pp，是 pp2 的 **2 倍**，是预测的 5 倍。

所以「blend 只是削掉 chunk 开头那 10 帧的超出」这个模型是错的。本脚本查它错在哪：

  **判别法**：blend 只改 chunk 开头 `有效blend` 帧的指令。
    · 若收益只在开头 10 帧 ⇒ profile 在第 10 帧之后应与 A 重合
    · 若中段也降 ⇒ 接缝跳变激起的瞬态**响铃**到 10 帧之外，
      即"能被 blend 消掉的能量"远大于"接缝处测到的超出"

口径与 analyze_ab 一致：gap = bandpass(real−ideal) 陀螺三轴，0.5–9 Hz 41 抽头 FIR，
ideal 从 chunks[].raw 逐 chunk 渲染，未对齐口径。

用法:
  python sim/ff_decompose.py --root outputs/ab_ins --task ins --arms A,B3,B3b13
  python sim/ff_decompose.py --root outputs/ab2    --task pp2 --arms A,B3,B3b13
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
FPS, CHUNK, SMAX = 20.0, 100, 6
BANDS = [(0.5, 2), (2, 4), (4, 6), (6, 8), (8, 9)]


def load(path, R, rend):
    d = json.loads(Path(path).read_text())
    imu = np.array([f["imu"] for f in d["frames"] if f["imu"] is not None], np.float32)
    outs, prev = [], None
    for c in d["chunks"]:
        raw = np.array(c["raw"], np.float32)
        outs.append(rend.render(raw, prev)); prev = raw
    ideal = np.concatenate(outs)[:len(imu)][:, 3:6] @ R.T
    return ideal.astype(np.float32), imu[:len(ideal), 3:6].astype(np.float32)


def eps(root, arm, min_frames, n):
    fs = sorted(Path(root).glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return [f for f in fs
            if json.loads(f.read_text())["summary"]["frames"] > min_frames][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/ab_ins")
    ap.add_argument("--task", default="ins")
    ap.add_argument("--arms", default="A,B3,B3b13")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-frames", type=int, default=900)
    a_ = ap.parse_args()
    arms = [s.strip() for s in a_.arms.split(",")]

    from analyze_ab import _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from scipy.signal import welch
    bp = _bp()
    R = np.array(json.loads((ROOT / f"outputs/p2_labels/{a_.task}-deploy/manifest.json"
                             ).read_text())["R"], np.float32)
    rend = IdealChunkRenderer()

    D = {}
    for arm in arms:
        fs = eps(ROOT / a_.root, arm, a_.min_frames, a_.n)
        profs, curves, ener, psd = [], [], [], []
        for f in fs:
            ideal, real = load(f, R, rend)
            n = len(ideal) - SMAX
            # ① 移位扫描（残余滞后）
            curves.append([float((bp(real[s:s + n] - ideal[:n]) ** 2).sum() / n)
                           for s in range(SMAX + 1)])
            # ② chunk 内位置剖面（用 s=0，与主判据同口径）
            g = bp(real[:n] - ideal[:n])
            e = (g ** 2).sum(1)
            pos = np.arange(n) % CHUNK
            profs.append([e[pos == p].mean() for p in range(CHUNK)])
            ener.append(float(e.mean()))
            fq, P = welch(np.linalg.norm(g, axis=1), fs=FPS, nperseg=256)
            psd.append(P)
        C = np.array(curves); pr = np.array(profs)
        k = int(np.argmin(C.mean(0)))
        if k in (0, SMAX):
            sub = float("nan")          # 触搜索边界, 抛物线插值无意义(勿当测量值引用)
        else:
            aa, bb, cc = C.mean(0)[k - 1], C.mean(0)[k], C.mean(0)[k + 1]
            den = aa - 2 * bb + cc
            sub = k + 0.5 * (aa - cc) / den if abs(den) > 1e-12 else float(k)
        D[arm] = {"prof": pr.mean(0), "prof_ep": pr, "L": sub, "Lint": k,
                  "e": np.array(ener), "psd": np.mean(psd, 0), "fq": fq}

    A = D[arms[0]]
    print("=" * 78)
    print(f"① 残余滞后（H2）—— FF 臂应为 0    [{a_.task}]")
    print(f"  {'臂':>8}{'整数最优':>10}{'= ms':>8}{'亚帧 L*':>12}")
    for arm in arms:
        s = D[arm]["L"]
        sub = "触下界(=0)" if np.isnan(s) else f"{s:.2f}"
        print(f"  {arm:>8}{D[arm]['Lint']:>10d}{D[arm]['Lint']*50:>8.0f}{sub:>12}")

    print("\n" + "=" * 78)
    print("② 收益来自哪里：chunk 内位置分解（gap²/帧，相对 A 的降幅）")
    seg = {"开头0-9": slice(0, 10), "10-19": slice(10, 20), "20-29": slice(20, 30),
           "中段30-89": slice(30, 90), "末尾90-99": slice(90, 100)}
    print(f"  {'臂':>8}" + "".join(f"{k:>12}" for k in seg) + f"{'全程':>10}")
    for arm in arms:
        p = D[arm]["prof"]
        row = "".join(f"{p[s].mean():>12.5f}" for s in seg.values())
        print(f"  {arm:>8}{row}{p.mean():>10.5f}")
    print("  " + "-" * 74)
    for arm in arms[1:]:
        p, pa = D[arm]["prof"], A["prof"]
        row = "".join(f"{(p[s].mean()/pa[s].mean()-1)*100:>11.1f}%" for s in seg.values())
        print(f"  {arm+'/A':>8}{row}{(p.mean()/pa.mean()-1)*100:>9.1f}%")

    print("\n  【判读】blend 只改 chunk 开头若干帧的**指令**。")
    print("   · 若只有'开头0-9'降 ⇒ 收益 = 削掉接缝超出，与原模型一致")
    print("   · 若中段30-89 也明显降 ⇒ 接缝跳变激起的瞬态**响铃到 10 帧之外**，")
    print("     可消能量 ≫ 接缝处测到的超出（原模型低估了 blend 的作用范围）")

    print("\n" + "=" * 78)
    print("③ 时间趋势（臂内前半 vs 后半，各 5 集）—— 排除采集顺序效应")
    print(f"  {'臂':>8}{'前5集':>11}{'后5集':>11}{'后/前':>10}")
    for arm in arms:
        e = D[arm]["e"]
        print(f"  {arm:>8}{e[:5].mean():>11.5f}{e[5:].mean():>11.5f}"
              f"{e[5:].mean()/e[:5].mean()*100-100:>9.1f}%")

    print("\n" + "=" * 78)
    print("④ 逐频段 PSD 比（相对 A，<1 = 改善）")
    print(f"  {'臂':>8}" + "".join(f"{f'{lo}-{hi}Hz':>11}" for lo, hi in BANDS))
    fq = A["fq"]
    for arm in arms[1:]:
        row = ""
        for lo, hi in BANDS:
            m = (fq >= lo) & (fq < hi)
            row += f"{D[arm]['psd'][m].sum()/A['psd'][m].sum():>11.3f}"
        print(f"  {arm:>8}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
