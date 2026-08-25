#!/usr/bin/env python3
"""Step 2 天花板测量 —— 真机自身的可重复性，按频段分解。

任何**确定性**模型 G 的保真度都不可能超过真机自身的可重复性：gap 里不重复的
部分谁也预测不了。本脚本用 `probe_b2` 的**六次裸复放**（同一段指令，跨三轮，
2026-07-28 18:44–20:11）在 **100 Hz 原生采样**上测这个上限。

为什么必须用 100 Hz 而不是落盘的 20 Hz：
    TSND151 以 100 Hz 流式输出（`tsnd_interval_ms=10`），而控制回路每 20 Hz
    调一次 `get_latest()` ⇒ **5 倍抽取、无抗混叠**。20 Hz 的 Nyquist 是 10 Hz，
    真机 11–29 Hz 的内容会折叠进 0.5–9 Hz 评价带。实测 >10 Hz 占陀螺能量
    **5.47%** —— 与 20 Hz 上测到的约 7% 不可复现量吻合，故混叠很可能就是
    那个「噪声」的主要来源。`replay_probe.py` 早已把 100 Hz 全样本另存为
    `imu100`（与控制侧环形缓冲分离，见 tsnd151_reader 2026-07-27 的修复）。

用法: python sim/step2_ceiling.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "outputs/probe_b2"
FS = 100.0
REPLAYS = {"A1": ("replay_ep0_A1.json", 1), "A2": ("replay_ep0_A2.json", 1),
           "A3": ("replay_ep0_r2_A3.json", 2), "A4": ("replay_ep0_r2_A4.json", 2),
           "A5": ("replay_ep0_r3_A5.json", 3), "A6": ("replay_ep0_r3_A6.json", 3)}
BANDS = ((0.5, 1), (1, 2), (2, 3.5), (3.5, 6), (6, 9), (9, 12),
         (12, 18), (18, 25), (25, 35), (35, 45))


def bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def main() -> int:
    D = {k: json.loads((P / f).read_text()) for k, (f, _) in REPLAYS.items()}
    cmds = {np.round(np.asarray(d["cmd"], float), 6).tobytes() for d in D.values()}
    assert len(cmds) == 1, "六次复放的指令序列不同 —— 不能当重复性测量"
    G = {k: np.asarray(d["imu100"], float)[:, 5:8] for k, d in D.items()}
    n = min(len(g) for g in G.values())
    G = {k: g[:n] for k, g in G.items()}
    ms = np.diff(np.asarray(D["A1"]["imu100"], float)[:, 1])
    print(f"六次裸复放 · 指令逐字节相同 · {n} 样本 @ {FS:.0f} Hz · 丢样 {(ms > 10).sum()}")

    full = np.concatenate(list(G.values()))
    tot = float(np.square(full - full.mean(0)).sum())
    print(f"\n{'频段(Hz)':<14}{'占能量':>8}{'重复性':>9}{'噪声':>8}   判读")
    for lo, hi in BANDS:
        B = {k: bp(g, lo, hi) for k, g in G.items()}
        e = float(np.mean([np.square(b).sum() for b in B.values()]))
        rs = [1 - float(np.square(B[x] - B[y]).sum()) / float(np.square(B[y]).sum())
              for x, y in itertools.combinations(G, 2)]
        r = float(np.mean(rs))
        tag = "**确定性**" if r > 0.85 else ("过渡" if r > 0.5 else "基本是噪声")
        print(f"{lo:>5.1f}–{hi:<7.1f}{e/tot*100:>7.2f}%{r*100:>8.1f}%{(1-r)*100:>7.1f}%   {tag}")

    print(f"\n{'带':<14}{'轮内':>9}{'跨轮':>9}{'差':>8}")
    for lo, hi in ((0.5, 9.0), (0.5, 40.0)):
        B = {k: bp(g, lo, hi) for k, g in G.items()}
        wi, ac = [], []
        for x, y in itertools.combinations(G, 2):
            r = 1 - float(np.square(B[x] - B[y]).sum()) / float(np.square(B[y]).sum())
            (wi if REPLAYS[x][1] == REPLAYS[y][1] else ac).append(r)
        print(f"{f'{lo}–{hi} Hz':<14}{np.mean(wi)*100:>+8.1f}%{np.mean(ac)*100:>+8.1f}%"
              f"{(np.mean(wi)-np.mean(ac))*100:>+7.1f}pp")
    print("  ⇒ 跨轮不比轮内差 ⇒ 这 1.5 小时内**测不到会话漂移**；变异由单次复放噪声主导。")
    print("     (六次同属 2026-07-28 一个晚上 ⇒ 只覆盖「~1.5h 内」，不覆盖跨天。)")

    f = np.fft.rfftfreq(n, d=1 / FS)
    Pw = np.abs(np.fft.rfft(G["A1"] - G["A1"].mean(0), axis=0)) ** 2
    print(f"\n混叠量: >10 Hz 占陀螺能量 {Pw[f >= 10].sum()/Pw.sum()*100:.2f}%"
          f" —— 20 Hz 无抗混叠抽取时，这部分折叠进 0.5–9 Hz 评价带")
    print("\n结论:")
    print("  · 确定性内容在 **9–12 Hz** 截止；冻结的 0.5–9 Hz 评价带恰好卡对位置")
    print("  · 加宽带宽**不会**帮助辨识，只会引入噪声")
    print("  · 100 Hz 落盘的价值在**抗混叠**，不在扩带宽")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
