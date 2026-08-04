#!/usr/bin/env python3
"""反馈式补正的**因果天花板**（2026-08-04）—— 与任何被控对象模型无关。

**问题**：决定第 i 帧的修正时，必须先发指令、机器人才响应，所以永远测不到自己造成的
后果。要抵消的误差 `e(i) = real[i+L] − ideal[i]` 只有在墙钟 i+L 才可观测。
⇒ **任何反馈方案能消掉的，最多就是 e(i) 中可由因果信息预测的那一部分。**

本脚本测的就是那个比例。**不涉及 S、不涉及 Ĝ、不涉及任何驱动模型** ——
只问"误差本身有多可预见"。

判据用**非中心化 R²**（参照基准是"预测 0"= 什么都不知道）：

    R² = 1 − ‖ê − e‖² / ‖e‖²

它的含义：若被控对象模型完美、执行也完美，施加修正后残差 = (1−R²)·原能量，
⇒ **gap 的降幅上限 = R²**。

**特征消融**（同一网络、同一划分，只换输入）回答一个附加问题：
可预测性主要来自**反馈**（误差历史）还是**前馈**（已知的未来指令）？

用法: python sim/causal_ceiling.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.policies.act.b_runtime import build_offline

ROOT = Path(__file__).resolve().parents[1]
LROOT = ROOT / "outputs/p2_labels"
TAGS = ["pp2a-deploy", "insa-deploy"]          # 裸 ACT，同会话
TRAIN_EPS, VAL_EPS = list(range(8)), [8, 9]
LSHIFT = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEG, BATCH, STEPS = 96, 32, 1500

# 51 维特征的分组（见 row_features 的布局）
GROUPS = {
    "① 仅误差历史 e(i−L)": list(range(0, 6)),
    "② 仅 ideal 前瞻(t,t+4,t+8)": list(range(6, 24)),
    "③ 仅指令 a(t,t+4,t+8)": list(range(24, 42)),
    "④ 仅编码器速度 q̇": list(range(42, 48)),
    "⑤ ②+③ 指令侧全部(前馈)": list(range(6, 42)),
    "⑥ ①+④ 测量侧全部(反馈)": list(range(0, 6)) + list(range(42, 48)),
    "⑦ 全部 51 维": list(range(0, 51)),
}


def bp_eval():
    from analyze_ab import _bp
    return _bp()


def load(eps_list, bp):
    out = []
    for tag in TAGS:
        for ei in eps_list:
            z = np.load(LROOT / tag / f"ep{ei:03d}.npz")
            X, _, _ = build_offline(z, L=LSHIFT)
            n = min(len(X), len(z["e"]))
            Y = bp(z["e"][:n, 3:6].astype(np.float32))       # 带通陀螺误差 = 要消的东西
            out.append({"X": torch.from_numpy(X[:n]).to(DEV),
                        "Y": torch.from_numpy(Y.astype(np.float32)).to(DEV),
                        "motion": torch.from_numpy(~z["mask_still"][:n]).to(DEV)})
    return out


class Net(torch.nn.Module):
    """与 π 同骨干（TCN，因果空洞卷积），只是输出 3 维（陀螺三轴）且不限幅。"""

    def __init__(self, d, h=32):
        super().__init__()
        self.c = torch.nn.ModuleList([
            torch.nn.Conv1d(d, h, 3, dilation=1), torch.nn.Conv1d(h, h, 3, dilation=2),
            torch.nn.Conv1d(h, h, 3, dilation=4)])
        self.p = [2, 4, 8]
        self.o = torch.nn.Linear(h, 3)

    def forward(self, u):
        x = u.transpose(1, 2)
        for c, p in zip(self.c, self.p):
            x = torch.nn.functional.gelu(c(torch.nn.functional.pad(x, (p, 0))))
        return self.o(x.transpose(1, 2))


def r2(P, L):
    """**非中心化** R²：参照基准是预测 0（= 无任何知识）。"""
    return float(1 - ((P - L) ** 2).sum() / (L ** 2).sum())


def run(cols, tr, va, seed=0, steps=STEPS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = torch.tensor(cols, device=DEV)
    m = Net(len(cols)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = -1e9

    def ev():
        m.eval(); P, L = [], []
        with torch.no_grad():
            for e in va:
                p = m(e["X"][:, idx].unsqueeze(0))[0]
                k = e["motion"]
                P.append(p[k].cpu().numpy()); L.append(e["Y"][k].cpu().numpy())
        return r2(np.concatenate(P), np.concatenate(L))

    for i in range(steps):
        m.train()
        bi = rng.integers(0, len(tr), BATCH)
        X, Y = [], []
        for j in bi:
            e = tr[j]; T = len(e["X"])
            o = int(rng.integers(0, max(T - SEG, 1)))
            X.append(e["X"][o:o + SEG, idx]); Y.append(e["Y"][o:o + SEG])
        X, Y = torch.stack(X), torch.stack(Y)
        loss = ((m(X) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (i + 1) % 50 == 0:
            best = max(best, ev())
    return best


def main():
    bp = bp_eval()
    tr, va = load(TRAIN_EPS, bp), load(VAL_EPS, bp)
    ntr = sum(len(e["X"]) for e in tr); nva = sum(len(e["X"]) for e in va)
    print(f"device={DEV}  train {ntr} / val {nva} 帧   标签源 {TAGS}")
    print("目标 = 带通(0.5–9 Hz)陀螺误差 e(i)，即补正要消掉的东西")
    print("判据 = **非中心化 R²**（基准 = 预测 0）⇒ 直接等于 gap 降幅的上限\n")
    print(f"  {'输入特征':<28}{'维数':>5}{'留出 R²':>10}{'⇒ gap 降幅上限':>16}")
    res = {}
    t0 = time.time()
    for name, cols in GROUPS.items():
        v = run(cols, tr, va)
        res[name] = v
        print(f"  {name:<28}{len(cols):>5}{v:>10.3f}{max(v,0)*100:>15.1f}%")
    print(f"  ({time.time()-t0:.0f}s)")

    full = res["⑦ 全部 51 维"]
    ff = res["⑤ ②+③ 指令侧全部(前馈)"]
    fb = res["⑥ ①+④ 测量侧全部(反馈)"]
    print(f"\n  前馈信息(指令侧) 单独 {ff:.3f}   反馈信息(测量侧) 单独 {fb:.3f}   "
          f"全部 {full:.3f}")
    print(f"  反馈的**边际贡献** = 全部 − 前馈 = {full-ff:+.3f}")
    print("\n  判读：")
    print("   · 全部 51 维的 R² = 任何**因果**补正方案(不论 model-based 与否)的降幅上限")
    print("   · 若上限 < FF 已有的 33% ⇒ 反馈路线在本平台上structurally 打不过前馈")
    print("   · 若反馈的边际贡献 ≈ 0 ⇒ 可预测性全部来自已知的未来指令，IMU 反馈无增益")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
