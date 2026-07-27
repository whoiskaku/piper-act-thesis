#!/usr/bin/env python3
"""汇报图 — Ĝ 输出波形三联(留出集): (A)场合一 实测vs合成 (B)柔性震铃对账
(C)场合二 单脉冲δa的响应(打一拳疼一阵)。输出 reports/figures/G_waveform.png"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_ideal_methods import load_episode
from fit_g_level2 import FlexL2, excitation, raw_feats
from fit_g_production import BP, OUT, bp_t, rigid_parts
from g_verdict import DATASETS, VAL_EPS, load

FIG = Path(__file__).resolve().parents[1] / "reports/figures"
FPS = 20.0

# ---- member-0 (全量数据) 参数重建 L2 模型 ----
z = np.load(OUT / "g_ensemble.npz")
c_enc = z["c_enc"][0]
mu, sd = z["mu"][0], z["sd"][0]
fx = FlexL2([3.1, 5.1, 5.7])
with torch.no_grad():
    for name in ["log_om", "raw_ze", "W", "B", "D", "theta"]:
        getattr(fx, name).copy_(torch.from_numpy(z[name][0]))

# ---- 留出集第一条 pp2 episode ----
val = load(VAL_EPS, with_gap=True)
i = 0
for repo, tag in DATASETS:
    for ei in VAL_EPS:
        d = load_episode(repo, ei)
        val[i]["a7"] = d["action"][:len(val[i]["a"])].astype(np.float32)
        val[i]["_ep"] = ei
        i += 1
e = val[0]
_, _, w, _ = rigid_parts(e, c_enc)                     # 刚体陀螺 (T,3)
u = excitation(e["a"], c_enc)
phi = ((raw_feats(e["a7"]) - mu) / sd).astype(np.float32)
with torch.no_grad():
    flex = fx(torch.from_numpy(u), torch.from_numpy(phi)).numpy()
meas = (e["g"] @ e["R"]).astype(np.float32)            # 实测陀螺(ideal系)
total = w + flex
resid_meas = bp_t(meas - w, BP)                        # 带通残差(震铃真值)

# 选能量最高的 8s 窗 + 最活跃通道
T = len(meas)
win = int(8 * FPS)
energy = np.convolve((meas ** 2).sum(1), np.ones(win), "valid")
s0 = int(np.argmax(energy)); s1 = s0 + win
ch = int(np.argmax((meas[s0:s1] ** 2).sum(0)))
t = np.arange(s0, s1) / FPS

fig, axes = plt.subplots(3, 1, figsize=(12.5, 10))
ax = axes[0]
ax.plot(t, meas[s0:s1, ch], color="#374151", lw=1.6, label="measured gyro (real IMU)")
ax.plot(t, total[s0:s1, ch], color="#dc2626", lw=1.3, alpha=0.9,
        label="$\\hat{G}$ synthesis = rigid + flex")
ax.plot(t, w[s0:s1, ch], color="#2563eb", lw=1.0, ls="--", alpha=0.8,
        label="rigid part only (Jacobian)")
ax.set_title(f"(A)  Use-case 1 output: synthetic gyro vs measurement — held-out episode "
             f"({e['tag']} ep{e['_ep']}, most active channel gyro-{'xyz'[ch]})",
             fontsize=11.5, weight="bold")
ax.set_ylabel("angular vel (rad/s)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25)

flex_energy = np.convolve((flex ** 2).sum(1), np.ones(win), "valid")
s2 = int(np.argmax(flex_energy)); s3 = s2 + win
ch2 = int(np.argmax((flex[s2:s3] ** 2).sum(0)))
t2 = np.arange(s2, s3) / FPS
ax = axes[1]
ax.plot(t2, resid_meas[s2:s3, ch2], color="#374151", lw=1.6,
        label="measured residual = BP(real $-$ rigid)")
ax.plot(t2, flex[s2:s3, ch2], color="#ea580c", lw=1.5, alpha=0.95,
        label="flex module prediction (3 modal resonators)")
ax.set_title(f"(B)  ringing slice (strongest-ringing segment, gyro-{'xyz'[ch2]}): flex owns "
             "only the resonant share — the rest is the documented 56% unexplained",
             fontsize=10.5, weight="bold")
ax.set_ylabel("angular vel (rad/s)")
ax.set_xlabel("time (s)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25)

# ---- (C) 场合二: 单帧 δa 脉冲的响应 ----
import train_pi_e2 as TP
G = TP.GEnsemble()
zl = np.load(TP.LROOT / e["tag"] / f"ep{e['_ep']:03d}.npz")
import json as _json
R = np.array(_json.loads((TP.LROOT / e["tag"] / "manifest.json").read_text())["R"],
             np.float32)
epJ, alpha, RB, RD = G.ep_context(zl["a_arm"], zl["grip"], R)
S = len(zl["a_arm"])
tau = s0 + win // 2                                     # 运动段中央打脉冲
delta = torch.zeros(1, S, 6, device=TP.DEV)
delta[0, tau, 1] = 0.005                                # j2 单帧 +5 mrad
with torch.no_grad():
    de = TP.apply_G(delta, epJ.unsqueeze(0), alpha.unsqueeze(0), RB.unsqueeze(0),
                    RD.unsqueeze(0), G)
de = de[0].cpu().numpy()                                # (K,S,3)
sl = slice(tau - 7, tau + 20)
# e-序列索引 i 对应墙钟 i+LSHIFT — 换算成墙钟时间轴, 保证图上因果
tm = (np.arange(sl.start, sl.stop) - tau + TP.LSHIFT) * 50.0
ax = axes[2]
for k in range(1, de.shape[0]):
    ax.plot(tm, de[k, sl, ch] * 1000, color="#9ca3af", lw=0.8, alpha=0.7,
            label="ensemble members 1-4" if k == 1 else None)
ax.plot(tm, de[0, sl, ch] * 1000, color="#b91c1c", lw=2.0, marker="o", ms=4,
        label="member 0 (full data)")
ax.axvline(0, color="#111827", lw=1, ls=":")
ax.text(0.02, 0.93, "$\\delta a$ pulse: +5 mrad on j2, one frame, sent at t=0\n"
        "(this response curve IS $\\pi$'s training gradient)",
        transform=ax.transAxes, fontsize=9, va="top")
ax.set_title("(C)  Use-case 2 output: response to a single $\\delta a$ pulse — "
             "servo delay $\\rightarrow$ first peak $\\approx$100 ms, tail $\\approx$800 ms",
             fontsize=11.5, weight="bold")
ax.set_ylabel("$\\Delta e$  (mrad/s)")
ax.set_xlabel("wall-clock time since pulse (ms)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(FIG / "G_waveform.png", dpi=160)

# 对账数字
def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
print(f"panelA window r2 total={r2(meas[s0:s1], total[s0:s1]):.3f}  "
      f"panelB window ringing r2={r2(resid_meas[s2:s3, ch2], flex[s2:s3, ch2]):.3f}  "
      f"episode r2 total={r2(meas, total):.3f}")
print(FIG / "G_waveform.png")
