#!/usr/bin/env python3
"""#75 — 命令动作平滑 操作点选择 (sim 交付物, 动作空间, 非IMU)。

评价目标 = 真机要部署的【命令动作】平滑度 (用户 2026-06-23 重申)。本脚本在 sim val
(phase1_dense.h5, 命令空间) 上对 absboth 残差的各操作点 (S0/Hattn × cut5.0/2.5/2.0)
算干净的绝对动作平滑 + 保真护栏, 选出最佳操作点作为上真机前的 sim 正结果。

度量 (active joints j1/j2/j3/grip):
  · 分带能量比 refined/ACT  (Hann-FFT, 去DC): 0-2(保主运动) / 2-4.5(透传/任务相关带) / 5-15(噪声)
    + clean/ACT 作"地板"参照(目标本身有多平滑)。
  · hp5_ratio = hp_rms(5Hz零相位高通) refined/ACT  (干净的绝对 jitter, <1 = 更平滑)。
  · 保真: L1(refined,clean) vs L1(ACT,clean), active joints。护栏: refined 不应比 ACT 更远离 clean。
最佳操作点判据: 2-4.5Hz 比最低, 且 (a) L1_ref ≤ L1_act(保真不劣于ACT) (b) 无带被放大>1.02。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/eval_action_smoothness.py
"""
from __future__ import annotations
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt

import sim.train_refine_net as T
from sim.train_refine_net import ACTIVE_JOINTS as AJ, FPS
from lerobot.common.policies.act.refine_net import RefineNet

REPO = Path(__file__).resolve().parents[1]
H5 = REPO / "outputs/sim/phase1_dense.h5"
RND = REPO / "outputs/train/refine_net"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = REPO / "outputs/eval/action_smoothness_oppoint.json"
OUT_PNG = REPO / "outputs/eval/action_smoothness_oppoint.png"
BANDS = [(0.0, 2.0), (2.0, 4.5), (5.0, 15.0)]
_HP5 = butter(4, 5.0 / (FPS / 2.0), "high")

# 操作点候选: 主方法 Hattn(token) + 对照 S0, × cutoff 5.0/2.5/2.0
CKPTS = {
    "S0 cut5.0":    "phase1_dense_S0+absboth_hf1.0_dl0.1_lr0.0003_bs64_ep150",
    "S0 cut2.5":    "phase1_dense_S0+absboth_cut2.5_hf1.0_dl0.1_lr0.0003_bs64_ep150",
    "S0 cut2.0":    "phase1_dense_S0+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150",
    "Hattn cut5.0": "phase1_dense_S0+Hattn+absboth_hf1.0_dl0.1_lr0.0003_bs64_ep150",
    "Hattn cut2.5": "phase1_dense_Hattn+absboth_cut2.5_hf1.0_dl0.1_lr0.0003_bs64_ep150",
    "Hattn cut2.0": "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150",
}


def band_energy(x):
    """x (N,T,A) → 每带 active-joint 能量和。Hann, 去DC。"""
    N, Tn, A = x.shape
    w = np.hanning(Tn)[None, :, None]
    X = np.fft.rfft((x - x.mean(1, keepdims=True)) * w, axis=1)
    P = (np.abs(X) ** 2)[:, :, AJ]
    f = np.fft.rfftfreq(Tn, d=1.0 / FPS)
    return {b: float(P[:, (f >= b[0]) & (f < b[1])].sum()) for b in BANDS}


def hp_rms(x):
    """5Hz 零相位高通后 RMS over active joints (绝对 jitter)。x (N,T,7)→标量。"""
    y = filtfilt(*_HP5, x, axis=1)
    return float(np.sqrt((y[:, :, AJ] ** 2).mean()))


def reconstruct(ckpt):
    a = ckpt.get("args", {}) or {}
    V = {"S0": (False, False, False), "full": (True, True, True)}
    us, uha, uhi = V.get(a.get("variant", "full"), (True, True, True))
    m = RefineNet(use_state=us, use_history_action=uha, use_history_imu=uhi,
                  use_act_hidden=a.get("use_act_hidden", False),
                  use_act_img_attn=a.get("use_act_img_attn", False),
                  n_img_readout=a.get("n_img_readout", 1),
                  use_act_rlt=a.get("use_act_rlt", False)).to(DEV).eval()
    sd = ckpt["model_state"]; te = sd["type_embed.weight"]
    if te.shape[0] < m.type_embed.weight.shape[0]:
        sd["type_embed.weight"] = torch.cat([te, m.type_embed.weight.data[te.shape[0]:].clone()], 0)
    m.load_state_dict(sd)
    return m, a


def run_model(m, ACT, IMG):
    out, z = [], torch.zeros
    with torch.no_grad():
        for b in range(0, len(ACT), 128):
            ao = torch.from_numpy(ACT[b:b+128]).to(DEV)
            ai = torch.from_numpy(IMG[b:b+128].astype(np.float32)).to(DEV)
            B = ao.shape[0]
            r, _ = m(ao, z(B, 600, 7, device=DEV), z(B, 600, 6, device=DEV),
                     z(B, 7, device=DEV), history_pad_mask=None, act_img=ai)
            out.append(r.cpu().numpy())
    return np.concatenate(out, 0)


def _resolve(val):
    """CKPTS 值: 目录名 → RND/dir/ckpt_best.pt; 直接 .pt 路径 → 原样。"""
    p = Path(val)
    return p if p.suffix == ".pt" else (RND / val / "ckpt_best.pt")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="只评估单个 ckpt(自训复现用, 传 ckpt_best.pt 路径); 省略=评估全部默认操作点")
    ap.add_argument("--name", type=str, default="my-repro", help="单 ckpt 评估时的显示名")
    cli = ap.parse_args()
    ckpts = {cli.name: str(cli.ckpt)} if cli.ckpt else CKPTS

    with h5py.File(H5, "r") as f:
        g = f["val"]
        ACT = g["ACT_output"][:].astype(np.float32)
        CLN = g["A_clean"][:].astype(np.float32)
        IMG = g["act_img_tokens"][:]
    print(f"[data] val n={len(ACT)}  active joints(j1/j2/j3/grip)={AJ}")
    Ea = band_energy(ACT)
    L1_act = float(np.abs(ACT[:, :, AJ] - CLN[:, :, AJ]).mean())
    hp_act = hp_rms(ACT)
    Ecln = band_energy(CLN)
    print(f"[ref] ACT hp_rms={hp_act:.5f}  L1(ACT,clean)={L1_act:.5f}")
    print(f"[floor] clean/ACT band: " + "  ".join(f"{b[0]:g}-{b[1]:g}={Ecln[b]/(Ea[b]+1e-12):.3f}" for b in BANDS))

    report, rows = {}, []
    for name, dirn in ckpts.items():
        p = _resolve(dirn)
        if not p.exists():
            print(f"[skip] {name}: {p} 不存在"); continue
        ckpt = torch.load(p, map_location=DEV, weights_only=False)
        m, a = reconstruct(ckpt)
        ref = run_model(m, ACT, IMG)
        Er = band_energy(ref)
        rb = {f"{b[0]:g}-{b[1]:g}Hz": Er[b] / (Ea[b] + 1e-12) for b in BANDS}
        hp = hp_rms(ref) / hp_act
        l1 = float(np.abs(ref[:, :, AJ] - CLN[:, :, AJ]).mean())
        amp = max(rb.values())                       # 是否有带被放大
        ok_fid = l1 <= L1_act * 1.001
        ok_noamp = amp <= 1.02
        report[name] = {"epoch": ckpt.get("epoch"), "band_ratio": rb, "hp5_ratio": hp,
                        "L1_ref": l1, "L1_act": L1_act, "fid_ok": ok_fid, "noamp_ok": ok_noamp}
        rows.append((name, rb, hp, l1, ok_fid, ok_noamp))
        print(f"  {name:>12} ep{ckpt.get('epoch'):>3} | "
              + "  ".join(f"{k}={v:.3f}" for k, v in rb.items())
              + f"  hp5={hp:.3f}  L1={l1:.5f}({'≤ACT✓' if ok_fid else '>ACT✗'})"
              + ("" if ok_noamp else "  ⚠放大"))

    # ---- 选最佳操作点: 2-4.5Hz 比最低, 且保真不劣于ACT + 无带放大 ----
    cand = [(n, rb["2-4.5Hz"]) for (n, rb, hp, l1, fok, nok) in rows if fok and nok]
    print("\n" + "=" * 70)
    if cand:
        best = min(cand, key=lambda t: t[1])
        print(f"[最佳操作点] {best[0]} — 2-4.5Hz(透传/任务带) refined/ACT={best[1]:.3f} "
              f"(↓{100*(1-best[1]):.0f}%), 保真≤ACT 且无带放大")
        b = report[best[0]]
        print(f"   完整: " + "  ".join(f"{k}={v:.3f}" for k, v in b['band_ratio'].items())
              + f"  hp5={b['hp5_ratio']:.3f}  L1 {b['L1_ref']:.5f} vs ACT {L1_act:.5f}")
        report["_best"] = best[0]
    else:
        print("[警告] 无操作点同时满足 保真≤ACT + 无带放大 — 见上表权衡")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {OUT_JSON}")

    # ---- 图 ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r[0] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(names)); w = 0.25
        for i, b in enumerate(BANDS):
            key = f"{b[0]:g}-{b[1]:g}Hz"
            vals = [report[n]["band_ratio"][key] for n in names]
            ax[0].bar(x + (i-1)*w, vals, w, label=key)
        ax[0].axhline(1.0, color="k", ls=":", lw=1, label="=ACT(无效)")
        ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax[0].set(ylabel="band energy refined/ACT (<1=平滑)", title="命令动作 分带平滑 (各操作点)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
        # 平滑(2-4.5) vs 保真(L1) 散点
        for n in names:
            ax[1].scatter(report[n]["band_ratio"]["2-4.5Hz"], report[n]["L1_ref"], s=60)
            ax[1].annotate(n, (report[n]["band_ratio"]["2-4.5Hz"], report[n]["L1_ref"]), fontsize=8)
        ax[1].axhline(L1_act, color="r", ls="--", lw=1, label=f"L1(ACT)={L1_act:.4f}")
        ax[1].set(xlabel="2-4.5Hz refined/ACT (←更平滑)", ylabel="L1(refined,clean) (↓更保真)",
                  title="平滑 ↔ 保真 权衡")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
        print(f"[saved] {OUT_PNG}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
