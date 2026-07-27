#!/usr/bin/env python3
"""#69 token 提取对比评估: S0 / 1q(#63a) / M-q / RLT-AE。

固定口径(独立于训练 cutoff), dense val:
  - 5-15Hz/ACT     : 真抖动衰减(被控制器阻挡, IMU 收益有限)
  - 2-4.5Hz/ACT    : 透传带衰减(传到 IMU 的频段)
  - 2-4.5Hz 相干保持: refined~demo 相干 / ACT~demo 相干(#66b 选择性=保住任务结构;
                     越接近 1 越好 = 去抖动同时不破坏 27% 任务运动)
  - L1             : 轨迹保真(vs demo)
架构按 ckpt['args'] 复原。用法: python sim/eval_token_extraction.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, filtfilt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.refine_net_dataset import RefinePhase1Dataset
from sim.train_refine_net import ACTIVE_JOINTS as AJ, FPS
from lerobot.common.policies.act.refine_net import RefineNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TR = REPO / "outputs/train/refine_net"
_b5, _a5 = butter(4, 5 / 15, "high")
_bb, _ab = butter(4, [2 / 15, 4.5 / 15], "band")


def _refine(ckdir, ds, ACT):
    ck = torch.load(TR / ckdir / "ckpt_best.pt", map_location=DEV, weights_only=False)
    a = ck.get("args", {})
    m = RefineNet(use_state=a.get("variant") not in ("S0", None) and False,  # 对比集均 S0
                  use_history_action=False, use_history_imu=False,
                  use_act_img_attn=a.get("use_act_img_attn", False),
                  n_img_readout=a.get("n_img_readout", 1),
                  use_act_rlt=a.get("use_act_rlt", False)).to(DEV).eval()
    m.load_state_dict(ck["model_state"])
    sm = np.asarray(ck["state_mean"], np.float32); ss = np.asarray(ck["state_std"], np.float32)
    N = len(ds.ACT_output); o = []
    for s in range(0, N, 128):
        e = min(s + 128, N)
        ao = torch.from_numpy(ds.ACT_output[s:e]).to(DEV)
        ha = torch.from_numpy(ds.history_action[s:e]).to(DEV)
        hi = torch.from_numpy(ds.history_imu[s:e]).to(DEV)
        st = ((torch.from_numpy(ds.start_state_7[s:e]) - torch.from_numpy(sm)) / (torch.from_numpy(ss) + 1e-8)).to(DEV)
        ai = torch.from_numpy(ds.act_img_tokens[s:e]).to(DEV) if a.get("use_act_img_attn") else None
        ar = torch.from_numpy(ds.act_rlt_tokens[s:e]).to(DEV) if a.get("use_act_rlt") else None
        with torch.no_grad():
            r, _ = m(ao, ha, hi, st, act_img=ai, act_rlt=ar)
        o.append(r.cpu())
    return torch.cat(o), ck.get("epoch", "?"), sum(p.numel() for p in m.parameters())


def _coh_band(Xs, Ys, freqs, lo, hi):       # avg mag-sq coherence over chunks & active joints, in band
    Pxy = (Xs * np.conj(Ys)).mean(0); Pxx = (np.abs(Xs) ** 2).mean(0); Pyy = (np.abs(Ys) ** 2).mean(0)
    C = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-20)        # (F, nAJ)
    m = (freqs >= lo) & (freqs < hi)
    return float(C[m].mean())


def _spec(X):
    w = np.hanning(100)
    x = (X.numpy() - X.numpy().mean(1, keepdims=True))[:, :, AJ] * w[None, :, None]
    return np.fft.rfft(x, axis=1)


def main():
    ds = RefinePhase1Dataset(REPO / "outputs/sim/phase1_dense.h5", "val",
                             np.zeros(7, np.float32), np.ones(7, np.float32))
    ACT = torch.from_numpy(ds.ACT_output); CLN = torch.from_numpy(ds.A_clean)
    freqs = np.fft.rfftfreq(100, d=1 / FPS)
    fA, fC = _spec(ACT), _spec(CLN)
    hp5A = np.sqrt((filtfilt(_b5, _a5, ACT.numpy(), axis=1) ** 2).mean(1))[:, AJ].mean()
    bpA = np.sqrt((filtfilt(_bb, _ab, ACT.numpy(), axis=1) ** 2).mean(1))[:, AJ].mean()
    cohAC = _coh_band(fA, fC, freqs, 2, 4.5)            # ACT~demo 2-4.5Hz 相干基准

    variants = [
        ("S0 (无token,对照)",   "phase1_dense_S0+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"),
        ("1q (#63a)",          "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"),
        ("M-q (8 query)",      "phase1_dense_Hattnx8+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"),
        ("RLT-AE (论文式)",     "phase1_dense_RLT+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"),
    ]
    print(f"dense val | ACT~demo 2-4.5Hz 相干基准={cohAC:.3f} | clean 2-4.5Hz/ACT=0.937 | L1(ACT)≈0.0084")
    print(f"{'变体':>20} {'参数':>9} {'5-15Hz/ACT':>11} {'2-4.5Hz/ACT':>12} {'相干保持%':>9} {'L1':>8}")
    print("-" * 80)
    for nm, d in variants:
        if not (TR / d / "ckpt_best.pt").exists():
            print(f"{nm:>20}  (缺 ckpt)"); continue
        X, ep, npar = _refine(d, ds, ACT)
        hp5 = np.sqrt((filtfilt(_b5, _a5, X.numpy(), axis=1) ** 2).mean(1))[:, AJ].mean()
        bp = np.sqrt((filtfilt(_bb, _ab, X.numpy(), axis=1) ** 2).mean(1))[:, AJ].mean()
        l1 = float((X[:, :, AJ] - CLN[:, :, AJ]).abs().mean())
        coh = _coh_band(_spec(X), fC, freqs, 2, 4.5)
        print(f"{nm:>20} {npar:>9,} {hp5/hp5A:>11.3f} {bp/bpA:>12.3f} "
              f"{100*coh/cohAC:>8.0f}% {l1:>8.5f}")
    print("-" * 80)
    print("选择性 = 相干保持%(refined~demo / ACT~demo, 2-4.5Hz): 越高=去抖动越少破坏任务结构")
    print("理想 token: 2-4.5Hz/ACT 低(去透传带抖动) + 相干保持高(留任务) + L1 低(保真)")


if __name__ == "__main__":
    main()
