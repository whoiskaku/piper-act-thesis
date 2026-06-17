#!/usr/bin/env python3
"""改进 I — 部署流训练(deployment-stream training)。

#69-real 头号失败: per-chunk 训练的残差在真机拼接时搞坏 chunk 边界 → 净高频反升、
有害。本脚本把损失从"逐 100 步 chunk 内"改成"相邻两 chunk 按 receding-horizon
拼成 200 步流"(含边界), 让残差被迫同时学会 chunk 内平滑 + 边界连续(把 blend 的
功能内化), 而非只优化无边界的孤立 chunk。

部署对 (i,j): 同 episode, src_start_j = src_start_i + 100(= deploy cadence,
n_action_steps=100=chunk_size, 无重叠)。stream = cat(refined_i, refined_j) (B,200,7);
边界跳变 = stream[:,100] - stream[:,99] 现在进入 absboth 平滑损失被惩罚。

teacher-forced 近似: chunk_j 的 ACT_output 取自 demo 第 s+100 帧观测(非闭环执行后),
与全程一致(残差小, 闭环漂移小)。

主方法 Hattn(1q, use_act_img_attn) cut2.0 absboth。
用法: python sim/train_refine_stream.py [--variant-img] [--epochs 150]
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

import sim.train_refine_net as T   # 复用 hf_power/hf_abs_loss/hf_absfir_loss/_fir_highpass/ACTIVE_JOINTS
from sim.train_refine_net import ACTIVE_JOINTS as AJ, FPS, EPS
from lerobot.common.policies.act.refine_net import RefineNet

REPO = Path(__file__).resolve().parents[1]
H5 = REPO / "outputs/sim/phase1_dense.h5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_pairs(grp):
    ep = grp["src_ep"][:]; st = grp["src_start"][:]
    idx = {(int(e), int(s)): k for k, (e, s) in enumerate(zip(ep, st))}
    return [(idx[(int(e), int(s))], idx[(int(e), int(s) + 100)])
            for e, s in zip(ep, st) if (int(e), int(s) + 100) in idx]


class StreamData:
    """部署对数据(全载入内存)。act_img fp16 省内存。"""
    def __init__(self, split):
        with h5py.File(H5, "r") as f:
            g = f[split]
            self.ACT = g["ACT_output"][:].astype(np.float32)        # (N,100,7)
            self.CLN = g["A_clean"][:].astype(np.float32)
            self.IMG = g["act_img_tokens"][:]                        # fp16 (N,300,512)
            self.pairs = build_pairs(g)
        self.n = len(self.pairs)

    def batches(self, bs, shuffle):
        order = np.random.permutation(self.n) if shuffle else np.arange(self.n)
        for b in range(0, self.n, bs):
            ii = [self.pairs[k][0] for k in order[b:b + bs]]
            jj = [self.pairs[k][1] for k in order[b:b + bs]]
            yield (torch.from_numpy(self.ACT[ii]), torch.from_numpy(self.ACT[jj]),
                   torch.from_numpy(self.IMG[ii].astype(np.float32)),
                   torch.from_numpy(self.IMG[jj].astype(np.float32)),
                   torch.from_numpy(self.CLN[ii]), torch.from_numpy(self.CLN[jj]))


def refine_chunk(model, ao, ai):
    B = ao.shape[0]
    z = torch.zeros
    dummy_ha = z(B, 600, 7, device=ao.device); dummy_hi = z(B, 600, 6, device=ao.device)
    dummy_st = z(B, 7, device=ao.device); dummy_pm = z(B, 600, dtype=torch.bool, device=ao.device)
    r, d = model(ao, dummy_ha, dummy_hi, dummy_st, history_pad_mask=None, act_img=ai)
    return r, d


def absboth_stream(stream, act_stream):
    return 0.5 * (T.hf_abs_loss(stream, act_stream) + T.hf_absfir_loss(stream, act_stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lambda-hf", type=float, default=1.0)
    ap.add_argument("--lambda-delta", type=float, default=0.1)
    ap.add_argument("--hf-cutoff", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--tag", type=str, default="stream_Hattn+absboth_cut2.0")
    args = ap.parse_args()

    # 设 HF_CUTOFF(同 train_refine_net: 改全局 + 重算 FIR taps)
    if args.hf_cutoff != T.HF_CUTOFF:
        T.HF_CUTOFF = args.hf_cutoff
        from scipy.signal import firwin
        T._FIR_TAPS = torch.from_numpy(firwin(31, T.HF_CUTOFF / (FPS / 2.0), pass_zero=False).astype(np.float32))
        print(f"[#I] HF_CUTOFF={T.HF_CUTOFF}Hz")

    tr = StreamData("train"); va = StreamData("val")
    print(f"[data] train pairs={tr.n} val pairs={va.n}")
    model = RefineNet(use_state=False, use_history_action=False, use_history_imu=False,
                      use_act_img_attn=True, n_img_readout=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    b5, a5 = butter(4, 5 / (FPS / 2), "high")

    def evaluate():
        model.eval()
        bj_r = bj_a = 0.0; n = 0
        sr = sa = 0.0  # stream filtfilt hp energy (active)
        l1 = 0.0
        with torch.no_grad():
            for ao_i, ao_j, ai_i, ai_j, cl_i, cl_j in va.batches(128, False):
                ao_i, ao_j, ai_i, ai_j = [x.to(DEV) for x in (ao_i, ao_j, ai_i, ai_j)]
                ri, _ = refine_chunk(model, ao_i, ai_i); rj, _ = refine_chunk(model, ao_j, ai_j)
                stream = torch.cat([ri, rj], 1).cpu().numpy()         # (B,200,7)
                act_s = torch.cat([ao_i, ao_j], 1).cpu().numpy()
                cln_s = torch.cat([cl_i, cl_j], 1).numpy()
                B = stream.shape[0]; n += B
                # 边界跳变 (step 100 vs 99), active joints
                bj_r += np.abs(stream[:, 100, AJ] - stream[:, 99, AJ]).mean() * B
                bj_a += np.abs(act_s[:, 100, AJ] - act_s[:, 99, AJ]).mean() * B
                # stream 5Hz 高通能量(含边界)
                sr += (filtfilt(b5, a5, stream, axis=1)[:, :, AJ] ** 2).mean() * B
                sa += (filtfilt(b5, a5, act_s, axis=1)[:, :, AJ] ** 2).mean() * B
                l1 += np.abs(stream[:, :, AJ] - cln_s[:, :, AJ]).mean() * B
        return dict(bj_ratio=float(bj_r / bj_a), bj_refined=float(bj_r / n), bj_act=float(bj_a / n),
                    stream_hp_ratio=float(np.sqrt(sr / sa)), l1=float(l1 / n))

    out = REPO / "outputs/train/refine_net" / f"phase1_{args.tag}_lr{args.lr}_bs{args.bs}_ep{args.epochs}"
    out.mkdir(parents=True, exist_ok=True)
    hist = []; t0 = time.time(); best = 1e9
    for ep in range(args.epochs):
        model.train()
        for ao_i, ao_j, ai_i, ai_j, cl_i, cl_j in tr.batches(args.bs, True):
            ao_i, ao_j, ai_i, ai_j, cl_i, cl_j = [x.to(DEV) for x in (ao_i, ao_j, ai_i, ai_j, cl_i, cl_j)]
            ri, di = refine_chunk(model, ao_i, ai_i); rj, dj = refine_chunk(model, ao_j, ai_j)
            stream = torch.cat([ri, rj], 1)                  # (B,200,7)
            act_s = torch.cat([ao_i, ao_j], 1)
            cln_s = torch.cat([cl_i, cl_j], 1)
            l_bc = nn.functional.l1_loss(stream, cln_s)
            l_hf = absboth_stream(stream, act_s)             # 含边界的平滑损失
            l_dl = (di.abs().mean() + dj.abs().mean()) * 0.5
            loss = l_bc + args.lambda_hf * l_hf + args.lambda_delta * l_dl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = evaluate(); hist.append({"epoch": ep, **v})
        if v["bj_ratio"] < best and v["l1"] <= 1.1 * 0.0084:   # BC 护栏(ACT~clean L1≈0.0084)
            best = v["bj_ratio"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "state_mean": np.zeros(7, np.float32), "state_std": np.ones(7, np.float32),
                        "args": dict(variant="S0", use_act_img_attn=True, n_img_readout=1,
                                     hf_mode="absboth", hf_cutoff=args.hf_cutoff, stream=True),
                        "val": v}, out / "ckpt_best.pt")
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] 边界跳变 refined/ACT={v['bj_ratio']:.3f} "
                  f"({v['bj_refined']:.4f} vs {v['bj_act']:.4f})  stream_hp/ACT={v['stream_hp_ratio']:.3f} "
                  f"L1={v['l1']:.5f}  [{time.time()-t0:.0f}s]")
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"[done] best 边界跳变 refined/ACT={best:.3f}  saved {out}/ckpt_best.pt")


if __name__ == "__main__":
    main()
