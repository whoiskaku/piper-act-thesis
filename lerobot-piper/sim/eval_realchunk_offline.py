#!/usr/bin/env python3
"""#76 — 离线重算: 在 B 录制(eval_rc_B_hattn_cut2)自己的观测上重跑 frozen ACT,
隔离 Hattn cut2.0 残差对【真机命令】的纯效果(去掉 A/B 配对依赖)。

每集第一个 chunk(frame 0, 单chunk):
  raw  = frozen ACT(同观测)                      ← 残差前
  ref  = raw + Hattn cut2.0 残差(refine_deploy wrapper, 同观测) ← 残差后
  rec  = B 录到的 action[0:100](部署时实下发)     ← 验证残差当时是否生效
比 2-4.5Hz RMS(臂关节 j1/j2/j3): ref/raw <1 = 残差在真机平滑; >1 = 变差。
ref≈rec(低 |ref-rec|) ⇒ 残差当时确生效 + 离线复现到位。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/eval_realchunk_offline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.deploy_eval_phase1 import act_chunk, C1_CKPT, DEVICE, CHUNK
from sim.precompute_act_output import load_frozen_act
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.refine_deploy import build_refine_deploy_policy

DS_B = "enolab-guo/eval_rc_B_hattn_cut2"
HATTN = REPO / "outputs/train/refine_net/phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150/ckpt_best.pt"
LR = Path.home() / ".cache/huggingface/lerobot" / DS_B
AJ = [0, 1, 2]
_BP = butter(4, [2.0 / 15.0, 4.5 / 15.0], "band")


def bp_rms(x):
    y = filtfilt(*_BP, x[:, AJ], axis=0)
    return float(np.sqrt((y ** 2).mean()))


def main():
    policy, cfg = load_frozen_act(C1_CKPT, DS_B, DEVICE)
    cfg.dataset.repo_id = DS_B
    dataset = make_dataset(cfg)
    ep_from = dataset.episode_data_index["from"].numpy()
    ep_to = dataset.episode_data_index["to"].numpy()
    n_ep = len(ep_from)
    print(f"[data] {DS_B}: {n_ep} eps")
    wrapped = build_refine_deploy_policy(policy, str(HATTN), DEVICE, mode="residual")

    by_ci = {}                                       # chunk-index -> list of (raw,ref,rec,|ref-rec|) RMS
    RAW, REF, CIDX, EPI = [], [], [], []             # 保存每 chunk 的 raw/ref(100,7) 供作图统一口径
    for ep in range(1, n_ep):                        # skip ep0 warmup
        f0, f1 = int(ep_from[ep]), int(ep_to[ep])
        pq = sorted((LR / "data").glob(f"chunk-*/episode_{ep:06d}.parquet"))[0]
        rec_all = np.stack(pd.read_parquet(pq)["action"].to_numpy()).astype(np.float64)
        for ci, s_local in enumerate(range(0, (f1 - f0) - CHUNK + 1, CHUNK)):
            sample = dataset[f0 + s_local]
            batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in sample.items()
                     if isinstance(v, torch.Tensor)}
            raw = act_chunk(policy, sample)          # (100,7) 裸 ACT(同观测)
            wrapped.reset()
            a0 = wrapped.select_action({k: v.clone() for k, v in batch.items()})
            ref = torch.stack([a0] + list(wrapped._queue)).reshape(-1, 7).cpu().numpy()
            rec = rec_all[s_local:s_local + CHUNK]
            by_ci.setdefault(ci, []).append(
                (bp_rms(raw), bp_rms(ref), bp_rms(rec), float(np.abs(ref - rec)[:, AJ].mean())))
            RAW.append(raw); REF.append(ref); CIDX.append(ci); EPI.append(ep)
    npz = REPO / "outputs/eval/realchunk_offline_chunks.npz"
    np.savez(npz, raw=np.array(RAW), ref=np.array(REF), ci=np.array(CIDX), ep=np.array(EPI))
    print(f"[saved] {npz}  ({len(RAW)} chunks)")

    print(f"\n{'chunk':>7} {'n':>3} | {'raw':>8} {'ref':>8} {'ref/raw':>8} | {'rec/raw':>8} {'|ref-rec|':>10}")
    print("-" * 60)
    allrows = []
    for ci in sorted(by_ci):
        R = np.array(by_ci[ci]); allrows.extend(by_ci[ci])
        raw_m, ref_m, rec_m, match_m = R.mean(0)
        tag = " (first)" if ci == 0 else ""
        print(f"{ci:>7}{tag:>0} {len(R):>3} | {raw_m*1e3:>7.3f}m {ref_m*1e3:>7.3f}m {ref_m/raw_m:>8.2f} | "
              f"{rec_m/raw_m:>8.2f} {match_m*1e3:>9.3f}m")
    print("-" * 60)
    def agg(rows):
        R = np.array(rows); return R[:, 0].mean(), R[:, 1].mean(), R[:, 2].mean(), R[:, 3].mean()
    for name, rows in [("ALL chunks", allrows),
                       ("LATER (chunk>=1)", [r for ci in by_ci if ci >= 1 for r in by_ci[ci]])]:
        if not rows:
            continue
        raw_m, ref_m, rec_m, match_m = agg(rows)
        print(f"{name:>16}: raw={raw_m*1e3:.3f}m ref={ref_m*1e3:.3f}m  ref/raw={ref_m/raw_m:.2f}  "
              f"rec/raw={rec_m/raw_m:.2f}  |ref-rec|={match_m*1e3:.3f}m")
    print("=" * 60)
    print("判读: ref/raw = 残差纯效果(同观测,隔离). <1 平滑/>1 变差。")
    print("      若 ref/raw 随 chunk 升高 → 闭环漂移累积使残差更糟; |ref-rec| 小 → 离线复现到位(残差确生效)。")


if __name__ == "__main__":
    main()
