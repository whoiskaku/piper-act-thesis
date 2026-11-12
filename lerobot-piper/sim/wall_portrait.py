#!/usr/bin/env python3
"""P1 — 墙画像: real − kin_ideal 的分层量化(从零重测, 不假设旧发现)。

口径(P0.4 落定):
  参照 = kin_ideal(指令运动学渲染, 20Hz 抗混叠) → 逐数据集 Kabsch(R)+acc零偏(b) 标定
  对齐 = 逐数据集自动估 lag(gyro 最优互相关中位数; 已知 demo≈2, deploy≈3=录制顺序伪影)
  分段 = **real 侧**运动判据(指令/ideal 侧掩码对 ACT 抖动失效, 2026-07-09 教训):
         still / free-move / hold-move(持物) / contact(夹爪开合±0.5s) ; boundary(deploy)单列
指标/类:
  LF(<2Hz) Δ-RMS         — 轨迹尺度偏差(滞后已对齐后)
  HF(>2Hz) RMS比 real/ideal — 真机在理想之外加的振动(=墙的振动分量)
  corr(gyro)             — 波形一致性
  boundary: HP能量 边界±0.25s / 段中 (real, ideal, Δ 各算)
用法: python sim/wall_portrait.py [--eps 10]
输出: outputs/sim/wall_portrait/{portrait.json} + stdout 表
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from compare_ideal_methods import kin_ideal_20, lag_corr_ch

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sim/wall_portrait"
DATASETS = [
    ("enolab-guo/pickplace2_train",     "pp2-demo"),
    ("enolab-guo/eval_pickplace2_100k", "pp2-deploy"),
    ("enolab-guo/insert_train",         "ins-demo"),
    ("enolab-guo/eval_insert_100k",     "ins-deploy"),
]
FPS = 20.0
INTENT_FC = 3.0   # 意图版理想: 指令零相位低通截止(Hz), 剔量化台阶/策略抖动, 保留真运动
BA_LP = butter(3, 2.0 / (FPS / 2))
BA_HP = butter(3, 2.0 / (FPS / 2), "high")


def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, "same")


def kabsch(gi, gr):
    H = gi.T @ gr
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T                      # real ≈ R @ ideal


def masks_for(real_a, ideal_a, grip_a):
    """real 侧运动判据分段。返回 dict[class] = bool mask。"""
    wr = smooth(np.linalg.norm(real_a[:, 3:], axis=1))
    wi = smooth(np.linalg.norm(ideal_a[:, 3:], axis=1))
    active = wr > 0.08                          # rad/s 绝对阈(真机确实在转)
    still = (wr < 0.04) & (wi < 0.04)
    g = grip_a - grip_a.min()
    rng = g.max()
    hold = np.zeros(len(g), bool)
    contact = np.zeros(len(g), bool)
    if rng > 1e-3:
        hold = g < 0.4 * rng                    # 指令闭合≈持物
        dg = np.abs(np.diff(g, prepend=g[0]))
        ev = np.where(smooth(dg, 3) > 0.02 * rng)[0]
        for i in ev:                            # 开合事件 ±0.5s = 接触窗
            contact[max(0, i - 10):i + 11] = True
    return {
        "still": still & ~contact,
        "free-move": active & ~hold & ~contact,
        "hold-move": active & hold & ~contact,
        "contact": contact,
    }


def masked_rms(x, m):
    return float(np.sqrt((x[m] ** 2).mean())) if m.sum() else float("nan")


def boundary_masks(n, period=100, halfw=5, guard=20):
    b = np.zeros(n, bool)
    mid = np.ones(n, bool)
    for k in range(1, n // period + 1):
        i = k * period
        if i < n:
            b[max(0, i - halfw):min(n, i + halfw + 1)] = True
        mid[max(0, i - guard):min(n, i + guard)] = False
    mid[:guard] = False
    return b, mid


def analyze_dataset(repo: str, tag: str, n_eps: int):
    root = Path.home() / ".cache/huggingface/lerobot" / repo
    if not root.exists():
        print(f"[skip] {repo}")
        return None
    total = json.loads((root / "meta/info.json").read_text())["total_episodes"]
    eps = list(range(min(n_eps, total)))

    # --- pass1: 渲染+缓存(字面版+意图版), 估 lag ---
    cache = []
    lag_votes = []
    for ep in eps:
        kin20, d, _ = kin_ideal_20(repo, ep)
        int20, _, _ = kin_ideal_20(repo, ep, smooth_cmd=INTENT_FC)
        T = min(len(d["imu_real"]), len(kin20))
        cache.append((kin20[:T], int20[:T], d["imu_real"][:T], d["action"][:T, 6]))
        lags, _ = lag_corr_ch(kin20[:T], d["imu_real"][:T])
        lag_votes.extend(lags[3:])
    L = int(np.median(lag_votes))

    # --- pass2: 逐会话标定 (gyro Kabsch + acc 零偏) ---
    Gi = np.concatenate([k[:-L, 3:] for k, i2, r, g in cache])
    Gr = np.concatenate([r[L:, 3:] for k, i2, r, g in cache])
    sel = np.linalg.norm(Gi, axis=1) > 0.15
    R = kabsch(Gi[sel], Gr[sel])
    Ai = np.concatenate([k[:-L, :3] for k, i2, r, g in cache])
    Ar = np.concatenate([r[L:, :3] for k, i2, r, g in cache])
    b = (Ar - Ai @ R.T).mean(0)

    def apply_rb(x):
        out = np.empty_like(x)
        out[:, :3] = x[:, :3] @ R.T + b
        out[:, 3:] = x[:, 3:] @ R.T
        return out

    # --- pass3: 分层指标 ---
    per_ep = []
    for (kin, intent, real, grip) in cache:
        ia, ii = apply_rb(kin)[:-L], apply_rb(intent)[:-L]
        ra, ga = real[L:], grip[:-L]
        lp_i, lp_r = filtfilt(*BA_LP, ia, axis=0), filtfilt(*BA_LP, ra, axis=0)
        hp_i, hp_r = filtfilt(*BA_HP, ia, axis=0), filtfilt(*BA_HP, ra, axis=0)
        hp_ii = filtfilt(*BA_HP, ii, axis=0)
        dlf = lp_r - lp_i
        m = masks_for(ra, ia, ga)
        row = {"n": len(ra)}
        # 噪声地板(still 段 real 的 HP RMS), 供地板归一
        if m["still"].sum() >= 40:
            row["floor_acc"] = masked_rms(hp_r[:, :3], m["still"])
            row["floor_gyr"] = masked_rms(hp_r[:, 3:], m["still"])
        for cls, mk in m.items():
            row[f"{cls}/frac"] = float(mk.mean())
            if mk.sum() < 40:
                continue
            row[f"{cls}/lf_dacc"] = float(np.sqrt((dlf[mk, :3] ** 2).mean()))
            row[f"{cls}/lf_dgyr"] = float(np.sqrt((dlf[mk, 3:] ** 2).mean()))
            row[f"{cls}/hp_real_acc"] = masked_rms(hp_r[:, :3], mk)
            row[f"{cls}/hp_real_gyr"] = masked_rms(hp_r[:, 3:], mk)
            row[f"{cls}/hf_acc"] = row[f"{cls}/hp_real_acc"] / max(masked_rms(hp_i[:, :3], mk), 1e-9)
            row[f"{cls}/hf_gyr"] = row[f"{cls}/hp_real_gyr"] / max(masked_rms(hp_i[:, 3:], mk), 1e-9)
            row[f"{cls}/hf_acc_int"] = row[f"{cls}/hp_real_acc"] / max(masked_rms(hp_ii[:, :3], mk), 1e-9)
            row[f"{cls}/hf_gyr_int"] = row[f"{cls}/hp_real_gyr"] / max(masked_rms(hp_ii[:, 3:], mk), 1e-9)
            cc = [np.corrcoef(ia[mk, c], ra[mk, c])[0, 1]
                  for c in range(3, 6) if ia[mk, c].std() > 1e-9]
            row[f"{cls}/corr_g"] = float(np.mean(cc)) if cc else float("nan")
        if "deploy" in tag:
            bm, mm = boundary_masks(len(ra))
            e = lambda x, mk2: float((x[mk2] ** 2).sum(1).mean()) if mk2.sum() else np.nan
            row["bnd/real"] = e(hp_r, bm) / max(e(hp_r, mm), 1e-12)
            row["bnd/ideal"] = e(hp_i, bm) / max(e(hp_i, mm), 1e-12)
            row["bnd/delta"] = e(hp_r - hp_i, bm) / max(e(hp_r - hp_i, mm), 1e-12)
        per_ep.append(row)
    return {"tag": tag, "repo": repo, "lag": L,
            "calib_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
            "R": R.tolist(), "acc_bias": b.tolist(), "per_ep": per_ep}


def med(rows, key):
    v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=10)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for repo, tag in DATASETS:
        r = analyze_dataset(repo, tag, args.eps)
        if r:
            out[tag] = r
            print(f"[{tag}] lag={r['lag']}帧  标定旋转={r['calib_rot_deg']:.1f}°  "
                  f"acc零偏={np.round(r['acc_bias'],2)}")
    (OUT_DIR / "portrait.json").write_text(json.dumps(out, indent=1))

    print(f"\n===== 墙画像(逐集中位数; LF=<2Hz Δ-RMS; HF比=real/参照, 字面|意图fc{INTENT_FC:g}Hz; 地板=÷still) =====")
    hdr = (f"{'数据集':11s}|{'类':9s}|{'占比':5s}|{'LFΔgyr':7s}|"
           f"{'HFa字面':7s}|{'HFa意图':7s}|{'HFg字面':7s}|{'HFg意图':7s}|{'HFg/地板':7s}|{'corr_g':6s}")
    print(hdr); print("-" * len(hdr))
    for tag, r in out.items():
        rows = r["per_ep"]
        floor_g = med(rows, "floor_gyr")
        for cls in ("still", "free-move", "hold-move", "contact"):
            fl = med(rows, f"{cls}/hp_real_gyr") / floor_g if np.isfinite(floor_g) else float("nan")
            print(f"{tag:11s}|{cls:9s}|{med(rows,f'{cls}/frac')*100:4.0f}% |"
                  f"{med(rows,f'{cls}/lf_dgyr'):7.4f}|"
                  f"{med(rows,f'{cls}/hf_acc'):7.2f}|{med(rows,f'{cls}/hf_acc_int'):7.2f}|"
                  f"{med(rows,f'{cls}/hf_gyr'):7.2f}|{med(rows,f'{cls}/hf_gyr_int'):7.2f}|"
                  f"{fl:7.2f}|{med(rows,f'{cls}/corr_g'):6.3f}")
        if "deploy" in tag:
            print(f"{tag:11s}|boundary : HP能量比(边界/段中) real={med(rows,'bnd/real'):.1f} "
                  f"ideal={med(rows,'bnd/ideal'):.1f} Δ={med(rows,'bnd/delta'):.1f}")
    print(f"\n→ {OUT_DIR/'portrait.json'}")


if __name__ == "__main__":
    main()
