#!/usr/bin/env python3
"""E-1b — Ĝ(变体A) 残差画像: 未解释的 ~56% 住在哪?

残差 ρ(t) = gap_true − gap_pred_A (真机系, gyro)。按六因子分箱看
「相对未解释能量」 U = Σρ²/Σgap² (全局=1−R²≈0.60; 分箱值≫全局 → 该处是重灾区):
  ①频段(0.5-2/2-5/5-9Hz) ②位形(q2×q3 3×3) ③速度(‖q̇̂‖五分位)
  ④夹爪态(开/合) ⑤chunk相位(边界头10/中段/尾段) ⑥通道(gx/gy/gz)
另: 逐集 R² 表(任务/会话系统差)。数据=全20集(诊断用途, 统计力优先; val 单独标出)。
用法: python sim/residual_portrait.py
"""
from __future__ import annotations
import json

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from compare_ideal_methods import load_episode
from fit_g_production import BP, FlexModal, OUT, rigid_parts
from g_verdict import DATASETS, TRAIN_EPS, VAL_EPS, load

Z = np.load(OUT / "g_production.npz")
c_enc = Z["c_enc"]
flex = FlexModal([1.0, 1.0, 1.0])
with torch.no_grad():
    flex.log_om.copy_(torch.from_numpy(Z["log_om"]))
    flex.raw_ze.copy_(torch.from_numpy(Z["raw_ze"]))
    flex.W.copy_(torch.from_numpy(Z["W"]))
    flex.B.copy_(torch.from_numpy(Z["B"]))
    flex.D.copy_(torch.from_numpy(Z["D"]))
flex.eval()

ALL_EPS = TRAIN_EPS + VAL_EPS


def band(x, lo, hi):
    return filtfilt(*butter(3, [lo / 10, hi / 10], "bandpass"), x, axis=0)


def main():
    eps = load(ALL_EPS, with_gap=True)
    # 附: 夹爪列 + 集号(load 顺序 = DATASETS × ALL_EPS)
    k = 0
    for repo, tag in DATASETS:
        for ei in ALL_EPS:
            d = load_episode(repo, ei)
            eps[k]["grip"] = d["action"][:len(eps[k]["a"]), 6].astype(np.float32)
            eps[k]["_ep"] = ei
            k += 1

    rows = []
    for e in eps:
        _, qddh, w_rigid, _ = rigid_parts(e, c_enc)
        with torch.no_grad():
            r = flex(torch.from_numpy(np.ascontiguousarray(qddh))).numpy()
        pred_real = (w_rigid + r) @ e["R"].T
        rho = e["g"] - pred_real                     # = gap − gap_pred
        qdh = np.vstack([np.zeros((1, 6)), np.diff(e["a"], axis=0)]) / 0.05
        rows.append({"e": e, "rho": rho, "gap": e["gap"], "spd": np.linalg.norm(qdh, axis=1),
                     "is_val": e["_ep"] in VAL_EPS})

    def U(mask_fn=None, xform=None):
        num = den = 0.0
        for r in rows:
            rho, gap = r["rho"], r["gap"]
            if xform:
                rho, gap = xform(rho), xform(gap)
            m = mask_fn(r) if mask_fn else np.ones(len(rho), bool)
            m = m[:len(rho)]
            num += (rho[m] ** 2).sum(); den += (gap[m] ** 2).sum()
        return num / max(den, 1e-12), int(den > 0)

    g_all, _ = U()
    print(f"全局未解释比 U = {g_all:.3f}  (=1−R², 参照)")
    print(f"val-only U = {U(lambda r: np.ones(len(r['rho']), bool) if r['is_val'] else np.zeros(len(r['rho']), bool))[0]:.3f}")

    print("\n① 频段:")
    for lo, hi in [(0.5, 2), (2, 5), (5, 9)]:
        u, _ = U(xform=lambda x, lo=lo, hi=hi: band(x, lo, hi))
        print(f"   {lo}-{hi}Hz: U={u:.3f}")

    print("\n② 位形 (q2 × q3, 3×3, 值=U, 括号=帧占比%):")
    q2a = np.concatenate([r["e"]["a"][:, 1] for r in rows])
    q3a = np.concatenate([r["e"]["a"][:, 2] for r in rows])
    e2 = np.quantile(q2a, [0, 1/3, 2/3, 1]); e3 = np.quantile(q3a, [0, 1/3, 2/3, 1])
    tot = sum(len(r["rho"]) for r in rows)
    for i in range(3):
        line = []
        for j in range(3):
            def mk(r, i=i, j=j):
                a = r["e"]["a"]
                return ((a[:, 1] >= e2[i]) & (a[:, 1] <= e2[i+1]) &
                        (a[:, 2] >= e3[j]) & (a[:, 2] <= e3[j+1]))
            u, _ = U(mk)
            occ = sum(mk(r).sum() for r in rows) / tot * 100
            line.append(f"{u:.2f}({occ:.0f}%)")
        print(f"   q2槽{i}: " + "  ".join(line))

    print("\n③ 速度五分位 (指令‖q̇‖):")
    spd_all = np.concatenate([r["spd"] for r in rows])
    qs = np.quantile(spd_all, [0, .2, .4, .6, .8, 1])
    for i in range(5):
        u, _ = U(lambda r, i=i: (r["spd"] >= qs[i]) & (r["spd"] <= qs[i+1]))
        print(f"   Q{i+1} [{qs[i]:.2f}-{qs[i+1]:.2f} rad/s]: U={u:.3f}")

    print("\n④ 夹爪态:")
    for name, fn in [("开(≥50%)", lambda r: r["e"]["grip"][:len(r['rho'])] >= 0.5 * 0.035),
                     ("合(<50%)", lambda r: r["e"]["grip"][:len(r['rho'])] < 0.5 * 0.035)]:
        u, _ = U(fn)
        print(f"   {name}: U={u:.3f}")

    print("\n⑤ chunk相位:")
    for name, lo, hi in [("边界头0-9", 0, 10), ("中段10-49", 10, 50), ("尾段50-99", 50, 100)]:
        u, _ = U(lambda r, lo=lo, hi=hi: (np.arange(len(r["rho"])) % 100 >= lo) &
                                          (np.arange(len(r["rho"])) % 100 < hi))
        print(f"   {name}: U={u:.3f}")

    print("\n⑥ 通道:")
    for c, n in enumerate(["gx", "gy", "gz"]):
        num = sum((r["rho"][:, c] ** 2).sum() for r in rows)
        den = sum((r["gap"][:, c] ** 2).sum() for r in rows)
        print(f"   {n}: U={num/den:.3f}")

    print("\n逐集 R²gap (× = val):")
    for r in rows:
        e = r["e"]
        r2 = 1 - (r["rho"] ** 2).sum() / ((r["gap"] - r["gap"].mean(0)) ** 2).sum()
        print(f"   {e['tag']:11s} ep{e['_ep']}{'×' if r['is_val'] else ' '}: {r2:.3f}")


if __name__ == "__main__":
    main()
