#!/usr/bin/env python3
"""**诊断（零机时）**：j5 的 3–9 Hz 关节残差，能不能靠**专门为它选的**滤波器压下去？

背景
----
`r_q` 选中 nlag=4 @ 20 Hz、`W_BAND=0.5`，但那是在
**5 个关节 + 宽带/带内混合损失**上胜出的**全局折中**。
j5 的 3–9 Hz 从来没有被单独优化过。而 j5 占关节层可得空间的 89%
（oracle 替换 ⇒ IMU 0.520 → 0.730）。

已排掉的三条（都指向「跨关节耦合不是主因」）
  · j5 带内运动几乎全由命令解释 ⇒ 不是「非指令运动」。
    ⚠ **口径订正**（用户 2026-09-02）：限幅后参考 / 真实编码器的带内 RMS =
    0.2552/0.2711 = **94.1%**，这是**幅值比**；对应**能量比 = 0.941² = 88.6%**。
    早先写的「94% 能量」是把幅值比当能量比，**错**。判定不受影响。
  · pp2 各关节带内输入近乎正交（非对角 |r| 中位 0.12）⇒ 单关节隔离买不到可辨识性
  · j5 残差 vs 其他关节**真实运动**的偏差修正多重相干仅 0.118/0.186

所以本脚本只问：**同一批数据、同一特征集，把目标窄化到「j5 + 3–9 Hz」，
再把 nlag / λ / 带内权重放开，折外能到多少？**

  · 明显好于现状 ⇒ 缺的是「为 j5 定制」，零机时可拿，**不必采探针**
  · 拿不到       ⇒ 现有命令+预测状态里确实没有这个信息，才谈采数据

口径：训练族折外选参（K=6，按集划分）；开发验证集只读一次、不参与任何选择。

用法:
    python -u sim/step4_gvnext/diag_j5_specialize.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, valid_files, FPS                            # noqa: E402
from train_rq import (features as rq_features, fir_design, fit,        # noqa: E402
                      BAND, KFOLD)


def band_err1(yt, yp, lo=BAND[0], hi=BAND[1]):
    """`train_rq.band_err` 的单通道版（原版内部按 `MOVING` 取 5 列）。"""
    a, b = bp(np.asarray(yt).reshape(-1, 1), lo, hi), bp(np.asarray(yp).reshape(-1, 1), lo, hi)
    return float(((a - b) ** 2).sum()), float((a ** 2).sum())

S1, S2 = ROOT / "outputs/step4_gvnext/s1", ROOT / "outputs/step4_gvnext/s2"
RQ = ROOT / "outputs/step4_gvnext/rq"
J5 = 4
LAGS = (4, 8, 16, 24, 40, 64)
LAMS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
WBS = (0.5, 1.0)                       # 0.5=现状口径；1.0=**只看带内**


def stack1(D, y, wb):
    """单通道版 stack_band：每项按自身尺度归一（与 train_rq 同规）。"""
    yb = bp(y[:, None], *BAND)[:, 0]
    s0 = np.sqrt(max(float(np.var(y)), 1e-30))
    s1 = np.sqrt(max(float(np.var(yb)), 1e-30))
    Db = bp(D, *BAND)
    return (np.vstack([D * np.sqrt(1 - wb) / s0, Db * np.sqrt(wb) / s1]),
            np.concatenate([y * np.sqrt(1 - wb) / s0, yb * np.sqrt(wb) / s1])[:, None])


def load_set(files):
    E = []
    for f in files:
        X, Y, _ = rq_features(load(f))
        E.append({"X": X, "y": Y[:, J5]})
    return E


def main() -> int:
    import gc
    gc.disable()
    zq = np.load(RQ / "rq_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    TR, DV = load_set(valid_files(S1)), load_set(valid_files(S2, "replay_pp2f*.json"))
    print(f"训练族 {len(TR)} 条 · 开发验证集 {len(DV)} 条 · 特征维 {TR[0]['X'].shape[1]}")
    print(f"现状 r_q: nlag={sq['nlag']} λ={sq['lam']:.0e} W_BAND=0.5（**全局折中**）\n")

    def red(E, W, nlag):
        n = d = 0.0
        for e in E:
            a, b = band_err1(e["y"], fir_design(e["X"], nlag) @ W)
            n += a; d += b
        return 1 - n / max(d, 1e-30)

    # 现状 r_q 的 j5 带内削减（折外不可得，用全量权重给个参照）
    cur_tr = red(TR, zq["W"][:, J5:J5+1], sq["nlag"])
    cur_dv = red(DV, zq["W"][:, J5:J5+1], sq["nlag"])
    print(f"  现状 r_q 的 j5 3–9 Hz 削减：训练族(样本内) {cur_tr*100:>5.1f}%   "
          f"开发验证集 {cur_dv*100:>5.1f}%\n")

    folds = [list(range(i, len(TR), KFOLD)) for i in range(KFOLD)]
    print(f"  {'W_BAND':>7}{'nlag':>6}{'lam':>8}{'折外 j5 3-9Hz 削减':>20}")
    best = None
    for wb in WBS:
        for nlag in LAGS:
            D = [fir_design(e["X"], nlag) for e in TR]
            for lam in LAMS:
                n = d = 0.0
                for te in folds:
                    tr = [i for i in range(len(TR)) if i not in te]
                    P = [stack1(D[i], TR[i]["y"], wb) for i in tr]
                    W = fit([p[0] for p in P], [p[1] for p in P], lam)
                    for i in te:
                        a, b = band_err1(TR[i]["y"], D[i] @ W)
                        n += a; d += b
                r = 1 - n / max(d, 1e-30)
                if best is None or r > best[0]:
                    best = (r, wb, nlag, lam)
            print(f"  {wb:>7}{nlag:>6}{'':>8}{'':>20}", end="\r")
    r, wb, nlag, lam = best
    print(f"  {'':>60}")
    print(f"  **折外最优：W_BAND={wb}, nlag={nlag}, λ={lam:.0e} —— j5 3–9 Hz 削减 "
          f"{r*100:.1f}%**\n")

    # 用折外选中的超参在全训练族拟合，一次性读开发验证集
    D = [fir_design(e["X"], nlag) for e in TR]
    P = [stack1(D[i], TR[i]["y"], wb) for i in range(len(TR))]
    W = fit([p[0] for p in P], [p[1] for p in P], lam)
    dv = red(DV, W, nlag)
    print(f"  {'':<24}{'训练族':>12}{'开发验证集':>14}")
    print(f"  {'现状 r_q (nlag=4)':<24}{cur_tr*100:>11.1f}%{cur_dv*100:>13.1f}%")
    print(f"  {'j5 专用':<24}{red(TR,W,nlag)*100:>11.1f}%{dv*100:>13.1f}%")
    print(f"\n  开发验证集上的增量：{(dv-cur_dv)*100:+.1f} 个百分点")
    print("\n  判读：明显为正 ⇒ 缺的是「为 j5 定制」，零机时可拿；"
          "\n        接近 0 或为负 ⇒ 现有命令+预测状态里没有这个信息，才谈采探针。")
    return 0




# ─────────────────────────────────────────────────────────────────────────
# 追加：**位形调度项**（用户 2026-09-02 提的 step 2）在 pp2 上零成本检验
#   r_q5(t) = Σ_i [H_i0(z) + s(q_t)·H_i1(z)] · x_i(t)
# 实现为特征增广 X_aug = [X, s·X]（是用户那个形式的超集）。
# s(q_t) = 标准化的臂伸展度 (q̂2+q̂3)，只用 G 的预测状态，推理时可得。
# ─────────────────────────────────────────────────────────────────────────
def sched_main() -> int:
    import gc; gc.disable()
    zq = np.load(RQ / "rq_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    TR, DV = load_set(valid_files(S1)), load_set(valid_files(S2, "replay_pp2f*.json"))
    # s = 臂伸展度，用训练族的均值/标差标准化（开发集用同一常数，不重估）
    def ext(e):                       # X 的列 6..11 = q̂（见 train_rq.features）
        return e["X"][:, 6 + 1] + e["X"][:, 6 + 2]
    allx = np.concatenate([ext(e) for e in TR])
    mu, sd = float(allx.mean()), float(allx.std() + 1e-12)
    print(f"  s(q) = 标准化 (q̂2+q̂3)：训练族 mu={np.degrees(mu):.1f}° "
          f"sd={np.degrees(sd):.1f}°  范围 "
          f"[{np.degrees(allx.min()):.1f}, {np.degrees(allx.max()):.1f}]°")
    for E in (TR, DV):
        for e in E:
            s = ((ext(e) - mu) / sd)[:, None]
            e["Xa"] = np.concatenate([e["X"], s * e["X"]], 1)
    print(f"  增广特征维 {TR[0]['Xa'].shape[1]}（原 {TR[0]['X'].shape[1]}）\n")

    folds = [list(range(i, len(TR), KFOLD)) for i in range(KFOLD)]
    best = None
    for nlag in (4, 8, 16):
        D = [fir_design(e["Xa"], nlag) for e in TR]
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            n = d = 0.0
            for te in folds:
                tr = [i for i in range(len(TR)) if i not in te]
                P = [stack1(D[i], TR[i]["y"], 1.0) for i in tr]
                W = fit([p[0] for p in P], [p[1] for p in P], lam)
                for i in te:
                    a, b = band_err1(TR[i]["y"], D[i] @ W); n += a; d += b
            r = 1 - n / max(d, 1e-30)
            print(f"    nlag={nlag:<3} λ={lam:.0e}  折外 {r*100:>5.1f}%")
            if best is None or r > best[0]:
                best = (r, nlag, lam)
    r, nlag, lam = best
    D = [fir_design(e["Xa"], nlag) for e in TR]
    P = [stack1(D[i], TR[i]["y"], 1.0) for i in range(len(TR))]
    W = fit([p[0] for p in P], [p[1] for p in P], lam)
    n = d = 0.0
    for e in DV:
        a, b = band_err1(e["y"], fir_design(e["Xa"], nlag) @ W); n += a; d += b
    dv = 1 - n / max(d, 1e-30)
    print(f"\n  **位形调度：折外最优 nlag={nlag} λ={lam:.0e} ⇒ 折外 {r*100:.1f}%，"
          f"开发验证集 {dv*100:.1f}%**")
    print(f"  对照：现状 r_q 38.8% · j5 专用(无调度) 40.1%")
    print("\n  判读：>45% ⇒ 位形调度项在 pp2 上就有效，零机时可拿；"
          "\n        ≈40% ⇒ 要么该项无效，要么 pp2 位形范围太窄辨不出 —— 后者才是采探针的理由。")
    return 0


if __name__ == "__main__":
    (sched_main if "--sched" in sys.argv else main)()
    sys.stdout.flush()
    import os
    os._exit(0)
