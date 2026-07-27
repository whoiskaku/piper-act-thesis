#!/usr/bin/env python3
"""#71 改进I' — 拟合低层控制器代理 C(·): commanded action → measured joint state。

控制器感知训练的基础。Stage5 坐实最大瓶颈在 action→state(控制器强低通: <5Hz≈1,
>5Hz 0.11-0.20), 残差在被挡掉的 >5Hz 上白学。本脚本把这条**实测**通路拟合成一个
可微 FIR 滤波器 C, 之后训练时把 HF 平滑损失搬到 C(refined) 上算 → 残差只在透传带
(2-4.5Hz)拿梯度。

方法(防过拟合/单位歧义):
  · H1 估计器 |H(f)| = |S_xy| / S_xx, 跨所有真机 episode 累加交叉/自谱后再除
    (比平均比值稳); coherence γ²=|S_xy|²/(S_xx·S_yy) 报告可信度。
  · 只拟合臂关节 j1/j2/j3(action 与 state 同为 rad, 干净); grip 及其余通道用
    **均值臂核**(同伺服带宽假设, 避开 grip 命令/状态单位歧义)。
  · firwin2 频率采样设计 → 线性相位对称 FIR(numtaps), 居中 'same' 卷积 = 零相位
    幅频, 与训练里 filtfilt/Hann 口径一致(只在乎幅度透传)。
  · 留出 episode 验证: C(action) 预测 state 的 per-joint 相关 + 5Hz 高通能量比。

输出: outputs/sim/controller_surrogate.npz (kernels (7,numtaps)+元数据) +
      outputs/eval/controller_surrogate_fit.png (幅频 + 验证)。

用法: python sim/fit_controller_surrogate.py [--numtaps 63] [--n-val 4]
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import csd, welch, firwin2, freqz

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
REPO = Path(__file__).resolve().parents[1]
OUT_NPZ = REPO / "outputs/sim/controller_surrogate.npz"
OUT_PNG = REPO / "outputs/eval/controller_surrogate_fit.png"
FS = 30.0
NYQ = FS / 2.0
ARM = [0, 1, 2]                       # 拟合通道(rad↔rad)
MDEG_TO_RAD = math.pi / 180.0 / 1000.0
NPERSEG = 128
# 所有真机数据集共享同一物理控制器 → 全部汇集拟合 plant
DATASETS = ["eval_refine_A", "eval_refine_B", "eval_s5_A_act", "eval_s5_B_blend",
            "eval_s5_C_full", "eval_s5tok_A", "eval_s5tok_1q", "eval_s5tok_Mq",
            "eval_s5tok_RLT"]


def load_eps():
    """返回 [(action(T,7) rad, state(T,7) rad)] 跨所有可用真机数据集。"""
    eps = []
    for ds in DATASETS:
        for pq in sorted(glob.glob(str(LR / ds / "data/chunk-*/episode_*.parquet"))):
            try:
                df = pd.read_parquet(pq)
                act = np.stack(df["action"].to_numpy()).astype(np.float64)
                st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
            except (KeyError, ValueError):
                continue
            if act.shape[1] < 7 or st.shape[1] < 7:
                continue
            st = st.copy(); st[:, :6] *= MDEG_TO_RAD
            eps.append((act, st))
    return eps


def transfer_fit(eps, j):
    """H1 估计 |H(f)| + coherence, 跨 ep 累加 Welch 谱。返回 (f, |H|, coh2)。"""
    Pxx = Pyy = Pxy = None; f = None
    for act, st in eps:
        x, y = act[:, j], st[:, j]
        if len(x) < NPERSEG:
            continue
        f, pxx = welch(x, fs=FS, nperseg=NPERSEG)
        _, pyy = welch(y, fs=FS, nperseg=NPERSEG)
        _, pxy = csd(x, y, fs=FS, nperseg=NPERSEG)   # complex
        Pxx = pxx if Pxx is None else Pxx + pxx
        Pyy = pyy if Pyy is None else Pyy + pyy
        Pxy = pxy if Pxy is None else Pxy + pxy
    H = np.abs(Pxy) / (Pxx + 1e-12)
    coh2 = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
    return f, H, coh2


def design_fir(f, mag, numtaps):
    """firwin2 频率采样: 把实测幅频 mag(f) 拟成线性相位对称 FIR。"""
    fn = f / NYQ
    fn[0], fn[-1] = 0.0, 1.0
    g = np.clip(mag, 0.0, 1.5).copy()
    # DC 处用最低 2-bin 均值(去单 bin 噪声); Nyquist 处保留实测(通常已≈0)
    g[0] = float(np.clip(mag[1:3].mean(), 0.0, 1.2))
    k = firwin2(numtaps, fn, g)
    return k.astype(np.float32)


def apply_fir_np(x, k):
    """零相位幅频: 对称 FIR 居中 'same' 卷积(反射边界)。x:(T,) k:(L,)。"""
    pad = len(k) // 2
    xp = np.pad(x, pad, mode="reflect")
    return np.convolve(xp, k, mode="valid")


def hp_energy(x, b, a):
    from scipy.signal import filtfilt
    y = filtfilt(b, a, x)
    return float((y ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numtaps", type=int, default=63)
    ap.add_argument("--n-val", type=int, default=6, help="留出验证 episode 数")
    args = ap.parse_args()

    eps = load_eps()
    print(f"[data] {len(eps)} real episodes pooled from {len(DATASETS)} datasets")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    val_eps = [eps[i] for i in order[:args.n_val]]
    fit_eps = [eps[i] for i in order[args.n_val:]]
    print(f"[split] fit={len(fit_eps)}  val={len(val_eps)}")

    # ---- 拟合每臂关节 + 设计核 ----
    kernels = np.zeros((7, args.numtaps), np.float32)
    mags = {}; cohs = {}; f_ref = None
    arm_kernels = []
    for j in ARM:
        f, H, coh2 = transfer_fit(fit_eps, j)
        f_ref = f
        k = design_fir(f, H, args.numtaps)
        kernels[j] = k; arm_kernels.append(k)
        mags[j] = H; cohs[j] = coh2
        # 带均值报告
        def band(lo, hi, arr):
            m = (f >= lo) & (f < hi); return float(arr[m].mean()) if m.any() else float("nan")
        print(f"  j{j+1}: |H| 0-2={band(0,2,H):.2f} 2-4.5={band(2,4.5,H):.2f} "
              f"5-10={band(5,10,H):.2f} 10-15={band(10,15,H):.2f} | "
              f"coh² 2-4.5={band(2,4.5,coh2):.2f} 5-15={band(5,15,coh2):.2f}")
    avg_arm = np.mean(arm_kernels, axis=0).astype(np.float32)
    for j in [3, 4, 5, 6]:                 # 其余通道用均值臂核
        kernels[j] = avg_arm

    # ---- 验证: C(action) 预测 state ----
    from scipy.signal import butter
    b5, a5 = butter(4, 5.0 / NYQ, "high")
    print("\n[validation] C(action)→state vs measured state (val eps, arm joints):")
    print(f"  {'joint':>6} {'corr(pred,meas)':>16} {'corr(act,meas)':>16} "
          f"{'HP5 C(act)/act':>16} {'HP5 meas/act':>14}")
    val_report = {}
    for j in ARM:
        cp, ca, hpc, hpm, hpa = [], [], 0.0, 0.0, 0.0
        for act, st in val_eps:
            x, y = act[:, j], st[:, j]
            xc = apply_fir_np(x, kernels[j])
            # 去均值后相关(关注波形, 非 DC)
            def dm(v): return v - v.mean()
            cp.append(np.corrcoef(dm(xc), dm(y))[0, 1])
            ca.append(np.corrcoef(dm(x), dm(y))[0, 1])
            hpc += hp_energy(xc, b5, a5); hpm += hp_energy(y, b5, a5); hpa += hp_energy(x, b5, a5)
        r = dict(corr_pred=float(np.nanmean(cp)), corr_act=float(np.nanmean(ca)),
                 hp_Cact_over_act=float(hpc / (hpa + 1e-12)),
                 hp_meas_over_act=float(hpm / (hpa + 1e-12)))
        val_report[f"j{j+1}"] = r
        print(f"  {'j'+str(j+1):>6} {r['corr_pred']:>16.3f} {r['corr_act']:>16.3f} "
              f"{r['hp_Cact_over_act']:>16.3f} {r['hp_meas_over_act']:>14.3f}")
    print("  判读: HP5 C(act)/act 应≈ HP5 meas/act(代理把命令>5Hz 削到与真机执行态相当)")

    # ---- 保存 ----
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, kernels=kernels, numtaps=args.numtaps, fs=FS,
             arm_joints=np.array(ARM), freq=f_ref,
             mag_j1=mags[0], mag_j2=mags[1], mag_j3=mags[2],
             coh_j1=cohs[0], coh_j2=cohs[1], coh_j3=cohs[2])
    (OUT_NPZ.with_suffix(".json")).write_text(json.dumps(
        {"numtaps": args.numtaps, "fs": FS, "n_fit": len(fit_eps), "n_val": len(val_eps),
         "validation": val_report,
         "note": "kernels (7,L): ch0-2 fitted per-joint, ch3-6 = mean arm kernel"}, indent=2))
    print(f"\n[saved] {OUT_NPZ}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        # 左: 实测 |H| + 拟合核幅频
        for j in ARM:
            axes[0].plot(f_ref, mags[j], alpha=0.4, label=f"meas |H| j{j+1}")
        wn, hh = freqz(avg_arm, worN=512, fs=FS)
        for j in ARM:
            wj, hj = freqz(kernels[j], worN=512, fs=FS)
            axes[0].plot(wj, np.abs(hj), "--", lw=1.5, label=f"FIR j{j+1}")
        axes[0].axvspan(2, 4.5, color="orange", alpha=0.12, label="2-4.5Hz 透传带")
        axes[0].axvline(5, color="r", ls=":", lw=1)
        axes[0].set(xlabel="Hz", ylabel="|H(f)|  (action→state gain)",
                    title="控制器代理: 实测传递幅频 vs 拟合 FIR", xlim=(0, 15), ylim=(0, 1.3))
        axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)
        # 右: coherence
        for j in ARM:
            axes[1].plot(f_ref, cohs[j], label=f"coh² j{j+1}")
        axes[1].axvspan(2, 4.5, color="orange", alpha=0.12)
        axes[1].axvline(5, color="r", ls=":", lw=1)
        axes[1].set(xlabel="Hz", ylabel="coherence²", title="action↔state 相干(拟合可信度)",
                    xlim=(0, 15), ylim=(0, 1))
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
        print(f"[saved] {OUT_PNG}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
