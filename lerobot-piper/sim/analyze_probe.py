#!/usr/bin/env python3
"""③a 探针分析（2026-07-27, 脉冲池版）— 从复放算真机增益 g, 与 Ĝ 预测对账。

支持**跨集合并**: 三集 pp2 各自 A1/B/A2, 帧号独立 → 在**脉冲层面**汇集后按关节符号平均。
    每发脉冲抽 LR 帧窗:  s = Δ_real[t:t+LR]   p = Δ_model[t:t+LR]   n = (A2−A1)[t:t+LR]
    按关节聚合(跨集、按符号翻正):  s_j = mean(sign·s)  等
    g = Σ_j⟨s_j,p_j⟩ / Σ_j‖p_j‖²      cos/snr/asym 同法

预注册判据(协议 ③a): ①信噪>3 ②cos>0.6(不过→停,回查Ĝ) ③g→按§8选β
    ④线性 asym<max(0.3, 2×噪声预期)(不过→增量诀窍前提动摇)
接触帧上的脉冲自动剔除(insert/pp2 插入接触会污染)。

用法:
  python sim/analyze_probe.py --multi                # pp2 三集(ep0/1/2)合并
  python sim/analyze_probe.py                        # 单集(probe_seq.npz + replay_{A1,B,A2})
  python sim/analyze_probe.py --self-test            # 无机时自检(注入已知 g)
  python sim/analyze_probe.py --noise-sweep          # 噪声裕度扫描
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs/probe"
FIG = ROOT / "reports/figures"
LR = 24                      # v2: 8帧小波 + 16帧尾巴
LABELS = ROOT / "outputs/p2_labels/pp2-deploy"


def load_arm(name):
    d = json.loads((PROBE / f"replay_{name}.json").read_text())
    return np.array(d["imu"], np.float32)[:, 3:6]


def synth(seq, g_true=1.35, noise=0.004, seed=0, reps=1):
    rng = np.random.default_rng(seed)
    pred = seq["pred"][0]
    T = len(pred)
    base = rng.normal(0, 0.25, (T, 3)).cumsum(0) * 0.02
    n = lambda: rng.normal(0, noise, (T, 3)) / np.sqrt(reps)
    return base + n(), base + g_true * pred + n(), base + n()


def extract_pulses(A1, B, A2, seq, drop_contact_ep=None):
    """一集 → 脉冲列表。每发: {j, sign, s(LR,3), p(K,LR,3), n(LR,3)}。"""
    T = min(len(A1), len(B), len(A2), seq["pred"].shape[1])
    d_real = (B - 0.5 * (A1 + A2))[:T]
    noise = (A2 - A1)[:T]
    pred = seq["pred"][:, :T]
    hits = seq["hits"]
    hits = hits[(hits[:, 0] >= 0) & (hits[:, 0] < T - LR)]
    contact = None
    if drop_contact_ep is not None:
        try:
            contact = np.load(LABELS / f"ep{drop_contact_ep:03d}.npz")["mask_contact"]
        except Exception:
            pass
    P = []
    for t, j, s in hits:
        if contact is not None and t < len(contact) and contact[t]:
            continue                                     # 接触帧剔除
        P.append({"j": int(j), "sign": int(s),
                  "s": d_real[t:t + LR], "p": pred[:, t:t + LR], "n": noise[t:t + LR]})
    return P


def analyze(runs, verbose=True):
    """runs: [(A1,B,A2,seq,ep), ...]。单集就是长度 1。"""
    pulses = []
    for A1, B, A2, seq, ep in runs:
        pulses += extract_pulses(A1, B, A2, seq, drop_contact_ep=ep)
    assert pulses, "无可用脉冲(全落在接触帧?)"
    K = pulses[0]["p"].shape[0]
    joints = sorted(set(p["j"] for p in pulses))

    def savg(key, j, member=None, signed=True):
        sel = [p for p in pulses if p["j"] == j]
        arr = []
        for p in sel:
            w = p[key][member] if member is not None else p[key]
            arr.append(w * (p["sign"] if signed else 1))
        return np.mean(arr, 0)

    out = {"per_joint": {}, "n_pulses": len(pulses), "n_joints": len(joints)}
    gs_member = []
    for k in range(K):
        nu = de = 0.0
        for j in joints:
            s, p = savg("s", j), savg("p", j, member=k)
            nu += float((s * p).sum())
            de += float((p ** 2).sum())
        gs_member.append(nu / max(de, 1e-12))
    out["g"] = [float(np.mean(gs_member)), float(np.std(gs_member))]
    out["g_members"] = [float(x) for x in gs_member]
    out["g_m0"] = float(gs_member[0])

    snrs, coss, asyms, nrs = [], [], [], []
    for j in joints:
        s, p, n = savg("s", j), savg("p", j, member=0), savg("n", j)
        snr = float(np.linalg.norm(s) / max(np.linalg.norm(n), 1e-12))
        cos = float((s * p).sum() / max(np.linalg.norm(s) * np.linalg.norm(p), 1e-12))
        # 奇对称: 不翻正地按符号分组求和(应≈0), 与噪声预期比
        acc, cnt = {}, {}
        for sgn in (1, -1):
            sub = [pp for pp in pulses if pp["j"] == j and pp["sign"] == sgn]
            if sub:
                acc[sgn] = np.mean([pp["s"] for pp in sub], 0)
                cnt[sgn] = len(sub)
        if len(acc) == 2:
            asym = float(np.linalg.norm(acc[1] + acc[-1]) / max(np.linalg.norm(acc[1]), 1e-12))
            sig1 = float(np.sqrt(np.mean([pp["n"] ** 2 for pp in pulses])))
            exp_num = sig1 * np.sqrt(1 / cnt[1] + 1 / cnt[-1]) * np.sqrt(LR * 3)
            asym_exp = float(exp_num / max(np.linalg.norm(acc[1]), 1e-12))
        else:
            asym = asym_exp = float("nan")
        # 范数比：g = cos × 范数比。必须**分开报** —— 方向差配上范数低估可以
        # 乘出一个「看着正常」的 g≈1（用户 2026-09-02 订正）。
        nr = float(np.linalg.norm(s) / max(np.linalg.norm(p), 1e-12))
        snrs.append(snr); coss.append(cos); asyms.append((asym, asym_exp)); nrs.append(nr)
        out["per_joint"][f"j{j+1}"] = {
            "n": sum(1 for p in pulses if p["j"] == j), "snr": snr, "cos": cos,
            "norm_ratio": nr,
            "asym": asym, "asym_noise_expected": asym_exp,
            "g": float((s * p).sum() / max((p ** 2).sum(), 1e-12)),
            "peak_real_mrad": float(np.abs(s).max() * 1e3),
            "peak_model_mrad": float(np.abs(p).max() * 1e3)}
    out["snr"] = float(np.mean(snrs))
    out["cos"] = float(np.mean(coss))
    out["norm_ratio"] = float(np.mean(nrs))
    out["noise_rms_mrad"] = float(np.sqrt(np.mean([p["n"] ** 2 for p in pulses])) * 1e3)
    nonlin = [a > max(0.3, 2 * e) for a, e in asyms if a == a]
    out["pass"] = bool(out["snr"] > 3 and out["cos"] > 0.6 and not any(nonlin))

    if verbose:
        print(f"脉冲池: {out['n_pulses']} 发(剔接触后) · {out['n_joints']} 关节 · K={K} 成员")
        print(f"复放噪声底(单帧) = {out['noise_rms_mrad']:.2f} mrad/s")
        print(f"\n{'关节':>5}{'发数':>5}{'信噪':>7}{'cos':>8}{'范数比':>8}{'g':>8}"
              f"{'奇对称':>8}{'(噪声预期)':>11}{'真实峰':>9}{'模型峰':>9}")
        for jn, v in out["per_joint"].items():
            f1 = "" if v["snr"] > 3 else " ✗①"
            f4 = "" if not (v["asym"] > max(0.3, 2 * v["asym_noise_expected"])) else " ✗④"
            print(f"{jn:>5}{v['n']:>5}{v['snr']:>7.1f}{v['cos']:>8.3f}"
                  f"{v['norm_ratio']:>8.3f}{v['g']:>8.3f}"
                  f"{v['asym']:>8.3f}{v['asym_noise_expected']:>11.3f}"
                  f"{v['peak_real_mrad']:>9.1f}{v['peak_model_mrad']:>9.1f}{f1}{f4}")
        print(f"\n增益 g   = {out['g'][0]:.3f} ± {out['g'][1]:.3f}  (K 成员均值±分歧)"
              f"   [成员0: {out['g_m0']:.3f}]")
        print(f"         逐成员: {np.round(out['g_members'], 3).tolist()}")
        print(f"方向 cos = {out['cos']:.3f}   {'✓' if out['cos'] > 0.6 else '✗ 判据② — 停下回查 Ĝ'}")
        print(f"范数比   = {out['norm_ratio']:.3f}  ‖Δ_real‖/‖Δ_model‖")
        print(f"         **g = cos × 范数比**（{out['cos']:.3f} × {out['norm_ratio']:.3f}"
              f" = {out['cos']*out['norm_ratio']:.3f}）—— g 把方向与尺度混在一起，"
              f"\n         方向差 + 范数低估可以乘出一个「看着正常」的 g≈1。"
              f"**g 不进通过条件**，判据是 ①信噪 ②cos>0.6 ④正负线性。")
        print(f"信噪     = {out['snr']:.1f}    {'✓' if out['snr'] > 3 else '✗ 判据① — 加大探针/增加复放'}")
        print(f"\n预注册判据: {'✅ 全部通过 — 可进 ③b' if out['pass'] else '❌ 未通过'}")
        g = out["g"][0]
        print(f"→ 按 WORKLOG §8:  g={g:.2f} ⇒ " +
              ("换 β=100（甚至 30）" if g < 1.2 else
               "留 β=300" if g < 1.9 else "连 β=300 都需重评"))
    return out, pulses, joints


def plot(pulses, joints):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(joints), figsize=(5.2 * len(joints), 4.2), squeeze=False)
    t = np.arange(LR) * 50
    for k, j in enumerate(joints):
        ax = axes[0, k]
        sel = [p for p in pulses if p["j"] == j]
        s = np.mean([p["s"] * p["sign"] for p in sel], 0)
        p_ = np.mean([p["p"][0] * p["sign"] for p in sel], 0)
        ch = int(np.argmax(np.abs(s).max(0)))
        ax.plot(t, s[:, ch] * 1e3, color="#2563eb", lw=2.2, marker="o", ms=4,
                label="measured  $B-(A1{+}A2)/2$")
        ax.plot(t, p_[:, ch] * 1e3, color="#dc2626", lw=1.8, ls="--", marker="s", ms=3.5,
                label="$\\hat{G}$ prediction")
        ax.axhline(0, color="#9ca3af", lw=0.8)
        ax.set_title(f"j{j+1}  ({len(sel)} pulses, sign-avg, gyro-{'xyz'[ch]})",
                     fontsize=11, weight="bold")
        ax.set_xlabel("time since pulse (ms)")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("$\\Delta$ gyro (mrad/s)")
            ax.legend(fontsize=8.5)
    fig.suptitle("Act 3a — does the real robot respond to $\\delta a$ as $\\hat{G}$ says?"
                 "   ·   replay probe, 6 mrad 5 Hz wavelets (pp2, 3 episodes)",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p_out = FIG / "probe_impulse_response.png"
    fig.savefig(p_out, dpi=155)
    return p_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", action="store_true", help="pp2 三集 ep0/1/2 合并")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--noise-sweep", action="store_true")
    ap.add_argument("--g-true", type=float, default=1.35)
    a = ap.parse_args()

    if a.noise_sweep:
        seq = np.load(PROBE / "probe_seq_pp2_ep0.npz", allow_pickle=True)
        print("噪声扫描(单集 pp2-ep0, 注入 g_true=1.35):")
        print(f"{'噪声σ(mrad/s)':>15}{'复放次数':>9}{'回收g(m0)':>11}{'相对误差':>10}{'信噪':>8}")
        for sig in (0.002, 0.004, 0.008, 0.015):
            for reps in (1, 3):
                A1, B, A2 = synth(seq, a.g_true, noise=sig, reps=reps)
                o, *_ = analyze([(A1, B, A2, seq, None)], verbose=False)
                e = abs(o["g_m0"] - a.g_true) / a.g_true
                print(f"{sig*1e3:>15.0f}{reps:>9}{o['g_m0']:>11.3f}{e*100:>9.1f}%"
                      f"{o['snr']:>8.1f}{'  ✓' if e < 0.1 else '  ✗'}")
        return 0

    if a.self_test:
        print("=== 自检: 合成三集 pp2 (注入 g_true=%.2f) ===" % a.g_true)
        runs = []
        for ep in (0, 1, 2):
            seq = np.load(PROBE / f"probe_seq_pp2_ep{ep}.npz", allow_pickle=True)
            A1, B, A2 = synth(seq, g_true=a.g_true, seed=ep)
            runs.append((A1, B, A2, seq, None))          # 自检不剔接触
        out, pulses, joints = analyze(runs)
        err = abs(out["g_m0"] - a.g_true) / a.g_true
        print(f"\n自检: 成员0 回收 g={out['g_m0']:.3f} vs 注入 {a.g_true}  "
              f"误差 {err*100:.1f}%  {'✅ 多集流程正确' if err < 0.1 else '❌ 有 bug'}")
        return 0 if err < 0.1 else 1

    if a.multi:
        runs = []
        for ep in (0, 1, 2):
            seq = np.load(PROBE / f"probe_seq_pp2_ep{ep}.npz", allow_pickle=True)
            A1 = np.array(json.loads((PROBE / f"replay_ep{ep}_A1.json").read_text())["imu"],
                          np.float32)[:, 3:6]
            B = np.array(json.loads((PROBE / f"replay_ep{ep}_B.json").read_text())["imu"],
                         np.float32)[:, 3:6]
            A2 = np.array(json.loads((PROBE / f"replay_ep{ep}_A2.json").read_text())["imu"],
                          np.float32)[:, 3:6]
            runs.append((A1, B, A2, seq, ep))
    else:
        seq = np.load(PROBE / "probe_seq.npz", allow_pickle=True)
        runs = [(load_arm("A1"), load_arm("B"), load_arm("A2"), seq, 0)]
    out, pulses, joints = analyze(runs)
    print("\n" + str(plot(pulses, joints)))
    (PROBE / "probe_result.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
