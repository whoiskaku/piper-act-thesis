#!/usr/bin/env python3
"""Ĝ 的带限校准（2026-07-28，B′ 三轮 z(f) 数据）。

**为什么是「扩不确定度」而不是「改中心值」**

三轮 18 条复放测出的核心事实不是「Ĝ 增益错了」，而是：

    Ĝ 在两个位形上预测的 6-8Hz 幅值几乎相同(~11.5 vs ~13 mrad/s)，
    而真机相差 3-4 倍(58-78 vs 11-21) —— **Ĝ 没有捕捉响应的位形依赖性**。

Ĝ 的柔性模态是一次辨识、位形无关的；刚体路径虽用解析 Jacobian 但在 6-8Hz 不主导。
我们只有 **2 个位形**(w170: j2=84.8° 臂近水平最柔; w280: j2=37.3° 更折叠更刚)，
**无法拟合位形依赖模型**。所以不能给出点校正 —— 任意时刻该用哪个因子是未知的。

诚实的做法: 在 6-8Hz 把 Ĝ 的不确定度扩到覆盖实测的 |z| 区间, 让下游
(π 的"均值训练/最差成员评估" 与 ε 界)看到真实的不确定度。

**适用边界(硬性)**
  · 只在 6.0-7.5 Hz 实测过。此带之外**没有任何信息**, 不得外推
  · 只有 2 个位形。位形依赖已确立(三轮 4/4 频点 CI 分离)但**无法参数化**
  · 方向残差 0.12-0.55 非零 ⇒ 纯标量缩放覆盖不了, 另给正交分量余量
  · 幅值无关已验证(4 vs 2 mrad 的 |z| CI 全重叠) ⇒ 线性修正可跨幅值迁移

用法:
  python sim/calib_G_band.py              # 出校准产物 + 留一工作点验证
  python sim/calib_G_band.py --report     # 只打印, 不写文件
"""
from __future__ import annotations
import argparse
import io
import contextlib
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_b2/calib_G_band.json"
ROUNDS = [(1, "4mrad"), (2, "4mrad"), (3, "2mrad")]
KEYS = ["j2_w170", "j2_w280", "j3_w170"]


def pooled_z(nboot=600):
    """三轮 z(f) 逆方差加权合并。返回 {key: {z_abs, sd, arg, resid}}。"""
    Z = {}
    for rnd, amp in ROUNDS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Z[rnd] = A.zwin(0, amp, rnd, nboot=nboot)
    out = {}
    for k in KEYS:
        rs = [r for r, _ in ROUNDS if k in Z[r]]
        za = np.array([Z[r][k]["z_abs"] for r in rs])
        ci = np.array([Z[r][k]["ci_abs"] for r in rs])
        sd = np.maximum((ci[:, 1, :] - ci[:, 0, :]) / 3.92, 1e-6)
        w = 1.0 / sd ** 2
        out[k] = {
            "n_round": len(rs),
            "z_abs": ((za * w).sum(0) / w.sum(0)).tolist(),
            "sd": (1.0 / np.sqrt(w.sum(0))).tolist(),
            "arg_deg": np.array([Z[r][k]["z_arg_deg"] for r in rs]).mean(0).tolist(),
            "resid_frac": np.array([Z[r][k]["resid_frac"] for r in rs]).mean(0).tolist(),
            "per_round_z": za.tolist(),
        }
    return out


def envelope(P, keys, kover=2.0):
    """由若干工作点建包络: [min(z−kσ), max(z+kσ)] 逐频点, 再按关节合并。"""
    env = {}
    for jn in ("j2", "j3"):
        ks = [k for k in keys if k.startswith(jn)]
        if not ks:
            continue
        lo = np.min([np.array(P[k]["z_abs"]) - kover * np.array(P[k]["sd"]) for k in ks], 0)
        hi = np.max([np.array(P[k]["z_abs"]) + kover * np.array(P[k]["sd"]) for k in ks], 0)
        env[jn] = {"lo": np.maximum(lo, 0.0).tolist(), "hi": hi.tolist(),
                   "resid_max": float(max(max(P[k]["resid_frac"]) for k in ks)),
                   "from": ks}
    return env


def loeo(P):
    """留一工作点: 用其余点建包络, 检查是否覆盖留出点。"""
    print("\n=== 留一工作点验证 (LOEO) ===")
    print("  用其余工作点建的包络, 能否覆盖留出的那个?")
    print(f"  {'留出':>10}{'其余建的包络(该关节)':>26}{'留出点 z±2σ':>26}{'覆盖':>8}")
    allok = True
    for k in KEYS:
        rest = [x for x in KEYS if x != k]
        jn = k[:2]
        env = envelope(P, rest)
        if jn not in env:                     # 该关节没有别的工作点 ⇒ 无法验证
            print(f"  {k:>10}{'—(该关节仅此一个工作点)':>26}{'':>26}{'n/a':>8}")
            continue
        lo, hi = np.array(env[jn]["lo"]), np.array(env[jn]["hi"])
        z, s = np.array(P[k]["z_abs"]), np.array(P[k]["sd"])
        cov = ((z + 2 * s >= lo) & (z - 2 * s <= hi)).mean()
        allok &= cov == 1.0
        print(f"  {k:>10}  [{lo.min():.2f},{hi.max():.2f}]{'':>12}"
              f"  [{(z-2*s).min():.2f},{(z+2*s).max():.2f}]{'':>10}"
              f"{cov*100:>7.0f}%")
    print(f"  ⇒ {'✓ 全部覆盖' if allok else '⚠ 有未覆盖 —— 包络对新位形不保证外推'}")
    print(f"  注意: j2 只有 2 个工作点、j3 只有 1 个 ⇒ LOEO 的说服力有限,")
    print(f"        它**不能**证明包络覆盖未测过的位形。")
    return allok


def config_dependence():
    """诊断: 真机 vs Ĝ 的位形依赖比 —— 校准形式的根据。"""
    import train_pi_e2 as TP
    from make_probe_b2 import NEXC
    tg = np.arange(0, A.NANA / A.FPS, 0.01)
    RN = [(1, "4mrad", "",), (2, "4mrad", "r2_"), (3, "2mrad", "r3_")]
    res = {}
    for jn, w in [("j2", 170), ("j2", 280), ("j3", 170)]:
        real, mod = [], []
        for rnd, amp, rt in RN:
            seq = np.load(A.PB / f"seq_ep0_{jn}_{amp}_p.npz", allow_pickle=True)
            if w not in [int(x) for x in seq["wins"]]:
                continue
            F = np.asarray(seq["freqs"], float)
            R = {kk: json.loads((A.PB / f"replay_ep0_{rt}{jn}_{amp}_{s}.json").read_text())
                 for kk, s in [("Bp", "p"), ("Bm", "m")]}
            W = {kk: A.arm_windows(R[kk], seq, +1 if kk == "Bp" else -1) for kk in R}
            LI = lambda kk, f: A.lockin_at(W[kk][w]["y"], W[kk][w]["t"], f, W[kk][w]["dt"])
            sc = 4.0 / float(amp[:-4])
            real.append(np.array([np.linalg.norm((LI("Bp", f) - LI("Bm", f)) / 2)
                                  for f in F]) * sc)
            pr = seq["pred"]
            mg = np.array([[np.linalg.norm(A.lockin_at(
                A._bl_eval(pr[k], w / A.FPS + tg), tg, f)) for f in F]
                for k in range(pr.shape[0])])
            mod.append(np.median(mg, 0) * sc)
        res[f"{jn}_w{w}"] = {"real": np.mean(real, 0).tolist(),
                             "model": np.mean(mod, 0).tolist()}
    z = np.load(TP.LROOT / "pp2-deploy" / "ep000.npz")
    a = z["a_arm"].astype(float)
    cfg = {f"w{w}": np.degrees(a[w:w + NEXC].mean(0)).round(1).tolist() for w in (170, 280)}
    print("\n=== 位形依赖诊断 (校准形式的根据) ===")
    print(f"  位形(窗内均值 °): w170 = {cfg['w170'][:3]}...  w280 = {cfg['w280'][:3]}...")
    r1 = np.mean(res["j2_w170"]["real"]); r2 = np.mean(res["j2_w280"]["real"])
    m1 = np.mean(res["j2_w170"]["model"]); m2 = np.mean(res["j2_w280"]["model"])
    print(f"  j2 真机 w170/w280 = {r1/r2:.2f}×   而 Ĝ 预测 w170/w280 = {m1/m2:.2f}×")
    print(f"  ⇒ **Ĝ 少了 {(r1/r2)/(m1/m2):.1f} 倍的位形依赖** —— 这才是失配的形态,")
    print(f"     不是'增益错了'。位形无关的模态参数无法表达它。")
    return res, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="只打印, 不写文件")
    a = ap.parse_args()
    P = pooled_z()
    F = [6.0, 6.5, 7.0, 7.5]

    print("=== 三轮合并的 z(f) (逆方差加权) ===")
    print(f"  {'工作点':>10}{'轮':>4}" + "".join(f"{f'{f}Hz':>13}" for f in F)
          + f"{'方向残差max':>12}")
    for k in KEYS:
        z, s = np.array(P[k]["z_abs"]), np.array(P[k]["sd"])
        print(f"  {k:>10}{P[k]['n_round']:>4}"
              + "".join(f"{zz:>8.2f}±{ss:.2f}" for zz, ss in zip(z, s))
              + f"{max(P[k]['resid_frac']):>12.2f}")

    res, cfg = config_dependence()
    env = envelope(P, KEYS)
    print("\n=== 校准产物: 6.0-7.5Hz 的乘性不确定度包络 ===")
    print("  (中心值**不动** —— 位形依赖无法参数化; 只把区间扩到覆盖实测)")
    for jn, e in env.items():
        print(f"  {jn}: |z| ∈ " + "  ".join(
            f"{f}Hz [{l:.2f},{h:.2f}]" for f, l, h in zip(F, e["lo"], e["hi"])))
        print(f"      方向残差余量 {e['resid_max']:.2f}  (来自 {e['from']})")
    ok = loeo(P)

    art = {"band_hz": [6.0, 7.5], "freqs_hz": F,
           "pooled_z": P, "envelope": env, "loeo_all_covered": bool(ok),
           "config_deg": cfg, "real_vs_model": res,
           "validity": {
               "identified_band_only": "6.0-7.5 Hz; 带外无任何信息, 不得外推",
               "n_operating_points": {"j2": 2, "j3": 1},
               "center_not_shifted": "位形依赖无法参数化 ⇒ 只扩区间, 不改中心",
               "amplitude_invariant": "4 vs 2 mrad 的 |z| CI 全重叠 ⇒ 线性修正可跨幅值",
               "direction_residual": "0.12-0.55 非零 ⇒ 标量缩放不足, 另给正交余量",
               "loeo_caveat": "工作点太少, LOEO 不能证明覆盖未测位形"}}
    if not a.report:
        OUT.write_text(json.dumps(art, indent=1, ensure_ascii=False))
        print(f"\n→ {OUT.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
