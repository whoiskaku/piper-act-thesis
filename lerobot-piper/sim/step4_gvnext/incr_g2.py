#!/usr/bin/env python3
"""**测 G-2 的增量通道**：`Ĝ_δ[δa] = ω̂(u+δa) − ω̂(u)`，并与**实测 H** 对标。

为什么这是「用新 G 重新做」的第一步
------------------------------------
旧 Ĝ 训出来的 π 真机失败，根因不是训练方法，是**旧 Ĝ 的增量方向被实测判为反的**：

    旧 Ĝ 说   δa 使 gap 能量 −0.0034（减小 6.5%）
    真机配对实测              +0.0137（增大 38%）      ⇒ 符号相反，量级差 ~5 倍

而 **G-2 的增量通道从未被测过** —— 0.5–3 Hz ρ²=0.836 / 3–9 Hz 0.520 是**绝对**
保真度，对 δa 响应什么都没说。老师的判据本来就是「**Ĝ 只需增量准**」。

做法
----
在 G-2 上做**外生 ± 配对**数值扰动（探针外部生成、不读任何实测量）：

    Δ_G2 = [ω̂(u+δa) − ω̂(u−δa)] / 2        ← 取奇分量，任务基线（偶）精确抵消
    Δ_H  = causal_conv(H, δa)               ← 已验证的实测被控对象响应

对标量（口径同 `reports/PROTOCOL_robot_validation.md` ③a）：

    g   = argmin_g ‖Δ_H − g·Δ_G2‖²   = ⟨Δ_H, Δ_G2⟩ / ‖Δ_G2‖²   「真实是 G-2 的几倍」
    cos = ⟨Δ_H, Δ_G2⟩ / (‖Δ_H‖‖Δ_G2‖)                           **门槛 > 0.6**

`H` 取自 `outputs/step1_H/step1_H_v1.npz`（ab4 P1 8 集拟合，P2 跨 seed β 中位 0.9546、
P3 幅值线性 1.0134，float64 sha 与 step1-linear-v1 逐比特一致）。
它是**独立的实测标尺**，不是被判否的那些候选。

⚠ 坐标系
--------
`H` 的输出在**真机传感器系**（`gap = real[t+3] − ideal[t]`，两端都已标定到该系）；
G-2 的 `ω̂` 在**机体系**（训练 `r_s` 时真机已逐集 Procrustes 旋入机体系）。
故 cos **必须同时报 raw 与最优正交旋转后**两个值，并报出旋转角 ——
raw 差而旋转后好 = **坐标系失配**，不是模型错；两者都差才是模型错。
逐系列最优位移同样要报（1 sample = 50 ms = 3–9 Hz 的 54°–162° 相位）。

用法:
    python -u sim/step4_gvnext/incr_g2.py
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

from data import load, bp, valid_files, FPS                          # noqa: E402
from train_rq import features as rq_features, fir_design             # noqa: E402
from train_rs import rs_features                                     # noqa: E402
from eval_gate import base_q                                         # noqa: E402
from ceiling_200hz import kin_gyro_fs                                # noqa: E402

S1, S2 = ROOT / "outputs/step4_gvnext/s1", ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
H_ART = ROOT / "outputs/step1_H/step1_H_v1.npz"
OUT = ROOT / "outputs/step4_gvnext/incr_g2.json"

AMP = 0.003                     # 3 mrad 峰值（ab4 P3 同量级，ε=10 的 30%）
FREQS = np.array([0.7, 1.3, 2.1, 3.4, 5.2, 7.1, 8.6])   # 0.5–9 Hz 覆盖
COS_GATE = 0.6                  # PROTOCOL ③a 判据②
SHIFTS = range(-4, 5)


def schroeder(freqs, n, fps, seed):
    """Schroeder 低峰值因子多正弦，峰值归一到 1（与 make_probe_b2 同式）。"""
    rng = np.random.default_rng(seed)
    k = np.arange(len(freqs))
    ph = -np.pi * k * (k - 1) / len(freqs) + rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / fps
    s = np.sum(np.cos(2 * np.pi * freqs[:, None] * t[None, :] + ph[:, None]), 0)
    return s / max(np.abs(s).max(), 1e-12)


def probe(n, seed):
    """(n,6) 外生探针：逐关节独立相位，**不读任何实测量**。"""
    return np.stack([AMP * schroeder(FREQS, n, FPS, seed * 97 + j)
                     for j in range(6)], 1)


def g2_omega(dd, t20, zq, sq, zs, ss):
    """G-2 前向：ω̂ = kin(G-q-v1(u)+r_q) + r_s。"""
    qh = base_q(dd, t20) + fir_design(rq_features(dd)[0], sq["nlag"]) @ zq["W"]
    n = len(t20)
    Xs = rs_features(dd, t20, qh)[:n]
    return kin_gyro_fs(qh, FPS)[:n] + fir_design(Xs, ss["nlag"]) @ zs["W"]


def perturb(dd, da):
    """命令流 u → u+δa（只动 6 个臂关节，夹爪不动）。"""
    cmd = np.array(dd["cmd"], float, copy=True)
    m = min(len(cmd), len(da))
    cmd[:m, :6] += da[:m]
    return {**dd, "cmd": cmd}


def best_align(A, B):
    """→ (g, cos_raw, cos_rot, 旋转角°, 最优位移)。A=Δ_H(参照), B=Δ_G2。"""
    from scipy.linalg import orthogonal_procrustes
    best = None
    for s in SHIFTS:
        a, b = (A[s:], B[:len(B) - s]) if s > 0 else (A[:len(A) + s], B[-s:]) if s < 0 else (A, B)
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        c = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        if best is None or c > best[0]:
            R, _ = orthogonal_procrustes(b, a)               # B → A
            br = b @ R
            best = (c, float((a * b).sum() / max((b * b).sum(), 1e-30)),
                    float((a * br).sum() / max(np.linalg.norm(a) * np.linalg.norm(br), 1e-30)),
                    float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), s)
    return best[1], best[0], best[2], best[3], best[4]


def main() -> int:
    import gc
    gc.disable()
    from fit_step1_candidate import causal_conv
    H = np.ascontiguousarray(np.load(H_ART)["H"], dtype=np.float64)
    zq, zs = np.load(RQ / "rq_fir.npz"), np.load(RS / "rs_fir.npz")
    sq, ss = (json.loads((p / "select.json").read_text()) for p in (RQ, RS))
    print(f"G-2: r_q FIR nlag={sq['nlag']} · r_s FIR nlag={ss['nlag']}")
    print(f"标尺 H: {H_ART.name}（ab4 P1 拟合，跨 seed β 0.9546 / 幅值线性 1.0134）")
    print(f"探针: 外生 ±配对 Schroeder 多正弦 {list(FREQS)} Hz，峰值 {AMP*1e3:.0f} mrad\n")

    rows = []
    for tag, d, pat in (("训练族", S1, "replay_*.json"),
                        ("开发验证集", S2, "replay_pp2f*.json")):
        for f in valid_files(d, pat):
            dd = load(f)
            t20 = dd["tg_i"][::5]
            n = len(t20)
            da20 = probe(n, seed=abs(hash(dd["name"])) % 9973)
            # δa 在 t_read 网格上注入（命令时基），再取 20 Hz 评价
            da_cmd = np.stack([np.interp(np.asarray(dd["t_read"], float), t20, da20[:, j])
                               for j in range(6)], 1)
            wp = g2_omega(perturb(dd, da_cmd), t20, zq, sq, zs, ss)
            wm = g2_omega(perturb(dd, -da_cmd), t20, zq, sq, zs, ss)
            m = min(len(wp), len(wm), n)
            d_g2 = 0.5 * (wp[:m] - wm[:m])                  # 奇分量
            d_h = causal_conv(H, da20[:m])                  # 实测被控对象响应
            a, b = bp(d_h, 0.5, 9.0), bp(d_g2, 0.5, 9.0)
            g, c0, c1, ang, sh = best_align(a, b)
            rows.append({"name": dd["name"], "set": tag, "g": g, "cos_raw": c0,
                         "cos_rot": c1, "rot_deg": ang, "shift": sh,
                         "rms_g2": float(np.sqrt(np.mean(b ** 2))),
                         "rms_h": float(np.sqrt(np.mean(a ** 2)))})

    print(f"  {'集':<16}{'增益 g':>9}{'cos raw':>10}{'cos 旋转后':>12}"
          f"{'旋转角°':>9}{'位移':>6}{'‖Δ_G2‖':>10}{'‖Δ_H‖':>10}")
    for r in rows:
        print(f"  {r['name']:<16}{r['g']:>9.3f}{r['cos_raw']:>10.3f}{r['cos_rot']:>12.3f}"
              f"{r['rot_deg']:>9.1f}{r['shift']:>6d}{r['rms_g2']:>10.5f}{r['rms_h']:>10.5f}")

    med = lambda k, s=None: float(np.median([r[k] for r in rows if s is None or r["set"] == s]))
    print(f"\n  {'':<16}{'g 中位':>9}{'cos raw':>10}{'cos 旋转后':>12}{'旋转角°':>9}")
    for s in ("训练族", "开发验证集"):
        print(f"  {s:<16}{med('g',s):>9.3f}{med('cos_raw',s):>10.3f}"
              f"{med('cos_rot',s):>12.3f}{med('rot_deg',s):>9.1f}")
    print(f"  {'全体':<16}{med('g'):>9.3f}{med('cos_raw'):>10.3f}"
          f"{med('cos_rot'):>12.3f}{med('rot_deg'):>9.1f}")

    c = med("cos_rot")
    print(f"\n  ── 判据（PROTOCOL ③a②，事前定死）──")
    print(f"  方向 cos（旋转后中位）= {c:.3f}   门槛 > {COS_GATE}   "
          f"{'✓ **G-2 的增量方向可信**' if c > COS_GATE else '**✗ 根基有问题，先查 G-2 的增量通道**'}")
    print(f"  增益 g 中位 = {med('g'):.3f}（报告值，不设通过线；>1 = G-2 低估）")
    if med("cos_raw") < COS_GATE <= c:
        print(f"  注：raw {med('cos_raw'):.3f} 差而旋转后 {c:.3f} 好"
              f"（旋转 {med('rot_deg'):.1f}°）⇒ **坐标系失配，不是模型错**")
    OUT.write_text(json.dumps({
        "rows": rows, "cos_gate": COS_GATE,
        "median": {k: med(k) for k in ("g", "cos_raw", "cos_rot", "rot_deg")},
        "probe": {"amp_rad": AMP, "freqs_hz": FREQS.tolist(),
                  "form": "exogenous +/- paired Schroeder multisine, per-joint phase"},
        "H_artifact": str(H_ART),
        "caliber": ("Δ_G2 取 ±配对奇分量(任务基线偶分量精确抵消); Δ_H = causal_conv(实测H, δa); "
                    "带通 0.5-9Hz; 逐集最优位移; cos 同时报 raw 与最优正交旋转后 —— "
                    "H 在真机传感器系、G-2 在机体系，raw 差而旋转后好属坐标系失配非模型错。"),
    }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
