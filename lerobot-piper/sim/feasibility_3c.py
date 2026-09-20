#!/usr/bin/env python3
"""③c 离线可行性检查（**零机时**）—— 「π 式补正」的真实增量响应能不能测得到？

本轮要回答的**唯一**一句话（用户 2026-09-20 定）：

> 对同一条名义指令，加入**策略实际会产生的补正**后，机器人究竟怎么变？
> **冻结的 G-2 和 H 能预测多少？**

本脚本**不上机**，只回答机时预算所依赖的三件事：

    ① 幅度    —— 冻结 π 实际产生多大的 δa？分布在 chunk 接缝还是内部？
    ② 限幅链路 —— u+δa 会不会撞关节限位 / 被限幅器非对称削掉？
    ③ 信噪    —— 用 ③a 实测的**固定指令复放噪声底**，要多少次重复才够？

⚠ **①②③ 全部只是可行性筛查，不判定 ③c 本身**（与 ③a 的 `feasibility.json` 同性质）。

复用（不重造轮子）
------------------
`③a` 已经把这套台架在真机上跑通过（13 次复放，`RESULT_3a.md`）：

| 件 | 复用程度 |
|---|---|
| `sim/replay_probe.py` | **直接用** —— 开环固定指令复放 + IMU + 墙钟时间戳 + 安全断言 + `--dry-run` |
| `sim/analyze_probe_3a.py` | **改一处** —— ③a 按「脉冲对齐叠加」算信噪，π 波形是**连续**的，要改成整段 |
| `±` 跨复放配对 · `(R,L)` 只由独立 `A0` 定死 | **原样继承**（见下「③a 的三个已知坑」） |
| `sim/make_probe_sequence.py` | **不复用** —— 它生成单帧脉冲；③c 要的是 π 的波形 |

### ⚠ ③a 的三个已知坑（③c 必须继承其解法，别重犯）

1. **`±` 必须跨复放配对，不能跨时间**：同一关节逐发的模型响应范数差 4×(j2)–27×(j3)
   （每发落在不同臂位形）⇒ 跨时间配对会被位形变化主导，判据④ 测不到符号非线性。
   解法：`_p`/`_m` 两条序列**帧号完全相同**，分别由不同复放跑。
2. **`(R, L)` 只能由独立的 `A0` 定，之后全程固定**：自由旋转会把 `cos` 变成自由参数
   （③a 实测同一份数据：冻结 R 给 0.084，自由旋转给 0.412）。
3. **`g = cos × 范数比` 会骗人**：方向不对时大幅低估范数也能乘出 `g≈1`
   ⇒ **波形 / cos / 范数比 / g 四者分开报**。

用法:
    python -u sim/feasibility_3c.py [--base outputs/ab4/P0_ep0.json] [--repeats 4,6,8,10]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PI_CKPT = ROOT / "outputs/p2_pi/pi_distill_v2.pt"
H_NPZ = ROOT / "outputs/step1_H/step1_H_v1.npz"
H_JSON = ROOT / "outputs/step1_H/step1_H_v1.json"
P3A = ROOT / "outputs/probe_3a"
OUT = ROOT / "outputs/probe_3c/feasibility_3c.json"
EPS_B = 0.010
CHUNK = 100
SEAM_W = 4          # chunk 头 4 帧 = 「接缝」（墙能量 52.2% 在这里）
SNR_GATE = 3.0
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


# ══════════════════════════════════════════════════════════════════════
#  ③ 噪声底：直接取 ③a 的裸复放 A1 / A2（**实测**，不是假设）
# ══════════════════════════════════════════════════════════════════════
def replay_sigma():
    """σ(单次复放) —— 固定指令下两次裸复放之差给出的**经验**噪声。

    与 `analyze_probe_3a.py` 同口径：`σ = RMS(m1−m2) / sqrt(1/E1 + 1/E2)`，
    此处逐 repeat 直接用单对 (E1=E2=1) ⇒ `σ = RMS(A1−A2)/sqrt(2)`。
    """
    rows, sig = [], []
    for r in sorted(P3A.glob("r*")):
        f1, f2 = r / "replay_A1.json", r / "replay_A2.json"
        if not (f1.exists() and f2.exists()):
            continue
        a1 = np.asarray(json.loads(f1.read_text())["imu"], float)[:, 3:6]
        a2 = np.asarray(json.loads(f2.read_text())["imu"], float)[:, 3:6]
        n = min(len(a1), len(a2))
        s = float(np.sqrt(np.mean((a1[:n] - a2[:n]) ** 2)) / np.sqrt(2))
        rows.append({"rep": r.name, "n": n, "sigma": s}); sig.append(s)
    return float(np.mean(sig)), rows


# ══════════════════════════════════════════════════════════════════════
#  ① 幅度：冻结 π 在给定名义指令上实际产生的 δa
# ══════════════════════════════════════════════════════════════════════
def pi_delta(base_json: Path):
    """跑冻结 π（`pi-distill-v2`）→ 整段 δa。**强制 δa_j4 = 0**（本轮范围限制）。"""
    import torch
    from train_pi_g2env import PiNet, build_inputs
    pay = json.loads(base_json.read_text())
    chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
    n = len(pay["frames"])
    u = np.zeros((max(int(c["start"]) + len(c["raw"]) for c in chunks), 7), np.float32)
    for c in chunks:
        s0 = int(c["start"]); r = np.asarray(c["raw"], np.float32)
        u[s0:s0 + len(r)] = r
    starts = np.asarray([int(c["start"]) for c in chunks])
    x, spans = build_inputs(u, starts, n)

    net = PiNet()
    ck = torch.load(PI_CKPT, map_location="cpu", weights_only=False)   # 自产件
    net.load_state_dict(ck["state_dict"])
    net.eval()
    # ⚠ **必须施加训练时的输入归一化** —— checkpoint 里带 norm_mu/norm_sd。
    #   漏掉它我第一版算出的 δa RMS 只有 0.115 mrad（真机实测是 0.65），
    #   机时预算会被整体算错。口径与 `pi_net_runtime.py:153` 逐字相同。
    mu = np.asarray(ck["norm_mu"], np.float32)
    sd_ = np.asarray(ck["norm_sd"], np.float32)
    with torch.no_grad():
        dch = net(torch.from_numpy(((x - mu) / sd_).astype(np.float32))).numpy()
    da = np.zeros((n, 6))
    seam = np.zeros(n, bool)
    for i, (s0, lc) in enumerate(spans):
        da[s0:s0 + lc] = dch[i, :lc]
        seam[s0:min(s0 + SEAM_W, n)] = True
    da[:, 3] = 0.0                                       # ⚠ 本轮固定 δa_j4 = 0
    return u[:n], da, seam, starts


# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/ab4/P0_ep0.json")
    ap.add_argument("--repeats", default="2,3,4,6,8,10")
    a = ap.parse_args()
    reps = [int(x) for x in a.repeats.split(",")]

    import fit_step1_candidate as S1
    H = np.ascontiguousarray(np.load(H_NPZ)["H"], np.float64)
    base = ROOT / a.base
    u, da, seam, starts = pi_delta(base)
    T = len(da)
    print(f"基准轨迹 {base.name}   {T} 帧 · {len(starts)} 个 chunk · "
          f"π = {PI_CKPT.name}（冻结）")

    # ── ① 幅度 ───────────────────────────────────────────────────────
    print("\n" + "=" * 72 + "\n① 幅度：冻结 π 实际产生的 δa\n" + "=" * 72)
    rms = np.sqrt((da ** 2).mean(0)) * 1e3
    print("  逐关节 RMS(mrad) " + " · ".join(f"{j} {v:.3f}" for j, v in zip(JOINTS, rms)))
    print(f"  全体 RMS {np.sqrt((da**2).mean())*1e3:.3f} mrad · "
          f"峰值 {np.abs(da).max()*1e3:.3f} mrad（ε = {EPS_B*1e3:.0f} mrad）")
    e_seam = float((da[seam] ** 2).sum()); e_tot = float((da ** 2).sum())
    print(f"  **接缝（每 chunk 头 {SEAM_W} 帧，占 {seam.mean()*100:.1f}% 的帧）"
          f"承载 {e_seam/e_tot*100:.1f}% 的 δa 能量**")
    print(f"  ⚠ δa_j4 已强制为 0（本轮范围限制，**不是**断言 H 无 j4 信息）")
    print(f"\n  对照 ③a：单帧脉冲 5 mrad。**π 的 RMS 小 {5.0/(np.sqrt((da**2).mean())*1e3):.1f}× **"
          f" ⇒ 信噪是本轮的主要风险。")

    # ── ② 限幅链路 ───────────────────────────────────────────────────
    print("\n" + "=" * 72 + "\n② 限幅链路\n" + "=" * 72)
    hit_eps = float(np.mean(np.abs(da) >= EPS_B * 0.999) * 100)
    print(f"  触 ε 界的帧比例 {hit_eps:.3f}%  "
          f"{'✓ 不触界 ⇒ ± 严格反对称' if hit_eps == 0 else '**⚠ 触界会破坏 ± 反对称**'}")
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/piper_model.xml"))
        lo, hi = m.jnt_range[:6, 0] - 0.035, m.jnt_range[:6, 1] + 0.035
        viol = {}
        for sgn in (+1, -1):
            c = u[:, :6] + sgn * da
            v = int(((c < lo) | (c > hi)).sum())
            viol[f"{sgn:+d}"] = v
        base_v = int(((u[:, :6] < lo) | (u[:, :6] > hi)).sum())
        print(f"  关节限位越界帧·关节数：裸 u {base_v} · u+δa {viol['+1']} · u−δa {viol['-1']}")
        ok_lim = viol["+1"] == base_v and viol["-1"] == base_v
        print(f"  {'✓' if ok_lim else '**⚠**'} δa 未引入**新的**越界"
              + ("" if ok_lim else "  ⇒ 该基准轨迹须换或须做逐关节掩码"))
    except Exception as ex:
        print(f"  ⚠ 限位检查跳过（{type(ex).__name__}: {ex}）—— **上机前必须补**")
        ok_lim = None
    print("  ⚠ ③a 的教训：j2/j3 在 pp2 上**任务已占满加速度预算**，6 mrad 脉冲必被"
          "非对称削波（5 个幅度判据④ 全部 0.46–0.91）。")
    print(f"     π 的 j2/j3 RMS 只有 {rms[1]:.3f}/{rms[2]:.3f} mrad，比那时小一个量级，"
          "**但削波器本身未建模** ⇒ 见下「未决」。")

    # ── ③ 信噪 ───────────────────────────────────────────────────────
    print("\n" + "=" * 72 + "\n③ 信噪（噪声底取 ③a 的**实测**裸复放）\n" + "=" * 72)
    sigma, rows = replay_sigma()
    for r in rows:
        print(f"    {r['rep']}: σ = {r['sigma']:.5f} rad/s   ({r['n']} 帧)")
    print(f"  σ(单次复放) = **{sigma:.5f} rad/s**（{len(rows)} 对裸复放，固定指令）")

    pred = S1.causal_conv(H, da)                 # 冻结 H 对 π 波形的预测响应
    sig_norm = float(np.linalg.norm(pred))
    print(f"  冻结 H 预测的 ‖Δ_odd‖ = {sig_norm:.5f}   "
          f"(RMS {np.sqrt((pred**2).mean())*1e3:.3f} mrad/s)")

    print(f"\n  {'每侧复放数 N':>12}{'奇分量噪声':>14}{'预测信噪':>12}   判（门槛 {SNR_GATE:g}）")
    tab = []
    for N in reps:
        n_odd = sigma * 0.5 * np.sqrt(1 / N + 1 / N)
        snr = sig_norm / (n_odd * np.sqrt(3 * T))
        tab.append({"N_per_side": N, "n_odd": n_odd, "snr": snr,
                    "pass": bool(snr > SNR_GATE)})
        print(f"  {N:>12}{n_odd:>14.6f}{snr:>12.2f}   "
              f"{'✓' if snr > SNR_GATE else '✗'}")
    need = next((r["N_per_side"] for r in tab if r["pass"]), None)

    # ── 机时预算 ─────────────────────────────────────────────────────
    print("\n" + "=" * 72 + "\n机时预算\n" + "=" * 72)
    ep_s = T / 20.0
    if need is None:
        print(f"  **✗ 在 N ≤ {max(reps)} 内达不到信噪 {SNR_GATE:g} ⇒ 不要按现状上机。**")
    else:
        # A0(标定) + A1 + A2 + Bp×N + Bm×N
        n_rep = 1 + 2 * need + 2 * need
        overhead = 1.8            # 归位/静置/换臂，按 ③a 的实际节奏取
        mins = n_rep * ep_s * overhead / 60
        print(f"  达门槛所需：**每侧 N = {need}** ⇒ 复放次数 "
              f"= A0 1 + A1 {need} + A2 {need} + Bp {need} + Bm {need} = **{n_rep} 次**")
        print(f"  单次 {ep_s:.1f} s × {overhead:g}（归位/静置/换序列）"
              f" ⇒ **净机时 ≈ {mins:.0f} 分钟**")
        print(f"  含起停/标定/异常重跑，建议预算 **{mins*2/60:.1f} 小时**（³a 实际 13 次复放）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"base": str(base), "T": T, "n_chunks": len(starts), "pi": PI_CKPT.name,
         "da_rms_mrad": rms.tolist(), "da_peak_mrad": float(np.abs(da).max() * 1e3),
         "seam_energy_frac": e_seam / e_tot, "eps_hit_pct": hit_eps,
         "sigma_replay": sigma, "sigma_rows": rows,
         "pred_norm": sig_norm, "snr_table": tab, "N_needed": need,
         "gates": {"snr": SNR_GATE},
         "caliber": "可行性筛查，用于锁定波形/幅度/机时；**不判定 ③c**。"
                    "σ 取自 ③a 的裸复放 —— 那是 j5 脉冲实验的基准轨迹，"
                    "换基准轨迹后 σ 需重估。"},
        indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")

    print("\n" + "=" * 72 + "\n⚠ 这份检查**没有**回答的（必须写进方案的未决项）\n" + "=" * 72)
    print("  1. σ 借自 ③a 的**那条**基准轨迹（j5 脉冲实验用的 pp2 复放）。"
          "换基准轨迹后 **σ 必须重估** —— 或把 A1/A2 本身当作重估手段。")
    print("  2. **削波器/加速度限幅未建模**：③a 正是栽在这里（j2/j3 结构性不可行）。"
          "上机前应先用 A1/A2 的 `cmd` 回读核对下发是否等于计划。")
    print("  3. 本估算用**冻结 H** 算信号。若真实响应比 H 预测的小，信噪会同比下降"
          "—— 而「H 预测不准」恰恰是本轮要测的东西 ⇒ **N 应留裕量**。")
    print("  4. 信噪够 ≠ 判据④（符号非线性）有判别力。③a 的 ④ 通过主要是"
          "**检出力不足**（噪声预期 0.364 > 0.3 的绝对线）。④ 的门槛须另算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
