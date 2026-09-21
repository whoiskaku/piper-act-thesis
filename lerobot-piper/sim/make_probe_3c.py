#!/usr/bin/env python3
"""③c 序列生成 + 次数规划（零机时）—— **两阶段设计**。

用户 2026-09-21 定死的设计
--------------------------
    第一阶段：A0 → A1×M          ⇒ 只估噪声，据此**锁定 N**
    第二阶段：(+ − − +)×(N/2) → A2×M

- `A1` / `A2` **各固定 M = 6 次**（`A2` **必须留在 B 之后**，保留前后基线）
- `Bp` / `Bm` **各 N 次**，`N ∈ {6, 8, 10}` ⇒ 总次数 `13 + 2N` = **25 / 29 / 33**
- **N=10 不必把裸臂也加到 10** ⇒ 不是 41 次

### N 的规则（**事前冻结**）

    S_H  = 最终冻结波形经 H 后，**预测奇分量的 RMS**
    σ̂    = 当日 M 条 A1 估出的单次复放噪声（同坐标、同滤波、同有效时间窗）

    取满足   S_H · sqrt(2N) / σ̂  ≥  3   的**最小偶数 N**，且 N ≥ 6

- 所需 `N ≤ 10` ⇒ **锁定次数后采 B**，**不根据中途模型得分追加**
- 所需 `N > 10` ⇒ **不启动 B**，记录「**当前预算下预计检出能力不足**」，
  另行决定预算，**不能判模型失败**

⚠ **这是次数规划，不是通过保证、也不是统计功效证明。**
噪声须从**重复轨迹之间的差异**估计 —— **不是**把整段运动的 IMU RMS 当噪声，
**也不能**把数千帧当作独立重复。六次估计本身仍有不确定性；
公式还依赖**重复噪声近似独立、同方差**。

### 偶分量必须**独立判读**

    E = Δ⁺ + Δ⁻   （**不除以二**）      σ_E = σ · sqrt(2/N + 2/M)

上面的 N 只解决**奇分量**的预计信噪。完全可能出现：
**奇响应足以比较 G/H，而偶分量仍不足以判断是否足够小。**
⇒ **允许这个分开的结论。不得因为噪声大就抬高容忍线，最后宣称「线性通过」。**

其余三条（用户 2026-09-20/21 定）
----------------------------------
1. 基准轨迹 = **ab5 的 A（裸 ACT）**，按固定规则选集：**满足既有有效性与安全检查的最小集号**。
   **不按墙大小、预测收益或信噪挑「最好的一集」。**
2. **保留部署接触门控并在生成序列时固定**，两侧共用同一掩码，不按各次复放实际状态重决。
   **门控边界的跳变必须进入速度/加速度/连续性检查。**
3. `δa_j4 = 0` —— 措辞只能是「**本轮主动排除 j4 增量方向**」。
   ⚠ **不得**写成「j4 静止所以屏蔽不损失覆盖」：**名义 j4 不动，不代表扰动 j4 没有响应**。

前置：`sim/verify_pi_replay_3c.py` 步骤 A 必须先通过（device=cuda 时 15/15 逐位相同）。
生成后**冻结序列与哈希**；之后复放**直接读冻结序列，不要求再跑 CUDA 网络**。

用法:
    python -u sim/make_probe_3c.py [--device cuda] [--m-bare 6]
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/probe_3c"
EPS_B = 0.010
J4 = 3
MIN_FRAMES = 550
SEAM_W = 4
SNR_GATE = 3.0
N_ALLOWED = (6, 8, 10)          # 事前冻结的可选次数
M_BARE = 6                      # A1 / A2 各固定 6 次
BLOCK = "+--+"                  # 平衡块；重复 N/4... 见 build_order
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


# ══════════════════════════════════════════════════════════════════════
def build_order(n_sign: int, m_bare: int) -> tuple[list[str], list[str]]:
    """两阶段臂序。**裸臂次数与符号次数分开**，并断言计数一致。

    ⚠ 旧版把 B 块写死成 `BLOCK * 3`（永远 6 次），改 N 只会改裸臂 —— 已修。
    """
    if n_sign not in N_ALLOWED:
        raise ValueError(f"N 必须取 {N_ALLOWED}，得到 {n_sign}")
    if n_sign % 2:
        raise ValueError("N 必须是偶数（+/− 各半）")
    # 平衡块 '+--+' 每块含 2 正 2 负 ⇒ 需要 n_sign/2 块
    nblk, rem = divmod(n_sign * 2, len(BLOCK))
    if rem:
        raise ValueError(f"N={n_sign} 无法用 {BLOCK} 整除平衡")
    stage1 = ["A0"] + ["A1"] * m_bare
    stage2 = [("Bp" if c == "+" else "Bm") for c in BLOCK * nblk] + ["A2"] * m_bare
    order = stage1 + stage2
    # ── 断言实际序列计数一致 ─────────────────────────────────────────
    cnt = {k: order.count(k) for k in ("A0", "A1", "A2", "Bp", "Bm")}
    assert cnt == {"A0": 1, "A1": m_bare, "A2": m_bare,
                   "Bp": n_sign, "Bm": n_sign}, f"臂序计数不符: {cnt}"
    assert len(order) == 1 + 2 * m_bare + 2 * n_sign, "总次数不符"
    return stage1, stage2


def plan_n(s_h: float, sigma: float, m_bare: int) -> dict:
    """事前冻结的次数规则：最小偶数 N≥6 使 `S_H·sqrt(2N)/σ̂ ≥ 3`；上限 10。"""
    rows = []
    for n in N_ALLOWED:
        snr = s_h * np.sqrt(2.0 * n) / sigma
        sig_e = sigma * np.sqrt(2.0 / n + 2.0 / m_bare)
        rows.append({"N": n, "snr_odd": float(snr), "sigma_even": float(sig_e),
                     "even_noise_over_signal": float(sig_e / max(s_h, 1e-30)),
                     "total_replays": 1 + 2 * m_bare + 2 * n,
                     "meets_gate": bool(snr >= SNR_GATE)})
    need = next((r["N"] for r in rows if r["meets_gate"]), None)
    # 不受 N_ALLOWED 限制的理论需求，用于「>10 时不启动 B」的判定
    n_theory = float((SNR_GATE * sigma / max(s_h, 1e-30)) ** 2 / 2.0)
    return {"rows": rows, "N_chosen": need, "N_theory_min": n_theory,
            "over_budget": bool(need is None),
            "rule": "最小偶数 N≥6 使 S_H·sqrt(2N)/σ̂ ≥ 3；上限 10；"
                    "所需 N>10 ⇒ 不启动 B，记「当前预算下预计检出能力不足」，"
                    "**不能判模型失败**"}


def robot_limits_safe():
    """安全限位 = 模型 jnt_range ±2° ∩ 真机实测编码器范围（与 replay_probe 同源）。"""
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    from make_probe_b2 import robot_limits
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mlo, mhi = m.jnt_range[:6, 0] - 0.035, m.jnt_range[:6, 1] + 0.035
    rlo, rhi = robot_limits()
    return np.maximum(mlo, rlo), np.minimum(mhi, rhi), mlo, mhi


def nominal(pay):
    """名义指令 u（chunks[].raw，**不含** δa）+ chunk 起始帧。"""
    chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
    n = len(pay["frames"])
    tot = max(int(c["start"]) + len(c["raw"]) for c in chunks)
    u = np.full((tot, 7), np.nan)
    prev_end = 0
    for c in chunks:
        s0 = int(c["start"]); r = np.asarray(c["raw"], float)
        if s0 != prev_end:
            raise ValueError(f"chunk 不连续: 期望 {prev_end}, 实得 {s0}")
        u[s0:s0 + len(r)] = r
        prev_end = s0 + len(r)
    if not np.isfinite(u[:n]).all():
        raise ValueError("名义指令有未覆盖帧")
    return u[:n], np.asarray([int(c["start"]) for c in chunks])


def gated_delta(rt, pay):
    """按**部署时序**驱动 PiNetRuntime ⇒ 门控后的 δa 与门控掩码。"""
    chunks = {int(c["start"]): np.asarray(c["raw"], np.float64) for c in pay["chunks"]}
    n = len(pay["frames"])
    rt.reset()
    da = np.zeros((n, 6)); gate = np.zeros(n, bool)
    for t in range(n):
        if t in chunks:
            rt.begin_chunk(chunks[t])
        fr = rt.step()
        da[t] = fr.delta
        gate[t] = bool(fr.contact_gated)
    return da, gate


def checks(u, da, gate, lo, hi, mlo, mhi, seam_frames=()):
    """安全 + 连续性。

    ⚠ **整集最大值之比不能证明每处跳变都小** —— 已加**逐时刻余量**：
    对每一帧比较 `|Δ(u±δa)[t]|` 与 `|Δu[t]|`，报告最坏的**绝对**增量与其位置。
    """
    r, msgs = {}, []
    ok = True

    def add(k, good, msg):
        nonlocal ok
        r[k] = bool(good); ok &= bool(good)
        msgs.append(("✓" if good else "**✗**") + " " + msg)

    add("finite", np.isfinite(u).all() and np.isfinite(da).all(), "指令/补正无 NaN")
    add("eps", np.abs(da).max() <= EPS_B + 1e-12,
        f"δa 峰值 {np.abs(da).max()*1e3:.3f} mrad ≤ ε {EPS_B*1e3:.0f} mrad")
    add("j4_zero", np.abs(da[:, J4]).max() == 0.0,
        "δa_j4 ≡ 0（**本轮主动排除 j4 增量方向**）")
    add("grip", da.shape[1] == 6, "夹爪不受补正影响")

    multi = int((np.abs(da) > 0).sum(1).max())
    r["max_joints_per_frame"] = multi
    msgs.append(f"ℹ 同帧最多 {multi} 个关节同时发 —— **③c 是多关节连续补正**，"
                f"复放器旧模式的「同帧仅一关节」断言对本轮不适用（需显式新模式）")

    for nm, sgn in (("plus", +1), ("minus", -1)):
        tot = u[:, :6] + sgn * da
        e_safe = int(((tot < lo) | (tot > hi)).sum())
        e_model = int(((tot < mlo) | (tot > mhi)).sum())
        r[f"limit_{nm}_safe"] = e_safe; r[f"limit_{nm}_model"] = e_model
        add(f"limit_{nm}", e_model == 0,
            f"u{'+' if sgn>0 else '−'}δa 越**模型**限位 {e_model} 处"
            f"（越实测包络 {e_safe} 处，仅提示）")

    # ── 连续性 ───────────────────────────────────────────────────────
    d1u = np.abs(np.diff(u[:, :6], axis=0))
    d2u = np.abs(np.diff(u[:, :6], 2, axis=0))
    r["base_abs_step_max_mrad"] = (d1u.max(0) * 1e3).tolist()
    r["base_abs_acc_max_mrad"] = (d2u.max(0) * 1e3).tolist()
    for nm, sgn in (("plus", +1), ("minus", -1)):
        tot = u[:, :6] + sgn * da
        d1 = np.abs(np.diff(tot, axis=0)); d2 = np.abs(np.diff(tot, 2, axis=0))
        # 整集最大值之比（**只说明相对峰值**）
        r[f"vel_ratio_{nm}"] = [None if d1u[:, j].max() == 0 else
                                float(d1[:, j].max() / d1u[:, j].max()) for j in range(6)]
        r[f"acc_ratio_{nm}"] = [None if d2u[:, j].max() == 0 else
                                float(d2[:, j].max() / d2u[:, j].max()) for j in range(6)]
        # ⚠ **逐时刻余量**：同一帧上补正把步长/加速度推大了多少（绝对值，mrad）
        ex1 = (d1 - d1u); ex2 = (d2 - d2u)
        i1 = np.unravel_index(np.argmax(ex1), ex1.shape)
        i2 = np.unravel_index(np.argmax(ex2), ex2.shape)
        r[f"worst_step_excess_{nm}"] = {
            "mrad": float(ex1[i1] * 1e3), "frame": int(i1[0]), "joint": JOINTS[i1[1]],
            "abs_step_mrad": float(d1[i1] * 1e3), "base_step_mrad": float(d1u[i1] * 1e3)}
        r[f"worst_acc_excess_{nm}"] = {
            "mrad": float(ex2[i2] * 1e3), "frame": int(i2[0]), "joint": JOINTS[i2[1]],
            "abs_acc_mrad": float(d2[i2] * 1e3), "base_acc_mrad": float(d2u[i2] * 1e3)}
        r[f"abs_step_max_{nm}_mrad"] = float(d1.max() * 1e3)
        r[f"abs_acc_max_{nm}_mrad"] = float(d2.max() * 1e3)

    edge = np.zeros(len(gate), bool)
    edge[1:] |= gate[1:] != gate[:-1]
    jump = np.abs(np.diff(da, axis=0))
    ei = np.where(edge[1:])[0]
    r["n_gate_edges"] = int(edge.sum())
    r["gate_edge_jump_max_mrad"] = float(jump[ei].max() * 1e3) if len(ei) else 0.0
    r["interior_jump_max_mrad"] = float(
        np.delete(jump, ei, axis=0).max() * 1e3) if len(ei) else float(jump.max() * 1e3)

    vr = [v for v in r["vel_ratio_plus"] + r["vel_ratio_minus"] if v is not None]
    ar = [v for v in r["acc_ratio_plus"] + r["acc_ratio_minus"] if v is not None]
    add("rel_peak", max(vr) < 1.5 and max(ar) < 1.5,
        f"**相对峰值检查**：速度比 {min(vr):.3f}–{max(vr):.3f} · "
        f"加速度比 {min(ar):.3f}–{max(ar):.3f}（整集最大值之比，门槛 1.5）")
    msgs.append("⚠ 上一条**只是相对峰值检查**，**不能证明每一处跳变都小**。"
                "绝对步长/加速度须对照固件实际限幅核对 ——")
    for nm in ("plus", "minus"):
        w1, w2 = r[f"worst_step_excess_{nm}"], r[f"worst_acc_excess_{nm}"]
        at_seam = "**chunk 接缝**" if w1["frame"] in seam_frames else "chunk 内部"
        ratio = w1["abs_step_mrad"] / max(w1["base_step_mrad"], 1e-9)
        msgs.append(f"   [{nm}] 逐时刻最坏**步长**增量 {w1['mrad']:+.3f} mrad "
                    f"@帧{w1['frame']}({at_seam}) {w1['joint']}"
                    f"（绝对 {w1['abs_step_mrad']:.3f}，基线仅 {w1['base_step_mrad']:.3f}"
                    f" ⇒ **局部 {ratio:.0f}×**）· 全集绝对步长上限 "
                    f"{r[f'abs_step_max_{nm}_mrad']:.3f} mrad")
        msgs.append(f"   [{nm}] 逐时刻最坏**加速度**增量 {w2['mrad']:+.3f} mrad "
                    f"@帧{w2['frame']} {w2['joint']}（绝对 {w2['abs_acc_mrad']:.3f}，"
                    f"基线 {w2['base_acc_mrad']:.3f}）")
    msgs.append(f"ℹ 门控边界 {r['n_gate_edges']} 处 · 边界处 δa 跳变最大 "
                f"{r['gate_edge_jump_max_mrad']:.3f} mrad · "
                f"非边界最大 {r['interior_jump_max_mrad']:.3f} mrad "
                f"⇒ **必须进入完整发送序列的绝对约束核对**")
    static = [JOINTS[j] for j in range(6) if d1u[:, j].max() == 0]
    if static:
        r["base_static_joints"] = static
        msgs.append(f"ℹ 基线上名义不动的关节：{static}。"
                    f"⚠ **名义 j4 不动，不代表扰动 j4 没有响应** —— "
                    f"正确表述是「本轮主动排除 j4 增量方向」，**不是**「没有覆盖损失」。")
    return ok, r, msgs


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m-bare", type=int, default=M_BARE)
    ap.add_argument("--src", default="outputs/ab5")
    ap.add_argument("--arm", default="A")
    a = ap.parse_args()

    import torch
    import fit_step1_candidate as S1
    from lerobot.common.policies.act.pi_net_runtime import PiNetRuntime
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    lo, hi, mlo, mhi = robot_limits_safe()
    ck = ROOT / "outputs/p2_pi/pi_distill_v2.pt"
    rt = PiNetRuntime(str(ck), device=a.device, verbose=True)

    paths = sorted((ROOT / a.src).glob(f"{a.arm}_ep*.json"),
                   key=lambda p: int(p.stem.split("ep")[-1]))
    print(f"\n候选 {len(paths)} 集（{a.src}/{a.arm}）· 规则 = "
          f"**满足既有有效性与安全检查的最小集号**\n" + "=" * 78)

    chosen = None
    for p in paths:
        ep = int(p.stem.split("ep")[-1])
        try:
            pay = json.loads(p.read_text())
            n = len(pay["frames"])
            if n < MIN_FRAMES:
                raise ValueError(f"帧数 {n} < {MIN_FRAMES}")
            if any(f.get("imu") is None for f in pay["frames"]):
                raise ValueError("有缺失 IMU 帧")
            if not np.isfinite(np.asarray([f["imu"] for f in pay["frames"]], float)).all():
                raise ValueError("IMU 非有限")
            u, starts = nominal(pay)
            da, gate = gated_delta(rt, pay)
            da[:, J4] = 0.0                                  # 主动排除 j4 增量方向
            ok, rep, msgs = checks(u, da, gate, lo, hi, mlo, mhi,
                                   seam_frames=set(int(x) for x in starts))
            print(f"  ep{ep:<3} {n:>4} 帧  {'✓ 通过' if ok else '✗ 未过'}")
            if ok:
                chosen = {"path": p, "ep": ep, "u": u, "da": da, "gate": gate,
                          "starts": starts, "report": rep, "msgs": msgs}
                break                                        # **取最小集号，不再比较**
        except Exception as ex:
            print(f"  ep{ep:<3}  ---   ✗ {type(ex).__name__}: {ex}")

    if chosen is None:
        print("\n**✗ 没有集通过 ⇒ 不生成序列**")
        return 1

    print(f"\n⇒ **选中 {chosen['path'].name}**（最小通过集号；"
          f"**不是**按墙大小/收益/信噪挑的）")
    for m in chosen["msgs"]:
        print("   " + m)

    u, da, gate = chosen["u"], chosen["da"], chosen["gate"]
    T = len(da)
    seam = np.zeros(T, bool)
    for s0 in chosen["starts"]:
        if s0 < T:
            seam[s0:min(s0 + SEAM_W, T)] = True

    print("\n" + "=" * 78 + "\n补正波形（门控后 + 排除 j4）\n" + "=" * 78)
    rms = np.sqrt((da ** 2).mean(0)) * 1e3
    print("  逐关节 RMS(mrad) " + " · ".join(f"{j} {v:.3f}" for j, v in zip(JOINTS, rms)))
    print(f"  全体 RMS {np.sqrt((da**2).mean())*1e3:.3f} mrad · 峰值 "
          f"{np.abs(da).max()*1e3:.3f} mrad · 门控 {gate.mean()*100:.1f}% 的帧")
    e_seam = float((da[seam] ** 2).sum()); e_tot = float((da ** 2).sum())
    print(f"  接缝（chunk 头 {SEAM_W} 帧，占 {seam.mean()*100:.1f}%）承载 "
          f"{e_seam/e_tot*100:.1f}% 的 δa 能量 ⇒ 全段复放同时覆盖接缝与内部")

    # ── 次数规划 ─────────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n次数规划（规则事前冻结）\n" + "=" * 78)
    from feasibility_3c import replay_sigma
    sigma_borrow, _ = replay_sigma()
    pred = S1.causal_conv(H, da)
    s_h = float(np.linalg.norm(pred) / np.sqrt(3 * T))       # 预测奇分量 RMS
    print(f"  S_H（冻结波形经 H 的预测奇分量 RMS）= **{s_h:.6f} rad/s**")
    print(f"  σ̂ 当日由 M={a.m_bare} 条 A1 估出 —— 此处**借用** ③a 的 "
          f"{sigma_borrow:.5f} rad/s 仅作预演")
    pl = plan_n(s_h, sigma_borrow, a.m_bare)
    print(f"\n  {'N':>4}{'奇分量信噪':>12}{'σ_E':>11}{'σ_E/S_H':>10}{'总次数':>8}  判")
    for r_ in pl["rows"]:
        print(f"  {r_['N']:>4}{r_['snr_odd']:>12.2f}{r_['sigma_even']:>11.5f}"
              f"{r_['even_noise_over_signal']:>10.2f}{r_['total_replays']:>8}  "
              f"{'✓' if r_['meets_gate'] else '✗'}")
    print(f"\n  规则：{pl['rule']}")
    print(f"  借用 σ 下理论最小 N = {pl['N_theory_min']:.1f} ⇒ "
          + (f"锁定 **N = {pl['N_chosen']}**（总 {1+2*a.m_bare+2*pl['N_chosen']} 次）"
             if pl["N_chosen"] else "**超出 10 的上限 ⇒ 不启动 B**"))
    print("  ⚠ 这是**次数规划**，不是通过保证、也不是统计功效证明。"
          "噪声须从**重复轨迹之间的差异**估计，")
    print("     **不是**整段运动的 IMU RMS，**也不能**把数千帧当独立重复；"
          "六次估计本身仍有不确定性。")

    # ── 偶分量：独立判读 ──────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n偶分量（E = Δ⁺ + Δ⁻，**独立判读**）\n" + "=" * 78)
    worst = pl["rows"][-1]
    print(f"  σ_E = σ·sqrt(2/N + 2/M)。即便 N=10、M={a.m_bare}："
          f"σ_E = {worst['sigma_even']:.5f} rad/s，"
          f"而预测奇响应 S_H = {s_h:.5f} rad/s")
    print(f"  ⇒ **σ_E/S_H = {worst['even_noise_over_signal']:.2f}** —— "
          f"偶分量的噪声底与信号同量级")
    # ⚠ **M 才是绑定约束**：σ_E ≥ σ·sqrt(2/M)，与 N 无关 ⇒ 光加 N 降不下来
    floor = sigma_borrow * np.sqrt(2.0 / a.m_bare)
    n_eq = 4.0 / max((0.3 * s_h / sigma_borrow) ** 2, 1e-30)      # N = M 时所需
    print(f"  ⚠ **加大 N 也没用**：σ_E 的下界（N→∞、M={a.m_bare} 固定）"
          f"= σ·sqrt(2/M) = {floor:.5f} rad/s")
    print(f"     ⇒ 比值下界 **{floor/s_h:.3f}**，**无论 N 多大都降不到 0.3 以下** "
          f"—— 绑定约束是**裸臂次数 M**，不是 N。")
    print(f"     要让「比值 < 0.3」有判别力且 N=M，需 **M = N ≈ {n_eq:.0f}**"
          f"（总 ≈ {1+4*n_eq:.0f} 次）⇒ **任何合理预算内都做不到**。")
    print("  ⇒ **事前接受这个分开的结论**：奇响应可能足以比较 G/H，"
          "而偶分量**仍不足以判断是否足够小**。")
    print("  ⚠ **不得因为噪声大就抬高容忍线，最后宣称「线性通过」。**")

    # ── 冻结落盘 ─────────────────────────────────────────────────────
    n_sign = pl["N_chosen"] or N_ALLOWED[-1]
    stage1, stage2 = build_order(n_sign, a.m_bare)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "artifact": "probe-3c-v2-draft",
        "base": str(chosen["path"].relative_to(ROOT)), "base_ep": chosen["ep"],
        "base_rule": "满足既有有效性与安全检查的**最小集号**；不按墙/收益/信噪挑选",
        "T": T, "n_chunks": int(len(chosen["starts"])),
        "pi_ckpt": str(ck.relative_to(ROOT)), "pi_ckpt_sha256": rt.ckpt_sha256,
        "gen_env": {"device": a.device, "torch": torch.__version__,
                    "numpy": np.__version__, "python": platform.python_version(),
                    "note": "冻结后复放**直接读序列，不要求再跑 CUDA 网络**"},
        "gating": "部署接触门控，**生成时固定**，两侧共用同一掩码",
        "gated_pct": float(gate.mean() * 100),
        "j4": "**本轮主动排除 j4 增量方向**（δa_j4 ≡ 0）。"
              "⚠ 名义 j4 不动**不代表**扰动 j4 没有响应 ⇒ 这是范围选择，不是「无覆盖损失」",
        "da_rms_mrad": rms.tolist(), "da_peak_mrad": float(np.abs(da).max() * 1e3),
        "seam_energy_frac": e_seam / e_tot,
        "checks": chosen["report"],
        "S_H": s_h, "sigma_borrowed": sigma_borrow, "plan": pl,
        "design": {"m_bare": a.m_bare, "n_sign": n_sign, "block": BLOCK,
                   "stage1": stage1, "stage2": stage2,
                   "total_replays": len(stage1) + len(stage2),
                   "note": "**A2 必须留在 B 之后**，保留前后基线；"
                           "裸臂次数与符号次数分开存储，计数已断言"},
        "scope": "选定轨迹上、D′ 波形经门控与**主动排除 j4 增量方向**后的组合扰动方向；"
                 "**不是全臂辨识，也不直接解释旧 D 臂的失败原因**。",
        "status": "**草案 —— 未放行上机**（预注册未冻结；复放器多关节模式未实现；"
                  "软件发送链路与绝对约束核对未做；N 须由当日 A1 重定）",
    }
    for tag, sgn in (("p", +1), ("m", -1)):
        f = OUT / f"probe_3c_{tag}.npz"
        np.savez(f, cmd7=u.astype(np.float32), da=(sgn * da).astype(np.float32),
                 gate=gate, seam=seam, starts=chosen["starts"],
                 meta_json=json.dumps({**meta, "sign": tag}, ensure_ascii=False))
        meta[f"sha256_{tag}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"\n  → {f.name}  sha {meta[f'sha256_{tag}'][:16]}…")
    (OUT / "probe_3c_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78 + "\n两阶段臂序\n" + "=" * 78)
    print(f"  第一阶段（{len(stage1)} 次）：A0 → A1×{a.m_bare}"
          f"   ⇒ **只估噪声，据此锁定 N**")
    print(f"  第二阶段（{len(stage2)} 次）：({BLOCK})×{n_sign*2//len(BLOCK)}"
          f" → A2×{a.m_bare}   ⇒ 预演 N={n_sign}")
    print(f"  合计 **{len(stage1)+len(stage2)} 次**（= 13 + 2N）· "
          f"**A2 留在 B 之后**，保留前后基线")
    print(f"  ⚠ 符号**不与采集时间绑定**；实际时间戳仍须记录 —— "
          f"**采集位置平衡不代表所有漂移都被消除**。")
    print(f"\n  **状态：草案，未放行上机。** 还缺：复放器多关节模式 · "
          f"发送链路与**绝对**步长/加速度约束核对 · 冻结 G-2/H 预测 · 预注册。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
