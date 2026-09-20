#!/usr/bin/env python3
"""③c 序列生成（零机时）—— 选基准集 → 门控后的 δa → `_p`/`_m` 两条序列 → 安全/连续性检查。

用户 2026-09-21 定死的三条，本脚本逐条实现
------------------------------------------
1. **基准轨迹 = ab5 的 A（裸 ACT）**，按**固定规则**选集：
   **满足既有有效性与安全检查的最小集号**。
   ⚠ **不按墙大小、预测收益或信噪挑「最好的一集」。**
   ③a 的轨迹继续用于分析流程自检，**不因为已有噪声估计就优先使用它**。
2. **保留部署接触门控，并在生成序列时固定**：用名义 chunk + 原部署逻辑
   （`PiNetRuntime`）算出门控后的 δa，然后生成 `u+δa` / `u−δa`，
   **两侧共用同一个门控掩码**，**不**根据各次复放的实际状态重新决定门控。
   ⇒ 测的是**部署门控后的固定补正波形**。
   **门控边界本身会产生跳变 ⇒ 必须进入速度/加速度/连续性检查。**
3. `δa_j4 = 0`（本轮范围限制，**不是**断言 H 无 j4 信息）。

依赖前置：`sim/verify_pi_replay_3c.py` 步骤 A 必须先通过
（**device=cuda 时 15/15 逐位相同**；CPU 会有 ~6e-6 rad 的卷积舍入差）。

⚠ 本脚本产出的信噪**不能用于放行上机**
--------------------------------------
它仍借用 ③a 的噪声底 σ，而 ③a 是**另一条轨迹**。
最终噪声**必须由当日的裸复放（A1/A2）估计**，且
**不得在看到模型得分之后临时加次数**。

用法:
    python -u sim/make_probe_3c.py [--device cuda] [--n-per-side 6]
"""
from __future__ import annotations
import argparse, hashlib, json, sys
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
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
# 冻结的平衡臂序（用户 2026-09-21 定）：符号不与采集时间绑定
BLOCK = "+--+"          # ×3 ⇒ Bp 6 次 / Bm 6 次


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


def checks(u, da, gate, lo, hi, mlo, mhi):
    """安全 + 连续性。**门控边界的跳变必须被这里抓到。**"""
    r, msgs = {}, []
    ok = True

    def add(k, good, msg):
        nonlocal ok
        r[k] = bool(good); ok &= bool(good)
        msgs.append(("✓" if good else "**✗**") + " " + msg)

    add("finite", np.isfinite(u).all() and np.isfinite(da).all(), "指令/补正无 NaN")
    add("eps", np.abs(da).max() <= EPS_B + 1e-12,
        f"δa 峰值 {np.abs(da).max()*1e3:.3f} mrad ≤ ε {EPS_B*1e3:.0f} mrad")
    add("j4_zero", np.abs(da[:, J4]).max() == 0.0, "δa_j4 ≡ 0")
    add("grip", da.shape[1] == 6, "夹爪不受补正影响")

    # ⚠ **多关节同帧** —— 这正是与 ③a 的关键差异，旧断言在这里必然不成立
    multi = int((np.abs(da) > 0).sum(1).max())
    r["max_joints_per_frame"] = multi
    msgs.append(f"ℹ 同帧最多 {multi} 个关节同时发 —— **③c 是多关节连续补正**，"
                f"复放器旧模式的「同帧仅一关节」断言对本轮不适用（见方案 §阻断项一）")

    # 关节限位：两条分开报（沿用 replay_probe 的口径）
    for nm, sgn in (("plus", +1), ("minus", -1)):
        tot = u[:, :6] + sgn * da
        e_safe = int(((tot < lo) | (tot > hi)).sum())
        e_model = int(((tot < mlo) | (tot > mhi)).sum())
        r[f"limit_{nm}_safe"] = e_safe; r[f"limit_{nm}_model"] = e_model
        add(f"limit_{nm}", e_model == 0,
            f"u{'+' if sgn>0 else '−'}δa 越**模型**限位 {e_model} 处"
            f"（越实测包络 {e_safe} 处，仅提示）")

    # ── 连续性：速度 / 加速度 / 门控边界跳变 ──────────────────────────
    d1u = np.abs(np.diff(u[:, :6], axis=0)).max(0)
    d2u = np.abs(np.diff(u[:, :6], 2, axis=0)).max(0)
    for nm, sgn in (("plus", +1), ("minus", -1)):
        tot = u[:, :6] + sgn * da
        d1 = np.abs(np.diff(tot, axis=0)).max(0)
        d2 = np.abs(np.diff(tot, 2, axis=0)).max(0)
        # ⚠ 基线本身不动的关节（ab5 上的 j4）比值无意义 ⇒ 记 None，别显示成 0.000
        r[f"vel_ratio_{nm}"] = [None if d1u[j] == 0 else float(d1[j] / d1u[j])
                                for j in range(6)]
        r[f"acc_ratio_{nm}"] = [None if d2u[j] == 0 else float(d2[j] / d2u[j])
                                for j in range(6)]
        r[f"base_static_joints"] = [JOINTS[j] for j in range(6) if d1u[j] == 0]
    # 门控边界：δa 在此处对 0 跳变
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
    add("continuity", max(vr) < 1.5 and max(ar) < 1.5,
        f"逐帧步长/加速度未被补正显著推大：速度比 {min(vr):.3f}–{max(vr):.3f} · "
        f"加速度比 {min(ar):.3f}–{max(ar):.3f}（相对基线自身，门槛 1.5）")
    if r["base_static_joints"]:
        msgs.append(f"ℹ 基线上**完全不动**的关节：{r['base_static_joints']}"
                    f" ⇒ 比值记 None（屏蔽 j4 在本轨迹上不损失覆盖）")
    msgs.append(f"ℹ 门控边界 {r['n_gate_edges']} 处 · 边界处 δa 跳变最大 "
                f"{r['gate_edge_jump_max_mrad']:.3f} mrad · "
                f"非边界最大 {r['interior_jump_max_mrad']:.3f} mrad")
    return ok, r, msgs


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-side", type=int, default=6)
    ap.add_argument("--src", default="outputs/ab5")
    ap.add_argument("--arm", default="A")
    a = ap.parse_args()

    import fit_step1_candidate as S1
    from lerobot.common.policies.act.pi_net_runtime import PiNetRuntime
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    lo, hi, mlo, mhi = robot_limits_safe()
    ck = ROOT / "outputs/p2_pi/pi_distill_v2.pt"
    rt = PiNetRuntime(str(ck), device=a.device, verbose=True)

    paths = sorted((ROOT / a.src).glob(f"{a.arm}_ep*.json"),
                   key=lambda p: int(p.stem.split("ep")[-1]))
    print(f"\n候选 {len(paths)} 集（{a.src}/{a.arm}）· 规则 = "
          f"**满足既有有效性与安全检查的最小集号**")
    print("=" * 78)

    chosen = None
    scan = []
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
            da[:, J4] = 0.0                                  # ⚠ j4 屏蔽
            ok, rep, msgs = checks(u, da, gate, lo, hi, mlo, mhi)
            scan.append({"ep": ep, "n": n, "pass": ok,
                         "why": None if ok else [m for m in msgs if m.startswith("**")]})
            print(f"  ep{ep:<3} {n:>4} 帧  {'✓ 通过' if ok else '✗ ' + str(scan[-1]['why'])}")
            if ok and chosen is None:
                chosen = {"path": p, "ep": ep, "u": u, "da": da, "gate": gate,
                          "starts": starts, "report": rep, "msgs": msgs, "pay": pay}
                break                                        # **取最小集号，不再比较**
        except Exception as ex:
            scan.append({"ep": ep, "n": None, "pass": False, "why": [f"{type(ex).__name__}: {ex}"]})
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

    print("\n" + "=" * 78 + "\n补正波形（门控后 + j4 屏蔽）\n" + "=" * 78)
    rms = np.sqrt((da ** 2).mean(0)) * 1e3
    print("  逐关节 RMS(mrad) " + " · ".join(f"{j} {v:.3f}" for j, v in zip(JOINTS, rms)))
    print(f"  全体 RMS {np.sqrt((da**2).mean())*1e3:.3f} mrad · 峰值 "
          f"{np.abs(da).max()*1e3:.3f} mrad · 门控 {gate.mean()*100:.1f}% 的帧")
    e_seam = float((da[seam] ** 2).sum()); e_tot = float((da ** 2).sum())
    print(f"  接缝（chunk 头 {SEAM_W} 帧，占 {seam.mean()*100:.1f}%）承载 "
          f"{e_seam/e_tot*100:.1f}% 的 δa 能量 ⇒ 全段复放同时覆盖接缝与内部")

    # ── 信噪重算（**仍借 ③a 的 σ ⇒ 不能放行**）────────────────────────
    print("\n" + "=" * 78 + "\n信噪重算\n" + "=" * 78)
    from feasibility_3c import replay_sigma
    sigma, _ = replay_sigma()
    pred = S1.causal_conv(H, da)
    sn = float(np.linalg.norm(pred))
    N = a.n_per_side
    n_odd = sigma * 0.5 * np.sqrt(2.0 / N)
    snr = sn / (n_odd * np.sqrt(3 * T))
    snr4 = sn / (sigma * 0.5 * np.sqrt(2.0 / 4) * np.sqrt(3 * T))
    print(f"  σ(借自 ③a) {sigma:.5f} rad/s · 冻结 H 预测 ‖Δ_odd‖ {sn:.5f}")
    print(f"  N={N}/侧 ⇒ 预测信噪 **{snr:.2f}**   (N=4 时 {snr4:.2f}；"
          f"N=6 相对 N=4 只多 {(snr/snr4-1)*100:.0f}% 的理论裕量)")
    print("  ⚠ **此数不能用于放行上机** —— σ 借自 ③a 的**另一条轨迹**。")
    print("     最终噪声必须由**当日裸复放 A1/A2** 估计，且"
          "**不得在看到模型得分后临时加次数**。")

    # ── 冻结落盘 ─────────────────────────────────────────────────────
    order = ["A0"] + ["A1"] * N + [("Bp" if c == "+" else "Bm") for c in BLOCK * 3] \
            + ["A2"] * N
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "artifact": "probe-3c-v1-draft",
        "base": str(chosen["path"].relative_to(ROOT)), "base_ep": chosen["ep"],
        "base_rule": "满足既有有效性与安全检查的**最小集号**；不按墙/收益/信噪挑选",
        "T": T, "n_chunks": int(len(chosen["starts"])),
        "pi_ckpt": str(ck.relative_to(ROOT)), "pi_ckpt_sha256": rt.ckpt_sha256,
        "device": a.device,
        "gating": "部署接触门控，**生成时固定**，两侧共用同一掩码；不按各次复放实际状态重决",
        "gated_pct": float(gate.mean() * 100),
        "j4": "δa_j4 ≡ 0（实验范围选择，**不是**断言 H 无 j4 信息）",
        "da_rms_mrad": rms.tolist(), "da_peak_mrad": float(np.abs(da).max() * 1e3),
        "seam_energy_frac": e_seam / e_tot,
        "checks": chosen["report"],
        "order": order, "n_per_side": N, "block": BLOCK,
        "snr_borrowed_sigma": {"sigma": sigma, "snr": snr, "snr_N4": snr4,
                               "valid_for_go": False,
                               "note": "σ 借自 ③a 的另一条轨迹；放行须用当日裸复放重估"},
        "scope": "选定轨迹上、D′ 波形经门控与 j4 屏蔽后的**组合扰动方向**；"
                 "**不是全臂辨识，也不直接解释旧 D 臂的失败原因**。",
        "status": "**草案 —— 未放行上机**（预注册未冻结；复放器新模式未实现；"
                  "软件发送链路核对未做）",
    }
    for tag, sgn in (("p", +1), ("m", -1)):
        f = OUT / f"probe_3c_{tag}.npz"
        np.savez(f, cmd7=chosen["u"].astype(np.float32),
                 da=(sgn * da).astype(np.float32), gate=gate,
                 seam=seam, starts=chosen["starts"],
                 meta_json=json.dumps({**meta, "sign": tag}, ensure_ascii=False))
        meta[f"sha256_{tag}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"\n  → {f.name}  sha {meta[f'sha256_{tag}'][:16]}…")
    (OUT / "probe_3c_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78 + f"\n臂序（**预先冻结的平衡顺序**，共 {len(order)} 次复放）\n" + "=" * 78)
    print(f"  A0 → A1×{N} → ({BLOCK})×3 → A2×{N}")
    print(f"  ⇒ 符号**不与采集时间绑定**。实际时间戳仍须记录 —— "
          f"**位置平衡不等于消除了所有漂移**。")
    print(f"\n  **状态：草案，未放行上机。** 还缺：复放器多关节新模式 · "
          f"软件发送链路核对 · 冻结 G-2/H 预测 · 预注册。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
