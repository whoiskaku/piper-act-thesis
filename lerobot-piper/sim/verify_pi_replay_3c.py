#!/usr/bin/env python3
"""③c 上机前校验 ①：**用部署路径逐帧复现 ab7 D′ 日志里的 δa**（零机时）。

用户 2026-09-21 的订正（本脚本就是照它写的）：

> 不同轨迹的 RMS **不必强行对齐到 0.65 mrad**。真正需要对齐的是：
> **同一条 ab7 D′ 记录、同一 checkpoint、同一部署路径，能否逐帧复现日志里的 δa。**
> 建议直接复用 `PiNetRuntime.begin_chunk()/step()` 生成补正，
> 避免再复制一份特征与归一化代码。**先复现原六关节日志，再施加本轮的 j4 屏蔽。**

⇒ 本脚本**不自己造特征、不自己做归一化**，只驱动 `PiNetRuntime`。
   我此前在 `feasibility_3c.py` 里手抄了一份特征构造并**漏掉了 norm_mu/norm_sd**，
   算出的 δa RMS 差 9 倍 —— 这正是「别复制第二份」的理由。

两步，顺序不能反
----------------
    步骤 A：**原样**（六关节，不屏蔽 j4）⇒ 必须逐帧等于日志的 `frames[].da`
    步骤 B：在 A 成立的前提下，再施加 `δa_j4 = 0`，报告它改变了多少

只有 A 通过，B 的波形才有资格作为 ③c 的补正源。

用法:
    python -u sim/verify_pi_replay_3c.py [--root outputs/ab7] [--arm D]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/probe_3c/verify_pi_replay.json"
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
J4 = 3


def replay_one(rt, pay):
    """按部署时序驱动 PiNetRuntime：chunk 起始帧调 begin_chunk，其余帧只 step。"""
    chunks = {int(c["start"]): np.asarray(c["raw"], np.float64) for c in pay["chunks"]}
    n = len(pay["frames"])
    rt.reset()
    da = np.zeros((n, 6))
    gate = np.zeros(n, bool)
    kk = np.zeros(n, int)
    for t in range(n):
        if t in chunks:
            rt.begin_chunk(chunks[t])
        fr = rt.step()
        da[t] = fr.delta
        gate[t] = bool(fr.contact_gated)
        kk[t] = int(fr.k)
    return da, gate, kk


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/ab7")
    ap.add_argument("--arm", default="D")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol-rad", type=float, default=1e-6)
    a = ap.parse_args()

    from lerobot.common.policies.act.pi_net_runtime import PiNetRuntime

    paths = sorted((ROOT / a.root).glob(f"{a.arm}_ep*.json"),
                   key=lambda p: int(p.stem.split("ep")[-1]))
    if not paths:
        print(f"**✗ 没找到 {a.root}/{a.arm}_ep*.json**")
        return 1
    ck = json.loads(paths[0].read_text())["summary"].get("pi_ckpt")
    print(f"{a.arm} 臂 {len(paths)} 集 · 日志记录的 checkpoint = {ck}")
    rt = PiNetRuntime(str(ROOT / ck), device=a.device, verbose=True)

    print("\n" + "=" * 76)
    print("步骤 A：**原样六关节** —— 必须逐帧等于日志的 frames[].da")
    print("=" * 76)
    print(f"  {'集':>10}{'帧':>6}{'δa 最大差(mrad)':>18}{'门控一致':>10}"
          f"{'k 一致':>8}{'日志RMS':>10}{'复现RMS':>10}")
    rows, ok_a = [], True
    store = {}
    for p in paths:
        pay = json.loads(p.read_text())
        n = len(pay["frames"])
        log_da = np.asarray([f["da"] for f in pay["frames"]], float)
        log_gate = np.asarray([bool(f.get("pi_contact", False)) for f in pay["frames"]])
        log_k = np.asarray([int(f.get("pi_k", -1)) for f in pay["frames"]])
        da, gate, kk = replay_one(rt, pay)
        dmax = float(np.abs(da - log_da).max())
        drms = float(np.sqrt(((da - log_da) ** 2).mean()))
        g_ok = bool((gate == log_gate).all())
        k_ok = bool((kk == log_k).all())
        good = dmax < a.tol_rad and g_ok and k_ok
        ok_a &= good
        store[p.stem] = {"da": da, "gate": gate, "pay": pay}
        rows.append({"ep": p.stem, "n": n, "max_diff_mrad": dmax * 1e3, "rms_diff_mrad": drms * 1e3,
                     "gate_match": g_ok, "k_match": k_ok,
                     "rms_log_mrad": float(np.sqrt((log_da ** 2).mean()) * 1e3),
                     "rms_rep_mrad": float(np.sqrt((da ** 2).mean()) * 1e3),
                     "gated_pct": float(gate.mean() * 100)})
        print(f"  {p.stem:>10}{n:>6}{dmax*1e3:>18.3e}{'✓' if g_ok else '**✗**':>10}"
              f"{'✓' if k_ok else '**✗**':>8}"
              f"{rows[-1]['rms_log_mrad']:>10.3f}{rows[-1]['rms_rep_mrad']:>10.3f}")

    npass = sum(r["max_diff_mrad"] < a.tol_rad * 1e3 for r in rows)
    print(f"\n  {'✓' if ok_a else '**✗**'} 步骤 A：{npass}/{len(rows)} 集过"
          f"（device={a.device} · 门槛 {a.tol_rad:g} rad）")
    print(f"  逐帧差：max 中位 {np.median([r['max_diff_mrad'] for r in rows]):.3e} mrad · "
          f"RMS 中位 {np.median([r['rms_diff_mrad'] for r in rows]):.3e} mrad "
          f"（δa 本身 RMS ≈ {np.median([r['rms_log_mrad'] for r in rows]):.3f} mrad）")
    print(f"  日志 δa RMS 中位 {np.median([r['rms_log_mrad'] for r in rows]):.3f} mrad · "
          f"门控占 {np.mean([r['gated_pct'] for r in rows]):.1f}%")
    if not ok_a:
        print("\n  **A 不通过 ⇒ 不进步骤 B。部署路径没有被复现，补正源不可信。**")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"step_A_pass": False, "rows": rows},
                                  indent=2, ensure_ascii=False))
        return 1

    # ── 步骤 B ────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("步骤 B：在 A 成立的前提下施加 δa_j4 = 0（本轮范围限制）")
    print("=" * 76)
    br = []
    for stem, v in store.items():
        da = v["da"]
        e_all = float((da ** 2).sum())
        e_j4 = float((da[:, J4] ** 2).sum())
        masked = da.copy(); masked[:, J4] = 0.0
        br.append({"ep": stem, "j4_energy_frac": e_j4 / max(e_all, 1e-30),
                   "rms_before_mrad": float(np.sqrt((da ** 2).mean()) * 1e3),
                   "rms_after_mrad": float(np.sqrt((masked ** 2).mean()) * 1e3),
                   "j4_rms_mrad": float(np.sqrt((da[:, J4] ** 2).mean()) * 1e3)})
    fr = np.array([r["j4_energy_frac"] for r in br])
    print(f"  j4 占 δa 总能量 {fr.mean()*100:.2f}%（范围 {fr.min()*100:.2f}–{fr.max()*100:.2f}%）")
    print(f"  屏蔽后全体 RMS {np.mean([r['rms_before_mrad'] for r in br]):.3f}"
          f" → {np.mean([r['rms_after_mrad'] for r in br]):.3f} mrad")
    print(f"  ⚠ 屏蔽 j4 使补正**不再等于 D′ 实际跑过的那个**"
          f" ⇒ ③c 测的是「D′ 波形经门控与 j4 屏蔽后的组合扰动方向」，**不是 D′ 本身**。")

    # 逐关节汇总（给序列生成器用）
    allda = np.concatenate([v["da"] for v in store.values()])
    rms = np.sqrt((allda ** 2).mean(0)) * 1e3
    print("\n  逐关节 δa RMS(mrad)（含门控帧）  "
          + " · ".join(f"{j} {x:.3f}" for j, x in zip(JOINTS, rms)))
    print(f"  峰值 {np.abs(allda).max()*1e3:.3f} mrad")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"step_A_pass": True, "ckpt": ck, "ckpt_sha256": rt.ckpt_sha256,
         "rows": rows, "j4_mask": br,
         "per_joint_rms_mrad": rms.tolist(),
         "peak_mrad": float(np.abs(allda).max() * 1e3),
         "caliber": "步骤 A 证明部署路径可逐帧复现；步骤 B 之后的波形**不再等于 D′ 实跑**。"
                    "本轮范围 = 选定轨迹上、D′ 波形经门控与 j4 屏蔽后的组合扰动方向；"
                    "**不是全臂辨识，也不直接解释旧 D 臂的失败原因**。"},
        indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
