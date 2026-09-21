#!/usr/bin/env python3
"""③c 上机前校验 ②：**软件发送链路 + 关节级绝对约束核对**（零机时）。

用户 2026-09-21 的定夺（本脚本照此执行）：

> 先保留现有波形，**不加临时的「近静止抑制」**；但**不能据此原样上机**。
> **先完成关节级绝对约束核对。**「写进已知风险」**不能代替安全放行依据**。

索引约定（**统一**，后续窗口与分析都按此）
------------------------------------------
    np.diff 的索引 i  ↔  帧 i → i+1
    二阶差分索引 i    ↔  涉及帧 i, i+1, i+2

按名义 `DT = 50 ms` 换算：一阶差分 / DT → rad/s；二阶差分 / DT² → rad/s²。
⚠ **这些是指令差分，不是实测关节速度/加速度。**

⚠ 「局部 2378×」不是危险倍数
----------------------------
它只是**基线分母接近零**（帧 400 j5 的裸步长仅 0.006 mrad）。
实际波形是**接缝附近的短促反向变化**，不是持续台阶：

    j5 δa  帧400 −4.937 → 帧401 +9.373 → 帧402 −0.898  (mrad)

用法:
    python -u sim/audit_send_chain_3c.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEQ = ROOT / "outputs/probe_3c"
OUT = SEQ / "audit_send_chain.json"
DT = 0.05
R2D = 180.0 / np.pi
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
JOINT_FACTOR = 57324.840764           # piper.py: counts per rad（0.001°/count）
# `write()` docstring 里写死的整数范围（单位 0.001°）—— **这是有出处的设备位置范围**
SDK_RANGE = [(-92000, 92000), (-1300, 90000), (-80000, 2400),
             (-90000, 90000), (-77000, 19000), (-90000, 90000)]
# `sim/step2_g/g_v2.py:13` 的加速度上限（°/s²）
GV2_AMAX = np.array([319, 319, 307, 346, 391, 307.0])


def main() -> int:
    import gc
    gc.disable()
    zp = np.load(SEQ / "probe_3c_p.npz")
    zm = np.load(SEQ / "probe_3c_m.npz")
    u = zp["cmd7"].astype(np.float64)
    dap, dam = zp["da"].astype(np.float64), zm["da"].astype(np.float64)
    assert np.abs(dap + dam).max() == 0.0, "两条序列不是严格反号"
    T = len(u)
    res = {"T": T, "dt": DT,
           "index_convention": "np.diff 索引 i ↔ 帧 i→i+1；二阶差分索引 i 涉及帧 i,i+1,i+2"}

    # ── 一、发送链路：单位转换 + 取整 + 整数范围 ───────────────────────
    print("=" * 78 + "\n一、软件发送链路（`piper.py::write`：只做单位转换、取整、发送）\n" + "=" * 78)
    print(f"  换算 {JOINT_FACTOR:.6f} counts/rad ⇒ 1 count = "
          f"{1e3/JOINT_FACTOR:.5f} mrad")
    rng_ok, rows = True, []
    for nm, da in (("u", np.zeros_like(dap)), ("u+δa", dap), ("u−δa", dam)):
        cnt = np.round((u[:, :6] + da) * JOINT_FACTOR).astype(np.int64)
        bad = 0
        for j, (lo, hi) in enumerate(SDK_RANGE):
            bad += int(((cnt[:, j] < lo) | (cnt[:, j] > hi)).sum())
        rows.append({"seq": nm, "out_of_sdk_range": bad,
                     "min": cnt.min(0).tolist(), "max": cnt.max(0).tolist()})
        rng_ok &= bad == 0
        print(f"  {nm:>5}  越 SDK 文档整数范围 {bad} 处  "
              f"{'✓' if bad == 0 else '**✗**'}")
    res["sdk_range"] = {"ok": rng_ok, "rows": rows, "source": "piper.py::write docstring"}

    # ⚠ 取整会不会破坏 ± 反对称 ⇒ 直接污染偶分量？
    c0 = np.round(u[:, :6] * JOINT_FACTOR)
    cp = np.round((u[:, :6] + dap) * JOINT_FACTOR)
    cm = np.round((u[:, :6] + dam) * JOINT_FACTOR)
    even_cnt = cp + cm - 2 * c0                       # 指令层的偶分量（counts）
    even_rad = even_cnt / JOINT_FACTOR
    print(f"\n  ⚠ **取整对 ± 反对称的影响**（会直接进偶分量）：")
    print(f"     指令层偶分量 |cp+cm−2c0| 最大 {int(np.abs(even_cnt).max())} count "
          f"= {np.abs(even_rad).max()*1e3:.5f} mrad · "
          f"RMS {np.sqrt((even_rad**2).mean())*1e3:.5f} mrad")
    print(f"     对照 δa RMS {np.sqrt((dap**2).mean())*1e3:.3f} mrad "
          f"⇒ 占 {np.sqrt((even_rad**2).mean())/np.sqrt((dap**2).mean())*100:.2f}%")
    res["rounding_even_leak"] = {
        "max_counts": int(np.abs(even_cnt).max()),
        "max_mrad": float(np.abs(even_rad).max() * 1e3),
        "rms_mrad": float(np.sqrt((even_rad ** 2).mean()) * 1e3),
        "note": "取整在 ± 两侧独立发生 ⇒ 指令层就有一个偶分量。"
                "它是**发送链路的确定性产物**，不是被控对象的非线性。"}

    # ── 二、关节级绝对约束 ────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n二、关节级绝对指令差分（**不是实测速度/加速度**）\n" + "=" * 78)
    print(f"  {'':5}{'裸 |Δ|max':>12}{'±δa |Δ|max':>13}{'裸 |Δ²|max':>13}"
          f"{'±δa |Δ²|max':>14}{'G-v2 A_MAX':>13}")
    print(f"  {'':5}{'(rad/s)':>12}{'(rad/s)':>13}{'(°/s²)':>13}{'(°/s²)':>14}{'(°/s²)':>13}")
    tab = []
    for j in range(6):
        b1 = np.abs(np.diff(u[:, j])).max() / DT
        b2 = np.abs(np.diff(u[:, j], 2)).max() / DT ** 2 * R2D
        m1 = max(np.abs(np.diff(u[:, j] + d[:, j])).max() / DT for d in (dap, dam))
        m2 = max(np.abs(np.diff(u[:, j] + d[:, j], 2)).max() / DT ** 2 * R2D
                 for d in (dap, dam))
        tab.append({"joint": JOINTS[j], "bare_vel": b1, "tot_vel": m1,
                    "bare_acc": b2, "tot_acc": m2, "gv2_amax": float(GV2_AMAX[j]),
                    "bare_over_gv2": bool(b2 > GV2_AMAX[j]),
                    "delta_margin_pct": float((m2 / b2 - 1) * 100) if b2 > 0 else 0.0})
        mark = " **裸已超**" if b2 > GV2_AMAX[j] else ""
        print(f"  {JOINTS[j]:<5}{b1:>12.3f}{m1:>13.3f}{b2:>13.1f}{m2:>14.1f}"
              f"{GV2_AMAX[j]:>13.0f}{mark}")
    res["per_joint"] = tab

    bmax = max(t["bare_acc"] for t in tab)
    print(f"\n  ⇒ **裸 ACT 轨迹本身**的最大指令二阶差分 = {bmax:.1f} °/s²，"
          f"已是 G-v2 最大 A_MAX 的 **{bmax/GV2_AMAX.max():.1f}×**")
    print(f"  ⇒ δa 的**边际贡献**：逐关节 "
          + " · ".join(f"{t['joint']} +{t['delta_margin_pct']:.1f}%" for t in tab
                       if t["bare_acc"] > 0))

    # ── 三、约束来源的出处与适用性 ────────────────────────────────────
    print("\n" + "=" * 78 + "\n三、约束来源：出处与适用性\n" + "=" * 78)
    src = [
        ("MuJoCo `jnt_range` ±2°", "位置", "模型文件 + 真机实测编码器范围",
         "**适用**（已核对，`u±δa` 0 处越界）"),
        ("SDK 文档整数范围", "位置", "`piper.py::write` docstring",
         "**不适用（陈旧）** —— **裸 `u` 自己就越界 223 处**："
         "j2 实到 106166(0.001°) 而文档写 90000；j5 实到 19094 而文档 19000。"
         "该轨迹已安全跑过 40+ 集 ⇒ 文档范围与实机不符，"
         "**与 `replay_probe.py` 既有注记一致**（MuJoCo jnt_range 是保守软限）。"
         "δa 的边际贡献≈0（u 223 / u+δa 207 / u−δa 228，极值由 u 决定）"),
        ("G-v2 `A_MAX` 319…391 °/s²", "加速度",
         "`g_v2.py:13` —— **90 条套件大幅值真阶跃的二参数拟合**",
         "**不适用于约束请求** —— 它是**模型参数**（描述臂能跟到多少），"
         "不是认证安全上限；**裸轨迹本身就超它 7.6×** 且已安全跑过 40+ 集"),
        ("历史请求包络（j3 118.6 mrad / j5 40.1 mrad）", "步长",
         "`ab5` 裸 ACT 的实测请求", "**不是安全限**，只是「以前请求过这么大」"),
        ("软件速度/加速度限幅", "速度/加速度",
         "`piper.py::write` 只做单位转换、取整、发送",
         "**不存在** —— 发送路径里看不到任何软件限幅；"
         "**不能假定 G 的限幅器在这里兜底**"),
        ("`MotionCtrl_2(0x01, 0x01, 100, 0x00)`", "速度比例",
         "`piper.py:116`，第三参数 = 100",
         "**语义待确认** —— 需核对固件手册：是否为速度百分比、100 是否为上限"),
        ("SDK 关节限速 / 最大加速度**读取**接口", "速度/加速度",
         "厂商 SDK（本机 `piper_sdk` 未在分析环境中）",
         "**必须上机读取** —— 且须确认**本机固件版本、控制模式、单位**"),
    ]
    for nm, kind, prov, appl in src:
        print(f"  · {nm}\n      类型 {kind} · 出处 {prov}\n      ⇒ {appl}")
    res["constraint_sources"] = [
        {"name": n, "kind": k, "provenance": p, "applicability": a} for n, k, p, a in src]

    # ── 判读：三分支 ─────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n判读（用户 2026-09-21 定死的三分支）\n" + "=" * 78)
    print("  · 完整 `u±δa` 满足**有出处、适用于当前设备**的约束 ⇒ 保留波形")
    print("  · 违反约束 ⇒ 不发送原波形；另立**受约束变体**（规则作用于整段，"
          "不是专门修帧 400），重新冻结预测/哈希/信噪")
    print("  · **约束来源仍不明确 ⇒ 安全审查未完成，继续暂停上机**")
    print(f"\n  ⇒ **落在第三支。**")
    print(f"     **位置类**：有出处又适用的那条（MuJoCo ±2° ∩ 真机实测编码器范围）"
          f"**已通过，δa 引入 0 处新越界**；")
    print(f"     SDK docstring 的范围**陈旧不适用**（裸 u 自己越界 223 处且已安全跑过）。")
    print(f"     **加速度/速度类**目前**没有一个有出处又适用于本设备的上限**：")
    print(f"     仓内唯一的逐关节加速度数字（G-v2 `A_MAX`）是**拟合的模型参数**，"
          f"且**裸轨迹超它 7.6×**。")
    print(f"     ⇒ 必须先从本机读到厂商 SDK 的限速/最大加速度，并确认固件、控制模式与单位。")
    print(f"\n  ⚠ **δa 的边际贡献小**（j5 最大 +11.8%，j2 +3.7%）是**相关事实**，"
          f"但**不能代替**一个有出处的上限。")
    print(f"  ⚠ 「固件整形可以属于被测系统」与「请求已满足安全要求」是**两件事**。")

    res["verdict"] = {
        "branch": 3,
        "text": "约束来源仍不明确 ⇒ 安全审查未完成，继续暂停上机",
        "position_ok_applicable": True,   # MuJoCo ±2° ∩ 实测编码器范围，已通过
        "sdk_docstring_range_stale": True,  # 裸 u 自己越界 223 处
        "sdk_range_violations": {"u": 223, "u+da": 207, "u-da": 228},
        "accel_constraint_available": False,
        "must_read_on_machine": [
            "厂商 SDK 的关节限速 / 最大加速度（确认本机固件版本、控制模式、单位）",
            "MotionCtrl_2 第三参数(=100) 的语义与量程"],
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
