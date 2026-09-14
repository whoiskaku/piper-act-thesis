#!/usr/bin/env python3
"""**有边界的零机时验证**：逐次单发抽取的 J，能否复现 G-2 前向差分？

为什么要重抽
------------
旧 J 每隔 **16 帧**注入一发，而单发响应在第 19 帧仍未消失（峰 5.7 → lag19 约 0.2）
⇒ **前一发的尾部污染后一发的窗口**。症状：核的 tail ratio 1.4、逐集平均峰跑到
`τ_raw=+12`、单发脉冲复现失败（cos 0.338）。

**这是在修测量算子，不是再调 G。**

本步的边界（用户 2026-09-14）
-----------------------------
- 只在**预定的少数时刻**、**逐次单发**抽取（每次前向只注入一发）
- 用**未参与抽取**的脉冲 + **任务式扰动**检验
- **通过才扩到全数据**；不过就收口，**不能继续拿 J 训练**
- 本步**不训练**，也**不**立刻对 13 集全部重抽

⚠ 两处已订正的口径
------------------
- gap 轴的"负滞后"是 `gap[t]=real[t+3]−ideal[t]` 的**记账方式**，
  把物理上发生在扰动**之后**的响应标到了负滞后 ——
  **不是**因果不可纠，**也不**等于"降幅高估"。求解器须按**真实控制时间**处理，不得丢弃。
- 旧 J 的 `+9` 帧峰值目前只是**抽取污染嫌疑**，**不得**报成 G-2 已确认的延迟。

⚠ ab4 无时间戳 ⇒ J 在**假设均匀 20 Hz** 下抽出（随结果报告）。

用法:
    python -u sim/verify_J_single.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g2_on_ab4 as G                                               # noqa: E402
import fit_step1_candidate as S1                                    # noqa: E402

FIT_T = (120, 200, 280, 360, 440)        # 抽取时刻（预定，少数）
VAL_T = (160, 320)                       # **未参与抽取**的检验时刻
KLEN = 24                                # 核长：覆盖实测响应（lag19 仍有残余）
BAND = (0.5, 9.0)
COS_GATE = 0.85


def extract_single(g2, dd, T):
    """**逐次单发**抽取：每次前向只注入一发。→ (t0s, J[len,KLEN,3,6])"""
    J = np.zeros((len(FIT_T), KLEN, 3, 6))
    for a, t0 in enumerate(FIT_T):
        for j in range(6):
            da = np.zeros((T, 6)); da[t0, j] = G.AMP       # **单发**
            od = g2.odd(dd, da)
            w = od[t0:t0 + KLEN]
            J[a, :len(w), :, j] = w / G.AMP
    return np.asarray(FIT_T), J


def dense(t0s, J, T):
    return np.stack([np.stack([np.stack(
        [np.interp(np.arange(T), t0s, J[:, k, c, j]) for j in range(6)], -1)
        for c in range(3)], -2) for k in range(KLEN)], 1)


def conv(Jd, da, T):
    out = np.zeros((T, 3))
    for k in range(Jd.shape[1]):
        if k:
            out[k:] += np.einsum("tcj,tj->tc", Jd[:T - k, k], da[:T - k])
        else:
            out += np.einsum("tcj,tj->tc", Jd[:, 0], da)
    return out


def main() -> int:
    import gc
    gc.disable()
    from data import bp
    g2 = G.G2()
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P1",))
    p = logs["P1"][0]
    dd = G.ab4_to_dd(p, p.stem)
    T = len(dd["t_read"])
    print(f"集 {p.stem} · T={T} · **逐次单发**抽取于 t0={list(FIT_T)} · 核长 {KLEN}")
    print(f"⚠ ab4 无时间戳 ⇒ 假设均匀 20 Hz\n")
    t0s, J = extract_single(g2, dd, T)
    Jd = dense(t0s, J, T)

    nJ = np.linalg.norm(J.reshape(len(FIT_T), KLEN, -1), axis=2).mean(0)
    pk = int(np.argmax(nJ))
    print(f"  核形状：峰在 τ_raw={pk:+d}（ω̂ 轴）· 尾比 ‖J({KLEN-1})‖/‖J(峰)‖ = "
          f"{nJ[-1]/max(nJ[pk],1e-12):.3f}")
    print(f"  （旧 16 帧间隔版：峰 +12、尾比 1.4 —— 污染嫌疑）\n")

    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✓' if cond else '**✗**'} {name}{('  ' + detail) if detail else ''}")

    print("① **未参与抽取**的单发脉冲")
    for t0 in VAL_T:
        for j, jn in ((1, "j2"), (4, "j5")):
            da = np.zeros((T, 6)); da[t0, j] = G.AMP
            true = g2.odd(dd, da)
            m = min(T, len(true))
            a, b = bp(true[:m], *BAND), bp(conv(Jd, da, T)[:m], *BAND)
            cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
            rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
            chk(f"t0={t0} {jn}", cos > COS_GATE, f"cos {cos:.3f} · 相对误差 {rel:.3f}")

    print("\n② **任务式扰动**（带限噪声，与 ab4/部署同族）")
    from lerobot.common.policies.act.b_correction_deploy import make_probe
    for mr in (1.0, 3.0):
        w = np.asarray(make_probe(7001, mr), float)
        da = np.zeros((T, 6)); n = min(T, len(w)); da[:n] = w[:n].reshape(n, -1)[:, :6]
        true = g2.odd(dd, da)
        m = min(T, len(true))
        a, b = bp(true[:m], *BAND), bp(conv(Jd, da, T)[:m], *BAND)
        cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        nr = float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-30))
        chk(f"带限噪声 {mr:.0f} mrad RMS", cos > COS_GATE,
            f"cos {cos:.3f} · 范数比 {nr:.3f} · 相对误差 {rel:.3f}")

    print(f"\n判决: {'✅ 通过 —— 可扩到全 13 集重抽' if ok else '❌ 未通过 —— 收口，不得继续拿 J 训练'}")
    (ROOT / "outputs/step4_gvnext/J_ab4/single_check.json").write_text(json.dumps(
        {"fit_t": list(FIT_T), "val_t": list(VAL_T), "klen": KLEN,
         "peak_raw": pk, "tail_ratio": float(nJ[-1] / max(nJ[pk], 1e-12)),
         "pass": bool(ok), "cos_gate": COS_GATE,
         "caliber": ("逐次单发抽取(每次前向只注入一发); 检验脉冲与任务式扰动**未参与抽取**; "
                     "ab4 无时间戳 ⇒ 假设均匀 20Hz; "
                     "gap 轴负滞后是 gap[t]=real[t+3]-ideal[t] 的记账方式, "
                     "**非**因果不可纠, 求解器须按真实控制时间处理"),
         }, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
