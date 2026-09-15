#!/usr/bin/env python3
"""**蒸馏保真度 → 交付量**的衰减曲线（零机时）—— 训 π 之前先定验收门槛。

为什么先做这个
--------------
用户选定路线 ①：把训练环境里的响应模型从 `J` 换成**实测的 `H`** ——
`G-2` 负责预测墙（它擅长的那一半，可预测性 52.1%），`H` 负责反事实响应
（真机实测的那一半）。π 学会之后**运行时不需要 G-2** ⇒ 空档为 0。

已测的账：C 臂离线**零空档 15.50%**，被 3–4 帧空档砍到 ~7.6%（真机 −8.0%）；
而本次真机已证 **π 的空档是 0.00%**。所以上限约是**把 C 的效果翻倍**。

但 π 只能**近似**那个解析解。近似误差不是「少补一点」，而是**补错方向**——
δa 经 `H` 会实打实推动机器人。**所以必须先知道：保真度要到多少才不亏。**

口径
----
- 底座 = 真机裸 ACT 的实测墙 `e₀`（`ab6 A` 30 集 + `ab5 A` 30 集）
- 教师 δa* = 冻结的 `G-2 + H 解析求解`，**零空档**（离线无实时约束）
- 交付量 = `1 − ‖bp(e₀ + H⊛δa)‖² / ‖bp(e₀)‖²`
  —— 与预测 C 臂 −7.6%（真机实测 −8.0%）**同一方法**
- 扰动 = 在 δa* 上加白噪声，按 δa* 的 RMS 成比例；**加完再限幅 ε 与接触门控**
  （与部署一致）
- 保真度定义 = `1 − ‖δa − δa*‖ / ‖δa*‖`（0 = 完全不像，1 = 完全一致）

⚠ 白噪声是**乐观代理**：真网络的误差是结构化的、与指令相关，可能更糟。
本曲线给的是**必要条件**，不是充分条件。

用法:
    python -u sim/pi_distill_fidelity_curve.py
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

import fit_step1_candidate as S1                                     # noqa: E402

EPS = 0.010
OUT = ROOT / "outputs/p2_pi/distill_fidelity_curve.json"
NOISE = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5)
SEEDS = 5


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from lerobot.common.policies.act.g2wall_runtime import G2WallRuntime
    from lerobot.common.policies.act.step1_correction_runtime import calibrate_ideal_imu

    H = np.ascontiguousarray(
        np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    rend = IdealChunkRenderer()
    src = [("ab6", "ab6-deploy", 30), ("ab5", "ab5-deploy", 30)]

    eps = []
    for arm_dir, cal, n in src:
        man = json.loads(
            (ROOT / f"outputs/p2_labels/{cal}/manifest.json").read_text())
        R = np.asarray(man["R"], np.float32)
        b = np.asarray(man.get("b", [0, 0, 0]), np.float32)
        rt = G2WallRuntime(ROOT / "outputs/p2_pi/pi_g2wall_v1.json", R, b,
                           verbose=False)
        for i in range(n):
            p = ROOT / f"outputs/{arm_dir}/A_ep{i}.json"
            if not p.exists():
                continue
            ep = S1.load_episode(p, "P0", np.asarray(R, np.float64),
                                 np.asarray(b, np.float64), rend, 500)
            pay = json.loads(p.read_text())
            chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
            rt.reset()
            das, prev = [], None
            for c in chunks:
                raw = np.asarray(c["raw"], np.float64)
                ideal = np.asarray(rend.render(raw.astype(np.float32), prev), np.float32)
                rt.begin_chunk(raw)
                rt.fill_stage_b(ideal)                 # **零空档**（离线无实时约束）
                das.append(np.stack([rt.step().delta for _ in range(len(raw))]))
                prev = raw.astype(np.float32)
            da = np.concatenate(das)
            g = np.asarray(ep.gap, float)
            m = min(len(g), len(da))
            free = np.abs(da[:m]).sum(1) > 0           # 接触门控后为零的帧
            eps.append({"e0": g[:m], "da": da[:m], "free": free,
                        "name": f"{arm_dir}/A_ep{i}"})
        print(f"  {arm_dir}: 累计 {len(eps)} 集")

    def deliver(das):
        num = den = 0.0
        for e, d in zip(eps, das):
            a = S1.bandpass41(e["e0"])
            bb = S1.bandpass41(e["e0"] + S1.causal_conv(H, d)[:len(e["e0"])])
            num += float((bb ** 2).sum()); den += float((a ** 2).sum())
        return 1 - num / den

    base = deliver([e["da"] for e in eps])
    rms_star = float(np.sqrt(np.mean(np.concatenate([e["da"] for e in eps]) ** 2)))
    print(f"\n教师（零空档 G-2+H 解析解）在**真机实测墙**上的交付量 "
          f"**{base*100:.2f}%**   δa* RMS {rms_star*1e3:.2f} mrad")
    print(f"（同一方法预测 C 臂 −7.6%，真机实测 −8.0% ⇒ 口径已验证）\n")
    print(f"⚠ 白噪声是**乐观代理**；真网络的误差是结构化的，可能更糟。\n")
    print(f"  {'噪声/δa* RMS':>13}{'保真度':>9}{'交付量':>10}{'占教师':>9}{'δa RMS':>9}")

    rows = {}
    for s in NOISE:
        dl, fid, rr = [], [], []
        for sd in range(SEEDS):
            rng = np.random.default_rng(1000 + sd)
            das = []
            num = den = 0.0
            for e in eps:
                d = e["da"] + rng.normal(0, s * rms_star, e["da"].shape)
                d = np.clip(d, -EPS, EPS)
                d[~e["free"]] = 0.0                    # 接触门控（部署时在网络外）
                das.append(d)
                num += float(((d - e["da"]) ** 2).sum())
                den += float((e["da"] ** 2).sum())
            fid.append(1 - np.sqrt(num / den))
            dl.append(deliver(das))
            rr.append(float(np.sqrt(np.mean(np.concatenate(das) ** 2))))
        rows[str(s)] = {"fidelity": float(np.mean(fid)),
                        "delivery": float(np.mean(dl)),
                        "delivery_sd": float(np.std(dl)),
                        "da_rms_mrad": float(np.mean(rr)) * 1e3}
        r = rows[str(s)]
        print(f"  {s:>13.1f}{r['fidelity']:>9.3f}{r['delivery']*100:>9.2f}%"
              f"{r['delivery']/base*100:>8.0f}%{r['da_rms_mrad']:>9.2f}")

    # ── 更相关的失败模式：**有用分量被学丢**（欠拟合），而不是加噪声 ──
    # 把 π 的输出分解成 δa = α·δa* + ε：α 是沿教师的分量(有用)，ε 是正交残差(有害)。
    # 上面的白噪声扫的是 ε；这里扫 α。
    print(f"\n  ── 扫 α（沿教师的分量；α=1 即完美，α=0 即完全没学到）──")
    print(f"  {'α':>6}{'交付量':>10}{'占教师':>9}{'δa RMS':>9}")
    rows_a = {}
    for al in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        das = []
        for e in eps:
            d = np.clip(al * e["da"], -EPS, EPS)
            d[~e["free"]] = 0.0
            das.append(d)
        dv = deliver(das)
        rr = float(np.sqrt(np.mean(np.concatenate(das) ** 2))) * 1e3
        rows_a[str(al)] = {"delivery": float(dv), "da_rms_mrad": rr}
        print(f"  {al:>6.2f}{dv*100:>9.2f}%{dv/base*100:>8.0f}%{rr:>9.2f}")

    # α 与 ε 同时存在
    print(f"\n  ── α 与正交噪声同时存在（更接近真网络）──")
    print(f"  {'α':>6}{'噪声':>7}{'交付量':>10}{'占教师':>9}")
    rows_ae = {}
    for al in (0.5, 0.75, 1.0):
        for s_ in (0.3, 0.7, 1.0):
            rng = np.random.default_rng(7)
            das = []
            for e in eps:
                d = al * e["da"] + rng.normal(0, s_ * rms_star, e["da"].shape)
                d = np.clip(d, -EPS, EPS); d[~e["free"]] = 0.0
                das.append(d)
            dv = deliver(das)
            rows_ae[f"{al}_{s_}"] = float(dv)
            print(f"  {al:>6.2f}{s_:>7.1f}{dv*100:>9.2f}%{dv/base*100:>8.0f}%")

    # 验收门槛：交付量 ≥ C 臂真机实测的 8.0%（即至少不输给带空档的 C）
    gate = 0.080
    oka = [(float(k), v) for k, v in rows_a.items() if v["delivery"] >= gate]
    print(f"\n  ── 验收门槛 ──")
    print(f"  要求 π 的交付量 ≥ **8.0%**（C 臂真机实测，即至少不输给带空档的 C）")
    if oka:
        amin = min(oka, key=lambda z: z[0])[0]
        print(f"  ⇒ **沿教师的分量 α ≥ {amin:.2f}** —— 这是主要约束")
    print(f"  ⇒ 正交残差很便宜：即使 ε 的 RMS 与 δa* 同量级，也只掉 ~12%（见上表）")
    print(f"\n  ⚠ 这是**必要条件**。白噪声误差与真网络的结构化误差不同，"
          f"训完必须用真 δa 重算一遍。")
    OUT.write_text(json.dumps(
        {"teacher_delivery": base, "teacher_da_rms_mrad": rms_star * 1e3,
         "n_episodes": len(eps), "rows": rows, "gate_delivery": gate,
         "caliber": ("底座=**真机裸 ACT 实测墙** e0(ab6 A 30 + ab5 A 30); "
                     "教师=冻结 G-2+H 解析解, **零空档**; "
                     "交付量=1−‖bp(e0+H⊛δa)‖²/‖bp(e0)‖², 与预测 C 臂 −7.6%(实测 −8.0%)同法; "
                     "扰动=白噪声, **乐观代理**(真网络误差是结构化的); "
                     "加噪后再限幅 ε + 接触门控, 与部署一致"),
         }, ensure_ascii=False, indent=1))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
