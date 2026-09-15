#!/usr/bin/env python3
"""评估蒸馏版 π：**α（沿教师的有用分量）+ 真机实测墙上的交付量**（零机时）。

为什么按 α 评估
---------------
`sim/pi_distill_fidelity_curve.py` 实测：把 π 的输出拆成 `δa = α·δa* + ε`，
交付量几乎只由 **α** 决定，正交残差 `ε` 很便宜（ε 的 RMS 与 δa* 同量级也只掉 ~12%）。

    α  0.50 → 7.34% | **0.75 → 10.33%** | 1.00 → 12.86%（教师，零空档）
    验收门槛 **α ≥ 0.75**（即不输给带空档的 C 臂真机 −8.0%）

⚠ **不要追 α > 1**
------------------
H 模型里 α=2.0 还在涨（17.10%），但那是**模型外推**：
真机 D 臂已证 **大幅值、结构化的 δa 会让 H 的预测失准并实际有害**
（`RESULT_pi_net_AD.md` §4：反解出的反事实墙比 A 高 48.7%，自相矛盾）。
H 的有效区间是它辨识时的激励形态（带限白噪声，1.5–3 mrad RMS）与 C 臂的 δa
（0.54 mrad、由 H 解出）。**π 的幅度应当留在这个区间内。**

口径
----
- **留出集**（训练时划分的那批），真机实测墙 `e₀` 来自 `ab5 A` / `ab6 A`
- 教师 δa* = 冻结 `G-2 + H 解析求解`，**零空档**
- 交付量 = `1 − ‖bp(e₀ + H⊛δa)‖²/‖bp(e₀)‖²` —— 与预测 C 臂 −7.6%（实测 −8.0%）同法
- α = `⟨δa_π, δa*⟩ / ‖δa*‖²`（逐集算，再按能量加权汇总）

用法:
    python -u sim/verify_pi_distill.py [--tag pi_distill_v1]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                     # noqa: E402
import train_pi_g2env as T                                           # noqa: E402

EPS, L = 0.010, S1.ALIGN_L
OUT = ROOT / "outputs/p2_pi/verify_pi_distill.json"
CAL = {"ab5": "ab5-deploy", "ab6": "ab6-deploy"}


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pi_distill_v1")
    a = ap.parse_args()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from lerobot.common.policies.act.g2wall_runtime import G2WallRuntime

    H = np.ascontiguousarray(
        np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    rend = IdealChunkRenderer()
    ck = torch.load(ROOT / f"outputs/p2_pi/{a.tag}.pt", map_location=T.DEV,
                    weights_only=False)
    net = T.PiNet().to(T.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
    mu, sd = ck["norm_mu"], ck["norm_sd"]
    # 只取**留出**的裸 ACT 集（有真机实测墙）
    val = [v for v in ck["val_episodes"] if "/A_ep" in v and v.split("/")[0] in CAL]
    print(f"评估 `{a.tag}` · 留出的裸 ACT 集 {len(val)} 条（有真机实测墙）")
    print(f"验收门槛 **α ≥ 0.75**；⚠ 不要追 α>1（H 在大幅值结构化 δa 上已知失准）\n")
    print(f"{'集':<14}{'α':>7}{'δa_π RMS':>10}{'δa* RMS':>9}"
          f"{'教师交付':>10}{'π 交付':>9}")

    num = den = 0.0
    tnum = 0.0
    aw_n = aw_d = 0.0
    rows = []
    rts = {}
    for name in val:
        arm_dir, stem = name.split("/")
        man = json.loads(
            (ROOT / f"outputs/p2_labels/{CAL[arm_dir]}/manifest.json").read_text())
        R = np.asarray(man["R"], np.float32); b = np.asarray(man.get("b", [0, 0, 0]), np.float32)
        if arm_dir not in rts:
            rts[arm_dir] = G2WallRuntime(ROOT / "outputs/p2_pi/pi_g2wall_v1.json",
                                         R, b, verbose=False)
        rt = rts[arm_dir]
        p = ROOT / f"outputs/{arm_dir}/{stem}.json"
        ep = S1.load_episode(p, "P0", np.asarray(R, np.float64),
                             np.asarray(b, np.float64), rend, 500)
        pay = json.loads(p.read_text())
        chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
        rt.reset(); star, prev = [], None
        for c in chunks:
            raw = np.asarray(c["raw"], np.float64)
            ideal = np.asarray(rend.render(raw.astype(np.float32), prev), np.float32)
            rt.begin_chunk(raw); rt.fill_stage_b(ideal)
            star.append(np.stack([rt.step().delta for _ in range(len(raw))]))
            prev = raw.astype(np.float32)
        star = np.concatenate(star)

        z = np.load(ROOT / "outputs/p2_pi/g2env" / (name.replace("/", "__") + ".npz"))
        x, spans = T.build_inputs(z["u"], z["starts"], int(z["n"]))
        with torch.no_grad():
            dch = net(torch.from_numpy(((x - mu) / sd).astype(np.float32)
                                       ).to(T.DEV)).cpu().numpy()
        dpi = np.zeros((int(z["n"]), 6))
        for i, (s0, lc) in enumerate(spans):
            dpi[s0:s0 + lc] = dch[i, :lc]
        dpi[z["contact"][:len(dpi)]] = 0.0

        g = np.asarray(ep.gap, float)
        m = min(len(g), len(star), len(dpi))
        g, star, dpi = g[:m], star[:m], dpi[:m]
        al = float((dpi * star).sum() / max((star ** 2).sum(), 1e-30))
        aw_n += float((dpi * star).sum()); aw_d += float((star ** 2).sum())
        e = lambda d: float((S1.bandpass41(g + S1.causal_conv(H, d)[:m]) ** 2).sum())
        b0 = float((S1.bandpass41(g) ** 2).sum())
        et, epi = e(star), e(dpi)
        num += epi; tnum += et; den += b0
        rows.append({"name": name, "alpha": al,
                     "da_pi_rms_mrad": float(np.sqrt((dpi ** 2).mean()) * 1e3),
                     "da_star_rms_mrad": float(np.sqrt((star ** 2).mean()) * 1e3),
                     "teacher": 1 - et / b0, "pi": 1 - epi / b0})
        r = rows[-1]
        print(f"{name:<14}{al:>7.2f}{r['da_pi_rms_mrad']:>10.2f}"
              f"{r['da_star_rms_mrad']:>9.2f}{r['teacher']*100:>9.2f}%{r['pi']*100:>8.2f}%")

    A = aw_n / max(aw_d, 1e-30)
    dt, dp = 1 - tnum / den, 1 - num / den
    print(f"\n  ── 汇总（留出集，真机实测墙）──")
    print(f"  **α = {A:.2f}**（门槛 0.75）")
    print(f"  教师（零空档解析解）  {dt*100:+.2f}%")
    print(f"  **π**                {dp*100:+.2f}%   = 教师的 {dp/dt*100:.0f}%")
    print(f"  δa RMS: π {np.mean([r['da_pi_rms_mrad'] for r in rows]):.2f} mrad  vs  "
          f"教师 {np.mean([r['da_star_rms_mrad'] for r in rows]):.2f} mrad")
    print(f"\n  ⚠ 对照 C 臂真机实测 **−8.0%**；π 的幅度若远超教师，"
          f"\n     则落在 **H 已知失准**的区间（D 臂教训），不可只看这里的数字。")
    OUT.write_text(json.dumps(
        {"tag": a.tag, "alpha": A, "teacher_delivery": dt, "pi_delivery": dp,
         "per_episode": rows,
         "caliber": ("留出集; 真机实测墙 e0; 交付量 = 1−‖bp(e0+H⊛δa)‖²/‖bp(e0)‖², "
                     "与预测 C 臂 −7.6%(实测 −8.0%)同法; "
                     "α = <δa_pi, δa*>/‖δa*‖²; 教师 = 零空档 G-2+H 解析解; "
                     "**H 对大幅值结构化 δa 已知失准(D 臂)** ⇒ 幅度远超教师时此数不可信"),
         }, ensure_ascii=False, indent=1))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
