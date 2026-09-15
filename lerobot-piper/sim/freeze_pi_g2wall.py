#!/usr/bin/env python3
"""**冻结 C 候选：G-2 预测墙 + H 解析求解**（`pi-g2wall-v1`）。

结构（**零个自由参数**）
------------------------
    ê₀[t] = calib(ω̂_G2(u))[t+ALIGN_L] − ideal[t]          ← G-2 预测「会错多少」
    δa[t] = clip( −(HᵀH+λI)⁻¹Hᵀ · (ê₀[t:t+K] + resp) , ±ε )，接触窗归零
            K = min(M, cap[t] − t)                         ← chunk 钳位

- **G-2**：冻结（`rq_fir` + `rs_fir`），拟合于 `step4_gvnext`
- **H**：冻结（`step1-H-v1`，float64 sha `89d118d4…`），辨识于 `ab4 P1`
- **求解器**：解析，`λ_K=1.0`（取自冻结 sidecar）、`ALIGN_L=3`、`ε=10 mrad`（峰值硬界）
- **本数据上不拟合任何东西** ⇒ 无过拟合风险、无需 train/test 划分；
  16.22% 是**纯样本外**数字（G-2 拟合于 step4，评价于 ab4）

⚠ 模型选择**事前声明**（不按结果挑）
------------------------------------
**用原始冻结的 G-2（v=0），不用 `g2-incr-v1`（v=0.4）。**
理由是**角色匹配**：v=0.4 的增量约束是为改善**增量通道**加的，而增量已不在本结构中；
且它**付了绝对保真度的代价**（ρ² 0.836→0.826 / 0.519→0.463）。
新角色是**绝对预测**，正是原始 G-2 被预注册选中并冻结时的判据
（见 `G2_FROZEN_SCOPE.md`）。
**两个都测并报告，但无论哪个数高都用原始 G-2。**

⚠ 分工与口径
------------
「**G-2 预测墙（绝对）+ H 算修正（增量）**」。H 只作**执行器模型**，
实质预测由 G 承担 —— 而预测正是已测定的瓶颈（W 5.08% → 完美 22.07%）。
用户 2026-09-15 裁定：**算「用 G 训练 π」**。

⚠ **FF 不在本线**（用户 2026-09-15）。两臂 = **A 裸 ACT** / **C 裸 ACT + π**，判据 `C−A`。

⚠ 已知风险（必须随结果报告）
----------------------------
**换会话掉 25.8 pp**：ab4 P0 裸 **52.1%** → ab2 A 裸 **26.3%**，
两边各用本会话既有标定、对齐均成立 ⇒ 干净比较。
成因未分（G-2 跨会话泛化 vs 标定不匹配）。**直接影响部署时能指望多少。**

用法:
    python -u sim/freeze_pi_g2wall.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                    # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402
import delivery_g2_predictor as DG                                  # noqa: E402

OUT = ROOT / "outputs/p2_pi/pi_g2wall_v1.json"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"


class G2v0(G.G2):
    """**原始冻结的 G-2**（v=0）：`r_s` 用 `rs_fir.npz`，而非增量约束版。"""

    def __init__(self):
        z = np.load(RS / "rs_fir.npz")
        self.W = np.ascontiguousarray(z["W"], float)
        self.nlag = int(z["nlag"])
        self.zq = np.load(RQ / "rq_fir.npz")
        self.sq = json.loads((RQ / "select.json").read_text())
        self.man = {"artifact": "G-2 (original, v=0)",
                    "W_sha256": hashlib.sha256(self.W.tobytes()).hexdigest()}


def evaluate(g2, E, H):
    """→ (可预测性, 交付降幅, δa RMS, 触界%)，用 DG 的同一求解器与同一约束。"""
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    preds = []
    for e in E:
        w = g2.omega(e["dd"])
        w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
        ws = S1.calibrate_ideal_imu(w6, e["R32"], e["b32"])[:, 3:]
        n = e["n"]
        preds.append(ws[S1.ALIGN_L:S1.ALIGN_L + n].astype(float) - e["ideal"][:n])
    n_ = d_ = 0.0
    for e, p in zip(E, preds):
        a, b = S1.bandpass41(e["e0"]), S1.bandpass41(p)
        n_ += float(np.square(a - b).sum()); d_ += float(np.square(a).sum())
    das = [DG.roll(e, H, p) for e, p in zip(E, preds)]
    r, rms, sa = DG.score(E, H, das)
    return 1 - n_ / max(d_, 1e-30), r, rms, sa


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R32 = np.asarray(calib["R"], np.float32)
    b32 = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    assert hashlib.sha256(H.tobytes()).hexdigest() == \
        "89d118d4b69837f05636c1a73f3aefbe8d8c2a2fe3f35e37ba647a6c0c4164c2"

    E = []
    for p in S1.discover_logs(ROOT / "outputs/ab4", ("P0",))["P0"]:
        ep = S1.load_episode(p, "P0", np.asarray(R32, np.float64),
                             np.asarray(b32, np.float64), rend, 550)
        pay = json.loads(p.read_text())
        _, ideal_sim, cap = S1._render_logged_chunks(
            pay.get("chunks", []), rend, len(pay["frames"]))
        ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R32, b32)[:, 3:6]
        n = min(len(ep.gap), len(ideal))
        E.append({"name": p.stem, "n": n, "e0": np.asarray(ep.gap, float)[:n],
                  "ideal": ideal[:n].astype(float),
                  "contact": np.asarray(ep.contact_mask, bool)[:n],
                  "cap": np.asarray(cap[:n], int),
                  "dd": G.ab4_to_dd(p, p.stem), "R32": R32, "b32": b32})

    print(f"ab4 P0 {len(E)} 集 · ε=10 mrad 峰值 · chunk 钳位 · 接触门控 · "
          f"ALIGN_L={S1.ALIGN_L} 固定")
    print(f"**事前声明：用原始 G-2(v=0)，不按结果挑**\n")
    print(f"  {'G-2 变体':<26}{'可预测性':>10}{'交付降幅':>10}{'δa RMS':>9}{'触界%':>8}")
    got = {}
    for tag, mdl in (("**原始 G-2（v=0）← 选用**", G2v0()),
                     ("g2-incr-v1（v=0.4）", G.G2())):
        pr, r, rms, sa = evaluate(mdl, E, H)
        got[tag.replace("*", "")] = {"predictability": pr, "reduction": r,
                                     "da_rms_mrad": rms, "sat_pct": sa,
                                     "W_sha256": mdl.man["W_sha256"]}
        print(f"  {tag:<26}{pr*100:>9.1f}%{r*100:>9.2f}%{rms:>9.2f}{sa:>8.2f}")

    k = list(got)
    sel = got[k[0]]
    man = {
        "artifact": "pi-g2wall-v1",
        "structure": ("e0_hat = calib(omega_G2(u))[t+3] - ideal[t]; "
                      "da = clip(-(H^T H + lam I)^-1 H^T (e0_hat[t:t+K] + resp), +-eps); "
                      "K = min(M, cap[t]-t); contact-gated"),
        "free_parameters": 0,
        "components": {
            "G2": {"variant": "original v=0 (rq_fir + rs_fir)",
                   "W_sha256": sel["W_sha256"],
                   "chosen_by": "**事前声明**: 角色=绝对预测, 与 G-2 原冻结判据一致; "
                                "不按结果挑; v=0.4 的增量约束已不在本结构中且代价 "
                                "rho2 0.836->0.826 / 0.519->0.463"},
            "H": {"artifact": "step1-H-v1",
                  "sha256": "89d118d4b69837f05636c1a73f3aefbe8d8c2a2fe3f35e37ba647a6c0c4164c2"},
            "solver": {"lambda_K": DG.KMULT, "align_L": S1.ALIGN_L, "eps_rad": DG.EPS,
                       "eps_meaning": "**瞬时峰值**硬界(非 RMS)",
                       "chunk_clamp": True, "contact_gating": True}},
        "offline_ab4_P0": {k[0].replace("*", ""): sel, k[1]: got[k[1]]},
        "arms": {"A": "裸 ACT", "C": "裸 ACT + pi", "gate": "C - A",
                 "note": "**FF 不在本线**(用户 2026-09-15)"},
        "known_risk": {
            "session_effect": ("**换会话掉 25.8 pp**: ab4 P0 裸 52.1% -> ab2 A 裸 26.3%; "
                               "两边各用本会话既有标定、对齐均成立 => 干净比较; "
                               "成因未分(G-2 跨会话泛化 vs 标定不匹配); "
                               "**直接影响部署时能指望多少**")},
        "caliber": ("**离线**数字, 被控对象是 H(模型), **非真机验证**; "
                    "本数据上不拟合任何东西 => 16.22% 是纯样本外(G-2 拟合于 step4, "
                    "评价于 ab4); ab4 无时间戳 => G 前向假设均匀 20Hz; "
                    "分工=G-2 预测墙(绝对)+H 算修正(增量), 用户裁定算「用 G 训练 π」"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"\n  **冻结 `pi-g2wall-v1`** —— 自由参数 **0** 个")
    print(f"  G-2(v=0) W sha {sel['W_sha256'][:16]}… · H sha 89d118d4…")
    print(f"  ⚠ 已知风险已写入 manifest：**换会话掉 25.8 pp**，成因未分")
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
