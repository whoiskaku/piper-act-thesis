#!/usr/bin/env python3
"""**诊断（非实验、非选参）**：柔性铰的贡献 Δ 是否**本来就落在 `r_s` 的张成里**？

阶段 2 未通过（0.520 → 0.524）。要判断这是
  (a) 结构本身没用，还是
  (b) 阶段 1 的选格判据（底座层 IMU ρ²）选错了点，
只需一个问题：**同一组 FIR 特征能不能自己造出 Δ？**

若能 ⇒ 底座层的 +13.7% 是 `r_s` 早已能提供的东西，不是增量 ⇒ 结构没用。
（项目里早有结论：**因果 FIR 张成 LTI 传递率**，而 `kin` 就是它的特征之一。）

⚠ 探针里那条「ρ(Δ, r_s 输出) = 0.202 ⇒ 几乎正交」是**错的测法**：
   与**已拟合输出**低相关，不等于落在**特征张成之外**。本脚本测的才是后者。

只用训练族、折外；**开发验证集完全不参与**。

用法:
    python -u sim/step4_gvnext/diag_flex_span.py
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

from data import bp, valid_files                                    # noqa: E402
from train_rq import fir_design, stack_band, fit, BAND, W_BAND, KFOLD  # noqa: E402
from train_rs import band_err3                                      # noqa: E402
from stage2_flex import build, S1, OUT                              # noqa: E402


def main() -> int:
    import gc
    gc.disable()
    from g_q_v1 import THETA_STAR
    from sim.step2_gap_baseline import MODEL
    from structure_compare import flex_xml

    g = json.loads((OUT / "grid.json").read_text())
    sel = max([r for r in g["rows"] if r["enc_ok"]], key=lambda r: r["r39"])
    xml = flex_xml(4, sel["w"], sel["z"], THETA_STAR)
    zq = np.load(ROOT / "outputs/step4_gvnext/rq/rq_fir.npz")
    sq = json.loads((ROOT / "outputs/step4_gvnext/rq/select.json").read_text())
    ss = json.loads((ROOT / "outputs/step4_gvnext/rs/select.json").read_text())
    TR = build(valid_files(S1), MODEL, xml, THETA_STAR, zq, sq, oof=zq)
    nlag = ss["nlag"]
    print(f"柔性铰 ω={sel['w']:.3f} Hz ζ={sel['z']}   训练族 {len(TR)} 条   "
          f"FIR nlag={nlag}（与 r_s 同一特征集/阶数）\n")

    D_ = [fir_design(e["Xs"], nlag) for e in TR]
    folds = [list(range(i, len(TR), KFOLD)) for i in range(KFOLD)]
    print(f"  {'lam':>8}{'折外可解释的 Δ（3–9 Hz）':>26}{'（0.5–3 Hz）':>16}")
    best = 0.0
    for lam in (1e-4, 1e-3, 1e-2):
        n39 = d39 = n13 = d13 = 0.0
        for te in folds:
            tr = [i for i in range(len(TR)) if i not in te]
            W = fit([stack_band(D_[i], TR[i]["D"], W_BAND)[0] for i in tr],
                    [stack_band(D_[i], TR[i]["D"], W_BAND)[1] for i in tr], lam)
            for i in te:
                P = D_[i] @ W
                a, b = band_err3(TR[i]["D"], P, *BAND); n39 += a; d39 += b
                a, b = band_err3(TR[i]["D"], P, 0.5, 3.0); n13 += a; d13 += b
        r39, r13 = 1 - n39 / max(d39, 1e-30), 1 - n13 / max(d13, 1e-30)
        best = max(best, r39)
        print(f"  {lam:>8.0e}{r39*100:>25.1f}%{r13*100:>15.1f}%")

    dE = np.mean([np.sqrt(np.mean(bp(e["D"], *BAND) ** 2)) for e in TR])
    yE = np.mean([np.sqrt(np.mean(bp(e["Y_old"], *BAND) ** 2)) for e in TR])
    print(f"\n  Δ 的带内 RMS {dE:.5f} rad/s = r_s 目标带内 RMS 的 {dE/yE*100:.0f}%")
    print(f"  **r_s 的特征张成能折外重现 Δ 的 {best*100:.1f}%（3–9 Hz）**")
    print("\n  判读：>80% ⇒ Δ 本来就在 r_s 能力之内，柔性铰对完整模型无增量；"
          "\n        <50% ⇒ Δ 确在张成之外，则阶段 2 的失败另有原因。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
