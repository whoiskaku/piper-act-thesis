#!/usr/bin/env python3
"""**导出并冻结 ab4 的开环 Ĥ** 为独立 artifact（不再依赖 Step-1 候选那一整套）。

为什么单独导出
--------------
`Ĥ` 是**被控对象属性**，已过两关（`reports/STEP1_RESULT_v1.md` §1）：
  · P2 跨 seed β = 0.774 / 0.955 / 1.005，中位 **0.9546**（门槛 ≥0.5 且全 >0）
  · P3 幅值线性 β_P3/median(β_P2) = **1.0134**（门槛 [0.7, 1.3]）
被**判否的是 Step-1 的候选构造方式**（2.65%），**不是 `Ĥ` 本身**。
所以把 `Ĥ` 与那个已判否的 θ/W 解耦，单独冻结、可独立引用。

口径（写死，不可改）
--------------------
- 数据 **只用 `outputs/ab4` 的 P1（8 集）拟合**；P2（6 集）/ P3（4 集）**只做验证**。
- P0 **不参与** `Ĥ` 的拟合（它无探针，是 baseline 层）。
- 外生开环：探针序列外部生成、**不读 IMU**⇒ 无回路回声。
- 对齐 `gap[t] = real[t+ALIGN_L] − ideal[t]`，`ALIGN_L=3`（141 ms 伺服滞后被它吸收）。
- 拟合空间 = **raw aligned gyro gap**；逐集常数作为 nuisance 扣除（不是带通）。
- ridge multiplier 取自**冻结的** sidecar，不在此重新决策。

⚠ 不做的事
-----------
- **不**给闭环 `sim/measure_da_response.py` 增加任何部署用途。
  D/E/F 闭环 `Ĥ` **只作诊断**，不用于辨识或验证（同一件事用闭环数据做时是灾难性失败：
  E 拟合 / F 评估从 −86.7% 翻成 +187.8%，见 `reports/STEP1_GATE.md`）。
- **不**改 `sim/fit_step1_candidate.py`（本脚本把它当模块导入）。
- **不**重采 `probe_b2`（6–8 Hz / j2·j3，被 ab4 的 0.5–9 Hz × 六关节完全覆盖）。

用法:
    python -u sim/export_step1_H.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                   # noqa: E402

FROZEN_SIDECAR = ROOT / "outputs/step1_candidate/step1_linear_v1_p2.json"
FROZEN_H_SHA = "89d118d4b69837f05636c1a73f3aefbe8d8c2a2fe3f35e37ba647a6c0c4164c2"
OUT = ROOT / "outputs/step1_H/step1_H_v1.npz"


def h_sha(H: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(H, dtype=np.float64).tobytes()).hexdigest()


def main() -> int:
    import gc
    gc.disable()
    frozen = json.loads(FROZEN_SIDECAR.read_text())
    h_mult = float(frozen["cv"]["chosen_multipliers"]["H"])
    calib_path = Path(frozen["calibration"]["path"])
    print(f"冻结决策来自 {FROZEN_SIDECAR.name}   h_multiplier={h_mult:g}"
          f"   （本脚本不重新决策 ridge）")

    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0", "P1", "P2", "P3"))
    calibration = json.loads(calib_path.read_text())
    R = np.asarray(calibration["R"], dtype=np.float64)
    b = np.asarray(calibration.get("b", [0.0, 0.0, 0.0]), dtype=np.float64)
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    renderer = IdealChunkRenderer()
    eps = {t: [S1.load_episode(p, t, R, b, renderer, 550) for p in ps]
           for t, ps in logs.items()}
    S1.validate_protocol(eps)
    print(f"  ab4 载入 " + " · ".join(f"{t} {len(v)} 集" for t, v in sorted(eps.items())))

    # ── 拟合：只用 P1 ────────────────────────────────────────────────────
    H, h_lambda = S1.fit_H(eps["P1"], h_mult)
    sha = h_sha(H)
    print(f"\n  H shape {H.shape}  dtype float64  λ_H = {h_lambda:.10e}")
    print(f"  H float64 sha256 {sha}")
    if sha != FROZEN_H_SHA:
        print(f"  **✗ 与冻结版不一致**（期望 {FROZEN_H_SHA}）")
        raise SystemExit(2)
    print(f"  **✓ 与 step1-linear-v1 的 H 逐比特一致**")

    # ── 验证：P2 跨 seed + P3 幅值线性 ──────────────────────────────────
    g = S1.evaluate_beta_gates(eps["P2"], eps["P3"], H)
    print(f"\n  P2 跨 seed β  " + " · ".join(
        f"seed{k} {v:.4f}" for k, v in sorted(g["p2_beta_by_seed"].items())))
    print(f"    中位 {g['p2_beta_median']:.4f}  门槛 ≥0.5 且全 >0   "
          f"{'✓' if g['p2_pass'] else '**✗**'}")
    print(f"  P3 幅值线性 β_P3 {g['p3_beta']:.4f}  比值 {g['p3_over_p2_median']:.4f}"
          f"  门槛 [0.7,1.3]   {'✓' if g['p3_pass'] else '**✗**'}")
    if not (g["p2_pass"] and g["p3_pass"]):
        raise SystemExit(2)

    # ── 冲激响应形状（旁证）────────────────────────────────────────────
    nrm = np.linalg.norm(H.reshape(len(H), -1), axis=1)
    print(f"\n  ‖H(τ)‖ 峰在 τ={int(np.argmax(nrm))}"
          f"（ALIGN_L={S1.ALIGN_L} 已吸收伺服滞后）"
          f"   尾比 ‖H(15)‖/‖H(0)‖ = {nrm[15]/nrm[0]:.3f}"
          f"   （闭环 Ĥ 是 0.25–0.50，更脏）")

    # ── 落盘 ───────────────────────────────────────────────────────────
    man = {
        "artifact": "step1-H-v1",
        "provenance": "outputs/ab4 P1 only; open-loop exogenous probe; no IMU in the loop",
        "identical_to": "step1-linear-v1 H (bit-identical, float64 sha256 verified)",
        "H": {"shape": list(H.shape), "dtype": "float64", "sha256": sha,
              "layout": "H[lag, gyro_axis, joint]"},
        "alignment": {"ALIGN_L": S1.ALIGN_L, "M": S1.M, "N_LAGS": S1.N_LAGS,
                      "gap": "real[t+ALIGN_L] - ideal[t]", "fs_hz": S1.FS},
        "ridge": {"multiplier": h_mult, "actual_lambda": h_lambda,
                  "scale_rule": "trace(X^T X)/dim",
                  "chosen_by": f"frozen sidecar {FROZEN_SIDECAR.name} (not re-decided here)",
                  "sidecar_sha256": S1._file_sha256(FROZEN_SIDECAR)},
        "fit_space": "raw_aligned_gyro_gap",
        "H_fit_nuisance": "per-episode constant removed from both sides",
        "metric_band_hz": list(S1.METRIC_BAND),
        "calibration": {"path": str(calib_path),
                        "sha256": S1._file_sha256(calib_path),
                        "R": R.tolist(), "b": b.tolist()},
        "fit_data": [{"tier": "P1", "ep": S1._episode_number(p), "path": str(p),
                      "sha256": S1._file_sha256(p)} for p in logs["P1"]],
        "validation_data": [{"tier": t, "ep": S1._episode_number(p), "path": str(p),
                             "sha256": S1._file_sha256(p)}
                            for t in ("P2", "P3") for p in logs[t]],
        "validation_gates": g,
        "scope": {
            "validated": ["cross-seed transferability (P2)", "amplitude linearity (P3)"],
            "never_validated": ("H has never been checked on the real robot AFTER applying a "
                                "delta_a solved from it -- Step-1 v1 was rejected offline"),
            "closed_loop_H": ("D/E/F closed-loop H is DIAGNOSTIC ONLY; never for identification "
                              "or validation (E-fit/F-eval flipped -86.7% -> +187.8%)"),
            "rejected_downstream": ("the Step-1 candidate CONSTRUCTION (theta/W, history-zero "
                                    "local inverse) delivered 2.65% and was rejected; correct "
                                    "causal rolling solve 3.16% (+-2mrad) / 4.34% (+-3mrad, "
                                    "6.5% saturated). H itself was NOT rejected."),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, H=np.ascontiguousarray(H, dtype=np.float64),
                        manifest_json=np.asarray(json.dumps(man, ensure_ascii=False,
                                                            sort_keys=True)))
    OUT.with_suffix(".json").write_text(
        json.dumps(S1._jsonable(man), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"\n  → {OUT}")
    print(f"  → {OUT.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
