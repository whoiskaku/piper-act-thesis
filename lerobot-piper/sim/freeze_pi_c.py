#!/usr/bin/env python3
"""**冻结 C 候选 π** —— 全 13 集训练一个最终 checkpoint，配置事前固定。

事前固定的训练配置（用户 2026-09-14 要求：架构/超参/轮数规则/随机种子都事前定）
------------------------------------------------------------------------------
    架构      MLP 171 → 96 → 96 → 48（GELU），无 dropout / 无 BN
    优化      AdamW, lr=3e-3, weight_decay=1e-3
    轮数规则  **固定 400 轮全批**（无早停、无按验证集选点）
    种子      0
    特征      `sim/pi_c_features.py`（训练与部署**共用同一实现**）
    目标      e₀ 的 16 帧 horizon（48 维），输入/输出各自标准化
这些在训练**之前**就定死，本脚本不做任何搜索。

⚠ 性能声明的口径（用户 2026-09-14 明确）
----------------------------------------
- **8.73%** 是「**该训练流程**的逐集留一折外估计」——
  **不是**本 checkpoint 的实测留出成绩（本 checkpoint 用了全部 13 集，无留出）。
- **11.91%** 是**样本内**结果。
- 两者都在「同一解析求解器 + 同一部署约束（ε=10 mrad 峰值硬界 / chunk 钳位 /
  接触门控）+ 被控对象 = 已验证的 H」下测得。
- **e₀ 是裸墙**（ab4 = 裸 ACT + 探针）；C 臂的 π 实际要修的是 **FF 之后**的墙。

用法:
    python -u sim/freeze_pi_c.py
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

import oracle_causal_feedback as O                                   # noqa: E402
import oracle_constraint_ceiling as CC                               # noqa: E402
import pi_feature_ablation as A                                      # noqa: E402
import train_pi_g2 as T                                              # noqa: E402
import pi_c_features as PF                                           # noqa: E402
import fit_step1_candidate as S1                                     # noqa: E402

OUT = ROOT / "outputs/p2_pi/pi_c_v1.npz"
LOO_OOF, IN_SAMPLE = 8.73, 11.91          # 已测，见 train_pi_g2.json


def main() -> int:
    import gc, torch
    gc.disable()
    H, E = O.load_all()
    E = A.add_state(E)

    # ── 启动时的单位一致性断言（运行中改为归零，不抛异常）──────────────
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0", "P1"))
    p0 = [p for t in ("P0", "P1") for p in logs[t]][0]
    fr = json.loads(p0.read_text())["frames"]
    st_raw = np.asarray([f["state"] for f in fr], float)
    PF.assert_units(st_raw, E[0]["x"][:, 18:24])
    print(f"✓ 单位一致性断言通过：mdeg × π/180000 → rad "
          f"(MDEG_TO_RAD = {PF.MDEG_TO_RAD:.12e})")

    # ── 训练/部署特征一致性自测（同一已知样本，两条路径逐点比较）────────
    e = E[0]
    enc_ref = e["terr"]                                   # 消融脚本里算的
    enc_new, why = PF.encoder_block(st_raw[:len(e["e0"])], e["x"][:, 18:24])
    d = float(np.abs(enc_ref[:len(enc_new)] - enc_new).max())
    print(f"✓ 训练/部署特征一致性：编码器块逐点最大差 {d:.3e}"
          f"（原因字段 {'空' if not why else why}）")
    assert d < 1e-9 and not why, "特征构造两条路径不一致"
    # 异常状态必须**归零而非抛异常**
    bad = st_raw.copy(); bad[10, 0] = np.nan
    z, why2 = PF.encoder_block(bad[:len(e["e0"])], e["x"][:, 18:24])
    print(f"✓ 异常状态处理：返回全零 {np.abs(z).max()==0}，原因 '{why2}'（未抛异常）")
    assert np.abs(z).max() == 0 and why2 == "state_nan"

    # ── 用事前固定的配置，在**全部 13 集**上训练最终 checkpoint ────────
    Wall = O.fit_w(E)
    for ep in E:
        ep["r_imu"] = ep["e0"] - ep["x"] @ Wall.T
    XY = T.build_xy(E)
    print(f"\n训练最终 checkpoint：全部 {len(E)} 集 · 特征维 {XY[0][0].shape[1]}"
          f"（PF.FEAT_DIM={PF.FEAT_DIM}）· MLP {T.HID}×2 · {T.EPOCHS} 轮 · seed {T.SEED}")
    net, nrm = T.train(XY, seed=T.SEED)
    t0 = T.JLAG + S1.ALIGN_L
    das = [T.roll_pred(E[i], H, T.predict(net, nrm, XY[i][0]), t0) for i in range(len(E))]
    r_i, rm_i, sa_i = CC.score(E, H, das)
    pk = max(float(np.abs(d).max()) for d, _ in das) * 1e3

    sd = net.state_dict()
    flat = np.concatenate([v.detach().cpu().numpy().ravel() for v in sd.values()])
    sha = hashlib.sha256(np.ascontiguousarray(flat, np.float64).tobytes()).hexdigest()
    man = {
        "artifact": "pi-c-v1",
        "trained_on": f"ab4 P0+P1 全 {len(E)} 集（**无留出**）",
        "config_fixed_in_advance": {
            "arch": f"MLP {XY[0][0].shape[1]}->{T.HID}->{T.HID}->{XY[0][1].shape[1]} (GELU)",
            "optimizer": "AdamW", "lr": T.LR, "weight_decay": T.WD,
            "epoch_rule": f"固定 {T.EPOCHS} 轮全批，无早停、无按验证集选点",
            "seed": T.SEED, "jlag": PF.JLAG,
            "features": "sim/pi_c_features.py（训练与部署共用）"},
        "weights_sha256": sha,
        "performance_claims": {
            "loo_oof_percent": LOO_OOF,
            "loo_oof_meaning": ("**该训练流程**的逐集留一折外估计；"
                                "**不是**本 checkpoint 的实测留出成绩"),
            "in_sample_percent": IN_SAMPLE,
            "this_checkpoint_in_sample_percent": r_i * 100,
            "linear_all_obs_baseline_percent": 5.08,
            "perfect_prediction_ceiling_percent": 22.07},
        "deployment_constraints": {
            "eps_rad": T.EPS, "eps_meaning": "**瞬时峰值**硬界（不是 RMS）",
            "chunk_clamp": True, "contact_gating": True,
            "frame_ms_p99_max": 60.0, "dropped_frames_max": 0},
        "measured": {"da_rms_mrad": rm_i, "da_peak_mrad": pk, "sat_pct": sa_i},
        "units": {"observation.state": "mdeg", "MDEG_TO_RAD": PF.MDEG_TO_RAD,
                  "runtime_policy": "state 缺失/异常 ⇒ 补正输出归零并记录原因，**不抛异常**"},
        "caliber": ("被控对象=已验证 H; e0 是**裸墙**(ab4=裸ACT+探针), "
                    "C 臂的 π 实际修的是 **FF 之后**的墙(已小 33% 且形状不同), 不可直接换算; "
                    "③a 只验证了 j5 主轴的增量保真度, **全臂增量未验证**"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "norm": [x.tolist() for x in nrm],
                "manifest": man}, OUT.with_suffix(".pt"))
    OUT.with_suffix(".json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"\n  本 checkpoint 样本内 {r_i*100:.2f}%  δa RMS {rm_i:.2f} mrad  "
          f"峰值 {pk:.2f} mrad  触界 {sa_i:.2f}%")
    print(f"  性能声明：折外 **{LOO_OOF}%** = 该**训练流程**的逐集留一估计（非本 ckpt 的留出成绩）")
    print(f"            样本内 {IN_SAMPLE}%（流程）/ {r_i*100:.2f}%（本 ckpt）")
    print(f"  权重 sha256 {sha[:16]}…")
    print(f"\n  → {OUT.with_suffix('.pt')}")
    print(f"  → {OUT.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
