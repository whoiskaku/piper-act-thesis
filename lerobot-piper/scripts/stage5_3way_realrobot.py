#!/usr/bin/env python3
"""Stage 5 真机三方 A/B 分析 — 解耦 blend vs 学到的残差(#64/#66)。

回答的问题: #64 的 absboth 残差让命令绝对平滑(sim 命令空间 0.74-0.94), 但 sim
执行流已触电机噪声地板(#66: refined+blend exec 0.905 ≈ demo 0.906)。真机执行流
地板是否更低 → 学到的残差能否在真机 IMU 上产生超过纯 blend 的收益?

三个 eval 数据集(control_robot record + REFINE_MODE 门控同一 absboth ckpt):
  A  bare ACT     : REFINE_MODE 不设(或 off)
  B  pure blend   : REFINE_MODE=blend   (确定性边界续接, 无学到的残差)
  C  blend+残差   : REFINE_MODE=full    (absboth ckpt)
B−A = blend 贡献; C−B = 学到的残差的增量贡献(这正是 Stage5 旧 A/B 混淆、本次要拆开的)。

指标(权威绝对口径, 非 fraction —— fraction 有分母陷阱见 STAGE3B):
  - 5Hz 零相位 Butterworth 高通后 RMS(hp_rms): action(active j1/j2/j3/grip) +
    真 IMU(accel 3轴 / gyro 3轴)。
  - #66 中频带 2-4.5Hz RMS: absboth 残差在命令上放大该带(损失5Hz截止盲区);
    真机 IMU 对加速度敏感, 专门查该带是否在 IMU 上变差。
  - 附带旧 fraction hf_ratio 仅作对照(勿作主结论)。

用法:
  python scripts/stage5_3way_realrobot.py \
    --ds-act   ~/.cache/huggingface/lerobot/local/eval_stage5_A_act \
    --ds-blend ~/.cache/huggingface/lerobot/local/eval_stage5_B_blend \
    --ds-full  ~/.cache/huggingface/lerobot/local/eval_stage5_C_full \
    --output-dir outputs/eval/stage5_3way
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FPS = 30.0
ACTIVE_JOINTS = [0, 1, 2, 6]
_HP_B, _HP_A = butter(4, 5.0 / (FPS / 2.0), btype="high")            # 5Hz 高通
_BP_B, _BP_A = butter(4, [2.0 / (FPS / 2.0), 4.5 / (FPS / 2.0)], btype="band")  # 2-4.5Hz 带通(#66)


def _hp_rms(x):                       # x: (T, C) → per-channel 5Hz 高通 RMS
    y = filtfilt(_HP_B, _HP_A, x, axis=0)
    return np.sqrt((y ** 2).mean(axis=0))


def _bp_rms(x):                       # x: (T, C) → per-channel 2-4.5Hz 带通 RMS
    y = filtfilt(_BP_B, _BP_A, x, axis=0)
    return np.sqrt((y ** 2).mean(axis=0))


def _hf_fraction(x):                  # 旧口径(仅对照): >5Hz 能量占比
    T = x.shape[0]
    freqs = np.fft.rfftfreq(T, d=1.0 / FPS)
    p = np.abs(np.fft.rfft(x - x.mean(0), axis=0)) ** 2
    return (p[freqs > 5.0].sum(0) / (p.sum(0) + 1e-9))


def load_dataset(ds_root: Path):
    parquets = sorted(ds_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"无 episode parquet: {ds_root}/data/chunk-*/")
    eps = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        eps.append({"name": pq.stem, "action": action, "imu": imu})
    return eps


def analyze(label, ds_root):
    eps = load_dataset(Path(ds_root).expanduser())
    rows = []
    for ep in eps:
        a, imu = ep["action"], ep["imu"]
        rows.append({
            "name": ep["name"], "frames": int(len(a)),
            "act_hp": float(_hp_rms(a)[ACTIVE_JOINTS].mean()),
            "act_bp": float(_bp_rms(a)[ACTIVE_JOINTS].mean()),
            "act_frac": float(_hf_fraction(a)[ACTIVE_JOINTS].mean()),
            "imu_accel_hp": float(_hp_rms(imu[:, :3]).mean()),
            "imu_gyro_hp": float(_hp_rms(imu[:, 3:]).mean()),
            "imu_accel_bp": float(_bp_rms(imu[:, :3]).mean()),
            "imu_gyro_bp": float(_bp_rms(imu[:, 3:]).mean()),
            "imu_frac": float(_hf_fraction(imu).mean()),
        })
    keys = [k for k in rows[0] if k not in ("name", "frames")]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    agg["n_episodes"] = len(rows)
    return {"label": label, "episodes": rows, "aggregate": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-act", required=True, help="A: 裸 ACT (REFINE_MODE off)")
    ap.add_argument("--ds-blend", required=True, help="B: 纯 blend (REFINE_MODE=blend)")
    ap.add_argument("--ds-full", required=True, help="C: blend+残差 (REFINE_MODE=full)")
    ap.add_argument("--output-dir", default="outputs/eval/stage5_3way")
    args = ap.parse_args()

    res = {c: analyze(c, d) for c, d in
           [("A_act", args.ds_act), ("B_blend", args.ds_blend), ("C_full", args.ds_full)]}
    A, B, C = (res["A_act"]["aggregate"], res["B_blend"]["aggregate"], res["C_full"]["aggregate"])

    def pct(base, v):
        return f"{(v-base)/base*100:+5.1f}%" if base else "n/a"

    metrics = [
        ("act_hp",       "Action hp_rms (5Hz, active)   [指令空间, 应大降]"),
        ("act_bp",       "Action 2-4.5Hz RMS (#66 放大带)[指令空间]"),
        ("imu_accel_hp", "IMU accel hp_rms (5Hz)         [真机收益, 关键]"),
        ("imu_gyro_hp",  "IMU gyro  hp_rms (5Hz)         [真机收益, 关键]"),
        ("imu_accel_bp", "IMU accel 2-4.5Hz RMS          [#66 中频副作用检查]"),
        ("imu_gyro_bp",  "IMU gyro  2-4.5Hz RMS          [#66 中频副作用检查]"),
        ("act_frac",     "Action >5Hz fraction (仅对照)"),
        ("imu_frac",     "IMU >5Hz fraction    (仅对照)"),
    ]
    print("\n" + "=" * 100)
    print(f"Stage 5 真机三方 A/B  (n_ep: A={A['n_episodes']} B={B['n_episodes']} C={C['n_episodes']})")
    print("  B−A=blend 贡献; C−B=学到的残差增量(Stage5 旧A/B所混淆的, 现拆开)")
    print("=" * 100)
    print(f"{'metric':46s} {'A:ACT':>10} {'B:blend':>10} {'C:full':>10} {'B-A':>8} {'C-A':>8} {'C-B':>8}")
    print("-" * 100)
    for k, desc in metrics:
        print(f"{desc:46s} {A[k]:10.5f} {B[k]:10.5f} {C[k]:10.5f} "
              f"{pct(A[k],B[k]):>8} {pct(A[k],C[k]):>8} {pct(B[k],C[k]):>8}")
    print("=" * 100)
    print("判读: IMU accel/gyro hp_rms 上 C−B 显著负 → 学到的残差在真机有超 blend 的增量收益(论文核心正面结论)")
    print("      C−B≈0 → 残差贡献被真机控制器/传感器地板吃掉, blend 已够(回到 Stage5 瓶颈叙事)")
    print("      若 IMU 2-4.5Hz 带 C 比 B 升 → #66 中频放大在真机有害, 需降 HF_CUTOFF 重训")

    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "stage5_3way_results.json").write_text(json.dumps(
        {c: res[c] for c in res}, indent=2))
    print(f"\nsaved -> {out / 'stage5_3way_results.json'}")


if __name__ == "__main__":
    main()
