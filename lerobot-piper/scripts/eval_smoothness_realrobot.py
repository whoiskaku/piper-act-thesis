#!/usr/bin/env python3
"""Step 6 真机平滑性评估：对比 vanilla ACT vs α-imu-only ACT 在实机上的表现。

读取两个 eval_* 数据集（由 control_robot record + policy 采集），计算：
  - 动作指标：RMS jerk、高频能量比 (>5Hz)
  - 真实 TSND151 IMU 指标：RMS 幅值、高频能量比、峰值频率
  - 逐 episode + 汇总
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(ds_root: Path):
    """读取一个 LeRobot v2 数据集的所有 episode 的 action + imu_real。"""
    parquets = sorted(ds_root.glob("data/chunk-*/episode_*.parquet"))
    episodes = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        t = df["timestamp"].to_numpy().astype(np.float64)
        episodes.append({"name": pq.stem, "action": action, "imu": imu, "t": t})
    return episodes


def action_metrics(action, dt):
    vel = np.diff(action, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    rms_jerk = np.sqrt((jerk ** 2).mean(axis=0))

    T = action.shape[0]
    freqs = np.fft.rfftfreq(T, d=dt)
    fft = np.fft.rfft(action - action.mean(axis=0), axis=0)
    power = np.abs(fft) ** 2
    high = freqs > 5.0
    hf_ratio = power[high].sum(axis=0) / (power.sum(axis=0) + 1e-9)

    return {
        "rms_jerk_per_ch": rms_jerk.tolist(),
        "rms_jerk_mean": float(rms_jerk.mean()),
        "hf_ratio_per_ch": hf_ratio.tolist(),
        "hf_ratio_mean": float(hf_ratio.mean()),
    }


def imu_metrics(imu, dt):
    """真实 IMU 的平滑/振动指标。
    accel 部分减去重力（取 z 轴均值作为 baseline）；gyro 减去均值。
    """
    # 去 DC
    imu_ac = imu - imu.mean(axis=0, keepdims=True)

    # RMS amplitude per channel
    rms = np.sqrt((imu_ac ** 2).mean(axis=0))

    # Total vibration magnitude: RMS of (|accel|, |gyro|) 分别
    accel_rms_total = float(np.sqrt((imu_ac[:, :3] ** 2).sum(axis=1).mean()))
    gyro_rms_total = float(np.sqrt((imu_ac[:, 3:] ** 2).sum(axis=1).mean()))

    # Spectral analysis
    T = imu.shape[0]
    freqs = np.fft.rfftfreq(T, d=dt)
    fft = np.fft.rfft(imu_ac, axis=0)
    power = np.abs(fft) ** 2

    high = freqs > 5.0
    hf_ratio = power[high].sum(axis=0) / (power.sum(axis=0) + 1e-9)

    # Peak freq (ignore DC; per channel)
    peak_freq = freqs[1:][power[1:].argmax(axis=0)]

    return {
        "rms_per_ch": rms.tolist(),
        "accel_rms_total": accel_rms_total,
        "gyro_rms_total": gyro_rms_total,
        "hf_ratio_per_ch": hf_ratio.tolist(),
        "hf_ratio_mean": float(hf_ratio.mean()),
        "peak_freq_per_ch": peak_freq.tolist(),
    }


def analyze_dataset(label, ds_root, dt=1.0 / 30.0):
    episodes = load_dataset(ds_root)
    rows = []
    for ep in episodes:
        am = action_metrics(ep["action"], dt)
        im = imu_metrics(ep["imu"], dt)
        rows.append({
            "name": ep["name"],
            "frames": int(len(ep["action"])),
            "action": am,
            "imu": im,
        })
    return {"label": label, "episodes": rows}


def aggregate(result):
    eps = result["episodes"]
    n = len(eps)
    def m(path):
        vals = []
        for r in eps:
            x = r
            for k in path:
                x = x[k]
            vals.append(x)
        return float(np.mean(vals))
    return {
        "n_episodes": n,
        "total_frames": int(sum(r["frames"] for r in eps)),
        "action_rms_jerk_mean": m(["action", "rms_jerk_mean"]),
        "action_hf_ratio_mean": m(["action", "hf_ratio_mean"]),
        "imu_accel_rms_total": m(["imu", "accel_rms_total"]),
        "imu_gyro_rms_total": m(["imu", "gyro_rms_total"]),
        "imu_hf_ratio_mean": m(["imu", "hf_ratio_mean"]),
    }


def print_comparison(agg_v, agg_a):
    def pct(v, a):
        if v == 0:
            return "n/a"
        change = (a - v) / v * 100
        sign = "↓" if change < 0 else "↑"
        return f"{sign}{abs(change):5.1f}%"

    print("\n" + "=" * 72)
    print(f"{'Metric':35s}  {'Vanilla':>12s}  {'α-imu-only':>12s}  Delta")
    print("-" * 72)
    for key, desc in [
        ("action_rms_jerk_mean",   "Action RMS jerk"),
        ("action_hf_ratio_mean",   "Action high-freq (>5Hz) %"),
        ("imu_accel_rms_total",    "Real IMU accel RMS (m/s²)"),
        ("imu_gyro_rms_total",     "Real IMU gyro RMS (rad/s)"),
        ("imu_hf_ratio_mean",      "Real IMU high-freq (>5Hz) %"),
    ]:
        v = agg_v[key]
        a = agg_a[key]
        print(f"{desc:35s}  {v:12.4f}  {a:12.4f}  {pct(v, a)}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_vanilla_v1"))
    ap.add_argument("--alpha-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_imu_only_v1"))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] vanilla: {args.vanilla_dataset}")
    res_v = analyze_dataset("vanilla", Path(args.vanilla_dataset))
    print(f"[load] alpha:   {args.alpha_dataset}")
    res_a = analyze_dataset("alpha_imu_only", Path(args.alpha_dataset))

    # per-episode summary
    print("\n--- per-episode ---")
    for res in [res_v, res_a]:
        print(f"\n[{res['label']}]")
        for r in res["episodes"]:
            print(f"  {r['name']} ({r['frames']:4d}f)  "
                  f"act_jerk={r['action']['rms_jerk_mean']:.2f}  "
                  f"act_hf={r['action']['hf_ratio_mean']:.4f}  "
                  f"imu_acc_rms={r['imu']['accel_rms_total']:.3f}  "
                  f"imu_gyr_rms={r['imu']['gyro_rms_total']:.3f}  "
                  f"imu_hf={r['imu']['hf_ratio_mean']:.4f}")

    agg_v = aggregate(res_v)
    agg_a = aggregate(res_a)
    print_comparison(agg_v, agg_a)

    out = {
        "vanilla":  {"per_episode": res_v["episodes"], "aggregate": agg_v},
        "alpha_imu_only": {"per_episode": res_a["episodes"], "aggregate": agg_a},
    }
    with open(out_dir / "real_robot_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_dir / 'real_robot_results.json'}")


if __name__ == "__main__":
    main()
