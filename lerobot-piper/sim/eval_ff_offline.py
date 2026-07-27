#!/usr/bin/env python3
"""P2 前馈层离线验证: 对已录 deploy 指令应用 FF2(边界衔接)+FF3(意图整形),
渲染后量: ①LF保真(vs意图参照) ②HF毛刺压制 ③边界集中度 ④指令步长安全性。
FF1(滞后前移)为纯平移, 不进这些指标(时间中性), 部署时叠加。

用法: python sim/eval_ff_offline.py [--eps 10]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from correction_ff import apply_to_episode
from compare_ideal_methods import load_episode
from ideal_imu_kinematic import MODEL_PATH, build_spline, ideal_imu_traj
from dyn_replay_imu import aa_downsample
from wall_portrait import boundary_masks

FPS, FC = 20.0, 3.0
BA_LP = butter(3, 2.0 / (FPS / 2))
BA_HP = butter(3, 2.0 / (FPS / 2), "high")
DATASETS = [("enolab-guo/eval_pickplace2_100k", "pp2-deploy"),
            ("enolab-guo/eval_insert_100k", "ins-deploy")]

_model = _data = _sid = None


def render(cmd: np.ndarray) -> np.ndarray:
    """指令(T,7) → 20Hz 抗混叠理想 IMU (T,6)。"""
    global _model, _data, _sid
    import mujoco
    if _model is None:
        _model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        _data = mujoco.MjData(_model)
        _sid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    T = len(cmd)
    t20 = np.arange(T) / FPS
    sp = build_spline(t20, cmd, "cubic")
    hi = 200.0
    t_hi = np.arange(int(round(t20[-1] * hi)) + 1) / hi
    _, imu_hi, _, _ = ideal_imu_traj(_model, _data, sp, t_hi, _sid)
    return aa_downsample(imu_hi, hi, FPS, T)


def bnd_metrics(imu: np.ndarray):
    """(集中度=边界/段中, 绝对边界能量) — 集中度在段中被清干净时会虚高, 须配绝对值看。"""
    hp = filtfilt(*BA_HP, imu, axis=0)
    e = (hp ** 2).sum(1)
    b, m = boundary_masks(len(imu))
    return (float(e[b].mean() / max(e[m].mean(), 1e-12)), float(e[b].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=10)
    args = ap.parse_args()
    out = {}
    for repo, tag in DATASETS:
        rows = []
        for ep in range(args.eps):
            d = load_episode(repo, ep)
            A = d["action"]
            A_shp = apply_to_episode(A, lag=0, blend=4, fc=FC)
            assert np.array_equal(A[:, 6], A_shp[:, 6]), "夹爪列被改动!"
            A_int = A.copy()
            A_int[:, :6] = filtfilt(*butter(2, FC / (FPS / 2)), A[:, :6], axis=0)  # 全序列意图参照
            R_lit, R_shp, R_int = render(A), render(A_shp), render(A_int)

            lp_s, lp_i = filtfilt(*BA_LP, R_shp, axis=0), filtfilt(*BA_LP, R_int, axis=0)
            corr = float(np.mean([np.corrcoef(lp_s[:, c], lp_i[:, c])[0, 1] for c in range(3, 6)]))
            lf_rms = float(np.sqrt(((lp_s[:, 3:] - lp_i[:, 3:]) ** 2).mean()))
            hp = lambda x: filtfilt(*BA_HP, x, axis=0)
            g = lambda x: float(np.sqrt((hp(x)[:, 3:] ** 2).mean()))
            hf_lit, hf_shp, hf_int = g(R_lit), g(R_shp), g(R_int)
            da = np.abs(np.diff(A[:, :6], axis=0)).max(1)
            da_s = np.abs(np.diff(A_shp[:, :6], axis=0)).max(1)
            bsteps = [k * 100 for k in range(1, len(A) // 100 + 1) if k * 100 < len(A)]
            bc_lit, be_lit = bnd_metrics(R_lit)
            bc_shp, be_shp = bnd_metrics(R_shp)
            rows.append({
                "lf_corr": corr, "lf_rms": lf_rms,
                "hf_ratio_before": hf_lit / hf_int, "hf_ratio_after": hf_shp / hf_int,
                "bnd_before": bc_lit, "bnd_after": bc_shp,
                "bnd_abs_before": be_lit, "bnd_abs_after": be_shp,
                "step_p99_before": float(np.percentile(da, 99)),
                "step_p99_after": float(np.percentile(da_s, 99)),
                "bnd_step_before": float(np.median([da[i - 1] for i in bsteps])) if bsteps else np.nan,
                "bnd_step_after": float(np.median([da_s[i - 1] for i in bsteps])) if bsteps else np.nan,
            })
            print(f"[{tag} ep{ep}] HF比 {rows[-1]['hf_ratio_before']:.2f}→{rows[-1]['hf_ratio_after']:.2f}  "
                  f"边界 {rows[-1]['bnd_before']:.1f}→{rows[-1]['bnd_after']:.1f}  LFcorr={corr:.4f}")
        out[tag] = rows

    print("\n========== FF2+FF3 离线验证(逐集中位数) ==========")
    med = lambda rs, k: float(np.median([r[k] for r in rs if np.isfinite(r[k])]))
    for tag, rows in out.items():
        print(f"\n[{tag}]")
        print(f"  ①LF保真:   整形后渲染 vs 意图参照  corr={med(rows,'lf_corr'):.4f}  ΔRMS={med(rows,'lf_rms'):.4f} rad/s")
        print(f"  ②HF毛刺:   渲染HF/意图HF  整形前 {med(rows,'hf_ratio_before'):.2f}× → 整形后 {med(rows,'hf_ratio_after'):.2f}×")
        print(f"  ③边界: 绝对能量 {med(rows,'bnd_abs_before'):.2f}→{med(rows,'bnd_abs_after'):.2f} "
              f"(降{med(rows,'bnd_abs_before')/max(med(rows,'bnd_abs_after'),1e-9):.1f}×);  "
              f"集中度 {med(rows,'bnd_before'):.1f}×→{med(rows,'bnd_after'):.1f}× (相对量,段中清零后会虚高)")
        print(f"  ④指令步长: P99 {med(rows,'step_p99_before')*1000:.1f}→{med(rows,'step_p99_after')*1000:.1f} mrad/帧;  "
              f"边界步 {med(rows,'bnd_step_before')*1000:.1f}→{med(rows,'bnd_step_after')*1000:.1f} mrad")
    p = Path(__file__).resolve().parents[1] / "outputs/sim/wall_portrait/ff_offline.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n→ {p}")


if __name__ == "__main__":
    main()
