#!/usr/bin/env python3
"""P2.2 — DAgger round-0 标签管线: 从已录 deploy rollout 构造 δa* 训练集。

对齐约定(关键): e(i) = real[i+L] − ideal_cal[i] — 命令帧 i 的修正经伺服在真机
时刻 i+L 生效, 所以对齐后的 e(i) 正是 δa(i) 应抵消的目标(对齐已编码 t+L)。

标签推导(v0, gyro-only — 画像显示墙的主载体是 gyro):
  δa 在一帧内保持 → 附加关节速度 δq̇ ≈ δa/dt → 附加 δω ≈ S·δa/dt
  其中 S(t) = R_cal · R_siteᵀ · Jr_arm(q_cmd(t)) ∈ R^{3×6}  (site系旋转雅可比→真机IMU系)
  抵消带限误差:  δa* = −dt · (SᵀS+λI)⁻¹ Sᵀ e_gyro_band(i)
  裁剪 ‖δa*‖∞ ≤ ε(硬界); still 段置零; 接触窗保留但打 flag(B 该学它)。
  承认有偏(伺服≠单帧速度模型) — 这正是 P3 RL 阶段存在的理由。

输出: outputs/p2_labels/<tag>/ep***.npz — 全流(训练时再开窗):
  e(6) ideal(6) real(6) a_arm(6) phase(3: 距边界/夹爪/gate电平) mask_still mask_contact
  label(6) label_clip_flag;  + manifest.json(R,b,L,ε,λ,频带,统计)
用法: python sim/build_dagger_labels.py [--eps 10]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from compare_ideal_methods import kin_ideal_20, lag_corr_ch
from ideal_imu_kinematic import MODEL_PATH, full_qpos
from wall_portrait import kabsch, masks_for, smooth

FPS, DT = 20.0, 0.05
CHUNK_RENDER = 100               # 2026-07-22: ideal 改**逐 chunk 渲染**, 与部署一致


def kin_ideal_chunked(repo, ep, chunk=CHUNK_RENDER):
    """逐 chunk 渲染 ideal —— 部署时拿不到下一 chunk, 训练必须同口径。

    2026-07-22 修: 原用整集样条(kin_ideal_20), 末尾有未来数据; 而部署逐 chunk 渲染,
    每 chunk 最后 ~5 帧的二阶导只能靠自然边界条件 → 误差 RMS 0.27 rad/s(墙本身仅
    0.11)。训练/部署失配, 与已修过的因果特征泄漏同类。现统一为逐 chunk。
    """
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    _, d, _ = kin_ideal_20(repo, ep)                    # 只借它读日志(kin 丢弃)
    cmd = d["action"].astype(np.float32)
    rend = IdealChunkRenderer()
    out, prev = [], None
    for s in range(0, len(cmd), chunk):
        c = cmd[s:s + chunk]
        out.append(rend.render(c, prev))
        prev = cmd[:s + chunk]
    return np.concatenate(out)[:len(cmd)], d, 0.0
BAND = (1.0, 8.0)
EPS = 0.010                      # δa 硬界 (rad)
OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs/p2_labels"
DATASETS = [("enolab-guo/eval_pickplace2_100k", "pp2-deploy"),
            ("enolab-guo/eval_insert_100k", "ins-deploy")]


def bandpass(x):
    ba = butter(3, [BAND[0] / (FPS / 2), BAND[1] / (FPS / 2)], "bandpass")
    return filtfilt(*ba, x, axis=0)


def site_rot_jac(model, data, sid, q7):
    """S_sim(3×6): site 系旋转雅可比(臂6关节)。"""
    import mujoco
    data.qpos[:] = full_qpos(q7)
    mujoco.mj_fwdPosition(model, data)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    R_site = data.site_xmat[sid].reshape(3, 3)
    return R_site.T @ jacr[:, :6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--repo", default=None, help="只跑这一个 repo(与 --tag 配对)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    ds = [(args.repo, args.tag)] if args.repo else DATASETS
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")

    for repo, tag in ds:
        # pass1: 渲染缓存 + L + (R,b)  (与 wall_portrait 同法)
        cache, lag_votes = [], []
        for ep in range(args.eps):
            kin, d, _ = kin_ideal_chunked(repo, ep)
            T = min(len(d["imu_real"]), len(kin))
            qenc = (d["state_mdeg"][:, :6] * (np.pi / 180 / 1000)).astype(np.float32)
            cache.append((kin[:T], d["imu_real"][:T], d["action"][:T], qenc[:T]))
            lags, _ = lag_corr_ch(kin[:T], d["imu_real"][:T])
            lag_votes.extend(lags[3:])
        L = int(np.median(lag_votes))
        Gi = np.concatenate([k[:-L, 3:] for k, r, a, _q in cache])
        Gr = np.concatenate([r[L:, 3:] for k, r, a, _q in cache])
        sel = np.linalg.norm(Gi, axis=1) > 0.15
        Rc = kabsch(Gi[sel], Gr[sel])
        Ai = np.concatenate([k[:-L, :3] for k, r, a, _q in cache])
        Ar = np.concatenate([r[L:, :3] for k, r, a, _q in cache])
        b = (Ar - Ai @ Rc.T).mean(0)

        out_dir = OUT_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = {"n_frames": 0, "gated": 0, "clipped": 0, "dircorr": [], "predred": [],
                 "label_rms_mrad": []}
        # 重跑口径注(2026-07-15): 增存 qw=编码器角(对齐同 aa, 供 π 的 q̇ 特征), 其余不变
        for ep, (kin, real, act, qenc) in enumerate(cache):
            ideal = np.empty_like(kin)
            ideal[:, :3] = kin[:, :3] @ Rc.T + b
            ideal[:, 3:] = kin[:, 3:] @ Rc.T
            ia, ra, aa, qw = ideal[:-L], real[L:], act[:-L], qenc[:-L]
            e = ra - ia
            e_g = bandpass(e[:, 3:])
            m = masks_for(ra, ia, aa[:, 6])
            n = len(ra)
            # 逐帧 S 与标签
            lab = np.zeros((n, 6)); dircorr_num = dircorr_den1 = dircorr_den2 = 0.0
            pr_num = pr_den = 0.0
            for i in range(n):
                S = Rc @ site_rot_jac(model, data, sid, aa[i])      # 真机IMU系
                lam = 0.05 * np.trace(S.T @ S) / 3 + 1e-9
                d0 = -DT * np.linalg.solve(S.T @ S + lam * np.eye(6), S.T @ e_g[i])
                lab[i] = d0
                dw = S @ d0 / DT
                dircorr_num += float(dw @ (-e_g[i]))
                dircorr_den1 += float(dw @ dw); dircorr_den2 += float(e_g[i] @ e_g[i])
                if not m["still"][i]:
                    pr_num += float(((e_g[i] + dw) ** 2).sum()); pr_den += float((e_g[i] ** 2).sum())
            clip_flag = (np.abs(lab) > EPS).any(1)
            lab = np.clip(lab, -EPS, EPS)
            lab[m["still"]] = 0.0
            phase = np.stack([
                (np.arange(n) % 100) / 100.0,                        # 距边界相位
                aa[:, 6] / max(np.abs(aa[:, 6]).max(), 1e-6),        # 夹爪
                smooth(np.linalg.norm(ra[:, 3:], axis=1)),           # gate 电平
            ], axis=1)
            np.savez_compressed(out_dir / f"ep{ep:03d}.npz",
                                e=e.astype(np.float32), ideal=ia.astype(np.float32),
                                real=ra.astype(np.float32), a_arm=aa[:, :6].astype(np.float32),
                                grip=aa[:, 6].astype(np.float32),   # 部署可用的固定尺度归一化(÷0.035)用
                                qw=qw.astype(np.float32),           # 编码器角(对齐同aa, 供π的q̇特征)
                                phase=phase.astype(np.float32),
                                mask_still=m["still"], mask_contact=m["contact"],
                                label=lab.astype(np.float32), label_clip=clip_flag)
            stats["n_frames"] += n
            stats["gated"] += int(m["still"].sum())
            stats["clipped"] += int(clip_flag.sum())
            stats["dircorr"].append(dircorr_num / max(np.sqrt(dircorr_den1 * dircorr_den2), 1e-9))
            stats["predred"].append(1 - pr_num / max(pr_den, 1e-9))
            stats["label_rms_mrad"].append(float(np.sqrt((lab[~m['still']] ** 2).mean()) * 1000))
        manifest = {"repo": repo, "lag": L, "R": Rc.tolist(), "b": b.tolist(),
                    "eps_rad": EPS, "band": BAND, "dt": DT,
                    "stats": {"n_frames": stats["n_frames"],
                              "gated_pct": 100 * stats["gated"] / stats["n_frames"],
                              "clipped_pct": 100 * stats["clipped"] / stats["n_frames"],
                              "dir_corr_median": float(np.median(stats["dircorr"])),
                              "pred_reduction_median": float(np.median(stats["predred"])),
                              "label_rms_mrad_median": float(np.median(stats["label_rms_mrad"]))}}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        s = manifest["stats"]
        print(f"[{tag}] L={L} 帧数={s['n_frames']}  gated={s['gated_pct']:.0f}%  "
              f"截断={s['clipped_pct']:.0f}%  方向corr={s['dir_corr_median']:.3f}  "
              f"模型内预测消减={s['pred_reduction_median']*100:.0f}%  "
              f"标签RMS={s['label_rms_mrad_median']:.1f} mrad")


if __name__ == "__main__":
    main()
