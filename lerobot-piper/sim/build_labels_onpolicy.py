#!/usr/bin/env python3
"""DAgger Round 1 — 从 **on-policy rollout**（π 已激活）构造 teacher δa*。

用户 2026-07-29 决定：停止扩展 C，立即做 DAgger 第一轮。
`outputs/ab/C_ep*.json` 是 10 条 π 激活的真机 rollout（`mode=full`），
正是效果从 13.9% 掉到 4.7% 的那批 —— 也就是**分布偏移后的 on-policy 状态**。

**关键：teacher 标签不依赖 Ĝ。**
    δa* = −dt·(SᵀS+λI)⁻¹·Sᵀ·e_gyro_band,   S = R_cal·R_siteᵀ·Jr_arm(q_cmd)
S 是 MuJoCo 的**解析**旋转雅可比。Ĝ 只在 `train_pi_e2` 里当**环境**（闭环合成）。
⇒ 重新求 teacher 这一步与 Ĝ 的辨识程度**无关**。

与 `build_dagger_labels.py` 的差别只在**入口**：
  · 那个从 HF repo 读；这个从本地 rollout JSON 读
  · ideal 从 `chunks[].raw`（ACT **原始**指令块）逐 chunk 渲染 —— 与部署同口径，
    且必须用 raw 而非实际发送的 a（a = raw + δa，π 的输出不属于"理想"）
  · (R, b) 与 L **在本批内自行标定**（逐会话 IMU 标定，不能借用旧会话的）

术语（用户 2026-07-29）：本轮只做 **DAgger-style on-policy data aggregation** ——
在访问状态上**重新求 teacher δa***，未查询人工专家。

用法: python sim/build_labels_onpolicy.py [--tag cnew-deploy]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from build_dagger_labels import (BAND, EPS, DT, FPS, OUT_ROOT, bandpass,
                                 site_rot_jac)
from compare_ideal_methods import lag_corr_ch
from ideal_imu_kinematic import MODEL_PATH
from wall_portrait import kabsch, masks_for, smooth

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "outputs/ab"
CHUNK_RENDER = 100


def load_rollout(ep):
    """读一条 on-policy rollout，返回 (ideal_kin, real_imu, a_sent, q_enc, raw_cmd)。"""
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    d = json.loads((AB / f"C_ep{ep}.json").read_text())
    fr = d["frames"]
    real = np.array([f["imu"] for f in fr], np.float32)
    a_sent = np.array([f["a"] for f in fr], np.float32)
    qenc = np.array([f["state"] for f in fr], np.float32)
    if qenc.max() > 100:                       # mdeg → rad
        qenc = qenc[:, :6] * (np.pi / 180 / 1000)
    else:
        qenc = qenc[:, :6]
    # ideal 从**原始 ACT 指令块**逐 chunk 渲染（部署同口径）
    raw = np.zeros_like(a_sent[:, :7])
    for c in d["chunks"]:
        r = np.asarray(c["raw"], np.float32)
        s = int(c["start"]); n = min(len(r), len(raw) - s)
        if n > 0:
            raw[s:s + n] = r[:n, :raw.shape[1]]
    rend = IdealChunkRenderer()
    out, prev = [], None
    for s in range(0, len(raw), CHUNK_RENDER):
        c = raw[s:s + CHUNK_RENDER]
        out.append(rend.render(c, prev))
        prev = raw[:s + CHUNK_RENDER]
    kin = np.concatenate(out)[:len(raw)]
    T = min(len(kin), len(real), len(a_sent), len(qenc))
    return kin[:T], real[:T], raw[:T], qenc[:T], a_sent[:T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnew-deploy")
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--lag", type=int, default=None,
                    help="强制 L。默认按本批投票; 但聚合训练集**必须**与部署口径一致(3)")
    a_ = ap.parse_args()
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")

    print(f"=== DAgger Round 1 · on-policy 标签 ({a_.eps} 条 C rollout) ===")
    cache, lag_votes = [], []
    for ep in range(a_.eps):
        kin, real, raw, qenc, a_sent = load_rollout(ep)
        cache.append((kin, real, raw, qenc, a_sent))
        lags, _ = lag_corr_ch(kin, real)
        lag_votes.extend(lags[3:])
    L_vote = int(np.median(lag_votes))
    L = a_.lag if a_.lag is not None else L_vote
    print(f"  本批投票 L = {L_vote} (分布 {dict(zip(*np.unique(lag_votes, return_counts=True)))})")
    if a_.lag is not None:
        print(f"  **强制 L = {L}** —— 聚合训练集须与部署口径(LSHIFT=3)一致;")
        print(f"    混用两种对齐是训练/部署口径失配, 与已修过的 chunk 渲染同类。")

    # (R, b) **在本批内标定** —— 逐会话, 不借旧会话
    Gi = np.concatenate([k[:-L, 3:] for k, r, *_ in cache])
    Gr = np.concatenate([r[L:, 3:] for k, r, *_ in cache])
    sel = np.linalg.norm(Gi, axis=1) > 0.15
    Rc = kabsch(Gi[sel], Gr[sel])
    Ai = np.concatenate([k[:-L, :3] for k, r, *_ in cache])
    Ar = np.concatenate([r[L:, :3] for k, r, *_ in cache])
    b = (Ar - Ai @ Rc.T).mean(0)
    print(f"  本批 IMU 标定: |R−I|={np.abs(Rc-np.eye(3)).max():.3f}  b={np.round(b,3)}")

    out_dir = OUT_ROOT / a_.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    st = {"n": 0, "gated": 0, "clip": 0, "dc": [], "pr": [], "rms": []}
    for ep, (kin, real, raw, qenc, a_sent) in enumerate(cache):
        ideal = np.empty_like(kin)
        ideal[:, :3] = kin[:, :3] @ Rc.T + b
        ideal[:, 3:] = kin[:, 3:] @ Rc.T
        ia, ra, aa, qw = ideal[:-L], real[L:], raw[:-L], qenc[:-L]
        e = ra - ia
        e_g = bandpass(e[:, 3:])
        m = masks_for(ra, ia, aa[:, 6])
        n = len(ra)
        lab = np.zeros((n, 6)); dn = d1 = d2 = pn = pd = 0.0
        for i in range(n):
            S = Rc @ site_rot_jac(model, data, sid, aa[i])
            lam = 0.05 * np.trace(S.T @ S) / 3 + 1e-9
            d0 = -DT * np.linalg.solve(S.T @ S + lam * np.eye(6), S.T @ e_g[i])
            lab[i] = d0
            dw = S @ d0 / DT
            dn += float(dw @ (-e_g[i])); d1 += float(dw @ dw); d2 += float(e_g[i] @ e_g[i])
            if not m["still"][i]:
                pn += float(((e_g[i] + dw) ** 2).sum()); pd += float((e_g[i] ** 2).sum())
        clip = (np.abs(lab) > EPS).any(1)
        lab = np.clip(lab, -EPS, EPS); lab[m["still"]] = 0.0
        phase = np.stack([(np.arange(n) % 100) / 100.0,
                          aa[:, 6] / max(np.abs(aa[:, 6]).max(), 1e-6),
                          smooth(np.linalg.norm(ra[:, 3:], axis=1))], 1)
        np.savez_compressed(out_dir / f"ep{ep:03d}.npz",
                            e=e.astype(np.float32), ideal=ia.astype(np.float32),
                            real=ra.astype(np.float32), a_arm=aa[:, :6].astype(np.float32),
                            grip=aa[:, 6].astype(np.float32), qw=qw.astype(np.float32),
                            phase=phase.astype(np.float32),
                            mask_still=m["still"], mask_contact=m["contact"],
                            label=lab.astype(np.float32), label_clip=clip,
                            da_applied=a_sent[:-L, :6] - raw[:-L, :6])  # π 当时实际施加的
        st["n"] += n; st["gated"] += int(m["still"].sum()); st["clip"] += int(clip.sum())
        st["dc"].append(dn / max(np.sqrt(d1 * d2), 1e-9))
        st["pr"].append(1 - pn / max(pd, 1e-9))
        st["rms"].append(float(np.sqrt((lab[~m["still"]] ** 2).mean()) * 1000))
    man = {"source": "outputs/ab/C_ep*.json (on-policy, mode=full)",
           "kind": "DAgger-style on-policy data aggregation (teacher 重新求解, 非人工专家)",
           "lag": L, "R": Rc.tolist(), "b": b.tolist(), "eps_rad": EPS,
           "band": list(BAND), "dt": DT, "n_ep": a_.eps,
           "stats": {"n_frames": st["n"], "gated_pct": 100*st["gated"]/st["n"],
                     "clipped_pct": 100*st["clip"]/st["n"],
                     "dir_corr_median": float(np.median(st["dc"])),
                     "pred_reduction_median": float(np.median(st["pr"])),
                     "label_rms_mrad_median": float(np.median(st["rms"]))}}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    s = man["stats"]
    print(f"\n  [{a_.tag}] 帧数={s['n_frames']}  gated={s['gated_pct']:.0f}%  "
          f"截断={s['clipped_pct']:.0f}%")
    print(f"    方向 corr={s['dir_corr_median']:.3f}  模型内预测消减="
          f"{s['pred_reduction_median']*100:.0f}%  标签 RMS={s['label_rms_mrad_median']:.1f} mrad")
    print(f"\n  对照 · 原训练集 pp2-deploy:")
    o = json.loads((OUT_ROOT / "pp2-deploy" / "manifest.json").read_text())["stats"]
    print(f"    帧数={o['n_frames']}  gated={o['gated_pct']:.0f}%  截断={o['clipped_pct']:.0f}%"
          f"  方向 corr={o['dir_corr_median']:.3f}  消减={o['pred_reduction_median']*100:.0f}%"
          f"  标签 RMS={o['label_rms_mrad_median']:.1f} mrad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
