#!/usr/bin/env python3
"""上机前的无机时回归测试：流式部署路径 vs 离线训练路径（2026-08-04）。

**为什么必须做**：上一版 π 失败的原因之一就是**训练/部署口径错配**
（`full` 模式下 π 看到的 ideal 是从 raw+FF 渲染的，与训练口径差 13.6 mrad）。
`FeatureAssembler` 的注释写"一致性靠构造"，但从未被实测验证过。本脚本用**录下来的真机数据**
把部署路径逐帧走一遍，与 `build_offline` 的离线结果逐元素比对。

三项检查：
  ① 特征一致性  流式 `FeatureAssembler.step()` 的第 i 行 == `build_offline` 的第 i 行
  ② δa 一致性   同一模型在两条路径上的输出
  ③ 安全性质    |δa| ≤ ε、无 NaN、still 段被门控置零

用法:
  python sim/dryrun_pi_direct.py --variant bare
  python sim/dryrun_pi_direct.py --variant ff
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.policies.act.b_runtime import (
    build_offline, FeatureAssembler, IdealChunkRenderer, apply_ff12, WIN, FEAT_DIM)

ROOT = Path(__file__).resolve().parents[1]
EPS_B, LSHIFT = 0.010, 3

VARIANT = {
    "bare": {"tag": "pp2a-deploy", "ck": "pi_direct_bare_s0.pt",
             "log": "outputs/ab2/A_ep0.json", "ff": None},
    "ff":   {"tag": "pp2ff-deploy", "ck": "pi_direct_ff_s0.pt",
             "log": "outputs/ab2/B3b13_ep0.json", "ff": (3, 13)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT), default="bare")
    a_ = ap.parse_args()
    V = VARIANT[a_.variant]
    from bakeoff_backbones import TCN

    man = json.loads((ROOT / "outputs/p2_labels" / V["tag"] / "manifest.json").read_text())
    R = np.array(man["R"], np.float32); b = np.array(man["b"], np.float32)
    print(f"[{a_.variant}]  标签 {V['tag']}  检查点 {V['ck']}  "
          f"FF={V['ff']}  manifest L={man['lag']}")
    assert man["lag"] == LSHIFT, f"manifest L={man['lag']} 与部署 LSHIFT={LSHIFT} 不一致"

    model = TCN(d=FEAT_DIM)
    model.load_state_dict(torch.load(ROOT / "outputs/p2_pi" / V["ck"], map_location="cpu"))
    model.eval()

    # ---------- 离线路径 ----------
    z = np.load(ROOT / "outputs/p2_labels" / V["tag"] / "ep000.npz")
    X_off, _, _ = build_offline(z, L=LSHIFT)
    with torch.no_grad():
        da_off = model(torch.from_numpy(X_off).unsqueeze(0))[0].numpy()

    # ---------- 流式路径（复现部署时序）----------
    d = json.loads((ROOT / V["log"]).read_text())
    imu = np.array([f["imu"] for f in d["frames"] if f["imu"] is not None], np.float32)
    st = np.array([f["state"] for f in d["frames"]], np.float64)[:, :6] * (np.pi / 180 / 1000)
    rend = IdealChunkRenderer()
    asm = FeatureAssembler(R, b, L=LSHIFT)
    n = min(len(imu), len(st), len(X_off))

    X_str, da_str, prev = [], [], None
    fed = 0
    for c in d["chunks"]:
        cmd = np.array(c["raw"], np.float32)
        if V["ff"] is not None:
            cmd = apply_ff12(cmd, prev, lag=V["ff"][0], blend=V["ff"][1])
        asm.on_new_chunk(cmd, rend.render(cmd, prev))
        prev = cmd.copy()
        for _ in range(len(cmd)):
            if fed >= n:
                break
            win, gate = asm.step(imu[fed], st[fed].astype(np.float32))
            X_str.append(win[-1])
            with torch.no_grad():
                da_str.append(model(torch.from_numpy(win).unsqueeze(0))[0, -1].numpy())
            fed += 1
        if fed >= n:
            break
    X_str = np.stack(X_str); da_str = np.stack(da_str)
    m = min(len(X_str), n)

    # ---------- 比对 ----------
    ok = True
    warm = WIN                                  # 前 WIN 帧窗口未填满, 单列
    # **集尾也必须单列**: 离线 build_offline 的 cap 被 L 裁到 n−1, 流式 cap 是整 chunk 末,
    # 于是最后若干帧的前瞻 j4/j8 夹取位置不同。实测差异全部落在末 12 帧(583 帧的集),
    # 且**部署侧才是对的**(它有完整的 chunk 可前瞻); 那些帧又是停放段, 会被 still 门控。
    tailskip = 16
    m2 = m - tailskip
    dx = np.abs(X_str[warm:m2] - X_off[warm:m2])
    print(f"\n① 特征一致性（跳过前 {warm} 帧预热窗 + 末 {tailskip} 帧）  "
          f"比对 {m2-warm} 帧 × {FEAT_DIM} 维")
    print(f"   最大绝对差 {dx.max():.3e}   中位 {np.median(dx):.3e}")
    if dx.max() > 1e-4:
        bad = np.argsort(dx.max(0))[::-1][:5]
        print(f"   ✗ 超差！最差的 5 个特征维: {bad.tolist()}  "
              f"各自最大差 {dx.max(0)[bad].round(5).tolist()}")
        ok = False
    else:
        print("   ✓ 一致")

    dd = np.abs(da_str[warm:m2] - da_off[warm:m2])
    print(f"\n② δa 一致性   最大绝对差 {dd.max()*1e3:.4f} mrad   "
          f"中位 {np.median(dd)*1e3:.4f} mrad")
    if dd.max() > 1e-4 * EPS_B * 1e3:
        print("   ✗ 两条路径的 δa 不一致"); ok = False
    else:
        print("   ✓ 一致")

    print(f"\n③ 安全性质")
    mx = np.abs(da_str).max()
    print(f"   |δa|max = {mx*1e3:.3f} mrad  (ε={EPS_B*1e3:.0f})   "
          f"{'✓' if mx <= EPS_B + 1e-9 else '✗ 越界'}")
    print(f"   NaN {int(np.isnan(da_str).sum())}   "
          f"{'✓' if not np.isnan(da_str).any() else '✗'}")
    still = z["mask_still"][:m]
    print(f"   RMS: 运动段 {np.sqrt((da_str[:m][~still]**2).mean())*1e3:.2f} mrad  "
          f"静止段 {np.sqrt((da_str[:m][still]**2).mean())*1e3:.2f} mrad "
          f"（still 由部署侧 gate 另行置零）")
    if np.isnan(da_str).any() or mx > EPS_B + 1e-9:
        ok = False

    print(f"\n{'✅ 全部通过, 可以上机' if ok else '❌ 有项目未通过, 不要上机'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
