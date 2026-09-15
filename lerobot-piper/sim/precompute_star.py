#!/usr/bin/env python3
"""预计算教师 δa*（冻结 `G-2 + H 解析求解`，**零空档**）并缓存。

⚠ **不另写求解器** —— 直接用跑过真机的 `G2WallRuntime`，保证与 C 臂**逐位同源**。
（本会话已两次栽在 J/H 的时基换算上：`Jd[t,k]` 在 ω 轴、`gap[i]` 在 gap 轴，差 L=3。）

零空档 = 离线没有实时约束，直接 `fill_stage_b` 算满整 chunk。

用法:  python -u sim/precompute_star.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ENV = ROOT / "outputs/p2_pi/g2env"
CAL = {"ab5": "ab5-deploy", "ab6": "ab6-deploy", "ab4": "pp2a-deploy",
       "ab2": "pp2-deploy"}


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from lerobot.common.policies.act.g2wall_runtime import G2WallRuntime
    idx = json.loads((ENV / "index.json").read_text())["episodes"]
    rend = IdealChunkRenderer()
    rts, t0 = {}, time.perf_counter()
    done = 0
    for name in idx:
        out = ENV / (name.replace("/", "__") + "__star.npy")
        if out.exists():
            done += 1; continue
        arm_dir = name.split("/")[0]
        cal = CAL.get(arm_dir, "pp2a-deploy")
        if arm_dir not in rts:
            man = json.loads((ROOT / f"outputs/p2_labels/{cal}/manifest.json").read_text())
            rts[arm_dir] = G2WallRuntime(
                ROOT / "outputs/p2_pi/pi_g2wall_v1.json",
                np.asarray(man["R"], np.float32),
                np.asarray(man.get("b", [0, 0, 0]), np.float32), verbose=False)
        rt = rts[arm_dir]
        p = ROOT / "outputs" / name.replace("/", "/") 
        p = ROOT / f"outputs/{name}.json"
        pay = json.loads(p.read_text())
        chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
        rt.reset(); das, prev = [], None
        for c in chunks:
            raw = np.asarray(c["raw"], np.float64)
            ideal = np.asarray(rend.render(raw.astype(np.float32), prev), np.float32)
            rt.begin_chunk(raw)
            rt.fill_stage_b(ideal)                 # **零空档**
            das.append(np.stack([rt.step().delta for _ in range(len(raw))]))
            prev = raw.astype(np.float32)
        np.save(out, np.concatenate(das).astype(np.float32))
        done += 1
        if done % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {done}/{len(idx)}  {el:.0f}s（剩 ~{el/max(done,1)*(len(idx)-done):.0f}s）")
    print(f"\n  完成 {done} 集 → {ENV}/*__star.npy")
    return 0


if __name__ == "__main__":
    main(); sys.stdout.flush()
    import os; os._exit(0)
