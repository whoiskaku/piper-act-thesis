"""B4 的 sim 侧共用：路径、在线配置、闭环一集的读取与指标。方案 `reports/PLAN_b4.md`，实现计划 `reports/PLAN_IMPL_B4.md`。

  python -u sim/b4_common.py cfg     # 生成 outputs/b4/online_cfg.json（已存在且内容相同就跳过，不同就停）
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmd_tracking as T  # noqa: E402
import repeat_reversal as RR  # noqa: E402
import shift_replay as S  # noqa: E402
from ideal_imu_kinematic import MODEL_PATH  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

OUT = HERE.parent / "outputs/b4"
CFG = OUT / "online_cfg.json"
CALIB = HERE.parent / "outputs/imu_calib/imu_calib.json"
MARGIN = max(S.CONDS.values()) + 0.05              # 非接触段末尾留白，同 ilc_route1.MARGIN


def make_cfg(path: Path = CFG) -> Path:
    """在线配置：目标中心 = 示教 pickcube_raw 里放下点的中位数（同 v0.10 的疑似接触目标区），限位 = 模型 jnt_range，其余取默认值。"""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    cfg = O.OnlineCfg(center=tuple(float(x) for x in T.target_center("pickcube_raw", model)),
                      jnt_lo=tuple(float(x) for x in model.jnt_range[:6, 0]),
                      jnt_hi=tuple(float(x) for x in model.jnt_range[:6, 1]), model_path=str(MODEL_PATH))
    text = json.dumps(dataclasses.asdict(cfg), indent=1)
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"⛔ {path} 已存在且内容不同，不覆盖")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_ctx(calib: Path = CALIB) -> SimpleNamespace:
    """算指标要用的：模型、IMU 安装点、标定（默认当前这份）及其哈希、目标中心（同在线配置）。"""
    import hashlib
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return SimpleNamespace(model=model, sid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site"),
                           calib=json.loads(Path(calib).read_text()), calib_sha=hashlib.sha256(Path(calib).read_bytes()).hexdigest()[:16],
                           center=np.asarray(O.load_cfg(CFG).center) if CFG.exists() else None)


def load_closed(repo: str, i: int) -> dict:
    """闭环一集：高频记录 + B4 包装的记录（_b4.npz）+ 数据集动作 → 同 cmd_tracking.load 的一集（指令 = 实发的 u），
    另加 a_pi（原始策略指令，评价的参考）、w、mode、act_mismatch（数据集动作与 u 的最大差）。记录不完整或帧数对不上就报错。"""
    import pandas as pd
    root = T.LR / repo
    h = json.loads((root / "high_rate" / f"episode_{i:06d}.json").read_text())
    log = np.load(root / "high_rate" / f"episode_{i:06d}_b4.npz")
    act = np.stack(pd.read_parquet(root / f"data/chunk-000/episode_{i:06d}.parquet")["action"].to_numpy())
    if not (h.get("complete") and bool(log["complete"])):
        raise ValueError(f"{repo} 第 {i} 集的记录不完整")
    if not len(h["t_send"]) == len(log["u"]) == len(act):
        raise ValueError(f"{repo} 第 {i} 集帧数对不上：高频 {len(h['t_send'])}、B4 {len(log['u'])}、数据集 {len(act)}")
    ep = T.from_record(h, log["u"].astype(float))
    ep.update(a_pi=log["a_pi"].astype(float), w=log["w"], mode=str(log["mode"]),
              act_mismatch=float(np.abs(act - log["u"]).max()))
    return ep


def closed_metrics(ep: dict, ctx) -> dict:
    """闭环一集的指标，参考 = 这一集自己的原始策略指令 a_π（不是实发的 u）：
    关节偏差、末端角速度偏差（只报告）、路径长度、关节速度（shift_replay.metrics，非接触段）；
    任务时长 = 第一次合拢后第一次张开的时刻 − 第一帧（s，没张开为 NaN）；策略指令自身速度 = a_π 五个活动关节的速度 RMS（rad/s）。"""
    a7 = np.asarray(ep["a_pi"], float)
    marked = RR.contact_frames(a7, ctx.model, ctx.center)
    fq, fi = S.masks(ep, marked, MARGIN)
    m = S.metrics(ep, a7[:, :6], fq, fi, ctx.calib, ctx.model, ctx.sid)
    hi = a7[:, 6] > T.GRIP_OPEN
    close = np.flatnonzero(hi[:-1] & ~hi[1:]) + 1
    opened = np.flatnonzero(~hi[:-1] & hi[1:]) + 1
    rel = opened[opened > close[0]] if len(close) else []
    v = np.diff(a7[:, list(O.ACTIVE)], axis=0) / np.diff(ep["t_send"])[:, None]
    return {**{k: m[k] for k in ("joint_raw", "ee_raw", "path", "speed")},
            "duration": float(ep["t_send"][rel[0]] - ep["t_send"][0]) if len(rel) else float("nan"),
            "cmd_speed": float(np.sqrt(np.mean(v ** 2)))}


def main(argv=None) -> int:
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser(description="B4 的 sim 侧共用")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cfg", help="生成在线配置")
    args = ap.parse_args(argv)
    if args.cmd == "cfg":
        p = make_cfg()
        print(f"目标中心 {np.round(O.load_cfg(p).center, 4)} m；在线配置 {p}，哈希 {O.cfg_sha(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
