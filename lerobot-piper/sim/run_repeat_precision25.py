#!/usr/bin/env python3
"""重复精度封顶验证（25 次）的专用采集入口。

依据：`reports/PLAN_REPEAT_PRECISION_CAPPED.md`（冻结的实验设计）与 `reports/PLAN_IMPL_REPEAT_PRECISION_25.md`（实施计划）。
一次调用只执行固定顺序里的下一条；没有重试、跳过、补跑或 `--force`。
release 是工程授权记录，不是安全证明；这些 Python 守卫防误操作，不是防篡改的安全边界。

    $PY -u sim/run_repeat_precision25.py --prepare        # 离线：从 pilot 复制并锁定冻结输入、写 manifest
    $PY -u sim/run_repeat_precision25.py --dry-run        # 离线核对；不连接设备，不开始现场时钟
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, HERE / "q1"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/repeat_precision25"
PILOT_SRC = ROOT / "outputs/pilot_response14"
RELEASE = ROOT / "reports/RELEASE_REPEAT_PRECISION_25.json"
RUN_SET_ID = "repeat-precision25"
SCHEMA = "repeat-precision25-v1"
RELEASE_SCHEMA = "repeat-precision25-release-v1"
BUDGET = dict(max_attempts=25, max_elapsed_s=3000)
PREHOLD_FRAMES = 4                       # 0.2 s 前置保持（20 Hz），单独记录，不进正式帧
COUNTS_PER_RAD = 57324.840764           # audit.json：float32 相加 → Python float → round(x × 本系数)
EPS_RAD = 0.010000001

# pilot manifest 里的冻结输入（同一名义轨迹、同一 P / Q 波形与幅值），字节级锁定
FROZEN_INPUTS = {
    "A.npz": "89985a6a46cac2857941f60362f008dfc936a98c188d10154be7bca061f6ba4f",
    "P_minus.npz": "e88d2eccdfb7ce7348606b17b16d2b9d1a0047ea063ba4fcac8f3c2e43876c2e",
    "P_plus.npz": "2b97231a2fa1c62ffb70cb9fafb9fa20b345d0864cf37642b9dc92e27db22c39",
    "P_sent_counts.npz": "54c9d22fec16428f6ce5835efd78758ac7eaa2295ed3349d37f50ac7a97534e6",
    "Q_minus.npz": "6230bfbaeaaa84ef0af00e8099cb524c511e22cb9e979786139ea0c23e965280",
    "Q_plus.npz": "3afd896a918cb4b77150a294ee0f7f3fae1e5c03d3051bf5d24a0c82d0ab7a89",
    "Q_sent_counts.npz": "a9e93c92c00d3677099922d3f68bf46affd7403b45f33fc2f83304a57d9543f9",
    "audit.json": "6bd00e13c1198c52935b6abc6778dbcf75115c452092937ce2b5780c920205e7",
    "calibration_reference.npz": "fa0f31002abd66ea7b6dd0e987a8d6ec22d868e26ccaed650baed942365611e4",
}
# 执行、分析及其直接依赖：release 里锁定，每次运行核对
CODE_FILES = tuple(ROOT / p for p in (
    "sim/run_repeat_precision25.py", "sim/replay_repeat_precision25.py", "sim/analyze_repeat_precision25.py",
    "sim/precision_chain.py", "sim/q1/q1_windows.py", "sim/q1/q1_data.py",
    "sim/analyze_pilot_response14.py", "sim/diag_pilot_response14_spikes.py",
    "sim/replay_probe.py", "sim/analyze_probe_3a.py", "sim/make_probe_b2.py", "sim/compare_ideal_methods.py",
    "sim/ideal_imu_kinematic.py", "sim/piper_model.xml", "sim/step3_canobs/can_hs_recorder.py",
    "lerobot/common/policies/act/b_runtime.py", "lerobot/common/robot_devices/motors/piper.py",
    "lerobot/common/robot_devices/robots/piper.py", "scripts/tsnd_real/tsnd151_reader.py"))

SEQ = {"P+": "P_plus.npz", "P-": "P_minus.npz", "Q+": "Q_plus.npz", "Q-": "Q_minus.npz"}
BLOCKS = (("bare", "P+", "Q-", "bare", "P-", "Q+"),     # 方案第 4 节：拉丁方轮换的 4 个区组
          ("bare", "Q+", "P-", "bare", "Q-", "P+"),
          ("bare", "P-", "Q+", "bare", "P+", "Q-"),
          ("bare", "Q-", "P+", "bare", "Q+", "P-"))


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_fingerprints() -> dict:
    return {str(Path(p).resolve()): sha(p) for p in CODE_FILES}


def _run(run_id, block, condition) -> dict:
    if condition in ("A0", "bare"):
        group, sign, seq = "A", 0, "A.npz"
    else:
        group, sign, seq = condition[0], (1 if condition[1] == "+" else -1), SEQ[condition]
    return dict(run_id=run_id, block=block, group=group, sign=sign, condition=condition,
                sequence=seq, use_probe=sign != 0)


def expected_runs() -> list:
    runs = [_run("A0", 0, "A0")]
    for b, conds in enumerate(BLOCKS, 1):
        k = 0
        for c in conds:
            if c == "bare":
                k += 1
                rid = f"b{b}_bare{k}"
            else:
                rid = f"b{b}_{c[0]}{'p' if c[1] == '+' else 'm'}"
            runs.append(_run(rid, b, c))
    return runs


def planned_counts(cmd7, da) -> np.ndarray:
    s = (np.asarray(cmd7)[:, :6] + np.asarray(da)).astype(np.float32)
    return np.rint(s.astype(np.float64) * COUNTS_PER_RAD).astype(np.int64)


def _artifact(name) -> Path:
    p = OUT / name
    if Path(name).name != name or p.resolve().parent != OUT.resolve():
        raise ValueError(f"冻结文件必须是本目录下的普通文件名: {name}")
    return p


def _write(path, data, exclusive=False):
    """耐久写入（输出产物，不是源码）。"""
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_inputs() -> int:
    """冻结序列的内容核验 → 名义帧数。"""
    with np.load(_artifact("A.npz"), allow_pickle=False) as z:
        base = z["cmd7"].copy()
    T = len(base)
    if T < 550 or base.shape != (T, 7) or base.dtype != np.float32 or not np.isfinite(base).all():
        raise ValueError("名义轨迹 / 帧数不合法")
    waves = {}
    for name in ("A.npz", *SEQ.values()):
        with np.load(_artifact(name), allow_pickle=False) as z:
            cmd, da, pc = z["cmd7"], z["da"], z["planned_counts"]
            if (cmd.dtype != np.float32 or da.dtype != np.float32 or not np.array_equal(cmd, base)
                    or da.shape != (T, 6) or not np.isfinite(da).all()
                    or np.abs(da).max() > EPS_RAD or np.any(da[:, 3])):
                raise ValueError(f"冻结序列不合法: {name}")
            if not np.array_equal(pc, planned_counts(cmd, da)):
                raise ValueError(f"计划发送计数与序列不符: {name}")
            waves[name] = da.copy()
    if np.any(waves["A.npz"]):
        raise ValueError("裸序列含探针")
    for g in "PQ":
        p, m = waves[f"{g}_plus.npz"], waves[f"{g}_minus.npz"]
        if not np.any(p) or not np.array_equal(m, -p):
            raise ValueError(f"{g} 探针的正负两侧不成对")
    with np.load(_artifact("calibration_reference.npz"), allow_pickle=False) as z:
        om = z["omega0"]
        if om.shape != (T, 3) or not np.isfinite(om).all():
            raise ValueError("标定参考不合法")
    return T


def _manifest(T: int) -> dict:
    import precision_chain as C
    return dict(schema=SCHEMA, run_set_id=RUN_SET_ID,
                design="reports/PLAN_REPEAT_PRECISION_CAPPED.md (76785f5)",
                implementation="reports/PLAN_IMPL_REPEAT_PRECISION_25.md",
                budget=BUDGET, nominal_frames=T, fps=20.0, prehold_frames=PREHOLD_FRAMES,
                probe_kind="multi_joint_continuous", counts_per_rad=COUNTS_PER_RAD,
                chain=dict(fir_sha16=C.FIR_SHA16, numtaps=C.NUMTAPS, cutoff_hz=C.CUTOFF_HZ,
                           padlen=C.PADLEN, offset_s=C.OFFSET_S),
                runs=expected_runs(), files=dict(FROZEN_INPUTS), preflight_complete=True)


def prepare(src=None) -> dict:
    """离线：从 pilot 复制冻结输入（复制前后都按锁定哈希核对），写 manifest；不覆盖不同的已有文件。"""
    src = Path(src or PILOT_SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "session.json").exists():
        raise ValueError("会话已经开始，不能重新准备")
    for name, digest in FROZEN_INPUTS.items():
        dst = _artifact(name)
        if sha(src / name) != digest:
            raise ValueError(f"源文件与锁定哈希不符: {name}")
        if dst.exists():
            if sha(dst) != digest:
                raise ValueError(f"目标目录已有不同的 {name}，拒绝覆盖")
            continue
        shutil.copyfile(src / name, dst)
        if sha(dst) != digest:
            raise ValueError(f"复制后哈希不符: {name}")
    m = _manifest(_validate_inputs())
    p = OUT / "manifest.json"
    if p.exists():
        if json.loads(p.read_text()) != m:
            raise ValueError("已存在不同的 manifest，拒绝覆盖")
        return m
    _write(p, m, exclusive=True)
    return m


def _check_manifest(m: dict) -> None:
    if (m.get("schema") != SCHEMA or m.get("run_set_id") != RUN_SET_ID or m.get("budget") != BUDGET
            or m.get("preflight_complete") is not True or m.get("prehold_frames") != PREHOLD_FRAMES):
        raise ValueError("manifest 身份 / 预算 / 前置保持不符")
    if m.get("runs") != expected_runs():
        raise ValueError("manifest 的顺序与冻结设计不符")
    if m.get("files") != FROZEN_INPUTS:
        raise ValueError("manifest 的冻结输入与锁定清单不符")
    for name, digest in m["files"].items():
        if sha(_artifact(name)) != digest:
            raise ValueError(f"冻结文件被改动: {name}")
    if _validate_inputs() != m.get("nominal_frames"):
        raise ValueError("名义帧数与 manifest 不符")


def load_frozen(require_motion: bool = True) -> tuple:
    """读取并核对 release 与 manifest；不导入任何设备后端。"""
    release = json.loads(RELEASE.read_text())
    if (release.get("schema") != RELEASE_SCHEMA or release.get("run_set_id") != RUN_SET_ID
            or release.get("budget") != BUDGET):
        raise ValueError("release 不属于本次 25 次封顶验证")
    if require_motion and (release.get("motion_authorized") is not True
                           or release.get("waive_missing_rate_limit_only") is not True):
        raise ValueError("运动未授权（motion_authorized 不为 true）：不开始会话、不连接设备")
    if release.get("manifest_sha256") != sha(OUT / "manifest.json"):
        raise ValueError("manifest 在 release 之后被改动")
    if release.get("code_sha256") != code_fingerprints():
        raise ValueError("执行 / 分析代码在 release 之后被改动，须重新离线核验并重新锁定指纹")
    m = json.loads((OUT / "manifest.json").read_text())
    _check_manifest(m)
    return release, m
