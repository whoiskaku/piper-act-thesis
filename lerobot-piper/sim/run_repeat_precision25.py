#!/usr/bin/env python3
"""重复精度封顶验证（25 次）的专用采集入口。

依据：`reports/PLAN_REPEAT_PRECISION_CAPPED.md`（冻结的实验设计）与 `reports/PLAN_IMPL_REPEAT_PRECISION_25.md`（实施计划）。
一次调用只执行固定顺序里的下一条；没有重试、跳过、补跑或 `--force`。
release 是工程授权记录，不是安全证明；这些 Python 守卫防误操作，不是防篡改的安全边界。

    $PY -u sim/run_repeat_precision25.py --prepare        # 离线：从 pilot 复制并锁定冻结输入、写 manifest
    $PY -u sim/run_repeat_precision25.py --dry-run        # 离线核对；不连接设备，不开始现场时钟
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 本机 OpenBLAS 默认 20 线程：负载下一次 182×182 求解（filtfilt 的初值）要约 3.5 s，并空转约 11 个核。
# 在 numpy 载入之前钉住单线程（只影响本进程）。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
MIN_START_S = 60.0                       # 剩余不足 60 s 不开始下一条
PREHOLD_FRAMES = 4                      # 0.2 s 前置保持（20 Hz），单独记录，不进正式帧
MAX_IMU_GAP_MS = 15.0                   # 方案第七节 3
MAX_SEND_INTERVAL_S = 0.100             # 方案第七节 5
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


# ── 会话、预算、锁停 ──────────────────────────────────────────────────────────
def now_mono() -> float:
    return time.monotonic()


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safety_audit(manifest) -> None:
    from replay_probe import safety_checks, MISSING_RATE_LIMIT_BLOCKER
    for name in sorted({r["sequence"] for r in manifest["runs"]}):
        with np.load(_artifact(name), allow_pickle=False) as z:
            _, degraded, blockers = safety_checks(z["cmd7"], z["da"], probe_kind="multi_joint_continuous")
        if degraded or any(b != MISSING_RATE_LIMIT_BLOCKER for b in blockers):
            raise ValueError(f"{name}: 未豁免的安全检查失败: {degraded!r}, {blockers!r}")


@contextlib.contextmanager
def _lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "session.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError("另一个进程持有本会话") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_session() -> dict:
    return json.loads((OUT / "session.json").read_text())


def _check_clock(s, required=MIN_START_S) -> float:
    if s["boot_id"] != boot_id():
        raise ValueError("机器重启过：本会话不能继续")
    remaining = min(s["deadline_monotonic"] - now_mono(),
                    s["started_unix"] + BUDGET["max_elapsed_s"] - time.time())
    if remaining < required or now_mono() < s["started_monotonic"]:
        raise ValueError("现场时间预算已用尽 / 时钟异常：不开始下一条运动")
    return remaining


def start_session(operator_ready=False) -> dict:
    if not operator_ready:
        raise ValueError("开始现场计时须 --operator-ready；时钟包含复位与等待")
    with _lock():
        _, m = load_frozen(require_motion=True)
        _safety_audit(m)
        if (OUT / "session.json").exists():
            raise ValueError("会话已存在；预算不能重置")
        if ((OUT / "records").exists() and any((OUT / "records").iterdir())) or (OUT / "alignment_A0.json").exists():
            raise ValueError("存在以前会话的记录或对齐文件；拒绝开始新预算")
        start = now_mono()
        s = dict(run_set_id=RUN_SET_ID, token=uuid.uuid4().hex, boot_id=boot_id(),
                 release_sha256=sha(RELEASE), manifest_sha256=sha(OUT / "manifest.json"),
                 started_utc=utc(), started_unix=time.time(), started_monotonic=start,
                 deadline_monotonic=start + BUDGET["max_elapsed_s"], attempts=[], halted=False)
        _write(OUT / "session.json", s, exclusive=True)
        return s


def _check_session(s) -> None:
    if (s.get("run_set_id") != RUN_SET_ID or s.get("release_sha256") != sha(RELEASE)
            or s.get("manifest_sha256") != sha(OUT / "manifest.json")):
        raise ValueError("会话与 release / manifest 的绑定被改动")
    if s.get("halted"):
        raise ValueError("已锁停：失败后不自动重试、不补跑")
    runs = expected_runs()
    if len(s["attempts"]) > len(runs):
        raise ValueError("尝试次数超出预算")
    for a, r in zip(s["attempts"], runs):
        if a.get("run_id") != r["run_id"]:
            raise ValueError("账本顺序与冻结设计不符")
    _check_clock(s)


def _validate_alignment() -> None:
    p = OUT / "alignment_A0.json"
    if not p.exists():
        raise ValueError("任何探针条之前必须先冻结独立的 A0 对齐")
    a = json.loads(p.read_text())
    if (a.get("A0_sha256") != sha(OUT / "records/replay_A0.json")
            or a.get("reference_sha256") != sha(OUT / "calibration_reference.npz")
            or a.get("determined_by") != "independent A0 only"):
        raise ValueError("A0 对齐或其来源被改动")
    if read_session().get("alignment_sha256") != sha(p):
        raise ValueError("入账后的 A0 对齐被改动")


def freeze_a0() -> dict:
    """只用已完成的 A0 与冻结的标定参考（与 pilot 同法），绝不用探针条。"""
    from analyze_probe_3a import fit_frame
    p = OUT / "records/replay_A0.json"
    a0 = np.asarray(json.loads(p.read_text())["imu"], float)[:, 3:6]
    with np.load(OUT / "calibration_reference.npz", allow_pickle=False) as z:
        om = z["omega0"]
    if a0.shape != om.shape or not np.isfinite(a0).all() or not np.isfinite(om).all():
        raise ValueError("A0 / 标定参考的形状或有限值核验失败")
    Rm, L, cos, angle = fit_frame(a0, om)
    if not np.isfinite(Rm).all() or np.linalg.det(Rm) < 0:
        raise ValueError("A0 没有给出有限的正常旋转；不进入探针条")
    a = dict(determined_by="independent A0 only", frozen_utc=utc(), A0_sha256=sha(p),
             reference_sha256=sha(OUT / "calibration_reference.npz"),
             method="analyze_probe_3a.fit_frame; SHIFTS=0..6; raw orthogonal Procrustes",
             R=Rm.tolist(), L=int(L), cosine=float(cos), angle_deg=float(angle))
    _write(OUT / "alignment_A0.json", a, exclusive=True)
    return a


def validate_replay_context(context, seqfile, run_id, cmd7, da, use_probe) -> dict:
    """执行器在构造设备之前调用：磁盘哈希、当前预留、条目与数组、持久预算都必须独立吻合。"""
    _, m = load_frozen(require_motion=True)
    s = read_session()
    _check_session(s)
    attempts = s["attempts"]
    if not attempts or len(attempts) > BUDGET["max_attempts"]:
        raise ValueError("没有为这次复放预留的尝试")
    a, want = attempts[-1], m["runs"][len(attempts) - 1]
    if (a.get("status") != "running" or a.get("run_id") != want["run_id"]
            or context.get("token") != s["token"] or context.get("run_id") != want["run_id"]
            or run_id != want["run_id"] or bool(use_probe) != want["use_probe"]
            or Path(seqfile).resolve() != _artifact(want["sequence"]).resolve()):
        raise ValueError("预留的复放 / 上下文不符")
    if any(old.get("status") != "complete" for old in attempts[:-1]):
        raise ValueError("之前的复放未完成")
    with np.load(_artifact(want["sequence"]), allow_pickle=False) as z:
        if not np.array_equal(cmd7, z["cmd7"]) or not np.array_equal(da, z["da"]):
            raise ValueError("载入的数组与冻结序列不同")
    if use_probe:
        _validate_alignment()
    return dict(release_id=RUN_SET_ID, run_id=want["run_id"], sequence=want["sequence"],
                seq_sha256=m["files"][want["sequence"]], deadline_monotonic=s["deadline_monotonic"])


# ── 记录完整性 ────────────────────────────────────────────────────────────────
def check_record(run, manifest) -> dict:
    """一条复放记录的完整性核验；不过即抛 ValueError（入口据此失败锁停，分析据此判无法判定）。"""
    import precision_chain as C
    p = OUT / "records" / f"replay_{run['run_id']}.json"
    raw = p.read_bytes()
    rec = json.loads(raw)
    T = manifest["nominal_frames"]
    if (rec.get("stopped") is not None or rec.get("return_blocked") or rec.get("return_deadline")
            or rec.get("return_error")):
        raise ValueError("采集未完成或归位未成功")
    imu, cmd = np.asarray(rec.get("imu", []), float), np.asarray(rec.get("cmd", []), float)
    if imu.shape != (T, 6) or cmd.shape != (T, 7) or not np.isfinite(imu).all() or not np.isfinite(cmd).all():
        raise ValueError("正式帧数或有限值不符")
    for key in ("t", "t_send"):
        st = np.asarray(rec.get(key, []), float)
        if st.shape != (T,) or not np.isfinite(st).all() or np.any(np.diff(st) <= 0):
            raise ValueError(f"时间戳不完整或不递增: {key}")
    ts = np.asarray(rec["t_send"], float)
    if np.diff(ts).max() > MAX_SEND_INTERVAL_S:
        raise ValueError("写总线间隔超过 100 ms")
    sc = rec.get("scope", {})
    if (sc.get("release_id") != RUN_SET_ID or sc.get("run_id") != run["run_id"]
            or sc.get("sequence") != run["sequence"] or sc.get("seq_sha256") != manifest["files"][run["sequence"]]):
        raise ValueError("记录的身份字段不符")
    # 方案第七节 2：发送计数与计划计数逐帧一致；指令与冻结波形逐帧完全一致
    with np.load(_artifact(run["sequence"]), allow_pickle=False) as z:
        cmd7, da, pc = z["cmd7"], z["da"], z["planned_counts"]
    sent = cmd7.copy()
    if run["use_probe"]:
        sent[:, :6] += da
    if not np.array_equal(planned_counts(cmd.astype(np.float32), np.zeros((T, 6), np.float32)), pc):
        raise ValueError("发送计数与计划计数不一致")
    if not np.array_equal(cmd, sent.astype(np.float64)):
        raise ValueError("记录的正式指令与冻结波形不同")
    # 第七节 3：100 Hz 设备时间戳单调、间隔不超过 15 ms
    i100 = np.asarray(rec.get("imu100", []), float)
    if i100.ndim != 2 or i100.shape[1] != 8 or len(i100) < 2 or not np.isfinite(i100).all():
        raise ValueError("100 Hz 流缺失或含非有限值")
    dms = np.diff(i100[:, 1])
    if np.any(dms <= 0):
        raise ValueError("100 Hz 设备毫秒不单调")
    if dms.max() > MAX_IMU_GAP_MS:
        raise ValueError("100 Hz 流有超过 15 ms 的间隔")
    # 第七节 4：每个 20 Hz 样本都能在 100 Hz 流中找到相同的值
    pool = {tuple(r) for r in i100[:, 2:8].tolist()}
    if any(tuple(r) not in pool for r in imu.tolist()):
        raise ValueError("有 20 Hz 样本在 100 Hz 流中找不到相同的值")
    # 前置保持：单独记录、目标 = 稳定段目标、全部早于第一个正式帧
    pre = rec.get("prehold")
    if not isinstance(pre, dict):
        raise ValueError("前置保持记录缺失")
    pts, pcmd = np.asarray(pre.get("t_send", []), float), np.asarray(pre.get("cmd", []), float)
    if (pre.get("frames") != PREHOLD_FRAMES or pts.shape != (PREHOLD_FRAMES,) or pcmd.shape != (PREHOLD_FRAMES, 7)
            or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0) or pts[-1] >= ts[0]):
        raise ValueError("前置保持记录不全，或没有早于第一个正式帧")
    hold = np.concatenate([cmd7[0, :6], cmd7[0, 6:7]]).astype(np.float32).astype(np.float64)
    if not np.array_equal(np.asarray(pre.get("target", []), float), hold) or not (pcmd == hold).all():
        raise ValueError("前置保持的目标不是稳定段目标")
    # 覆盖：按实际时间戳，处理链的全部取样点都在 100 Hz 流之内（不外推）
    tb = C.time_base(i100)
    q = ts + C.OFFSET_S
    lead, tail = float(q[0] - tb[0]), float(tb[-1] - q[-1])
    if lead < 0 or tail < 0:
        raise ValueError("处理链取样点超出 100 Hz 覆盖范围（不外推）")
    C.frame_locked(rec)                                                 # 整条处理链能跑通
    return dict(sha256=hashlib.sha256(raw).hexdigest(), coverage_lead_s=lead, coverage_tail_s=tail,
                timebase_residual_sd_s=float(np.std(i100[:, 0] - tb)), imu100_max_gap_ms=float(dms.max()),
                send_interval_max_s=float(np.diff(ts).max()), n_imu100=int(len(i100)))


# ── 单条执行 ──────────────────────────────────────────────────────────────────
def run_next(operator_ready=False, replay_fn=None) -> int:
    if not operator_ready:
        raise ValueError("确认本条的现场与复位已就绪：--operator-ready")
    with _lock():
        _, m = load_frozen(require_motion=True)
        _safety_audit(m)
        s = read_session()
        _check_session(s)
        if len(s["attempts"]) >= BUDGET["max_attempts"]:
            raise ValueError("25 次尝试已用完；不授权任何额外或补跑的复放")
        if any(a.get("status") != "complete" for a in s["attempts"]):
            raise ValueError("有未完成或失败的尝试；不自动重试")
        run = m["runs"][len(s["attempts"])]
        if run["use_probe"]:
            _validate_alignment()
        output = OUT / "records" / f"replay_{run['run_id']}.json"
        if output.exists():
            raise ValueError("结果已存在；不覆盖、不静默重复")
        output.parent.mkdir(parents=True, exist_ok=True)
        attempt = dict(run_id=run["run_id"], status="running", started_utc=utc(),
                       started_monotonic=now_mono(), sequence=run["sequence"])
        s["attempts"].append(attempt)
        _write(OUT / "session.json", s)              # 连接设备之前记账：初始化失败也消耗一次
        code = 5
        try:
            if replay_fn is None:
                from replay_repeat_precision25 import run as replay_fn
            code = replay_fn(run=run, seqfile=str(_artifact(run["sequence"])), outdir=output.parent,
                             context=dict(token=s["token"], run_id=run["run_id"]))
            if code == 0:
                info = check_record(run, m)
                attempt.update(record_sha256=info["sha256"], record_check=info)
                if run["run_id"] == "A0":
                    freeze_a0()
                    if (OUT / "alignment_A0.json").exists():
                        s["alignment_sha256"] = sha(OUT / "alignment_A0.json")
                _check_clock(s, required=0)              # 复放结束时已过截止 ⇒ 失败
            attempt["status"] = "complete" if code == 0 else "failed"
        except BaseException as e:
            attempt.update(status="failed", exception=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            code = 130 if isinstance(e, KeyboardInterrupt) else 5
        finally:
            attempt.update(exit_code=int(code), ended_utc=utc(), ended_monotonic=now_mono())
            s["halted"] = attempt["status"] != "complete"
            _write(OUT / "session.json", s)
        print(f"{run['run_id']}: {attempt['status']}；已用尝试 {len(s['attempts'])}/{BUDGET['max_attempts']}。"
              "没有自动开始下一条；先查看现场并复位，再执行下一条命令。")
        return int(code)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--prepare", "--dry-run", "--status", "--fingerprints", "--start-session", "--run-next"):
        action.add_argument(flag, action="store_true")
    ap.add_argument("--operator-ready", action="store_true")
    return ap


def main(argv=None) -> int:
    import gc
    gc.disable()
    a = build_parser().parse_args(argv)
    try:
        if a.prepare:
            m = prepare()
            print(f"离线准备完成：{len(m['runs'])} 条、{len(m['files'])} 个冻结输入。未连接设备，未开始现场时钟。")
            return 0
        if a.fingerprints:
            print(json.dumps(code_fingerprints(), ensure_ascii=False, indent=2))
            return 0
        if a.dry_run:
            release, m = load_frozen(require_motion=False)
            _safety_audit(m)
            print("离线核对通过：release、manifest、冻结输入、代码指纹一致；只有已知的速率上限阻断项。")
            if release.get("motion_authorized") is True:
                print("motion_authorized: true（本命令未连接设备、未开始会话、没有运动）")
            else:
                print("motion_authorized: false —— 运动未授权；--start-session 会拒绝。未连接设备、没有运动。")
            return 0
        if a.status:
            print(json.dumps(read_session() if (OUT / "session.json").exists() else
                             {"status": "not_started", "motion_performed_by_this_command": False},
                             ensure_ascii=False, indent=2))
            return 0
        if a.start_session:
            s = start_session(a.operator_ready)
            print(f"现场时钟已开始 {s['started_utc']}；最多 50 分钟（含复位与等待）。未连接设备。"
                  "下一步：--run-next --operator-ready（只跑 A0）。")
            return 0
        return run_next(a.operator_ready)
    except (OSError, ValueError, KeyError, AssertionError) as e:
        print(f"拒绝：{type(e).__name__}: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
