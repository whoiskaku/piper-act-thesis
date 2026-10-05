"""重复精度 25 次各组测试共用的合成数据（只供测试；不属于执行或分析代码，不进指纹）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FILES = ("A.npz", "P_minus.npz", "P_plus.npz", "P_sent_counts.npz", "Q_minus.npz", "Q_plus.npz",
         "Q_sent_counts.npz", "audit.json", "calibration_reference.npz")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def counts(cmd7, da) -> np.ndarray:
    """audit.json 的 planned_counts 语义：float32 相加 → Python float → round(x × 57324.840764)。"""
    s = (np.asarray(cmd7)[:, :6] + np.asarray(da)).astype(np.float32)
    return np.array([[round(float(v) * 57324.840764) for v in row] for row in s], np.int64)


def omega_ref(T: int) -> np.ndarray:
    t = np.arange(T) / 20.0
    return np.stack([0.3 * np.sin(2 * np.pi * 0.7 * t), 0.2 * np.cos(2 * np.pi * 1.3 * t),
                     0.25 * np.sin(2 * np.pi * 2.1 * t + 0.4)], 1)


def make_inputs(d, T=560, *, minus_scale=-1.0, bare_probe=False, j4=False, peak=0.001) -> dict:
    """合成的 9 个冻结输入 → {文件名: sha256}。"""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t = np.arange(T) / 20.0
    cmd = np.zeros((T, 7), np.float32)
    cmd[:, 1] = (0.3 * np.sin(2 * np.pi * 0.1 * t)).astype(np.float32)
    cmd[:, 4] = 0.52
    cmd[:, 6] = 0.02
    wave = np.zeros((T, 6), np.float32)
    wave[:, 0] = (peak * np.sin(2 * np.pi * 3.0 * t)).astype(np.float32)
    wave[:, 4] = (0.5 * peak * np.cos(2 * np.pi * 5.0 * t)).astype(np.float32)
    if j4:
        wave[:, 3] = 0.0005
    gate = np.zeros(T, bool)
    gate[100:187] = True
    seqs = {"A.npz": wave * (1.0 if bare_probe else 0.0), "P_plus.npz": wave, "P_minus.npz": wave * minus_scale,
            "Q_plus.npz": wave * 0.7, "Q_minus.npz": wave * 0.7 * minus_scale}
    for name, da in seqs.items():
        da = np.asarray(da, np.float32)
        np.savez(d / name, cmd7=cmd, da=da, gate=gate, planned_counts=counts(cmd, da),
                 starts=np.zeros(6, np.int64), meta_json=np.array("{}"))
    for g in "PQ":
        np.savez(d / f"{g}_sent_counts.npz", c0=counts(cmd, 0 * wave), cp=counts(cmd, wave), cm=counts(cmd, -wave))
    np.savez(d / "calibration_reference.npz", omega0=omega_ref(T), planned_counts=counts(cmd, 0 * wave),
             cmd7_quantized=cmd.astype(np.float64))
    (d / "audit.json").write_text(json.dumps({"planned_counts_semantics": "float32 replay sum"}))
    return {name: sha(d / name) for name in FILES}


def write_release(path, out_dir, code_files, **over) -> dict:
    rel = dict(schema="repeat-precision25-release-v1", run_set_id="repeat-precision25",
               budget=dict(max_attempts=25, max_elapsed_s=3000), motion_authorized=True,
               waive_missing_rate_limit_only=True, manifest_sha256=sha(Path(out_dir) / "manifest.json"),
               code_sha256={str(Path(p).resolve()): sha(p) for p in code_files})
    rel.update(over)
    Path(path).write_text(json.dumps(rel))
    return rel


def good_record(run, out_dir, deadline, t0=100.0, seed=0, gyro=None, rotation=None) -> dict:
    """一条按最终记录格式完全合法的合成复放记录。

    100 Hz 流从首帧前 0.25 s 开始（覆盖前置保持与取样偏移），20 Hz 样本取读数时刻的最新 100 Hz 样本。
    `gyro(t_rel) -> (n, 3)` 给出相对首帧时刻的陀螺信号；`rotation` 把参考信号转到 IMU 坐标系（A0 对齐测试用）。
    """
    with np.load(Path(out_dir) / run["sequence"], allow_pickle=False) as z:
        cmd7, da = z["cmd7"], z["da"]
    T = len(cmd7)
    sent = cmd7.copy()
    if run["use_probe"]:
        sent[:, :6] += da
    rng = np.random.default_rng(seed)
    t_send = t0 + 0.05 * np.arange(T)
    t_read = t_send + 0.002
    t_start = t0 - 0.25
    n100 = int(np.ceil((t_send[-1] + 0.06 - t_start) / 0.01))
    host = t_start + 0.01 * np.arange(n100)
    ms = 5000 + 10 * np.arange(n100)
    if gyro is None:
        def gyro(tr):
            ref = omega_ref(T)
            f = np.stack([np.interp(tr * 20.0, np.arange(T), ref[:, j]) for j in range(3)], 1)
            return f + 0.01 * rng.standard_normal((len(tr), 3))
    g = np.asarray(gyro(host - t0), np.float64)
    if rotation is not None:
        g = g @ rotation
    acc = np.tile([0.0, 0.0, 9.8], (n100, 1)) + 0.001 * rng.standard_normal((n100, 3))
    i100 = np.column_stack([host, ms, acc, g])
    idx = np.searchsorted(host, t_read, side="right") - 1
    hold = np.concatenate([cmd7[0, :6], cmd7[0, 6:7]]).astype(np.float32)
    pre_ts = t0 - 0.05 * np.arange(4, 0, -1)
    return {
        "arm": run["run_id"], "clock": "time.monotonic", "use_probe": bool(run["use_probe"]),
        "scope": {"release_id": "repeat-precision25", "run_id": run["run_id"], "sequence": run["sequence"],
                  "seq_sha256": sha(Path(out_dir) / run["sequence"]), "deadline_monotonic": deadline},
        "t": t_read.tolist(), "t_send": t_send.tolist(), "t_send_span": [0.0002] * T,
        "cmd": sent.astype(np.float32).tolist(), "imu": i100[idx, 2:8].tolist(),
        "state": np.column_stack([sent[:, :6] * 180 / np.pi * 1000, sent[:, 6:7]]).tolist(),
        "imu100": i100.tolist(), "can_hs": [], "can_q200": [],
        "prehold": {"s": 0.2, "frames": 4, "target": hold.tolist(), "t": (pre_ts + 0.002).tolist(),
                    "t_send": pre_ts.tolist(), "t_send_span": [0.0002] * 4, "cmd": [hold.tolist()] * 4},
        "stopped": None, "return_blocked": False, "return_deadline": False,
        "cleanup": {"wrapper_destructor_disarmed": True, "motion_or_disable_sent_by_cleanup": False, "errors": []},
    }


def response_gyro(run, i, amp=0.1, noise=0.01, q_amp=None):
    """合成陀螺：所有条共有的名义运动 + 条件响应（P / Q 各一条波形，正负号反向）+ 逐条独立噪声。"""
    rng = np.random.default_rng(1000 + i)
    a = {"P": amp, "Q": amp if q_amp is None else q_amp}

    def g(tr):
        base = np.stack([0.3 * np.sin(2 * np.pi * 0.7 * tr), 0.2 * np.cos(2 * np.pi * 1.3 * tr),
                         0.25 * np.sin(2 * np.pi * 2.1 * tr + 0.4)], 1)
        resp = np.zeros_like(base)
        if run["use_probe"]:
            f = 3.0 if run["group"] == "P" else 4.5
            w = np.stack([np.sin(2 * np.pi * f * tr), np.cos(2 * np.pi * (f + 0.7) * tr), np.sin(2 * np.pi * (f + 1.9) * tr)], 1)
            resp = run["sign"] * a[run["group"]] * w
        return base + resp + noise * rng.standard_normal(base.shape)
    return g


def build_session(R, gyro_for=response_gyro, t0=100.0, start=50.0) -> dict:
    """在 R.OUT（已 prepare、已写 release）里写出 25 条合法记录与完成的账本。"""
    m = json.loads((R.OUT / "manifest.json").read_text())
    (R.OUT / "records").mkdir(exist_ok=True)
    deadline = start + 3000.0
    attempts = []
    for i, run in enumerate(m["runs"]):
        ti = t0 + 60.0 * i
        rec = good_record(run, R.OUT, deadline, t0=ti, seed=i, gyro=gyro_for(run, i))
        p = R.OUT / "records" / f"replay_{run['run_id']}.json"
        p.write_text(json.dumps(rec))
        attempts.append(dict(run_id=run["run_id"], status="complete", record_sha256=sha(p), exit_code=0,
                             sequence=run["sequence"], started_monotonic=ti - 20.0, ended_monotonic=ti + 40.0))
    s = dict(run_set_id="repeat-precision25", token="t", boot_id="b", release_sha256=sha(R.RELEASE),
             manifest_sha256=sha(R.OUT / "manifest.json"), started_utc="2026-10-05T00:00:00+00:00",
             started_unix=0.0, started_monotonic=start, deadline_monotonic=deadline, attempts=attempts, halted=False)
    (R.OUT / "session.json").write_text(json.dumps(s))
    return s
