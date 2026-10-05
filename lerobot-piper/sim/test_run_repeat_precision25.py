"""重复精度 25 次的采集入口测试：全部离线，不导入设备后端，不读写真实的 outputs/repeat_precision25。"""
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import run_repeat_precision25 as R   # noqa: E402

T = 560
PILOT = HERE.parent / "outputs/pilot_response14"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _counts(cmd7, da):
    """audit.json 的 planned_counts 语义：float32 相加 → Python float → round(x × 57324.840764)。"""
    s = (cmd7[:, :6] + da).astype(np.float32)
    return np.array([[round(float(v) * 57324.840764) for v in row] for row in s], np.int64)


def make_inputs(d, *, minus_scale=-1.0, bare_probe=False, j4=False, peak=0.001):
    """合成的 9 个冻结输入；返回 {文件名: sha256}。"""
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
        da = da.astype(np.float32)
        np.savez(d / name, cmd7=cmd, da=da, gate=gate, planned_counts=_counts(cmd, da),
                 starts=np.zeros(6, np.int64), meta_json=np.array("{}"))
    for g in "PQ":
        np.savez(d / f"{g}_sent_counts.npz", c0=_counts(cmd, 0 * wave), cp=_counts(cmd, wave), cm=_counts(cmd, -wave))
    np.savez(d / "calibration_reference.npz", omega0=np.ones((T, 3)), planned_counts=_counts(cmd, 0 * wave),
             cmd7_quantized=cmd.astype(np.float64))
    (d / "audit.json").write_text(json.dumps({"planned_counts_semantics": "float32 replay sum"}))
    return {name: _sha(d / name) for name in sorted(R.FROZEN_INPUTS)}


@pytest.fixture
def env(tmp_path, monkeypatch):
    src, out = tmp_path / "pilot", tmp_path / "out"
    pins = make_inputs(src)
    monkeypatch.setattr(R, "PILOT_SRC", src)
    monkeypatch.setattr(R, "OUT", out)
    monkeypatch.setattr(R, "RELEASE", tmp_path / "release.json")
    monkeypatch.setattr(R, "FROZEN_INPUTS", pins)
    monkeypatch.setattr(R, "CODE_FILES", (Path(__file__),))
    return tmp_path


def write_release(env, **over):
    rel = dict(schema=R.RELEASE_SCHEMA, run_set_id=R.RUN_SET_ID, budget=dict(max_attempts=25, max_elapsed_s=3000),
               motion_authorized=True, waive_missing_rate_limit_only=True,
               manifest_sha256=_sha(R.OUT / "manifest.json"),
               code_sha256={str(Path(__file__).resolve()): _sha(Path(__file__))})
    rel.update(over)
    R.RELEASE.write_text(json.dumps(rel))
    return rel


# ── 顺序 ──────────────────────────────────────────────────────────────────────
DESIGN_TABLE = [  # 方案第 4 节，逐字抄录
    ["bare", "P+", "Q-", "bare", "P-", "Q+"],
    ["bare", "Q+", "P-", "bare", "Q-", "P+"],
    ["bare", "P-", "Q+", "bare", "P+", "Q-"],
    ["bare", "Q-", "P+", "bare", "Q+", "P-"],
]
SEQ_OF = {"bare": "A.npz", "P+": "P_plus.npz", "P-": "P_minus.npz", "Q+": "Q_plus.npz", "Q-": "Q_minus.npz"}


def test_order_is_the_frozen_design_table():
    runs = R.expected_runs()
    assert len(runs) == 25 and len({r["run_id"] for r in runs}) == 25
    a0 = runs[0]
    assert (a0["run_id"], a0["block"], a0["condition"], a0["sequence"], a0["use_probe"]) == ("A0", 0, "A0", "A.npz", False)
    for b, row in enumerate(DESIGN_TABLE, 1):
        got = runs[1 + 6 * (b - 1): 1 + 6 * b]
        assert [r["condition"] for r in got] == row
        assert all(r["block"] == b for r in got)
        assert [r["sequence"] for r in got] == [SEQ_OF[c] for c in row]
        assert [r["use_probe"] for r in got] == [c != "bare" for c in row]
        assert [r["sign"] for r in got] == [0 if c == "bare" else (1 if c.endswith("+") else -1) for c in row]


def test_order_balances_positions_by_sequence_number():
    runs = R.expected_runs()
    pos = {}
    for i, r in enumerate(runs, 1):
        pos.setdefault(r["condition"], []).append(i)
    for c in ("P+", "P-", "Q+", "Q-"):
        assert len(pos[c]) == 4 and np.mean(pos[c]) == 14.0
    assert len(pos["bare"]) == 8 and np.mean(pos["bare"]) == 12.5


# ── prepare ───────────────────────────────────────────────────────────────────
def test_prepare_copies_inputs_with_pinned_hashes_and_writes_manifest(env):
    m = R.prepare()
    for name, digest in R.FROZEN_INPUTS.items():
        assert _sha(R.OUT / name) == digest
    disk = json.loads((R.OUT / "manifest.json").read_text())
    assert disk == m
    assert m["runs"] == R.expected_runs() and m["files"] == R.FROZEN_INPUTS
    assert m["budget"] == {"max_attempts": 25, "max_elapsed_s": 3000}
    assert m["nominal_frames"] == T and m["prehold_frames"] == 4 and m["preflight_complete"] is True


def test_prepare_refuses_tampered_source(env):
    with (R.PILOT_SRC / "Q_plus.npz").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        R.prepare()
    assert not (R.OUT / "manifest.json").exists()


def test_prepare_is_idempotent_but_never_overwrites_a_different_manifest(env):
    R.prepare()
    R.prepare()                                                         # 相同内容：允许
    p = R.OUT / "manifest.json"
    m = json.loads(p.read_text())
    m["budget"]["max_attempts"] = 26
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        R.prepare()
    assert json.loads(p.read_text())["budget"]["max_attempts"] == 26


# ── load_frozen ───────────────────────────────────────────────────────────────
def test_load_frozen_accepts_consistent_release(env):
    R.prepare()
    write_release(env)
    release, m = R.load_frozen()
    assert release["run_set_id"] == R.RUN_SET_ID and len(m["runs"]) == 25


def test_load_frozen_rejects_changed_artifact_bytes(env):
    R.prepare()
    write_release(env)
    with (R.OUT / "P_minus.npz").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="冻结文件被改动"):
        R.load_frozen()


def test_load_frozen_rejects_changed_code(env):
    R.prepare()
    write_release(env, code_sha256={str(Path(__file__).resolve()): "0" * 64})
    with pytest.raises(ValueError, match="代码"):
        R.load_frozen()


def test_load_frozen_rejects_changed_manifest(env):
    R.prepare()
    write_release(env)
    p = R.OUT / "manifest.json"
    m = json.loads(p.read_text())
    m["note"] = "edited"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="manifest"):
        R.load_frozen()


def test_motion_needs_explicit_authorization_but_offline_check_does_not(env):
    R.prepare()
    write_release(env, motion_authorized=False, waive_missing_rate_limit_only=False)
    with pytest.raises(ValueError, match="运动未授权"):
        R.load_frozen(require_motion=True)
    release, _ = R.load_frozen(require_motion=False)
    assert release["motion_authorized"] is False


def test_escaping_artifact_name_rejected_even_if_pinned(env, monkeypatch):
    R.prepare()
    pins = dict(R.FROZEN_INPUTS, **{"../outside.npz": "0" * 64})
    monkeypatch.setattr(R, "FROZEN_INPUTS", pins)
    p = R.OUT / "manifest.json"
    m = json.loads(p.read_text())
    m["files"] = pins
    p.write_text(json.dumps(m))
    write_release(env)
    with pytest.raises(ValueError, match="普通文件名"):
        R.load_frozen()


@pytest.mark.parametrize("kw,why", [(dict(minus_scale=-0.9), "不成对"), (dict(bare_probe=True), "裸序列含探针"),
                                    (dict(j4=True), "不合法"), (dict(peak=0.011), "不合法")])
def test_invalid_sequences_rejected(env, monkeypatch, kw, why):
    pins = make_inputs(R.PILOT_SRC, **kw)
    monkeypatch.setattr(R, "FROZEN_INPUTS", pins)
    with pytest.raises(ValueError, match=why):
        R.prepare()


# ── 真实冻结输入与指纹清单（只读）────────────────────────────────────────────
@pytest.mark.skipif(not (PILOT / "manifest.json").exists(), reason="pilot 原始输入不在本机")
def test_real_pilot_inputs_match_the_pins_and_pass_validation(tmp_path, monkeypatch):
    for name, digest in R.FROZEN_INPUTS.items():
        assert _sha(PILOT / name) == digest, name
    pilot_manifest = json.loads((PILOT / "manifest.json").read_text())
    assert {k: v for k, v in pilot_manifest["files"].items()} == R.FROZEN_INPUTS
    monkeypatch.setattr(R, "OUT", tmp_path / "out")
    m = R.prepare()                                                     # 复制到临时目录并逐项核验
    assert m["nominal_frames"] == 587
