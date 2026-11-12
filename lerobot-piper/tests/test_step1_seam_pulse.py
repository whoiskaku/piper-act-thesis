"""Step 1 v2 接缝脉冲的契约测试 —— 逐条核对 PREREG_STEP1_V2.md 声称的性质。

按文件路径加载, 不 import lerobot 顶层包(因此不需要 draccus/torch/MuJoCo)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# step1_seam_pulse 只依赖 step1_correction_runtime 的 command_contact_mask,
# 后者是纯 NumPy —— 先把它塞进 sys.modules 以避开 lerobot 包的 __init__。
_rt = _load("lerobot.common.policies.act.step1_correction_runtime",
            "lerobot/common/policies/act/step1_correction_runtime.py")
sp = _load("_seam_pulse_test", "lerobot/common/policies/act/step1_seam_pulse.py")


def _chunk(n: int, grip: float = 0.0, j: np.ndarray | None = None) -> np.ndarray:
    c = np.zeros((n, 7), dtype=np.float64)
    if j is not None:
        c[:, :6] = j
    c[:, 6] = grip
    return c


def test_theta_roundtrip_and_bounds():
    th = sp.SeamPulseTheta(a_prox=0.0015, a_wrist=-0.002, tau=8.0)
    np.testing.assert_allclose(
        sp.SeamPulseTheta.from_x(th.to_x()).to_x(), th.to_x(), atol=1e-12)
    # 归一化坐标的端点正好映射到冻结范围的端点
    assert sp.SeamPulseTheta.from_x([1, 1, 1]).tau == pytest.approx(sp.TAU_MAX)
    assert sp.SeamPulseTheta.from_x([-1, -1, -1]).tau == pytest.approx(sp.TAU_MIN)
    assert sp.SeamPulseTheta.from_x([1, -1, 0]).a_prox == pytest.approx(sp.A_LIMIT)
    for bad in (dict(a_prox=0.003, a_wrist=0.0, tau=6.0),
                dict(a_prox=0.0, a_wrist=0.0, tau=1.0),
                dict(a_prox=0.0, a_wrist=0.0, tau=11.0),
                dict(a_prox=float("nan"), a_wrist=0.0, tau=6.0)):
        with pytest.raises(sp.SeamPulseError):
            sp.SeamPulseTheta(**bad)


def test_from_x_clips_instead_of_raising():
    """SPSA 可能把 x 推出界; from_x 必须夹住而不是抛异常。"""
    th = sp.SeamPulseTheta.from_x([5.0, -5.0, 5.0])
    assert th.a_prox == pytest.approx(sp.A_LIMIT)
    assert th.a_wrist == pytest.approx(-sp.A_LIMIT)
    assert th.tau == pytest.approx(sp.TAU_MAX)


def test_first_chunk_gets_no_correction():
    """预注册 §2.5: 首个 chunk 无 q_prev,last ⇒ δa 恒零。"""
    th = sp.SeamPulseTheta(a_prox=0.002, a_wrist=0.002, tau=6.0)
    da, _, _ = sp.seam_pulse_chunk(th, None, _chunk(40))
    assert np.all(da == 0.0)


def test_direction_comes_from_the_seam_jump_not_a_fixed_vector():
    """符号翻转的接缝 ⇒ δa 也翻转; 这是 v_G 设计的核心性质。"""
    th = sp.SeamPulseTheta(a_prox=0.001, a_wrist=0.001, tau=6.0)
    prev = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.20])
    nxt = _chunk(30)
    da_pos, _, _ = sp.seam_pulse_chunk(th, prev, nxt)
    da_neg, _, _ = sp.seam_pulse_chunk(th, -prev, nxt)
    np.testing.assert_allclose(da_pos, -da_neg, atol=1e-15)
    assert da_pos[0, 0] != 0.0 and da_pos[0, 5] != 0.0


def test_infinity_norm_normalisation_bounds_delta_by_A():
    """结构性安全: ‖v_G‖∞ ≤ 1 且 |A|≤2mrad ⇒ |δa| ≤ 2mrad, 裁剪永不触发。"""
    rng = np.random.default_rng(0)
    th = sp.SeamPulseTheta(a_prox=sp.A_LIMIT, a_wrist=-sp.A_LIMIT, tau=sp.TAU_MAX)
    worst = 0.0
    for _ in range(400):
        prev = rng.normal(0.0, 0.3, 6)          # 跳变量级远超 2 mrad
        da, _, _ = sp.seam_pulse_chunk(th, prev, _chunk(60))
        worst = max(worst, float(np.abs(da).max()))
    assert worst <= sp.DA_LIMIT + 1e-12
    # 且确实用满了界(说明不是因为恒零才没触界)
    assert worst == pytest.approx(sp.DA_LIMIT, rel=1e-9)


def test_tiny_jump_is_not_amplified_by_the_floor():
    """‖d_G‖∞ < 2mrad 时分母取下限 ⇒ ‖v_G‖∞ < 1, 小跳变不被放大成满幅。"""
    th = sp.SeamPulseTheta(a_prox=sp.A_LIMIT, a_wrist=0.0, tau=6.0)
    prev = np.array([2e-5, 0, 0, 0, 0, 0])       # 0.02 mrad, 远小于 2 mrad
    da, _, _ = sp.seam_pulse_chunk(th, prev, _chunk(20))
    # d=0.02mrad, 分母取下限 2mrad ⇒ v=0.01 ⇒ δa = 0.01×A_LIMIT
    assert abs(da[0, 0]) == pytest.approx(0.01 * sp.A_LIMIT, rel=1e-9)


def test_decay_follows_exp_minus_k_over_tau():
    th = sp.SeamPulseTheta(a_prox=0.001, a_wrist=0.0, tau=5.0)
    prev = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    da, _, _ = sp.seam_pulse_chunk(th, prev, _chunk(30))
    k = np.arange(30)
    np.testing.assert_allclose(da[:, 0], da[0, 0] * np.exp(-k / 5.0), rtol=1e-9)


def test_gripper_channel_is_never_touched():
    th = sp.SeamPulseTheta(a_prox=0.002, a_wrist=0.002, tau=6.0)
    da, _, _ = sp.seam_pulse_chunk(th, np.full(6, 0.1), _chunk(25, grip=0.02))
    assert da.shape[1] == 6            # 只有 6 个臂关节, 夹爪不在输出里


def test_contact_window_zeroes_the_pulse():
    th = sp.SeamPulseTheta(a_prox=0.002, a_wrist=0.0, tau=10.0)
    grip = np.zeros(60); grip[30:] = sp.GRIP_FULL          # 第 30 帧夹爪跃变
    c = _chunk(60); c[:, 6] = grip
    da, contact, _ = sp.seam_pulse_chunk(th, np.full(6, 0.1), c)
    assert contact[20:41].all()                            # ±10 帧全覆盖
    assert np.all(da[contact] == 0.0)
    assert np.any(da[~contact] != 0.0)


def test_runtime_resets_between_episodes():
    """每集清零 ⇒ 新一集的第一个 chunk 又变成"首 chunk", 不补正。"""
    th = sp.SeamPulseTheta(a_prox=0.002, a_wrist=0.002, tau=6.0)
    r = sp.SeamPulseRuntime(th, verbose=False)
    r.set_chunk(_chunk(10))                                # 首 chunk
    first = np.array([r.step() for _ in range(10)])
    assert np.all(first == 0.0)
    r.set_chunk(_chunk(10, j=np.full(6, 0.05)))            # 第二个 chunk 有跳变
    second = np.array([r.step() for _ in range(10)])
    assert np.any(second != 0.0)

    r.reset()
    r.set_chunk(_chunk(10, j=np.full(6, 0.05)))            # 新一集的首 chunk
    again = np.array([r.step() for _ in range(10)])
    assert np.all(again == 0.0)
    assert r.summary()["seam_chunks" if False else "chunks"] == 1


def test_runtime_summary_reports_zero_saturation():
    """预注册声称 sat_pct 应恒为 0(结构保证), 用满幅 θ 也要成立。"""
    th = sp.SeamPulseTheta(a_prox=sp.A_LIMIT, a_wrist=-sp.A_LIMIT, tau=sp.TAU_MAX)
    r = sp.SeamPulseRuntime(th, verbose=False)
    rng = np.random.default_rng(1)
    for i in range(6):
        r.set_chunk(_chunk(100, j=rng.normal(0, 0.3, 6)))
        for _ in range(100):
            r.step()
    s = r.summary()
    assert s["sat_pct"] == 0.0
    assert s["frames"] == 600
    assert s["first_chunk_skipped"] == 100


def test_theta_file_roundtrip(tmp_path: Path):
    th = sp.SeamPulseTheta(a_prox=-0.0012, a_wrist=0.0007, tau=3.5)
    f = tmp_path / "theta.json"
    th.save(f, round=7, note="test")
    back = sp.SeamPulseTheta.load(f)
    assert back.to_dict() == pytest.approx(th.to_dict())
    assert json.loads(f.read_text())["round"] == 7
    with pytest.raises(sp.SeamPulseError):
        (tmp_path / "bad.json").write_text('{"theta": {"a_prox": 0.0}}')
        sp.SeamPulseTheta.load(tmp_path / "bad.json")
