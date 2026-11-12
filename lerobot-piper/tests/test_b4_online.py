"""`b4_online`（B4 的在线量：TE 重建、在线 Shift、在线窗、网络输入）的测试。方案 `reports/PLAN_b4.md`，实现计划 `reports/PLAN_IMPL_B4.md`。
不连机械臂、不读数据集。"""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import lerobot.common.policies.act.b4_online as O
from lerobot.common.policies.act.modeling_act import ACTTemporalEnsembler

MODEL = Path(__file__).resolve().parents[1] / "sim/piper_model.xml"


@pytest.fixture(scope="module")
def ep():
    """合成一集（120 帧）：关节在限位中段小幅摆动；夹爪第 60 帧合拢、第 80 帧张开；预测 = 最终值。"""
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
    t = np.arange(120) * O.DT
    a = np.zeros((120, 7))
    a[:, :6] = 0.5 * (lo + hi) + 0.05 * (hi - lo) * np.sin(t[:, None] * np.arange(1, 7))
    a[:, 6] = np.where((np.arange(120) >= 60) & (np.arange(120) < 80), 0.0, 0.06)
    cfg = O.OnlineCfg(center=(9.0, 9.0, 9.0), jnt_lo=tuple(lo), jnt_hi=tuple(hi), model_path=str(MODEL))
    return SimpleNamespace(a=a, ens=[a[k + 1:k + 100] for k in range(120)], cfg=cfg, tcp=O.Tcp(str(MODEL)))


def test_te_states_matches_torch_ensembler():
    """离线复算的 TE 与 ACT 自带的 ACTTemporalEnsembler 逐帧相同：当前帧动作与之后各帧的部分融合值。"""
    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(40, 100, 7)).astype(np.float32)
    te, acts, ens = ACTTemporalEnsembler(0.01, 100), [], []
    for c in chunks:
        acts.append(te.update(torch.from_numpy(c[None]))[0].numpy())
        ens.append(te.ensembled_actions[0].numpy().copy())
    a_pi, e = O.te_states(chunks, 0.01)
    assert np.allclose(a_pi, acts, atol=1e-6)
    assert all(np.allclose(e[k], ens[k], atol=1e-6) for k in range(40))


def test_online_shift_is_0p8_0p2():
    """提前 60 ms = 1.2 帧：第 k+1 帧（ens_k[0]）占 0.8，第 k+2 帧占 0.2。"""
    ens_k = np.arange(21, dtype=float).reshape(3, 7)
    assert np.allclose(O.online_shift(ens_k), 0.8 * ens_k[0] + 0.2 * ens_k[1])


def test_online_shift_equals_offline_shift_when_all_chunks_agree():
    """每段预测都是同一条真实轨迹时，部分融合 = 最终值，在线 Shift 应等于回放用的离线 Shift（np.interp 提前 60 ms）。"""
    t = np.arange(160) * O.DT
    x = np.stack([np.sin(t * (j + 1)) for j in range(7)], 1).astype(np.float32)
    _, ens = O.te_states(np.stack([x[k:k + 100] for k in range(60)]))
    for k in range(59):
        off = [np.interp(t[k] + 0.060, t, x[:, j]) for j in range(7)]   # 同 shift_replay.shifted
        assert np.allclose(O.online_shift(ens[k]), off, atol=1e-6)


def test_online_window_ignores_information_after_k(ep):
    """约束 2：第 k 帧的窗只看第 k 帧当时已知的指令。k = 40 在第 60 帧合拢前的过渡段里（要靠预测的未来才知道）；
    改掉第 k 帧之后的最终指令（夹爪 +0.3 = 一直张开），窗不变。"""
    a, ens, k = ep.a, ep.ens, 40
    w = O.online_window(a, ens[k], k, ep.cfg, ep.tcp)
    assert np.all((w > 0) & (w < 1))
    a2 = a.copy()
    a2[k + 1:] += 0.3
    assert np.array_equal(O.online_window(a2, ens[k], k, ep.cfg, ep.tcp), w)


def test_online_input_layout_and_causality(ep):
    """网络输入 = 过去 P 帧最终指令 + 未来 F 帧部分融合值，展平成 float32；改第 k 帧之后的最终指令，输入不变。"""
    a, ens, k = ep.a, ep.ens, 40
    x = O.online_input(a, ens[k], k, P=3, F=2)
    assert np.array_equal(x[:21].reshape(3, 7), a[k - 2:k + 1].astype(np.float32))
    assert np.array_equal(x[21:].reshape(2, 7), ens[k][:2].astype(np.float32))
    a2 = a.copy()
    a2[k + 1:] += 1
    assert np.array_equal(O.online_input(a2, ens[k], k, 3, 2), x)


def test_short_future_is_padded_not_an_error(ep):
    """Review Focus 3：未来只剩 3 帧时用最后一行补齐，不抛错。"""
    w = O.online_window(ep.a, ep.ens[40][:3], 40, ep.cfg, ep.tcp)
    x = O.online_input(ep.a, ep.ens[40][:3], 40, 8, 8)
    assert w.shape == (5,) and np.all((w >= 0) & (w <= 1)) and x.shape == (112,)
    assert np.array_equal(x[-7:], ep.ens[40][2].astype(np.float32))


def test_saturation_is_per_joint_and_looks_one_frame_ahead(ep):
    """限位饱和逐关节、看下一帧指令：j2 只在第 21 帧越过上限 → 零帧是第 20 帧（前瞻）和第 21 帧；
    第 16 帧离第 20 帧 4 帧 → j2 的窗 = 0.5 − 0.5 cos(0.4π)，其余关节不受影响（= 1）。"""
    a = ep.a.copy()
    a[21, 1] = ep.cfg.jnt_hi[1] + 0.01
    w = O.online_window(a, a[17:116], 16, ep.cfg, ep.tcp)
    assert w[1] == pytest.approx(0.5 - 0.5 * np.cos(0.4 * np.pi))
    assert np.array_equal(w[[0, 2, 3, 4]], np.ones(4))
