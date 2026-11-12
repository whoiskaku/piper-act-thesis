"""B4 部署包装（`b4_deploy.B4Deploy`）与实发补正（`b4_online.apply_correction`）的测试。方案 `reports/PLAN_b4.md`。
小 ACT 随机初始化、建在 CPU 上；`ChunkReplayer` 是按顺序返回给定整段预测的假策略（TE 用 ACT 自带的）。不连机械臂。"""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import lerobot.common.policies.act.b4_online as O
from lerobot.common.policies.act.b4_deploy import B4Deploy, build_b4_policy
from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from tests.test_act_cuda_graph import IMG, tiny_config

MODEL = Path(__file__).resolve().parents[1] / "sim/piper_model.xml"
_M = mujoco.MjModel.from_xml_path(str(MODEL))
CFG = O.OnlineCfg(center=(9.0, 9.0, 9.0), jnt_lo=tuple(_M.jnt_range[:6, 0]), jnt_hi=tuple(_M.jnt_range[:6, 1]),
                  model_path=str(MODEL))
TCP = O.Tcp(str(MODEL))


def _chunks(n=60, c=100):
    """每帧一段预测（c 帧）：关节在限位中段摆动，夹爪第 30 帧合拢、第 45 帧张开；各段加 1e-4 的噪声（部分融合 ≠ 最终值）。"""
    lo, hi = np.asarray(CFG.jnt_lo), np.asarray(CFG.jnt_hi)
    t = np.arange(n + c) * O.DT
    x = np.zeros((n + c, 7))
    x[:, :6] = 0.5 * (lo + hi) + 0.05 * (hi - lo) * np.sin(t[:, None] * np.arange(1, 7))
    x[:, 6] = np.where((np.arange(n + c) >= 30) & (np.arange(n + c) < 45), 0.0, 0.06)
    rng = np.random.default_rng(0)
    return [(x[k:k + c] + 1e-4 * rng.standard_normal((c, 7))).astype(np.float32) for k in range(n)]


CHUNKS = _chunks()


class ChunkReplayer:
    """假的 ACT：predict_chunk 按顺序返回给定的整段预测；TE 是 ACT 自带的 ACTTemporalEnsembler。"""

    def __init__(self, chunks):
        self.chunks, self.i = chunks, 0
        self.config = SimpleNamespace(temporal_ensemble_coeff=0.01)
        self.temporal_ensembler = ACTTemporalEnsembler(0.01, chunks[0].shape[0])

    def predict_chunk(self, batch):
        self.i += 1
        return torch.from_numpy(self.chunks[self.i - 1][None])

    def reset(self):
        self.temporal_ensembler.reset()

    def eval(self):
        return self


class Oracle:
    """「照抄教师」的假网络：第 i 次调用返回教师补正的第 i 帧。"""

    def __init__(self, d):
        self.d, self.i = d, 0

    def __call__(self, x):
        self.i += 1
        return self.d[self.i - 1]


def run(dep, n, path):
    for _ in range(n):
        dep.select_action({})
    dep.save_episode_log(path)
    return dict(np.load(path))


def tiny_act():
    torch.manual_seed(0)
    stats = {"observation.state": {"mean": torch.randn(7), "std": torch.rand(7) + 0.5},
             "observation.images.two": {"mean": torch.rand(3, 1, 1), "std": torch.rand(3, 1, 1) + 0.5},
             "action": {"mean": torch.randn(7), "std": torch.rand(7) + 0.5}}
    return ACTPolicy(tiny_config(n_action_steps=1, temporal_ensemble_coeff=0.01), dataset_stats=stats).eval()


def test_raw_mode_equals_act_select_action():
    """raw 模式发出的就是 ACT（开 TE）自己的输出，逐位相同；其余属性透传给 ACT。"""
    a, b = tiny_act(), tiny_act()
    dep = B4Deploy(a, "raw", CFG, "x")
    g = torch.Generator().manual_seed(1)
    for _ in range(10):
        batch = {"observation.state": torch.randn(1, 7, generator=g), "observation.images.two": torch.rand(1, *IMG, generator=g)}
        assert torch.equal(dep.select_action(batch), b.select_action(batch))
    assert dep.config is a.config


def test_log_reproduces_online_quantities_offline(tmp_path):
    """约束 2：每集记下的整段预测足以离线复算当时的动作、Shift、窗和网络输入；shift 模式发出的就是在线 Shift。"""
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x"), 60, tmp_path / "a.npz")
    a_pi, ens = O.te_states(log["chunk"])
    assert np.allclose(a_pi, log["a_pi"], atol=1e-6)
    for k in range(60):
        assert np.allclose(O.online_shift(ens[k]), log["shift"][k], atol=1e-6)
        assert np.array_equal(O.online_window(log["a_pi"], ens[k], k, CFG, TCP), log["w"][k])
        assert np.allclose(O.online_input(log["a_pi"], ens[k], k, 8, 8), log["x"][k], atol=1e-6)
    assert np.array_equal(log["u"], log["shift"].astype(np.float32))
    assert str(log["mode"]) == "shift" and str(log["cfg_sha"]) == "x" and bool(log["complete"])


def test_net_mode_does_not_multiply_by_window(tmp_path):
    """约束 1：窗的过渡段（0 < w < 1）里，网络给多少就发多少；w = 0 的帧一律不发。"""
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=lambda x: np.full(5, 0.01)), 60, tmp_path / "a.npz")
    taper = (log["w"] > 0) & (log["w"] < 1)
    assert taper.any() and (log["w"] == 0).any()
    assert np.allclose(log["delta"][taper], 0.01)
    assert not log["delta"][log["w"] == 0].any()


def test_net_mode_clips_to_15_mrad(tmp_path):
    """网络输出超过 15 mrad 时只发 15 mrad。"""
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=lambda x: np.full(5, 0.05)), 60, tmp_path / "a.npz")
    assert np.allclose(log["delta"][log["w"] > 0], O.CLIP)


def test_net_mode_oracle_reproduces_teacher_cmd7_bitwise(tmp_path):
    """约束 1：教师补正本身已含窗；网络照抄它时，部署发出的指令与教师的 cmd7（同 ilc_route1.write_seq 的 float32 加法）逐位相同。"""
    base = run(B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x"), 60, tmp_path / "a.npz")
    d_star = base["w"] * 0.005 * np.sin(np.arange(60)[:, None] * np.arange(1, 6) * 0.3)
    cmd7 = base["u"].copy()
    cmd7[:, list(O.ACTIVE)] += d_star.astype(np.float32)
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=Oracle(d_star)), 60, tmp_path / "b.npz")
    assert np.array_equal(log["u"], cmd7)


def test_reset_clears_ensembler_and_log(tmp_path):
    """Review Focus 1：新的一集从头开始：记录清空，TE 也清空（第一帧动作 = 第一段预测的第 0 行）。"""
    dep = B4Deploy(ChunkReplayer(CHUNKS + CHUNKS), "shift", CFG, "x")
    run(dep, 60, tmp_path / "a.npz")
    dep.reset()
    log = run(dep, 5, tmp_path / "b.npz")
    assert len(log["a_pi"]) == 5 and np.allclose(log["a_pi"][0], CHUNKS[0][0], atol=1e-7)


def test_build_reads_cfg_and_rejects_unknown_mode(tmp_path):
    """build_b4_policy 读在线配置并记下它的哈希；模式写错就报错。"""
    p = tmp_path / "cfg.json"
    p.write_text('{"center": [9, 9, 9], "jnt_lo": %s, "jnt_hi": %s, "model_path": "%s"}'
                 % (list(CFG.jnt_lo), list(CFG.jnt_hi), MODEL))
    dep = build_b4_policy(ChunkReplayer(CHUNKS), "shift", str(p))
    assert dep.cfg == O.load_cfg(p) and dep.cfg_sha == O.cfg_sha(p)
    with pytest.raises(ValueError):
        build_b4_policy(ChunkReplayer(CHUNKS), "shfit", str(p))


def test_net_mode_inputs_use_frozen_P_F(tmp_path):
    """约束 2：net 模式按冻结网络的 P、F 算输入，与离线复算一致；网络输出 0 时发的就是 Shift。"""
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=lambda x: np.zeros(5), P=4, F=4), 30, tmp_path / "a.npz")
    a_pi, ens = O.te_states(log["chunk"])
    assert log["x"].shape[1] == 56 and int(log["P"]) == 4
    assert all(np.allclose(log["x"][k], O.online_input(a_pi, ens[k], k, 4, 4), atol=1e-6) for k in range(30))
    assert np.array_equal(log["u"], log["shift"].astype(np.float32))


def test_build_net_policy_loads_frozen(tmp_path, monkeypatch):
    """build_b4_policy 的 net 模式：读冻结文件，用其中的 P、F 与网络；冻结文件记下的哈希写进每集记录。"""
    calls = []
    monkeypatch.setattr(O, "load_frozen", lambda p: (lambda x: calls.append(len(x)) or np.zeros(5), "abc", 4, 4))
    p = tmp_path / "cfg.json"
    p.write_text('{"center": [9, 9, 9], "jnt_lo": %s, "jnt_hi": %s, "model_path": "%s"}'
                 % (list(CFG.jnt_lo), list(CFG.jnt_hi), MODEL))
    dep = build_b4_policy(ChunkReplayer(CHUNKS), "net", str(p), net_path="frozen.json")
    log = run(dep, 3, tmp_path / "a.npz")
    assert calls == [56, 56, 56] and str(log["net_sha"]) == "abc"


def test_non_finite_network_output_sends_no_correction(tmp_path):
    """审阅（升为 Important）：网络输出 NaN / inf 时这一帧不发补正，不能把 NaN 发给机械臂。"""
    out = iter([np.full(5, np.nan), np.array([np.inf, 0, 0, 0, 0])] + [np.zeros(5)] * 10)
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=lambda x: next(out)), 3, tmp_path / "a.npz")
    assert np.all(np.isfinite(log["u"])) and not log["delta"][:2].any()


def test_rerecorded_episode_keeps_the_earlier_attempt(tmp_path):
    """审阅 Important 3：同一集号再存一次（中断后补录、按 ← 重录）时，之前那次的记录改名留档，不被覆盖。"""
    p = tmp_path / "episode_000003_b4.npz"
    dep = B4Deploy(ChunkReplayer(CHUNKS + CHUNKS), "shift", CFG, "x")
    for _ in range(4):
        dep.select_action({})
    dep.save_episode_log(p, error="KeyboardInterrupt: ")
    dep.reset()
    for _ in range(5):
        dep.select_action({})
    dep.save_episode_log(p)
    old = np.load(tmp_path / "episode_000003_b4.try1.npz")
    assert not bool(old["complete"]) and len(old["u"]) == 4 and len(np.load(p)["u"]) == 5
