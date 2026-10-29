"""`b4_net.py` 的测试（方案 `reports/PLAN_b4.md` §5.3；实现计划任务 8、9）：标签 = 教师实发的补正（约束 1），
输入 = 当时在线可得的量（约束 2），测试集与验收没通过的教师不进数据。合成数据，不连机械臂。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_net as N  # noqa: E402
import b4_teacher as B  # noqa: E402
import repeat_reversal as RR  # noqa: E402

from lerobot.common.policies.act.b4_deploy import B4Deploy  # noqa: E402
from tests.test_b4_deploy import CFG, CHUNKS, MODEL, ChunkReplayer  # noqa: E402

HOLD, ACT = RR.HOLD, [0, 1, 2, 4, 5]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    dep = B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x")
    for _ in range(len(CHUNKS)):
        dep.select_action({})
    p = tmp_path_factory.mktemp("b4") / "ep_b4.npz"
    dep.save_episode_log(p)
    log = dict(np.load(p))
    seq = B.build_seq(log, log["u"], CFG, mujoco.MjModel.from_xml_path(str(MODEL)))
    delta = seq["w"] * 0.003 * np.sin(np.arange(len(seq["w"]))[:, None] * 0.2 + np.arange(5))   # 像教师那样已含窗
    cmd7 = seq["s7"].copy()
    cmd7[:, ACT] += delta.astype(np.float32)
    return SimpleNamespace(log=log, seq=seq, teacher={"cmd7": cmd7, "delta": delta})


def test_label_is_teacher_sent_delta(case):
    """约束 1：标签就是教师实发的补正（实发指令 − 冻结的在线 Shift），按录制帧下标；窗 = 0 处标签为 0。"""
    d = N.assemble(case.log, case.seq, case.teacher, 8, 8)
    sent = case.teacher["cmd7"][HOLD:, ACT].astype(float) - case.seq["s7"][HOLD:, ACT].astype(float)
    assert np.array_equal(d["Y"], sent) and np.array_equal(d["Wt"], case.log["w"])
    assert d["Y"].any() and not d["Y"][d["Wt"] == 0].any()


def test_inputs_equal_logged_online_inputs(case):
    """约束 2：由整段预测复算的网络输入 = 控制回路当时记下的输入；换窗口长度时照样能复算。"""
    assert np.allclose(N.assemble(case.log, case.seq, case.teacher, 8, 8)["X"], case.log["x"], atol=1e-6)
    assert N.assemble(case.log, case.seq, case.teacher, 4, 16)["X"].shape == (len(case.log["x"]), 140)


def test_rejects_inputs_that_differ_from_log(case):
    log = {**case.log, "x": case.log["x"] + np.float32(1e-3)}
    with pytest.raises(SystemExit, match="网络输入"):
        N.assemble(log, case.seq, case.teacher, 8, 8)


def test_build_excludes_test_split_and_failed_teachers(case, tmp_path):
    rows = {"a": {"split": "train", "passed": True}, "b": {"split": "val", "passed": True},
            "c": {"split": "test", "passed": True}, "d": {"split": "train", "passed": False},
            "e": {"split": "train", "passed": None}}
    p = N.build(8, 8, rows=rows, loader=lambda tid: (case.log, case.seq, case.teacher), out=tmp_path)
    z = np.load(p)
    assert sorted(set(z["traj_id"])) == ["a", "b"] and set(z["split"]) == {"train", "val"}
    assert len(z["X"]) == 2 * len(case.log["x"]) and str(z["data_sha"])


# ── 任务 9：网络、训练、冻结 ───────────────────────────────────────────────
import json  # noqa: E402

import torch  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402


def make_data(path, n=64, P=4, F=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2 * n, (P + F) * 7)).astype(np.float32)
    Y = 0.003 * np.tanh(X[:, :5] + 0.5 * X[:, 5:10])
    np.savez(path, X=X, Y=Y, Wt=np.ones_like(Y), traj_id=np.array(["a"] * n + ["b"] * n),
             split=np.array(["train"] * n + ["val"] * n), data_sha="x", P=P, F=F)
    return path


def test_weighted_mse_ignores_zero_weight_frames():
    y = torch.zeros(4, 5)
    p = y.clone()
    p[0] = 9.0
    w = torch.ones(4, 5)
    w[0] = 0
    assert N.weighted_mse(p, y, w).item() == 0.0


def test_overfits_tiny_dataset(tmp_path):
    d = make_data(tmp_path / "d.npz", n=32)
    r = N.train(d, seed=0, hidden=64, epochs=3000, patience=3000, out=tmp_path, val_split="train")
    assert r["train_loss"] < 0.01 * float(np.var(np.load(d)["Y"]))


def test_same_seed_same_weights(tmp_path):
    d = make_data(tmp_path / "d.npz")
    a = torch.load(N.train(d, 0, 64, epochs=5, out=tmp_path / "a")["path"])["state_dict"]
    b = torch.load(N.train(d, 0, 64, epochs=5, out=tmp_path / "b")["path"])["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_one_step_changes_params_with_nonzero_grad():
    m = O.B4Net(56, 16)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x, y, w = torch.randn(8, 56), 0.003 * torch.randn(8, 5), torch.ones(8, 5)
    N.weighted_mse(m(x), y, w).backward()
    assert any(p.grad.abs().sum() > 0 for p in m.parameters())
    before = [p.detach().clone() for p in m.parameters()]
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_freeze_once_and_load_frozen_checks_hash(tmp_path):
    d = make_data(tmp_path / "d.npz")
    run = N.train(d, 0, 64, epochs=5, out=tmp_path)["path"]
    fz = N.freeze(Path(run), out=tmp_path)
    fn, sha, P, F = O.load_frozen(str(fz))
    m = torch.load(run)
    net = O.B4Net(m["n_in"], m["hidden"])
    net.load_state_dict(m["state_dict"])
    x = np.load(d)["X"][0]
    assert (P, F) == (4, 4) and np.allclose(fn(x), net(torch.from_numpy(x)[None])[0].detach().numpy())
    with pytest.raises(SystemExit):
        N.freeze(Path(run), out=tmp_path)                              # 只冻结一次
    Path(json.loads(fz.read_text())["weights"]).write_bytes(Path(run).read_bytes() + b"x")
    with pytest.raises(SystemExit):
        O.load_frozen(str(fz))                                         # 权重被改过


def test_saved_weights_are_the_best_validation_epoch(tmp_path):
    """早停：存下的是验证损失最低的那一轮（训练和验证分布不同、练久了会过拟合，最后一轮不是最好的）。"""
    d = make_data(tmp_path / "d.npz", n=32)
    z = dict(np.load(d))
    z["Y"][32:] = -z["Y"][32:]                                          # 验证集的规律和训练集相反：越练越差
    np.savez(d, **z)
    r = N.train(d, 0, 64, epochs=200, patience=200, out=tmp_path)
    m = torch.load(r["path"])
    net = O.B4Net(m["n_in"], m["hidden"])
    net.load_state_dict(m["state_dict"])
    with torch.no_grad():
        vl = N.weighted_mse(net(torch.from_numpy(z["X"][32:])), torch.from_numpy(z["Y"][32:].astype(np.float32)),
                            torch.ones(32, 5)).item()
    assert vl == pytest.approx(r["val_loss"], rel=1e-5)


def test_build_counts_passed_spare_teachers_as_training(case, tmp_path):
    """审阅 Important 1：备用位置是用来补训练集的；验收通过的备用教师要进训练集（标成 train）。"""
    rows = {"a": {"split": "train", "passed": True}, "s": {"split": "spare", "passed": True},
            "t": {"split": "spare", "passed": False}}
    z = np.load(N.build(8, 8, rows=rows, loader=lambda tid: (case.log, case.seq, case.teacher), out=tmp_path))
    assert sorted(set(z["traj_id"])) == ["a", "s"] and set(z["split"]) == {"train"}


def test_select_rebuilds_every_candidate_dataset(monkeypatch, tmp_path):
    """审阅 Important 4：选模型前三份候选数据都重建，免得新加教师后各候选用的数据不一样。"""
    built = []
    monkeypatch.setattr(N, "NET", tmp_path)
    monkeypatch.setattr(N, "build", lambda P, F, **k: built.append((P, F)) or make_data(tmp_path / f"data_P{P}_F{F}.npz", P=P, F=F))
    monkeypatch.setattr(N, "train", lambda d, s, h, **k: {"path": str(d), "train_loss": 1.0, "val_loss": 1.0 + h + s, "val_r2": [0.0] * 5})
    (tmp_path / "data_P4_F4.npz").write_text("旧的")
    N.select()
    assert sorted(built) == sorted(N.CANDIDATES)
