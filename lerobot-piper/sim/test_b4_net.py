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


# ── 整段版 ①（方案 reports/PLAN_b4_seq.md；实现计划 reports/PLAN_IMPL_b4_seq.md 任务 1）────────────────
def test_seq_first_frame_recovers_teacher_delta(case):
    """第一帧的标签减去 D（Shift − a_π）就是教师补正 Δ*；输入与一帧版相同。"""
    d = N.assemble_seq(case.log, case.seq, case.teacher, 8, 8, 10)
    sent = case.teacher["cmd7"][HOLD:, ACT].astype(float) - case.seq["s7"][HOLD:, ACT].astype(float)
    assert np.allclose(d["Y"][:, :5] - d["D"], sent, rtol=0, atol=1e-12)    # 只差 float64 舍入
    assert np.allclose(d["X"], case.log["x"], atol=1e-6)


def test_seq_label_column_j_is_frame_k_plus_j(case):
    """按帧优先展平：第 j 组 5 列是第 k+j 帧；参考 + 标签 = 教师实发指令；权重是目标帧（不是第 k 帧）的窗。"""
    d = N.assemble_seq(case.log, case.seq, case.teacher, 8, 8, 10)
    n = len(d["X"])
    Y, W = d["Y"].reshape(n, 10, 5), d["Wt"].reshape(n, 10, 5)
    cmd = case.teacher["cmd7"][HOLD:, ACT].astype(float)
    for k, j in ((0, 0), (2, 7), (n - 10, 9)):                         # 第 2、9 帧的窗不同：分得出用的是哪一帧
        assert np.allclose(Y[k, j] + d["R"][k, j], cmd[k + j], rtol=0, atol=1e-12)
        assert np.array_equal(W[k, j], case.log["w"][k + j])


def test_seq_targets_past_the_end_are_zero(case):
    """Review Focus 1：最后 N−1 帧的目标超出轨迹末尾 → 标签、权重都是 0，不报错。"""
    d = N.assemble_seq(case.log, case.seq, case.teacher, 8, 8, 10)
    n = len(d["X"])
    Y, W = d["Y"].reshape(n, 10, 5), d["Wt"].reshape(n, 10, 5)
    assert not Y[n - 1, 1:].any() and not W[n - 1, 1:].any() and not W[n - 3, 3:].any()


def test_seq_reference_uses_only_chunks_up_to_k(case):
    """约束 2：改第 31 帧及以后的整段预测，第 k < 31 帧的参考和输入都不变（P、F 与录制时不同，走复算）。"""
    chunk = case.log["chunk"].copy()
    chunk[31:] += 0.05
    a, b = (N.assemble_seq({**case.log, "chunk": c}, case.seq, case.teacher, 4, 16, 10)
            for c in (case.log["chunk"], chunk))
    assert np.array_equal(a["R"][:31], b["R"][:31]) and np.array_equal(a["X"][:31], b["X"][:31])
    assert not np.array_equal(a["R"][31:], b["R"][31:])


def test_build_with_N_writes_a_separate_file(case, tmp_path):
    rows = {"a": {"split": "train", "passed": True}, "b": {"split": "val", "passed": True}}
    p = N.build(8, 8, rows=rows, loader=lambda t: (case.log, case.seq, case.teacher), out=tmp_path, N=10)
    z = np.load(p)
    assert p.name == "data_P8_F8_N10.npz" and z["Y"].shape[1] == 50 and z["D"].shape[1] == 5 and int(z["N"]) == 10


# ── 整段版 ①：多帧输出的网络、训练、选种子（实现计划任务 2）──────────────────────────────────────
def make_seq_data(path, n=64, P=4, F=4, N_=3, seed=0):
    """同 make_data，但标签是 N_ 帧 × 5（Y、Wt 为 (2n, 5·N_)），另存 D、N。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2 * n, (P + F) * 7)).astype(np.float32)
    Y = 0.003 * np.tanh(X[:, :5 * N_] + 0.5 * X[:, 5 * N_:10 * N_])
    np.savez(path, X=X, Y=Y, Wt=np.ones_like(Y), D=np.zeros((2 * n, 5)), traj_id=np.array(["a"] * n + ["b"] * n),
             split=np.array(["train"] * n + ["val"] * n), data_sha="x", P=P, F=F, N=N_)
    return path


def test_net_output_size_defaults_to_five():
    assert O.B4Net(56, 16)(torch.zeros(3, 56)).shape == (3, 5)
    assert O.B4Net(56, 16, 50)(torch.zeros(3, 56)).shape == (3, 50)


def test_old_checkpoint_without_n_out_loads_as_five(tmp_path):
    """Review Focus 2：一帧版的旧权重文件没有 n_out 键 → 仍按 5 维加载。"""
    r = N.train(make_data(tmp_path / "d.npz"), 0, 16, epochs=2, out=tmp_path)
    m = torch.load(r["path"])
    m.pop("n_out")
    torch.save(m, r["path"])
    fn, *_ = O.load_frozen(str(N.freeze(Path(r["path"]), out=tmp_path)))
    assert fn(np.zeros(56, np.float32)).shape == (5,)


def test_multi_frame_training_saves_n_out_and_first_frame_loss(tmp_path):
    r = N.train(make_seq_data(tmp_path / "d.npz"), 0, 16, epochs=5, out=tmp_path)
    assert torch.load(r["path"])["n_out"] == 15
    assert 0 < r["val_loss_first"] != r["val_loss"]


def test_overfits_tiny_multi_frame_dataset(tmp_path):
    d = make_seq_data(tmp_path / "d.npz", n=32)
    r = N.train(d, seed=0, hidden=64, epochs=3000, patience=3000, out=tmp_path, val_split="train")
    assert r["train_loss"] < 0.01 * float(np.var(np.load(d)["Y"]))


def test_select_seq_picks_lowest_first_frame_loss(monkeypatch, tmp_path):
    monkeypatch.setattr(N, "NET_SEQ", tmp_path)
    monkeypatch.setattr(N, "build", lambda *a, **k: tmp_path / "d.npz")
    fake = {0: (1.0, 0.3), 1: (0.5, 0.4), 2: (0.9, 0.2)}               # 种子：(N 帧验证损失, 第一帧验证损失)
    monkeypatch.setattr(N, "train", lambda d, s, h, **k: {"path": f"s{s}", "train_loss": 0.0, "val_loss": fake[s][0],
                                                          "val_loss_first": fake[s][1], "val_r2": []})
    assert N.select_seq()["chosen"]["seed"] == 2
    assert json.loads((tmp_path / "selection.json").read_text())["chosen"]["seed"] == 2


# ── 整段版 ①：离线检查与上真机门槛（实现计划任务 3）────────────────────────────────────────────
def test_act_average_weights_the_oldest_segment_most():
    C = np.zeros((6, 3, 5))
    C[:, 0], C[:, 1], C[:, 2] = 1.0, 2.0, 3.0                             # 第 k 段对第 k+j 帧给 j+1
    w = np.exp(-0.01 * np.arange(3))                                      # 最旧的一段（i = 2，值 3）权重最大
    out = N.act_average(C)
    assert np.allclose(out[5], (w[0] * 3 + w[1] * 2 + w[2] * 1) / w.sum())
    assert np.allclose(out[1], (w[0] * 2 + w[1] * 1) / w[:2].sum()) and np.allclose(out[0], 1.0)


def test_act_average_of_consistent_segments_is_exact():
    cmd = np.random.default_rng(0).normal(size=(8, 5))
    C = np.stack([cmd[np.minimum(np.arange(k, k + 4), 7)] for k in range(8)])
    assert np.allclose(N.act_average(C), cmd)


def test_gate_is_per_joint_against_the_exact_baseline():
    """用户 11-06：逐关节 R² ≥ 一帧版 − 0.10、限幅触发率 ≤ 1%；只看平均会放过腕部变差。"""
    base = np.array([0.5581268668174744, 0.7319753766059875, 0.6685474216938019,
                     0.20204311609268188, 0.35309410095214844])
    assert N.gate(base - 0.10, np.full(5, 0.01), base) == (True, [])
    r2 = base.copy()
    r2[3] -= 0.1001                                                       # 平均几乎不变，只有 j5 不过
    ok, why = N.gate(r2, np.zeros(5), base)
    assert not ok and len(why) == 1 and "j5" in why[0]
    ok, why = N.gate(base, np.array([0, 0, 0, 0, 0.0101]), base)
    assert not ok and len(why) == 1 and "j6" in why[0]


# ── 整段版 v0.3：相对输入、第一帧加权（设计 v0.3；实现计划任务 3b）───────────────────────────────
def test_relative_input_is_frames_minus_current_plus_current():
    x = torch.arange(56, dtype=torch.float32)[None]
    net = O.B4Net(56, 16, 15, rel_P=4)
    f = x.reshape(1, 8, 7)
    want = torch.cat([(f - f[:, 3:4]).reshape(1, -1), f[:, 3]], 1)
    assert torch.equal(net.prep(x), want) and net.mu.shape == (63,)
    assert torch.equal(O.B4Net(56, 16).prep(x), x)                       # 一帧版：不变换


def test_frozen_relative_net_reproduces_the_trained_net(tmp_path):
    r = N.train(make_seq_data(tmp_path / "d.npz"), 0, 16, epochs=3, out=tmp_path, rel_P=4, first=True)
    m = torch.load(r["path"])
    net = O.B4Net(m["n_in"], m["hidden"], m["n_out"], m["rel_P"])
    net.load_state_dict(m["state_dict"])
    fn, *_ = O.load_frozen(str(N.freeze(Path(r["path"]), out=tmp_path)))
    x = np.load(tmp_path / "d.npz")["X"][0]
    assert m["rel_P"] == 4 and m["first"] is True
    assert np.allclose(fn(x), net(torch.from_numpy(x)[None])[0].detach().numpy())


def test_first_frame_early_stopping_keeps_a_better_first_frame(tmp_path):
    """验证集第一帧的规律与训练相反（越练越差），后两帧一致（越练越好）：按第一帧早停留下的第一帧更好。
    标签取量级 1 的线性函数、样本 256 条：网络能学会且能推广，冲突才压过「把随机初值缩小」的效果
    （用 make_seq_data 的 0.003 量级标签时，200 步内各列损失一起降，测不出早停按哪一帧）。"""
    d = make_seq_data(tmp_path / "d.npz", n=256)
    z = dict(np.load(d))
    z["Y"] = z["X"][:, :15] + 0.5 * z["X"][:, 15:30]
    z["Y"][256:, :5] *= -1
    np.savez(d, **z)
    a = N.train(d, 0, 64, epochs=200, lr=1e-2, patience=200, out=tmp_path / "a", first=True)
    b = N.train(d, 0, 64, epochs=200, lr=1e-2, patience=200, out=tmp_path / "b")
    assert a["val_loss_first"] < b["val_loss_first"]


def test_seq_command_passes_N(monkeypatch):
    """设计 v0.4（用户 11-06 选 A2）：`seq --N 5` 只改段长。"""
    got = {}
    monkeypatch.setattr(N, "select_seq", lambda **k: got.update(k) or {"chosen": {"path": "x"}})
    assert N.main(["seq", "--N", "5"]) == 0 and got == {"N": 5}
