"""重新生成网络 ② 的测试（方案 `reports/PLAN_b4_regen.md` v0.2；实现计划 `reports/PLAN_IMPL_b4_regen.md`）。
合成数据，不连机械臂。任务 1：训练和在线回放共用的实测窗口、输入单元；任务 2：训练数据；任务 3：网络与训练。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_net as N  # noqa: E402
import b4_regen as R  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

HOLD, N_FR = 20, 80               # 合成轨迹：开头保持 20 帧，共 80 帧


# ── 合成数据的工具 ──────────────────────────────────────────────────────────
def q200_rows(f, t_end: float, dt: float = 0.005) -> np.ndarray:
    """三组 200 Hz 的 can_q200 行 [t, 组号, q_a, q_b]（0.001°），按到达时刻排好；f(t, 组号) → (q_a, q_b)。"""
    t = np.arange(0.0, t_end, dt)
    rows = np.concatenate([np.c_[t, np.full(len(t), g), np.stack(f(t, g), 1)] for g in (1, 3, 5)])
    return rows[np.argsort(rows[:, 0], kind="stable")]


def q200_ramp(t_end: float) -> np.ndarray:
    return q200_rows(lambda t, g: (1000 * t + 100 * g, 2000 * t - 100 * g), t_end)


def q200_const(v: float, t_end: float) -> np.ndarray:
    return q200_rows(lambda t, g: (np.full_like(t, v), np.full_like(t, v)), t_end)


def imu_rows(gyro, acc, t_end: float, dt: float = 0.01) -> np.ndarray:
    """100 Hz 的 imu100 行 [到达时刻, 设备 ms, 加速度 3 维, 陀螺 3 维]，读数恒定。"""
    t = np.arange(0.0, t_end, dt)
    return np.c_[t, t * 1000, np.tile(acc, (len(t), 1)), np.tile(gyro, (len(t), 1))]


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def act_case(seed: int, change_after: int | None = None, n_log: int = 60, c: int = 12, hold: int = 20):
    """(a7, a_pi, ens)：由合成的 ACT 整段预测经 TE 得到；a7 = 开头补 hold 帧首帧保持的 a_pi（同回放序列）。
    change_after：从这一录制帧起的 chunk 都改掉（检验只用到第 k 帧以前的 chunk）。"""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.003, (n_log + c, 7)), 0)
    chunk = np.stack([base[k:k + c] for k in range(n_log)]) + rng.normal(0, 0.001, (n_log, c, 7))
    if change_after is not None:
        chunk[change_after:] += 0.05
    a_pi, ens = O.te_states(chunk)
    return np.concatenate([np.repeat(a_pi[:1], hold, 0), a_pi]), a_pi, ens


def layout_inputs():
    """(过去 ACT 指令, 已发指令, 编码器窗口, IMU 窗口, ACT 指令段)，每个元素都不同，便于核对位置。"""
    return (np.arange(48.0).reshape(8, 6), 100 + np.arange(48.0).reshape(8, 6), 1000 + np.arange(480.0).reshape(80, 6),
            5000 + np.arange(240.0).reshape(40, 6), 9000 + np.arange(60.0).reshape(10, 6))


# ── 任务 1 ────────────────────────────────────────────────────────────────
def test_enc_window_ignores_rows_at_or_after_cutoff():
    rows = q200_ramp(t_end=1.2)
    a, _ = O.enc_window(rows[rows[:, 0] < 1.0], 0.0, 1.0)
    b, _ = O.enc_window(rows, 0.0, 1.0)
    assert np.array_equal(a, b)


def test_enc_window_converts_millidegrees_to_rad():
    w, _ = O.enc_window(q200_const(1000, t_end=1.0), 0.0, 1.0)
    assert w.shape == (80, 6) and np.allclose(w, np.radians(1.0))


def test_enc_window_holds_last_value_and_reports_age():          # Review Focus 1
    rows = q200_ramp(t_end=0.95)
    w, age = O.enc_window(rows, 0.0, 1.0)
    grid = 1.0 - O.WIN_S + O.ENC_DT * np.arange(1, 81)
    late = grid > rows[:, 0].max()
    assert late.any() and np.allclose(w[late], w[late][0]) and age >= 0.05 - 1e-9


def test_enc_window_missing_group_is_nan_and_stale():
    rows = q200_ramp(t_end=1.0)
    w, age = O.enc_window(rows[rows[:, 1] != 3], 0.0, 1.0)
    assert np.isnan(w[:, 2:4]).all() and np.isfinite(w[:, [0, 1, 4, 5]]).all() and age == np.inf


def test_imu_window_rotates_and_removes_bias():
    R = rot_z(90)
    calib = {"R": R.tolist(), "gyro_bias": [0.1, 0, 0], "b": [0, 0, 0.2]}
    rows = imu_rows(gyro=R @ [1, 2, 3] + [0.1, 0, 0], acc=R @ [4, 5, 6] + [0, 0, 0.2], t_end=1.0)
    w, _ = O.imu_window(rows, calib, 1.0)
    assert w.shape == (40, 6) and np.allclose(w, [1, 2, 3, 4, 5, 6])


def test_regen_act_uses_only_chunks_up_to_k():
    A, B_ = act_case(seed=0), act_case(seed=0, change_after=31)
    for k in (25, 50):                                  # 回放帧；HOLD = 20 ⇒ 录制帧 5、30
        assert all(np.array_equal(x, y) for x, y in zip(O.regen_act(*A, k, 20), O.regen_act(*B_, k, 20)))
    assert not np.array_equal(O.regen_act(*A, 52, 20)[1], O.regen_act(*B_, 52, 20)[1])   # 录制帧 32 起确实变了


def test_regen_act_pads_with_frame_zero():                      # Review Focus 3
    a7, a_pi, ens = act_case(seed=0)
    past, seg = O.regen_act(a7, a_pi, ens, 3, 20)
    assert np.array_equal(past[:5], np.repeat(a7[:1, :6], 5, 0)) and np.array_equal(seg[0], a_pi[0, :6])


def test_regen_tokens_layout():
    past, fut = O.regen_tokens(*layout_inputs())
    assert past.shape == (8, 102) and fut.shape == (10, 6) and past.dtype == fut.dtype == np.float32
    assert np.array_equal(past[3, 12:72], layout_inputs()[2][30:40].ravel())
    assert np.array_equal(past[3, :12], np.r_[layout_inputs()[0][3], layout_inputs()[1][3]])
    assert np.array_equal(past[3, 72:], layout_inputs()[3][15:20].ravel())


# ── 任务 2：合成的一条完整轨迹 ──────────────────────────────────────────────
def replay(t_send: np.ndarray, sent7: np.ndarray, off: float, rng, lag: float = 0.06) -> dict:
    """一遍回放记录（字段同 replay_probe）：实际关节角 = 已发指令按发令时刻零阶保持、再滞后 lag 秒；
    编码器三组 200 Hz、IMU 100 Hz，到达时刻带不到 1 ms / 2 ms 的抖动（仍然递增）。"""
    n = len(t_send)
    t0, t1 = t_send[0] - 0.02, t_send[-1] + 0.05
    te = np.arange(t0, t1, 0.005)
    q = sent7[np.clip(np.searchsorted(t_send, te - lag, side="right") - 1, 0, n - 1), :6]
    q200 = np.concatenate([np.c_[te + rng.uniform(0, 0.0008, len(te)) + off, np.full(len(te), g),
                                 np.degrees(q[:, list(cols)]) * 1000] for g, cols in O.CAN_GROUPS])
    ti = np.arange(t0, t1, 0.01) + rng.uniform(0, 0.002, len(np.arange(t0, t1, 0.01)))
    imu = np.c_[ti, np.round((ti - t0) * 1000), rng.normal(0, 0.1, (len(ti), 3)) + [0, 0, 9.8], rng.normal(0, 0.05, (len(ti), 3))]
    return {"t_send": t_send, "cmd": sent7, "can_q200": q200[np.argsort(q200[:, 0], kind="stable")],
            "can_hs_meta": {"clock": {"offset_realtime_minus_monotonic_s": off}}, "imu100": imu, "stopped": None, "cmd_hash": "h"}


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(0)
    n, n_log, c = N_FR, N_FR - HOLD, 12
    steps = rng.uniform(-0.002, 0.002, (n_log + c, 7)) * [1, 1, 1, 0, 1, 1, 0]          # j4、夹爪不动
    base_traj = np.cumsum(steps, 0) + [0.1, 0.5, -0.4, 0.0, 0.1, 0.0, 0.06]
    chunk = np.stack([base_traj[k:k + c] for k in range(n_log)]) + rng.normal(0, 0.0005, (n_log, c, 7)) * [1, 1, 1, 0, 1, 1, 0]
    a_pi, _ = O.te_states(chunk)
    a7 = np.concatenate([np.repeat(a_pi[:1], HOLD, 0), a_pi])
    s7 = a7[np.minimum(np.arange(n) + 1, n - 1)].copy()                                  # 提前一帧，夹爪列同
    w = np.ones((n, 5))
    w[:HOLD], w[45:53], w[30, 2] = 0, 0, 0
    teacher7 = s7.copy()
    teacher7[:, list(O.ACTIVE)] += (rng.uniform(-0.003, 0.003, (n, 5)) * w).astype(np.float32)
    t_send = 1.0 + 0.05 * np.arange(n)
    sent1 = s7.copy()
    sent1[:, :6] += rng.uniform(-0.002, 0.002, (n, 6)).astype(np.float32) * [1, 1, 1, 0, 1, 1]
    reps = [replay(t_send, s7, 1000.0, rng), replay(t_send, sent1, 1000.0, rng)]
    base, log = {"a7": a7, "s7": s7, "w": w}, {"a_pi": a_pi, "chunk": chunk}
    calib = {"R": rot_z(30).tolist(), "gyro_bias": [0.01, 0.0, 0.0], "b": [0.0, 0.0, 0.1]}
    return SimpleNamespace(n=n, base=base, log=log, teacher7=teacher7, calib=calib, replays=reps,
                           loader_tuple=(base, log, teacher7, calib, [("t_as01", reps[0]), ("t_c01", reps[1])]))


def with_rows_after(h: dict, frame: int, seconds: float) -> dict:
    """在第 frame 帧的截止时刻与发令时刻之间（截止 + 0.5 ms、截止 + seconds/2）插入数值极大的编码器、IMU 采样。"""
    cut = h["t_send"][frame] - O.CUT_S
    off = h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    dts = (0.0005, seconds / 2)
    q = np.concatenate([h["can_q200"], [[cut + dt + off, g, 1e6, -1e6] for dt in dts for g, _ in O.CAN_GROUPS]])
    i = np.concatenate([h["imu100"], [[cut + dt, 0, *[1e6] * 6] for dt in dts]])
    return {**h, "can_q200": q[np.argsort(q[:, 0], kind="stable")], "imu100": i[np.argsort(i[:, 0], kind="stable")]}


def write_replays(d: Path, **kinds) -> None:
    """按种类写回放记录与回放序列：ok 有效；stopped 中途停了；no_can 没有编码器；bad_hash 序列哈希不符；short 帧数不符；
    unsorted 编码器到达时刻没排好。"""
    for kind, rid in kinds.items():
        q200 = [[1.0, 1, 0, 0], [1.001, 3, 0, 0]] if kind != "unsorted" else [[1.001, 1, 0, 0], [1.0, 3, 0, 0]]
        h = {"t_send": list(np.arange(N_FR) * 0.05), "cmd": [[0.0] * 7] * N_FR, "can_q200": [] if kind == "no_can" else q200,
             "imu100": [[1.0, 0, 0, 0, 0, 0, 0, 0]], "stopped": "x" if kind == "stopped" else None, "cmd_hash": "ok",
             "can_hs_meta": {"clock": {"offset_realtime_minus_monotonic_s": 0.0}}}
        if kind == "short":
            h["t_send"] = h["t_send"][:-1]
        (d / f"replay_{rid}_r1.json").write_text(json.dumps(h))
        np.savez(d / f"seq_{rid}.npz", cmd7=np.zeros((N_FR, 7), np.float32),
                 source=json.dumps({"cmd_hash": "bad" if kind == "bad_hash" else "ok"}))


def two_rows() -> dict:
    return {"t1": {"split": "train", "passed": True, "session": "s1"}, "t2": {"split": "val", "passed": True, "session": "s1"}}


# ── 任务 2 ────────────────────────────────────────────────────────────────
def test_labels_are_the_teacher_commands_for_every_replay(case):
    for h in case.replays:
        d = R.assemble_regen(h, case.base, case.log, case.teacher7, case.calib)
        for j in (0, 4, 9):
            kk = np.minimum(d["k"] + j, case.n - 1)
            assert np.array_equal(d["Y"][:, j], case.teacher7[kk, :6])


def test_weights_follow_window_and_stop_at_the_end(case):     # Review Focus 3
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    i = list(d["k"]).index(case.n - 3)
    assert d["W"][i, 3:].sum() == 0 and d["W"][i, 0].sum() > 0
    i = list(d["k"]).index(30)
    assert d["W"][i, 0, 2] == 0 and d["W"][i, 0, 3] == 0 and d["W"][i, 0, [0, 1, 4, 5]].all()   # j3 窗为 0 ⇒ j4 也为 0


def test_frames_without_any_weight_are_dropped(case):
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    assert d["W"].reshape(len(d["k"]), -1).any(1).all() and 0 not in d["k"]


def test_first_kept_frame_is_hold_minus_nine(case):           # Review Focus 3
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    assert d["k"].min() == HOLD - (O.REGEN_N - 1)


def test_inputs_use_only_samples_before_cutoff(case):          # Review Focus 1
    h = case.replays[0]
    a = R.assemble_regen(h, case.base, case.log, case.teacher7, case.calib)
    b = R.assemble_regen(with_rows_after(h, frame=40, seconds=0.010), case.base, case.log, case.teacher7, case.calib)
    i = list(a["k"]).index(40)
    assert np.array_equal(a["past"][i], b["past"][i])
    assert not np.array_equal(a["past"][i + 1], b["past"][i + 1])      # 下一帧就看得到这些采样：插入确实生效


def test_samples_carry_dtypes_and_shapes(case):
    d = R.assemble_regen(case.replays[1], case.base, case.log, case.teacher7, case.calib)
    m = len(d["k"])
    assert d["past"].shape == (m, 8, 102) and d["fut"].shape == (m, 10, 6) and d["Y"].shape == d["W"].shape == (m, 10, 6)
    assert {d[k].dtype for k in ("past", "fut", "Y", "W")} == {np.dtype(np.float32)}
    i = list(d["k"]).index(40)
    assert np.allclose(d["past"][i, -1, 6:12], case.replays[1]["cmd"][39, :6])          # 最后一个过去单元 = 第 39 帧已发出的指令


def test_cases_skip_bad_sessions_failed_teachers_and_test_split():
    rows = {"a": {"split": "train", "passed": True, "session": "20261104e"},
            "b": {"split": "spare", "passed": True, "session": "20261105"},
            "c": {"split": "val", "passed": True, "session": "20261104b"},
            "d": {"split": "train", "passed": False, "session": "20261105"},
            "e": {"split": "test", "passed": True, "session": "20261105"}}
    assert R.cases(rows) == [("a", "train", "20261104e"), ("b", "train", "20261105")]


def test_invalid_replays_are_skipped_with_a_reason(tmp_path, capsys):   # Review Focus 2
    write_replays(tmp_path, ok="p1_as01", stopped="p1_as02", no_can="p1_as03", bad_hash="p1_at01", short="p1_b01")
    assert [r for r, _ in R.valid_replays(tmp_path, "p1", N_FR)] == ["p1_as01"]
    assert capsys.readouterr().out.count("跳过") == 4


def test_unsorted_arrival_times_are_skipped(tmp_path, capsys):          # 入口单次校验：窗口函数假定已排好
    write_replays(tmp_path, ok="p1_as01", unsorted="p1_as02")
    assert [r for r, _ in R.valid_replays(tmp_path, "p1", N_FR)] == ["p1_as01"]
    assert "到达时刻" in capsys.readouterr().out


def test_build_writes_splits_and_never_mixes_val_into_train(case, tmp_path):
    p = R.build_regen(rows=two_rows(), out=tmp_path, loader=lambda t: case.loader_tuple)
    z = np.load(p)
    assert set(z["split"]) == {"train", "val"}
    assert not set(z["traj_id"][z["split"] == "train"]) & set(z["traj_id"][z["split"] == "val"])
    assert len(str(z["data_sha"])) == 16 and len(z["rid"]) == len(z["past"])


# ── 任务 3：网络与训练 ────────────────────────────────────────────────────────
def tiny_data(d: Path, n: int = 64, j4_zero: bool = False, val_noise: float = 0.0, val_copy: bool = False, seed: int = 0) -> Path:
    """小数据：标签 = ACT 指令段 + 已发指令的最后一帧 × 0.5（网络学得会的简单关系）；后四分之一是验证集。
    j4_zero：j4 的标签恒为 0（同真实数据）；val_noise：给验证集的标签加噪声（验证损失会上下波动）；
    val_copy：n 个样本都是训练集，验证集是它们的原样副本（早停留下的那一轮就是拟合得最好的那一轮）。"""
    rng = np.random.default_rng(seed)
    past = rng.normal(0, 1, (n, 8, 102)).astype(np.float32)
    fut = rng.normal(0, 1, (n, 10, 6)).astype(np.float32)
    Y = fut + 0.5 * past[:, -1:, 6:12]
    if j4_zero:
        Y[..., 3] = 0
    split = np.where(np.arange(n) < 3 * n // 4, "train", "val")
    if val_copy:
        past, fut, Y = (np.concatenate([x, x]) for x in (past, fut, Y))
        split = np.r_[["train"] * n, ["val"] * n]
    Y[split == "val"] += rng.normal(0, val_noise, Y[split == "val"].shape)
    p = d / f"data_{n}_{int(j4_zero)}_{val_noise}_{int(val_copy)}.npz"
    np.savez(p, past=past, fut=fut, Y=Y.astype(np.float32), W=np.ones((len(Y), 10, 6), np.float32), split=split, data_sha="t")
    return p


class _Echo(torch.nn.Module):
    """测试用的「网络」：输出 = 之后单元（测试里放的就是标签），第 2 帧起再加 1。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("sd_y", torch.ones(6))

    def forward(self, past, fut):
        return fut + (torch.arange(10) > 0).float()[None, :, None]


def test_no_path_from_inputs_to_output_when_head_is_zero():     # 输出不是调整量
    net = O.RegenNet()
    torch.nn.init.zeros_(net.head.weight)
    torch.nn.init.zeros_(net.head.bias)
    net.mu_y.copy_(torch.arange(6.0))
    y = net(torch.randn(3, 8, 102), torch.randn(3, 10, 6))
    assert torch.equal(y, torch.arange(6.0).expand(3, 10, 6))


def test_dropped_groups_do_not_change_the_output():
    net = O.RegenNet(drop=("enc", "imu")).eval()
    p = torch.randn(2, 8, 102)
    q = p.clone()
    q[..., 12:102] += 5.0
    f = torch.randn(2, 10, 6)
    assert torch.equal(net(p, f), net(q, f))
    assert not torch.equal(net(p, f), net(p, f + 1.0))


def test_inputs_are_relative_and_r_is_kept():                   # 用户 11-11 选 B：输入改成相对量，输出仍是绝对指令
    past, fut = torch.randn(2, 8, 102), torch.randn(2, 10, 6)
    p, f = O.RegenNet().prep(past, fut)
    r = fut[:, :1]                                                # 第 k 帧的 ACT 指令
    assert torch.allclose(p[..., :6], past[..., :6] - r) and torch.allclose(p[..., 6:12], past[..., 6:12] - r)
    for c in (12, 66):                                            # 每帧第 1、第 10 个编码器点 − 同一帧已发出的指令
        assert torch.allclose(p[..., c:c + 6], past[..., c:c + 6] - past[..., 6:12])
    assert torch.equal(p[..., 72:], past[..., 72:])               # IMU 不变
    assert torch.allclose(f[..., :6], fut - r) and torch.equal(f[..., 6:], r.expand(-1, 10, -1))


def test_normalization_is_computed_on_the_relative_inputs(tmp_path):
    p = tiny_data(tmp_path)
    sd = torch.load(R.train_regen(p, 0, epochs=1, device="cpu", out=tmp_path)["path"])["state_dict"]
    z = np.load(p)
    tr = z["split"] == "train"
    pp, ff = O.RegenNet().prep(torch.from_numpy(z["past"][tr]), torch.from_numpy(z["fut"][tr]))
    assert torch.allclose(sd["mu_p"], pp.reshape(-1, 102).mean(0), atol=1e-5)
    assert torch.allclose(sd["mu_f"], ff.reshape(-1, 12).mean(0), atol=1e-5)


def test_constant_j4_labels_give_finite_outputs(tmp_path):      # Review Focus 4
    r = R.train_regen(tiny_data(tmp_path, j4_zero=True), seed=0, epochs=2, device="cpu", out=tmp_path)
    m = torch.load(r["path"], map_location="cpu")
    assert torch.isfinite(m["state_dict"]["sd_y"]).all() and m["state_dict"]["sd_y"][3] == 1e-3 and np.isfinite(r["val_first"])


def test_first_frame_loss_scores_only_the_first_frame():        # 曾经的错：只给第一帧标签时，10 帧预测都拿去和它比
    Y, W, P = torch.randn(5, 10, 6), torch.ones(5, 10, 6), torch.zeros(5, 8, 102)
    assert R.weighted_loss(_Echo(), P, Y, Y[:, :1], W[:, :1]) == 0.0
    assert R.weighted_loss(_Echo(), P, Y, Y, W) == pytest.approx(0.9e6)        # 9/10 的帧差 1 rad = 1000 mrad


def test_loss_weights_joints_equally_in_mrad():                # 用户 11-11：损失按 mrad 计、各关节等权（同主指标）
    net = _Echo()
    net.sd_y.copy_(torch.tensor([0.185, 0.667, 0.395, 0.001, 0.095, 0.160]))   # 真实数据里各关节标签的标准差差几倍
    Y, W, P = torch.zeros(4, 1, 6), torch.ones(4, 1, 6), torch.zeros(4, 8, 102)
    err = lambda j: R.weighted_loss(net, P, Y + 1e-3 * (torch.arange(6) == j), Y, W)   # noqa: E731  只有第 j 个关节差 1 mrad
    assert err(1) == pytest.approx(err(4)) == pytest.approx(1.0 / 6)


def test_overfits_a_tiny_dataset(tmp_path):
    p = tiny_data(tmp_path, n=48, val_copy=True)
    r = R.train_regen(p, seed=0, epochs=100, patience=100, lr=1e-3, batch=8, device="cpu", out=tmp_path)
    z = np.load(p)
    assert r["train_loss"] < 1e-2 * (1e3 * z["Y"][z["split"] == "train"]).var()   # 训练误差 < 标签方差的 1%


def test_first_frame_weight_is_n_minus_one():
    W = torch.ones(1, 10, 6)
    assert R.frame_weights(W)[0, 0].sum() == 9 * W[0, 1].sum() and torch.equal(W, torch.ones(1, 10, 6))   # 不改输入


def test_same_seed_same_weights(tmp_path):
    a, b = (R.train_regen(tiny_data(tmp_path), 0, epochs=3, device="cpu", out=tmp_path / x) for x in ("a", "b"))
    sa, sb = torch.load(a["path"])["state_dict"], torch.load(b["path"])["state_dict"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_default_runs_every_epoch_and_keeps_the_best(tmp_path):   # 用户 11-11：跑满、不早停（余弦衰减前段验证损失波动大）
    p = tiny_data(tmp_path, n=32)
    z = dict(np.load(p))
    z["Y"][z["split"] == "val"] *= -1                             # 验证集反着来：越训越差，最好的是开头几轮
    np.savez(p, **z)
    r = R.train_regen(p, 0, epochs=25, lr=1e-3, batch=4, device="cpu", out=tmp_path)
    assert r["epochs"] == 25 and r["val_first"] == min(r["history_val_first"])


def test_saved_state_is_the_best_validation_epoch(tmp_path):
    r = R.train_regen(tiny_data(tmp_path, val_noise=1.0), 0, epochs=30, patience=30, lr=1e-3, device="cpu", out=tmp_path)
    assert r["val_first"] == min(r["history_val_first"]) and len(r["history_val_first"]) == r["epochs"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="没有 GPU")
def test_cuda_graph_matches_eager_and_cpu(tmp_path):           # 用户 11-12：在线推理改用 GPU + CUDA graph
    m = torch.load(R.train_regen(tiny_data(tmp_path), 0, epochs=1, device="cpu", out=tmp_path)["path"], map_location="cpu")
    g, c = O.regen_fn(m, device="cuda"), O.regen_fn(m)
    net = O.RegenNet(**m["arch"], drop=tuple(m["drop"]))
    net.load_state_dict(m["state_dict"])
    net.eval().cuda()
    rng = np.random.default_rng(0)
    xs = [(rng.normal(0, 1, (8, 102)).astype(np.float32), rng.normal(0, 1, (10, 6)).astype(np.float32)) for _ in range(2)]
    ys = [g(*x) for x in xs] + [g(*xs[0])]                      # 换了输入，回放就要跟着变
    with torch.no_grad():
        eager = net(*(torch.from_numpy(v)[None].cuda() for v in xs[0]))[0].double().cpu().numpy()
    assert ys[0].shape == (10, 6) and ys[0].dtype == np.float64
    assert np.array_equal(ys[0], ys[2]) and not np.allclose(ys[0], ys[1])
    assert np.allclose(ys[0], eager, rtol=0, atol=1e-6) and np.allclose(ys[0], c(*xs[0]), rtol=0, atol=1e-5)


def test_load_regen_rejects_changed_weights(tmp_path):
    f = R.freeze_regen(Path(R.train_regen(tiny_data(tmp_path), 0, epochs=1, device="cpu", out=tmp_path)["path"]), out=tmp_path)
    Path(json.loads(f.read_text())["weights"]).write_bytes(b"x")
    with pytest.raises(SystemExit):
        O.load_regen(str(f))


# ── 任务 4：逐帧检查、指令切换与界 ──────────────────────────────────────────────
def toy_cfg(**kw) -> O.OnlineCfg:
    return O.OnlineCfg(**{"center": (0.0, 0.0, 0.0), "jnt_lo": (-1.0,) * 6, "jnt_hi": (1.0,) * 6, "model_path": "", **kw})


def test_each_rule_triggers_and_in_range_passes():
    lo, hi = -np.ones(6), np.ones(6)
    ok = np.zeros(6)
    assert O.step_check(ok, ok, ok, lo, hi) is None
    for u, prev, a, rule in ((np.r_[1.1, 0, 0, 0, 0, 0], ok, ok, "位置"),
                             (np.r_[0, 0, 0, 0, 0.03, 0], ok, np.r_[0, 0, 0, 0, 0.03, 0], "每帧变化"),
                             (np.r_[0.05, 0, 0, 0, 0, 0], np.r_[0.05, 0, 0, 0, 0, 0], np.r_[-0.06, 0, 0, 0, 0, 0], "与 ACT 指令的差"),
                             (np.r_[np.nan, 0, 0, 0, 0, 0], ok, ok, "非有限")):
        assert O.step_check(u, prev, a, lo, hi).startswith(rule)


def test_regen_command_switches_per_joint_and_keeps_shift_gripper():
    first, s7 = np.arange(6.0), -np.arange(7.0)
    u = O.regen_command(first, s7, np.array([1, 0, 1, 1, 1.0]))
    assert u.dtype == np.float32 and list(u) == [0, -1, 2, -3, 4, 5, -6]   # j2 窗为 0 ⇒ Shift；j4 取最小值 ⇒ Shift


def test_perfect_teacher_prediction_reproduces_the_teacher_bitwise(case):   # 设计 §6
    u = np.stack([O.regen_command(case.teacher7[k, :6], case.base["s7"][k], case.base["w"][k]) for k in range(case.n)])
    assert np.array_equal(u, case.teacher7.astype(np.float32))


def test_bounds_intersect_sent_range_and_model_limits():
    cfg = toy_cfg(jnt_lo=[-1] * 6, jnt_hi=[1] * 6)
    lo, hi = O.regen_bounds(cfg, np.full(6, -2.0), np.full(6, 0.1))
    assert np.allclose(hi, 0.15) and np.allclose(lo, -1 - 0.035 + np.radians(0.2))


# ── 任务 5：离线检查与冻结 ──────────────────────────────────────────────────────
def teacher_oracle(case, noise_after_first: float = 0.0):
    """测试用的「网络」：按调用次数（= 帧号）返回教师 cmd7 第 k … k+9 帧的 6 个关节（超出末尾取最后一帧）；
    noise_after_first：第 2 帧起另加的噪声（只发第一帧，这些帧不该影响发出的指令）。"""
    k, rng = iter(range(case.n)), np.random.default_rng(1)

    def fn(past, fut):
        y = np.asarray(case.teacher7, float)[np.minimum(next(k) + np.arange(10), case.n - 1), :6]
        return y + np.r_[np.zeros((1, 6)), rng.normal(0, noise_after_first, (9, 6))]
    return fn


def recording_fn(seen: list, out):
    """记下每次调用的输入 (past, fut)，输出 out(past, fut)。"""
    def fn(past, fut):
        seen.append((past, fut))
        return out(past, fut)
    return fn


def test_r2_against_the_teacher_is_one_for_a_perfect_net(case):
    first = R.first_frames(teacher_oracle(case), case.replays[0], case.base, case.log, case.calib)
    act = list(O.ACTIVE)
    star = case.teacher7[:, act] - case.base["s7"][:, act]
    assert np.allclose(N._r2(first[:, act] - case.base["s7"][:, act], star, case.base["w"]), 1.0)


def test_preview_feeds_back_its_own_commands(case):
    seen = []
    fn = recording_fn(seen, out=lambda past, fut: fut + 0.001)
    R.preview(fn, case.replays[0], case.base, case.log, case.calib, -np.ones(6) * 9, np.ones(6) * 9)
    k = 40
    assert np.allclose(seen[k][0][-1, 6:12], seen[k - 1][1][0] + 0.001)   # 第 k 帧输入里的上一帧已发指令 = 上一帧自己的输出


def test_preview_counts_every_violation_without_stopping(case):
    fn = lambda past, fut: fut + 0.2                              # noqa: E731  用网络的帧都和 ACT 指令差 0.2 rad
    r = R.preview(fn, case.replays[0], case.base, case.log, case.calib, -np.ones(6) * 9, np.ones(6) * 9)
    assert len(r["u7"]) == case.n and len(r["stops"]) >= int((case.base["w"].max(1) > 0).sum())


def test_freeze_only_once(tmp_path):
    run = Path(R.train_regen(tiny_data(tmp_path), 0, epochs=1, device="cpu", out=tmp_path)["path"])
    R.freeze_regen(run, out=tmp_path)
    with pytest.raises(SystemExit):
        R.freeze_regen(run, out=tmp_path)


def test_gate1_fails_on_nan_and_non_positive_r2():               # 用户 11-12：R²（相对 shift84）> 0；NaN 曾被判成通过
    assert R.gate1(np.full(5, 0.01)) == []
    why = R.gate1(np.array([0.3, np.nan, 0.0, -0.2, 0.4]))
    assert [w[:2] for w in why] == ["j2", "j3", "j5"]


# ── B：B4 + 写死的比例反馈（设计 `reports/PLAN_b3.md` §9）：「理想 − 实测」────────────────────────
def err_case(lag: float = 0.03, k: int = 30, n: int = 60) -> dict:
    """理想：六个关节都是斜率 0.2 rad/s 的斜线（网格间隔取 ideal_motion 的）；实测 = 理想整体晚 lag 秒。
    已发 k 帧，发令时刻 1.0 + 0.05·i；截止时刻在最后一帧之后 28 ms。"""
    t_send = 1.0 + 0.05 * np.arange(k)
    cutoff = t_send[-1] + 0.028
    t = cutoff - 0.4 + 0.005 * np.arange(1, 81)
    x = 0.2 * R.IM.DTG * np.arange((n - 1) * 10 + 1)[:, None] * np.ones(6)
    return dict(enc=0.2 * (t - 1.0 - lag)[:, None] * np.ones(6), cutoff=cutoff, t_send=t_send, x=x, w=np.ones((n, 5)))


def test_ideal_err_recovers_a_pure_delay():
    e = O.ideal_err(**err_case())
    assert e.shape == (80, 6) and np.allclose(e, 0.2 * 0.03)


def test_ideal_err_is_zero_where_the_window_is_zero():
    a = err_case()
    a["w"][22:26] = 0                                              # 第 22–25 帧所有关节的窗为 0
    a["w"][27, 2] = 0                                              # 第 27 帧只有 j3 的窗为 0
    e = O.ideal_err(**a)
    f = np.floor((a["cutoff"] - 0.4 + 0.005 * np.arange(1, 81) - 1.0) / 0.05).astype(int)   # 采样所在的帧
    assert not e[(f >= 22) & (f <= 25)].any()
    assert not e[f == 27][:, [2, 3]].any() and e[f == 27][:, [0, 1, 4, 5]].all()          # j4 取五个关节里最小的窗


def test_ideal_err_extrapolates_after_the_last_sent_frame():
    a = err_case(lag=0.0)
    a["t_send"] = np.r_[a["t_send"][:-1], a["t_send"][-1] + 0.004]   # 最后一帧晚发 4 ms
    e = O.ideal_err(**a)
    late = a["cutoff"] - 0.4 + 0.005 * np.arange(1, 81) > a["t_send"][-1]
    assert late.any() and np.allclose(e[late], -0.2 * 0.004)


def test_ideal_err_is_zero_before_any_frame_is_sent():
    a = err_case()
    assert not O.ideal_err(**{**a, "t_send": a["t_send"][:0]}).any()
