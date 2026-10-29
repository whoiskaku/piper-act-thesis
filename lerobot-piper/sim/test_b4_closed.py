"""`b4_common.closed_metrics`（闭环一集的指标）与 `b4_closed.check_results`（第一步检查的判定）的测试。
合成数据，不读数据集；用到 mujoco 模型。"""
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
import b4_closed as B  # noqa: E402
import b4_common as C  # noqa: E402
from ideal_imu_kinematic import MODEL_PATH  # noqa: E402

_M = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
CTX = SimpleNamespace(model=_M, sid=mujoco.mj_name2id(_M, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site"),
                      calib={"R": np.eye(3).tolist(), "b": [0, 0, 0], "gyro_bias": [0, 0, 0], "dp": [0, 0, 0], "delay_s": 0.0},
                      center=np.array([9.0, 9.0, 9.0]))


def synthetic_closed(offset_rad=0.0, v=0.1):
    """100 帧：a_π 五个活动关节以 v rad/s 匀速；夹爪第 30 帧合拢、第 60 帧张开；发出的 u = a_π + offset，
    编码器完全跟着 u（按 t_send 零阶保持）。IMU 全 0（末端指标这里不看）。"""
    n = 100
    t_send = 0.05 * np.arange(n)
    a = np.zeros((n, 7))
    a[:, :6] = 0.3 + v * t_send[:, None]
    a[:, 6] = np.where((np.arange(n) >= 30) & (np.arange(n) < 60), 0.0, 0.06)
    u = a.copy()
    u[:, :6] += offset_rad
    tq = np.arange(t_send[0], t_send[-1], 1 / 200)
    q = u[np.searchsorted(t_send, tq, side="right") - 1, :6]
    ti = np.arange(t_send[0], t_send[-1], 1 / 100)
    return {"t_send": t_send, "cmd": u[:, :6], "grip": u[:, 6], "tq": tq, "q": q, "ti": ti, "imu": np.zeros((len(ti), 6)),
            "a_pi": a}


def test_closed_metrics_reference_is_policy_command():
    """偏差相对原始策略指令 a_π，不是相对实际发出的 u：实测完全跟着 u = a_π + 2 mrad ⇒ 关节偏差 2 mrad。"""
    assert C.closed_metrics(synthetic_closed(offset_rad=0.002), CTX)["joint_raw"] == pytest.approx(2.0, abs=0.05)


def test_closed_metrics_duration_and_command_speed():
    """任务时长 = 第一次合拢后第一次张开的时刻（第 60 帧，3.0 s）；策略指令自身速度 = a_π 五个活动关节的速度 RMS。"""
    m = C.closed_metrics(synthetic_closed(v=0.1), CTX)
    assert m["duration"] == pytest.approx(3.0) and m["cmd_speed"] == pytest.approx(0.1)


def _side(jq, stages):
    return {f"c{i}": {"metrics": {"joint_raw": j}, "stage": s} for i, (j, s) in enumerate(zip(jq, stages))}


def test_check_results_pass_and_fail():
    """第一步检查：10 对齐全、成功率严格计数不降、关节偏差显著下降才算通过。"""
    raw = _side([10.0 + 0.1 * i for i in range(10)], [3] * 10)
    shift = _side([3.5 + 0.1 * i for i in range(10)], [3] * 10)
    r = B.check_results(raw, shift)
    assert r["passed"] and r["joint_raw"]["ci95"][1] < 0 and r["n_pairs"] == 10
    assert not B.check_results(raw, _side([3.5] * 10, [3] * 9 + [1]))["passed"]          # Shift 失败一条
    assert not B.check_results(raw, _side([10.0 + 0.1 * i for i in range(10)], [3] * 10))["passed"]   # 没下降
    missing = dict(shift)
    missing["c4"] = {"metrics": None, "stage": 3}
    assert not B.check_results(raw, missing)["passed"] and B.check_results(raw, missing)["n_pairs"] == 9


# ── 任务 11：闭环最终评价 ──────────────────────────────────────────────────
from collections import Counter  # noqa: E402

import b4_positions as P  # noqa: E402


def test_final_plan_counts_and_positions():
    """12 块 × 5 条：位置块 A（测试位置 1–5）、B（6–10）按 A、B、A、B；块内三个条件轮换；每个条件 20 条、每个测试位置各 2 次。"""
    d = P.generate()
    p = P.final_plan(d)
    assert len(p) == 12 and all(len(cm) == 5 for _, cm in p)
    assert Counter(c for c, _ in p) == {"shift": 4, "net": 4, "raw": 4}
    test = [x for x in d["order"] if next(q["split"] for q in d["pool"] if q["id"] == x) == "test"]
    cm = {q["id"]: q["cm"] for q in d["pool"]}
    for c in ("shift", "net", "raw"):
        got = [x for cc, xs in p if cc == c for x in xs]
        assert got == [cm[t] for t in test[:5] + test[5:] + test[:5] + test[5:]]
    assert [c for c, _ in p[:6]] == ["shift", "net", "raw", "net", "raw", "shift"]


def test_final_next_positions_resume_by_dataset_count():
    """第 b 块的位置按这个条件的数据集已录集数接着排：已录满的块跳过，录到一半的从断处接着。"""
    d = P.generate()
    full = P.final_plan(d)
    assert [c for _, c in P.next_positions(d, "final", n_done=0, block=1)] == full[0][1]
    assert [c for _, c in P.next_positions(d, "final", n_done=3, block=1)] == full[0][1][3:]
    assert P.next_positions(d, "final", n_done=5, block=1) == []
    assert [c for _, c in P.next_positions(d, "final", n_done=5, block=6)] == full[5][1]   # 块 6 是 shift 的第 2 块


def runs(net=0.35, fail=(), raw_jq=8.0):
    """合成：每个条件 4 个位置块 × 5 条；Shift 关节偏差 3 mrad，网络 3 × net 倍，原始 ACT raw_jq；fail 里的 (条件, 块, 序号) 失败。"""
    out = {}
    rng = np.random.default_rng(0)
    for pb in range(4):
        for s in range(5):
            for c, jq in (("shift", 3.0), ("net", 3.0 * net), ("raw", raw_jq)):
                m = {"joint_raw": jq * (1 + 0.01 * rng.normal()), "path": 1.0, "speed": 1.0, "duration": 12.0,
                     "cmd_speed": 0.16, "ee_raw": 0.06}
                out[(c, pb, s)] = {"metrics": m, "stage": 2 if (c, pb, s) in fail else 3}
    return out


def test_final_results_effective_and_raw_never_judged():
    a = B.final_results(runs())
    assert a["verdict"] == "有效" and a["n_pairs"] == 20 and a["joint_raw"]["n_groups"] == 10
    assert B.final_results(runs(raw_jq=99.0))["verdict"] == a["verdict"]        # 原始 ACT 组只报告
    assert a["raw_vs_shift"]["effect"] > 100


def test_pair_with_missing_side_is_dropped_and_not_judged():                  # Review Focus 5
    r = runs()
    r[("net", 2, 4)] = None
    out = B.final_results(r)
    assert out["n_pairs"] == 19 and "未完成" in out["verdict"]


def test_success_drop_fails_and_success_is_stage_3():
    assert B.success(3) and not B.success(2) and not B.success(None)
    assert B.final_results(runs(fail={("net", 1, 2)}))["verdict"] != "有效"
    assert B.final_results(runs(fail={("net", 1, 2), ("shift", 3, 0)}))["verdict"] == "有效"   # 一对换一对：不算下降


def test_interrupted_attempt_excludes_the_pair_and_rerecord_is_reported():
    """审阅 Important 3：某一边有被中断的尝试（不完整）→ 这一对不进统计；完整的重录（←）只报告次数。"""
    r = runs()
    r[("net", 1, 2)]["tries"] = [False]                                  # 先有一次中断，后补录成功
    r[("shift", 0, 0)]["tries"] = [True]                                  # 按 ← 重录过一次
    out = B.final_results(r)
    assert out["n_pairs"] == 19 and "未完成" in out["verdict"]
    assert out["attempts"] == {"interrupted": 1, "rerecorded": 1}
