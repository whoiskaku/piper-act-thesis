"""断开前分段归位: 两处定义必须一致, 且 full_timeline 要真的覆盖它。

2026-08-26 加。起因: 第 54 条 S_j4_p2_rise10_a8.2 在**归位段**碰撞, 而该段
从未被任何检查覆盖 —— full_timeline 建模的是「2s 插值回 HOME」, 真机执行的是
「固件一次性无插值写 safe_disable」, 起点/终点/路径/有无插值四者全不同。
修好之后, 两个文件各存一份定义, 最容易悄悄漂移, 故用断言钉住。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))


def test_stage_definitions_agree():
    """replay_probe 与 sequences 的分段归位定义必须逐项一致。"""
    import replay_probe as RP
    import sequences as SQ
    assert tuple(tuple(idx) for _, idx in RP.RETURN_STAGES) == SQ.RETURN_STAGES
    assert np.allclose(RP.SAFE_DISABLE, np.deg2rad(SQ.SAFE_DISABLE_DEG), atol=1e-4)
    assert abs(RP.RET_S - SQ.RETURN_STAGE_S) < 1e-9
    # 每个关节恰好被归位一次
    flat = [j for idx in SQ.RETURN_STAGES for j in idx]
    assert sorted(flat) == list(range(6)), flat


def test_full_timeline_starts_and_ends_at_park():
    """时间线的起点与终点都必须是**真机的停放位姿**, 不是 HOME。"""
    import sequences as SQ
    cmd = np.zeros((40, 7))
    cmd[:, :6] = np.deg2rad([-4.9, 51.5, -25.2, 0.0, 30.0, -110.0])
    full, spans = SQ.full_timeline(cmd)
    park = np.deg2rad(SQ.SAFE_DISABLE_DEG)
    assert np.allclose(full[0, :6], park, atol=1e-6), full[0, :6]
    assert np.allclose(full[-1, :6], park, atol=1e-6), full[-1, :6]


def test_return_moves_one_group_at_a_time():
    """归位段内, 每一段只能有该组的关节在动 —— 这是本次修复的安全性依据。"""
    import sequences as SQ
    cmd = np.zeros((40, 7))
    cmd[:, :6] = np.deg2rad([-4.9, 51.5, -25.2, 0.0, 61.9, -113.8])   # 出事那条的末帧
    full, spans = SQ.full_timeline(cmd)
    for a, b, nm in spans:
        if not nm.startswith("归位"):
            continue
        moved = np.where(np.abs(full[a:b, :6] - full[a, :6]).max(0) > 1e-9)[0]
        want = {int(x) - 1 for x in nm.replace("归位 j", "").split("/j")}
        assert set(moved.tolist()) <= want, (nm, moved, want)


def test_j6_rolls_first():
    """j6 必须**最先单独**归零 —— 它是模型看不见的那个(转动时 link6 原点移动 0 mm),
    且此刻臂正处在刚跑完整条激励、已知安全的位形上。"""
    import sequences as SQ
    assert SQ.RETURN_STAGES[0] == (5,), SQ.RETURN_STAGES
