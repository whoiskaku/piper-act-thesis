"""阶段一（Q1）数据层 —— `reports/DESIGN_G_PI_V2.md` §2.3/§2.4。

100 集裸 ACT（5 会话）→ `Q1Episode`。标签直接取 `fit_step1_candidate.load_episode`
（`gap[t] = ω_real[t+3] − ω_ideal[t]`），与 `train_fabs_g0.load_all` 逐集相同。
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs",
           ROOT / "sim/step4_gvnext", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SESSIONS = (("ab5", "A"), ("ab6", "A"), ("ab7", "Apost"), ("ab2", "A"), ("ab4", "P0"))
EXPECTED_COUNTS = {"ab5/A": 40, "ab6/A": 30, "ab7/Apost": 15, "ab2/A": 10, "ab4/P0": 5}
MIN_FRAMES = 550
FS = 20.0
ALIGN_L = 3
EDGE = 20
HIST = 20                     # n = 20 帧（1 s）：初始设计选择，不扫描（用户 2026-10-02）
CHUNK_MAX = 100
Q_VALID_FROM = 2              # 核实 2026-10-02：100 集编码器 state 前 2 帧全为零
MDEG = np.pi / 180.0 / 1000.0
CALIB_JSON = ROOT / "outputs/step1_H/step1_H_v1.json"
# 冻结的集名名单（各会话 40/30/15/10/5；ab2/A 的 ep8、ep9 因帧数/IMU 不合格不在内）——
# q1_roster.json 在代码指纹里，prepare 与汇总都按它核对
ROSTER = json.loads((HERE / "q1_roster.json").read_text())


class DataContractError(RuntimeError):
    """数据不满足已核实的契约（chunk 结构、编码器前 2 帧）。**不被 `load_bare` 吞掉。**"""


@dataclass
class Q1Episode:
    name: str                 # "ab5/A_ep0"
    sess: str                 # "ab5/A"
    path: str                 # 相对 ROOT
    T: int                    # 记录帧数
    cmd: np.ndarray           # (Tc,7) 完整 chunk 指令（含未执行尾部），rad
    ideal: np.ndarray         # (Tc,3) 渲染理想陀螺，已旋到传感器系
    caps: np.ndarray          # (Tc,) 每帧所在 chunk 的末帧
    q: np.ndarray             # (T,6) 实测关节 rad；无效行 = 0
    q_valid: np.ndarray       # (T,) bool
    w: np.ndarray             # (T,3) 实测陀螺（传感器系）
    w_valid: np.ndarray       # (T,) bool
    gap: np.ndarray           # (n,3) = w[t+3] − ideal[t]，n = T − 3

    @property
    def n(self) -> int:
        return len(self.gap)


def calibration() -> tuple[np.ndarray, np.ndarray]:
    cal = json.loads(CALIB_JSON.read_text())["calibration"]
    return (np.asarray(cal["R"], np.float64),
            np.asarray(cal.get("b", [0, 0, 0]), np.float64))


_RENDERER = None


def renderer():
    global _RENDERER
    if _RENDERER is None:
        from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
        _RENDERER = IdealChunkRenderer()
    return _RENDERER


def check_chunks(chunks: list[dict]) -> list[dict]:
    """chunk 契约：从 0 起连续、每个长度 1…CHUNK_MAX。→ 按起点排序的列表。"""
    cs = sorted(chunks, key=lambda c: int(c["start"]))
    end = 0
    for c in cs:
        if int(c["start"]) != end:
            raise DataContractError(f"chunk 不连续：期望起点 {end}，实际 {c['start']}")
        n = len(c["raw"])
        if not 1 <= n <= CHUNK_MAX:
            raise DataContractError(f"chunk 长度 {n} 不在 1…{CHUNK_MAX}")
        end += n
    return cs


def measured_joints(state) -> tuple[np.ndarray, np.ndarray]:
    """记录的 state（mdeg，取前 6 维）→ (q rad, 有效位)。全零行 = 无效（未就绪/掉线）。"""
    st = np.asarray(state, np.float64)[:, :6]
    nz = np.abs(st).sum(1) > 0
    if nz[:Q_VALID_FROM].any():
        raise DataContractError(f"前 {Q_VALID_FROM} 帧 state 非零 —— 与核实的数据事实不符")
    valid = nz & (np.arange(len(st)) >= Q_VALID_FROM)
    return np.where(valid[:, None], st * MDEG, 0.0), valid


def load_episode(path, sess: str) -> Q1Episode:
    """一集 → `Q1Episode`。帧数不足/缺 IMU 抛 `ValueError`；违反契约抛 `DataContractError`。"""
    import fit_step1_candidate as S1
    path = Path(path)
    pay = json.loads(path.read_text())
    fr = pay["frames"]
    T = len(fr)
    if T < MIN_FRAMES or any(f.get("imu") is None for f in fr):
        raise ValueError(f"{path}: 少于 {MIN_FRAMES} 帧或缺 IMU")
    chunks = check_chunks(pay["chunks"])
    R, b = calibration()
    e = S1.load_episode(path, "P0", R, b, renderer(), MIN_FRAMES)    # 标签的唯一来源
    cmd, ideal_sim, caps = S1._render_logged_chunks(chunks, renderer(), T)
    ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R.astype(np.float32),
                                   b.astype(np.float32))[:, 3:6].astype(np.float64)
    q, q_valid = measured_joints([f["state"] for f in fr])
    w = np.asarray([f["imu"] for f in fr], np.float64)[:, 3:6]
    gap = np.asarray(e.gap, np.float64)
    if gap.shape != (T - ALIGN_L, 3):
        raise DataContractError(f"{path}: gap 形状 {gap.shape}")
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:                                 # ROOT 之外的文件：记绝对路径
        rel = str(path)
    return Q1Episode(name=f"{sess.split('/')[0]}/{path.stem}", sess=sess,
                     path=rel, T=T, cmd=cmd, ideal=ideal, caps=caps,
                     q=q, q_valid=q_valid, w=w, w_valid=np.isfinite(w).all(1), gap=gap)


def load_bare(sessions=SESSIONS) -> list[Q1Episode]:
    """5 会话的裸 ACT 集；与 `train_fabs_g0.load_all` 同筛选、同顺序。"""
    out = []
    for s, arm in sessions:
        for p in sorted((ROOT / "outputs" / s).glob(f"{arm}_ep*.json"),
                        key=lambda q: int(q.stem.split("ep")[-1])):
            try:
                out.append(load_episode(p, f"{s}/{arm}"))
            except ValueError:
                continue
    return out


def check_roster(eps) -> None:
    """载入的集必须与冻结名单逐会话、逐集名完全一致（总数相同但会话间挪了一集也要拦下）。"""
    got = {}
    for e in eps:
        got.setdefault(e.sess, []).append(e.name)
    bad = []
    for s in sorted(set(got) | set(ROSTER)):
        a, b = set(got.get(s, [])), set(ROSTER.get(s, []))
        if a != b or len(got.get(s, [])) != len(a):
            bad.append(f"{s}: 缺 {sorted(b - a)[:3]} 多 {sorted(a - b)[:3]}（{len(got.get(s, []))} 集，应为 {len(b)}）")
    if bad:
        raise DataContractError("数据与冻结名单不符：" + "；".join(bad))


def by_session(eps) -> dict:
    d = {}
    for e in eps:
        d.setdefault(e.sess, []).append(e)
    return d
