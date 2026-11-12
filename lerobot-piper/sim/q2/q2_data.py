"""阶段二（Q2 开发检验）数据层 —— `reports/PREREG_Q2_STAGE2.md` §1。

近期装置三个会话（ab5 / ab6 / ab7）的裸跑与补正运行，以及 ab4 P2/P3 辅助参考 → `Q2Episode`。
标签、渲染与阶段一的数据契约全部沿用 `sim/q1/q1_data.py`（只读复用）；本模块只加：
- `da`：日志逐帧的 `da`，即**日志记录的请求补正量**（不是执行位移）；
- 日志契约：日志动作 a = 名义指令 u + δa（|差| ≤ 1e-6；只证明日志的组合关系，不证明设备执行了同样的补正）；
  裸跑的 δa 必须全为 0，补正运行与探针必须有非零 δa；
- 冻结的数据名单（`q2_roster.json`，在代码指纹里），按「会话/臂」逐集名核对。
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q1_data as D1           # noqa: E402

ROOT = D1.ROOT
FOLDS = ("ab5", "ab6", "ab7")
BARE = {"ab5": ("A",), "ab6": ("A",), "ab7": ("Apost",)}
CORR = {"ab5": ("C", "Csafe"), "ab6": ("D",), "ab7": ("D",)}
AUX = {"ab4": ("P2", "P3")}
LOG_TOL = 1e-6
ROSTER_FILE = HERE / "q2_roster.json"
ROSTER = json.loads(ROSTER_FILE.read_text()) if ROSTER_FILE.exists() else {}


class LogContractError(D1.DataContractError):
    """日志不满足阶段二的契约（a = u + δa、裸跑 δa = 0、补正运行 δa 非零）。不被加载循环吞掉。"""


@dataclass
class Q2Episode(D1.Q1Episode):
    fold: str = ""             # 会话：ab5 / ab6 / ab7 / ab4
    arm: str = ""              # A / Apost / C / Csafe / D / P2 / P3
    kind: str = ""             # bare / corr / aux
    da: np.ndarray = None      # (T, 6) 日志记录的请求补正量（rad）


def log_da(path, T: int, kind: str) -> np.ndarray:
    """从日志读 δa，并核对日志契约 → (T, 6)。"""
    pay = json.loads(Path(path).read_text())
    fr = pay["frames"]
    a = np.asarray([f["a"] for f in fr], np.float64)[:, :6]
    da = np.asarray([f["da"] for f in fr], np.float64)
    chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
    raw = np.concatenate([np.asarray(c["raw"], np.float64) for c in chunks])[:len(fr), :6]
    if da.shape != (T, 6) or not np.isfinite(da).all():
        raise LogContractError(f"{path}: δa 形状 {da.shape} 或含非有限值")
    gap = float(np.abs(a - raw - da).max())
    if gap > LOG_TOL:
        raise LogContractError(f"{path}: 日志动作 a ≠ 名义指令 + δa（最大差 {gap:.2e}）")
    nz = bool(np.abs(da).max() > 0)
    if kind == "bare" and nz:
        raise LogContractError(f"{path}: 裸跑的 δa 应全为 0")
    if kind in ("corr", "aux") and not nz:
        raise LogContractError(f"{path}: 补正运行 / 探针的 δa 全为 0")
    return da


def from_q1(e1, fold: str, arm: str, kind: str, da) -> Q2Episode:
    base = {f.name: getattr(e1, f.name) for f in fields(D1.Q1Episode)}
    return Q2Episode(**base, fold=fold, arm=arm, kind=kind, da=np.asarray(da, np.float64))


def load_episode(path, fold: str, arm: str, kind: str) -> Q2Episode:
    """一集 → `Q2Episode`。帧数不足 / 缺 IMU 抛 `ValueError`（同阶段一筛选）；违反契约抛 `DataContractError`。"""
    e1 = D1.load_episode(path, f"{fold}/{arm}")
    return from_q1(e1, fold, arm, kind, log_da(path, e1.T, kind))


def arm_paths(fold: str, arm: str) -> list:
    return sorted((ROOT / "outputs" / fold).glob(f"{arm}_ep*.json"), key=lambda q: int(q.stem.split("ep")[-1]))


def load_arms(spec: dict, kind: str) -> list:
    out = []
    for fold, arms in spec.items():
        for arm in arms:
            for p in arm_paths(fold, arm):
                try:
                    out.append(load_episode(p, fold, arm, kind))
                except ValueError:
                    continue
    return out


def spec_for(folds=FOLDS, kinds=("bare", "corr", "aux")) -> list:
    """→ [(spec, kind)]：给定会话与类别要加载的臂。"""
    out = []
    if "bare" in kinds:
        out.append(({f: BARE[f] for f in folds if f in BARE}, "bare"))
    if "corr" in kinds:
        out.append(({f: CORR[f] for f in folds if f in CORR}, "corr"))
    if "aux" in kinds:
        out.append((AUX, "aux"))
    return out


def load(folds=FOLDS, kinds=("bare", "corr", "aux"), check=True) -> list:
    """加载指定会话与类别的集；`check` 时按冻结名单核对这些「会话/臂」组。"""
    eps = []
    for spec, kind in spec_for(folds, kinds):
        eps += load_arms(spec, kind)
    if check:
        keys = [f"{f}/{a}" for spec, _ in spec_for(folds, kinds) for f, arms in spec.items() for a in arms]
        check_roster(eps, keys)
    return eps


def groups(eps) -> dict:
    g = {}
    for e in eps:
        g.setdefault(f"{e.fold}/{e.arm}", []).append(e.name)
    return g


def check_roster(eps, keys=None, roster=None) -> None:
    """载入的集必须与冻结名单逐组、逐集名一致（`keys` 指定要核对的「会话/臂」组；缺省 = 名单里的全部组）。"""
    roster = ROSTER if roster is None else roster
    keys = sorted(roster) if keys is None else sorted(keys)
    got = groups(eps)
    bad = []
    for k in sorted(set(keys) | (set(got) - set(roster))):
        a, b = got.get(k, []), roster.get(k, [])
        if sorted(a) != sorted(b):
            bad.append(f"{k}: 缺 {sorted(set(b) - set(a))[:3]} 多 {sorted(set(a) - set(b))[:3]}（{len(a)} 集，应为 {len(b)}）")
    if bad:
        raise D1.DataContractError("数据与阶段二冻结名单不符：" + "；".join(bad))


def select(eps, fold=None, kind=None) -> list:
    return [e for e in eps if (fold is None or e.fold == fold) and (kind is None or e.kind == kind)]
