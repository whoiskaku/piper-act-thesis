#!/usr/bin/env python3
"""把 **pp2 家族的 30 条评估回放**转成 `replay_probe.py` 能吃的序列。

来源 `outputs/p2_labels/{pp2,pp2a,pp2f,pp2f60}-deploy/ep*.npz`
（字段 `a_arm` 弧度 (T,6)、`grip` 米 (T,)），输出 `cmd7 = [a_arm, grip]`、`da = 0`
（**裸复放，不加探针**）。

为什么要复放而不是直接用原 npz
------------------------------
原 npz 是 **20 Hz 落盘、无 `imu100`、无 `can_q200`** —— 在修正后的抗混叠口径下
不能当训练目标（`RESULT_CAN_OBSERVABILITY.md`）。复放的唯一目的是
**用同样的指令重新采一遍，把 100 Hz IMU 与 200 Hz 关节角落下来**。

多样性（这是选它们而不是 step1_v3 那 17 条的理由）
--------------------------------------------------
step1_v3 的 17 条逐集指令差异只有 **8.2%**，j4 从不动，时长固定 584–587 帧
⇒ 同一条轨迹重复，学不出上下文相关的残差，只会记住时间模板。
pp2 家族有跨族多样性（`pp2f60` 与训练族的轨迹相关仅约 0.70–0.73）。

⚠ **j4 在四个族里全程恒为 0** —— 逐关节判据只在 5 个可动关节上成立。

用法:
    python -u sim/step4_gvnext/make_pp2_replay_seqs.py            # 写序列
    python -u sim/step4_gvnext/make_pp2_replay_seqs.py --check    # 只做安全体检
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SRC = ROOT / "outputs/p2_labels"
OUT = ROOT / "outputs/step4_gvnext/seqs"
FAMILIES = ("pp2", "pp2a", "pp2f", "pp2f60")
GRIP_MAX = 0.035                      # 与 replay_probe 的夹爪钳位一致


def build(fam: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for f in sorted((SRC / f"{fam}-deploy").glob("ep*.npz")):
        z = np.load(f)
        a = np.asarray(z["a_arm"], float)
        g = np.clip(np.asarray(z["grip"], float), 0.0, GRIP_MAX)
        cmd7 = np.concatenate([a, g[:, None]], 1).astype(np.float32)
        da = np.zeros((len(cmd7), 6), np.float32)          # 裸复放
        out.append((f"{fam}_{f.stem}", cmd7, da))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做安全体检，不落盘")
    a = ap.parse_args()
    from replay_probe import safety_checks

    OUT.mkdir(parents=True, exist_ok=True)
    tot = ok = 0
    print(f"{'序列':<18}{'帧':>6}{'时长s':>7}   安全检查")
    worst = []
    for fam in FAMILIES:
        for nm, cmd7, da in build(fam):
            tot += 1
            try:
                msgs, degraded = safety_checks(cmd7, da)
                bad = [m for m in msgs if not m.startswith("✓")]
                mar = [m for m in msgs if "模型限位" in m]
                ok += 1
                note = mar[0].replace("✓ ", "") if mar else ""
                if degraded:
                    note += f"  ⚠降级:{degraded[0][:30]}"
                print(f"{nm:<18}{len(cmd7):>6}{len(cmd7)/20:>7.1f}   ✓ {note}")
                if mar:
                    worst.append((nm, float(mar[0].split()[-2].rstrip("°(j123456"))
                                  if False else 0.0))
            except AssertionError as e:
                print(f"{nm:<18}{len(cmd7):>6}{len(cmd7)/20:>7.1f}   **✗ {e}**")
            if not a.check:
                np.savez(OUT / f"{nm}.npz", cmd7=cmd7, da=da,
                         src=str(SRC / f"{fam}-deploy"), family=fam)
    print(f"\n  {ok}/{tot} 条通过安全检查"
          + ("（--check：未落盘）" if a.check else f"  → {OUT}"))
    if not a.check:
        (OUT / "manifest.json").write_text(json.dumps(
            {"families": {f: len(build(f)) for f in FAMILIES},
             "note": "裸复放, da=0; j4 在所有族里恒为 0",
             "grip_clip": GRIP_MAX}, indent=1, ensure_ascii=False))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
