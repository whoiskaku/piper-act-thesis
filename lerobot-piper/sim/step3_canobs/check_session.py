#!/usr/bin/env python3
"""**采集后立即跑**：数据健康检查 + 记录器修复的回归检验。

2026-09-01 会话1 的教训：CAN 记录器初版在后台线程里调 SDK 解码
(26.4 µs/帧 × 1800 帧/s)，把 20 Hz 控制环拖成 19.3 Hz、帧间隔 p99 84.5 ms。
记录器已改为热路径只存原始字节(0.21 µs/帧)，**但必须在真机上复核**。

判据（不过就别拿去做 20 Hz 口径的分析）:
    ① 20 Hz 环: 中位 ≤ 50.5 ms 且 p99 ≤ 55 ms  （无记录器时实测 50.15/≤55）
    ② TSND 设备端 ms 步长恒为 10、零丢样        （数据完整性的底线）
    ③ 六路 CAN 各 ~200 Hz、溢出丢弃 = 0
    ④ 时钟偏移在窗口内漂移 < 1 ms

②③④ 只要过，即使 ① 不过，数据仍可用**设备/内核时基**分析；
① 不过意味着落盘的 20 Hz 字段不可信。

用法:
    python -u sim/step3_canobs/check_session.py --arms A1
    python -u sim/step3_canobs/check_session.py --arms A1 A3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EPS = ("ep0", "ep1")
TOL_MED_MS, TOL_P99_MS = 50.5, 55.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1"])
    ap.add_argument("--dir", type=Path, default=None,
                    help="改为体检该目录下所有 replay_*.json（忽略 --arms）")
    a = ap.parse_args()
    allok = True
    if a.dir:
        d = a.dir if a.dir.is_absolute() else ROOT / a.dir
        items = [(d.name, f.stem[len("replay_"):], f)
                 for f in sorted(d.glob("replay_*.json")) if ".orphan_" not in f.name]
    else:
        items = [(arm, nm, ROOT / f"outputs/probe/replay_{nm}_{arm}.json")
                 for arm in a.arms for nm in EPS]
    for arm, nm, p in items:
        if True:
            if not p.exists():
                print(f"  {arm}/{nm}: **文件不存在** {p}")
                allok = False
                continue
            d = json.loads(p.read_text())
            t = np.asarray(d["t"], float)
            dt = np.diff(t) * 1e3
            I = np.asarray(d["imu100"], float)
            H = np.asarray(d["can_hs"], float)
            Q = np.asarray(d["can_q200"], float)
            M = d.get("can_hs_meta", {})
            dur = t[-1] - t[0]
            print(f"\n=== {arm} / {nm}  ({len(t)} 帧, {dur:.1f}s, "
                  f"stopped={d.get('stopped')}) ===")

            # ① 20 Hz 环
            ok1 = np.median(dt) <= TOL_MED_MS and np.percentile(dt, 99) <= TOL_P99_MS
            print(f"  ① 20 Hz 环   中位 {np.median(dt):6.2f} ms  p99 "
                  f"{np.percentile(dt,99):6.2f}  最大 {dt.max():6.2f}  "
                  f"→ {1000/dt.mean():5.2f} Hz   {'✓' if ok1 else '**✗ 记录器仍在拖慢控制环**'}")
            # ② TSND 完整性
            dms = np.diff(I[:, 1]); dms = dms[dms >= 0]
            lost = int(np.round(np.clip(dms / 10.0 - 1, 0, None)).sum())
            ok2 = lost == 0 and set(np.unique(dms.astype(int))) <= {10}
            print(f"  ② TSND       {len(I)} 样本  设备 ms 步长 "
                  f"{sorted(set(dms.astype(int)))[:4]}  丢样 {lost}   "
                  f"{'✓' if ok2 else '**✗ 设备端真丢样**'}   "
                  f"(主机到达 p99 {np.percentile(np.diff(I[:,0]),99)*1e3:.1f} ms)")
            # ③ CAN
            hz = [1.0 / np.median(np.diff(H[H[:, 1] == j, 0])) for j in range(1, 7)]
            nd = M.get("n_dropped_overflow", None)
            ok3 = min(hz) > 150 and nd == 0
            print(f"  ③ CAN        电机 {len(H)} ({len(H)/dur:.0f}/s) 逐关节 Hz "
                  f"{np.min(hz):.0f}–{np.max(hz):.0f}   关节角 {len(Q)} "
                  f"({len(Q)/dur:.0f}/s)   溢出丢弃 {nd}   {'✓' if ok3 else '**✗**'}")
            # ④ 时钟
            c = M.get("clock", {})
            drift = (c.get("offset_at_stop_s") or 0) - (
                c.get("offset_realtime_minus_monotonic_s") or 0)
            ok4 = abs(drift) < 1e-3
            print(f"  ④ 时钟       REALTIME−MONOTONIC 漂移 {drift*1e3:+.3f} ms   "
                  f"{'✓' if ok4 else '**✗**'}")
            print(f"     固件 {M.get('firmware')}  SDK {M.get('sdk')}")
            allok &= ok1 and ok2 and ok3 and ok4
    print(f"\n  总判: {'全部通过' if allok else '**有未通过项 —— 见上面的 ✗**'}")
    print("  ①不过 ⇒ 落盘的 20 Hz 字段不可信；但②③④过则仍可用设备/内核时基分析。")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
