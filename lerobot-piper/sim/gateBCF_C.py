#!/usr/bin/env python3
"""门禁 B / C / F 阈值确认（c-pilot-v1，2026-07-29）。

从既有 A1–A6 的 100 Hz 裸复放记录里读出**实测分布**，据此定门槛，
而不是把 B′ 的值直接搬过来。

  B 记录链路: 缺样率、时间回退/重复、最大采样间隔、晶振 ppm、传输延迟 p95、t_send_span
  C 起始状态: 预移动后偏差、停放位姿、起点一致性
  F 物理停止门: gyro 增量包络

用法: python sim/gateBCF_C.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_c"
BARE = {"A1": "replay_ep0_A1", "A2": "replay_ep0_A2", "A3": "replay_ep0_r2_A3",
        "A4": "replay_ep0_r2_A4", "A5": "replay_ep0_r3_A5", "A6": "replay_ep0_r3_A6"}
NOM_MS = 10.0                      # 100 Hz 标称采样周期


def load():
    return {k: json.loads((A.PB / f"{v}.json").read_text()) for k, v in BARE.items()}


def main():
    R = load()
    nm = ["j1", "j2", "j3", "j4", "j5", "j6"]
    print("=== 门禁 B/C/F 阈值确认 (c-pilot-v1) ===")
    print(f"  数据源: {len(R)} 条裸复放 A 臂 (3 个 session × 2)\n")

    # ================= B 记录链路 =================
    print("  【B 记录链路】逐条实测")
    print(f"  {'臂':>4}{'样本':>7}{'缺样率':>9}{'回退':>6}{'重复':>6}"
          f"{'最大间隔ms':>12}{'晶振ppm':>9}{'ok':>4}{'延迟p95ms':>11}{'send_span ms':>13}")
    rows = []
    for k in BARE:
        d = R[k]
        z = np.array(d["imu100"], float)
        host, dev = z[:, 0], z[:, 1]
        dms = np.diff(dev)
        back = int((dms < 0).sum()); dup = int((dms == 0).sum())
        # 缺样率: 以标称 10ms 为单位, 统计"应有样本"中缺掉的比例
        span_ms = dev[-1] - dev[0]
        miss = 1.0 - len(dev) / max(span_ms / NOM_MS + 1.0, 1.0)
        t, info = A.fit_clock(host, dev, k)
        # fit_clock 内部已把 device_ms/1000 ⇒ a 无量纲, 且直接给出 rate_ppm。
        # 初版自己按 a/1e-3 重算, 单位错了 ⇒ 得到 9.99e8 ppm 的荒谬值(2026-07-29 修)。
        ppm = info.get("rate_ppm", np.nan)
        lat = host - t                                  # 传输/调度延迟(≥0)
        span = np.array(d["t_send_span"], float) * 1e3
        rows.append(dict(arm=k, n=len(dev), miss=miss, back=back, dup=dup,
                         gap=float(dms.max()), ppm=float(ppm),
                         lat95=float(np.percentile(lat, 95) * 1e3),
                         span95=float(np.percentile(span, 95)),
                         res_rms=float(np.sqrt(np.mean(lat ** 2)) * 1e3),
                         ok=bool(info.get("ok", False))))
        print(f"  {k:>4}{len(dev):>7}{miss:>9.2%}{back:>6}{dup:>6}"
              f"{dms.max():>12.1f}{ppm:>9.1f}{'✓' if rows[-1]['ok'] else '✗':>4}"
              f"{rows[-1]['lat95']:>11.2f}"
              f"{rows[-1]['span95']:>13.3f}")

    mx = lambda f: max(r[f] for r in rows)
    print(f"\n  实测上界: 缺样率 {mx('miss'):.2%}  最大间隔 {mx('gap'):.1f} ms  "
          f"|ppm| {max(abs(r['ppm']) for r in rows):.0f}  "
          f"延迟p95 {mx('lat95'):.2f} ms  send_span p95 {mx('span95'):.3f} ms")
    print("  **建议门槛**（实测上界留 ~2× 余量，除缺样率按用户已定的 2%）:")
    print(f"    B1 缺样率        < 2%            (实测最差 {mx('miss'):.2%})")
    print(f"    B2 回退/重复     = 0              (实测 {max(r['back'] for r in rows)}"
          f" / {max(r['dup'] for r in rows)})")
    print(f"    B3 最大采样间隔  ≤ {max(2*NOM_MS, 2*mx('gap')):.0f} ms"
          f"       (实测最差 {mx('gap'):.1f} ms)")
    print(f"    B4 |晶振 ppm|    ≤ {max(500, 2*max(abs(r['ppm']) for r in rows)):.0f}"
          f"          (实测 {[round(r['ppm']) for r in rows]})")
    print(f"    B5 延迟 p95      ≤ {2*mx('lat95'):.1f} ms"
          f"      (实测最差 {mx('lat95'):.2f} ms)")
    print(f"       t_send_span p95 ≤ {2*mx('span95'):.2f} ms"
          f"   (实测最差 {mx('span95'):.3f} ms)")
    print(f"    B6 三套原始时间戳落盘 = 必须 (二值)")
    print(f"       注: 'RMS(host−t)' = **传输延迟**的 RMS ({mx('res_rms'):.3f} ms), "
          f"**不是**映射残差 ——")
    print(f"       最小延迟线的残差按定义就是延迟本身, 无真值就测不出映射误差。")
    print(f"       映射器精度已由合成自检确立(恢复采样时刻 RMS 0.001 ms vs 原始 4.77 ms)。")
    print(f"       ⇒ 门禁只查 fit_clock 的 ok 标志(|ppm|<5000 且 延迟p95<50ms): "
          f"实测 {sum(r['ok'] for r in rows)}/{len(rows)} 通过\n")

    # ================= C 起始状态 =================
    print("  【C 起始状态一致性】预移动后的实测偏差")
    dev_all = np.array([R[k]["premove"]["dev_deg"] for k in BARE])
    parked = np.array([R[k]["premove"]["q_parked"] for k in BARE])
    print(f"  {'关节':>5}{'偏差最大(°)':>13}{'偏差 spread(°)':>16}"
          f"{'停放位姿 spread(°)':>19}")
    for j in range(6):
        print(f"  {nm[j]:>5}{dev_all[:, j].max():>13.3f}"
              f"{np.ptp(dev_all[:, j]):>16.3f}"
              f"{np.degrees(np.ptp(parked[:, j])):>19.3f}")
    print(f"    settle_vmax 实测 = {[R[k]['premove']['settle_vmax_dps'] for k in BARE]}"
          f" °/s")
    print("  **建议门槛**:")
    print(f"    C1 预移动后偏差 ≤ 该关节「指令超出可达范围的量」+ 1.0°"
          f"  (沿用 B′ 判据, 非固定值)")
    print(f"       且末段速度 < 0.5°/s  (实测全为 "
          f"{max(R[k]['premove']['settle_vmax_dps'] for k in BARE):.1f} °/s)")
    pmax = float(np.max(np.ptp(dev_all, 0)))
    print(f"    C4 B± 与对应 A 的起点差 ≤ **{max(0.2, 2*pmax):.2f}°**")
    print(f"       依据 = **预移动后**偏差的最大 spread {pmax:.3f}° (j1), 取 2× 并设 0.2° 地板。")
    print(f"       初版误用**停放位姿** spread(j1 达 5.09°)得 10.19° —— 那正是预移动")
    print(f"       要吸收掉的量, 不能当起点一致性门槛(2026-07-29 修)。")
    print("    C2 慢动相位 / C3 进场方向: **C 特有, 现有 B′ 数据无对应量**")
    print("       ⇒ 只能在 C pilot 里首次实测, 本轮无法定阈值\n")

    # ================= F 物理停止门 =================
    print("  【F 物理停止门】gyro 增量包络（A 臂之间的差, 逐轴 rad/s）")
    g = {k: np.array(R[k]["imu100"], float)[:, 5:8] for k in BARE}
    pairs = [("A1", "A2"), ("A3", "A4"), ("A5", "A6")]
    print(f"  {'对':>8}{'轴0 p99.9':>12}{'轴1 p99.9':>12}{'轴2 p99.9':>12}{'最大':>10}")
    inc = []
    for a_, b_ in pairs:
        n = min(len(g[a_]), len(g[b_]))
        d = np.abs(g[b_][:n] - g[a_][:n])
        q = np.percentile(d, 99.9, 0)
        inc.append(q)
        print(f"  {a_+'−'+b_:>8}{q[0]:>12.4f}{q[1]:>12.4f}{q[2]:>12.4f}"
              f"{d.max():>10.4f}")
    inc = np.array(inc)
    base = inc.max(0)
    print(f"\n    A 对 A 的 gyro 增量 p99.9 上界 = {np.round(base, 4).tolist()} rad/s")
    print(f"  **建议门槛**: B 相对同位姿 A 的 gyro 增量 p99.9 "
          f"≤ **3×** 该基线 = {np.round(3*base, 4).tolist()} rad/s")
    print(f"    (3× 而非 2×: A−A 基线只含复放变异, B 还合法地含探针响应本身)")
    print(f"    绝对护栏 / 异常声音 / 接触 / IMU stale / 时钟异常: 二值条件, 无需阈值\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gateBCF_C.json").write_text(json.dumps({
        "per_arm": rows,
        "B1_miss_max": 0.02, "B2_back_dup": 0,
        "B3_max_gap_ms": float(max(2 * NOM_MS, 2 * mx("gap"))),
        "B4_ppm_abs_max": float(max(500, 2 * max(abs(r["ppm"]) for r in rows))),
        "B5_lat95_ms": float(2 * mx("lat95")),
        "B5_send_span95_ms": float(2 * mx("span95")),
        "B6_latency_rms_ms": float(mx("res_rms")),
        "B6_clock_ok_required": True,
        "C4_start_diff_deg": float(max(0.2, 2 * float(np.max(np.ptp(dev_all, 0))))),
        "C_premove_dev_max_deg": dev_all.max(0).tolist(),
        "C_parked_spread_deg": np.degrees(np.ptp(parked, 0)).tolist(),
        "F_gyro_inc_base": base.tolist(), "F_gyro_inc_gate": (3 * base).tolist(),
        "undetermined": ["C2 慢动相位一致性", "C3 进场方向一致性"],
    }, indent=1, ensure_ascii=False))
    print("→ outputs/probe_c/gateBCF_C.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
