"""B4 的闭环分析（方案 `reports/PLAN_b4.md` §5.1、§5.5；实现计划任务 5、11）。

  python -u sim/b4_closed.py check    # 第一步：闭环不提前（raw）vs 闭环 Shift（shift），各 10 条，按位置配对
  python -u sim/b4_closed.py final    # 第五步：闭环 Shift vs 闭环 Shift＋网络（判定），原始 ACT 只报告；各 20 条

先用 scripts/review_eval.py 给两个数据集逐条标阶段（0–3，3 = 方块中心在框内）。
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_common as C  # noqa: E402
import b4_positions as P  # noqa: E402
import b4_stats as ST  # noqa: E402
import shift_replay as S  # noqa: E402

REPLAY_SHIFT = -66.8          # 回放实测：提前 60 ms 的关节偏差（相对原始指令）变化 %（reports/PLAN_shift_replay.md §6）


def check_results(raw: dict, shift: dict, n_required: int = 10) -> dict:
    """raw / shift = {位置: {"metrics": 指标 | None, "stage": 阶段 | None}}。配对 = 同一位置两边都有指标、都标了阶段。
    通过 = 配对正好 n_required 对、成功率严格计数不降、Shift 相对不提前的关节偏差变化（两级自助，组 = 位置）区间上限 < 0。"""
    labels = [k for k in raw if k in shift and raw[k]["metrics"] and shift[k]["metrics"]
              and raw[k]["stage"] is not None and shift[k]["stage"] is not None]
    rel = {k: [ST.rel_change(shift[k]["metrics"]["joint_raw"], raw[k]["metrics"]["joint_raw"])] for k in labels}
    r = ST.hier2(rel) if rel else None
    n_sf, n_fs, ok = ST.strict_count([(raw[k]["stage"] == 3, shift[k]["stage"] == 3) for k in labels])
    return {"n_pairs": len(labels), "joint_raw": r, "strict": {"success_to_fail": n_sf, "fail_to_success": n_fs, "ok": ok},
            "passed": len(labels) == n_required and ok and r is not None and r["ci95"][1] < 0}


def success(stage) -> bool:
    return stage == 3


def final_results(runs: dict) -> dict:
    """runs = {(条件, 位置块 0–3, 序号 0–4): {"metrics", "stage", "tries"} | None}；tries = 同一集号之前各次尝试的记录是否完整。
    配对 = 同一位置块、同一位置的 Shift 与 Shift＋网络，两边都有指标、都标了阶段、且之前没有被中断的尝试才算；
    缺一边或有中断（可能是安全停机）的不进统计（不满 20 对 = 未完成，由人看过再定）；中断、重录次数照报。各指标「网络相对 Shift」→ 两级自助（组 = 测试位置，单位 = 配对）→ judge_closed
    （20 对齐全、成功率严格计数不降、任务时长与策略指令速度护栏）。原始 ACT 只报告相对 Shift、相对网络的关节偏差变化。"""
    ok = lambda e: (e is not None and e.get("metrics") is not None and e.get("stage") is not None   # noqa: E731
                    and all(e.get("tries", ())))
    tries = [t for e in runs.values() if e for t in e.get("tries", ())]

    def rel(a, b, key):
        g = {}
        for pb in range(4):
            for s in range(5):
                x, y = runs.get((a, pb, s)), runs.get((b, pb, s))
                if ok(x) and ok(y) and np.isfinite(x["metrics"][key]) and np.isfinite(y["metrics"][key]):
                    g.setdefault(f"pos{(pb % 2) * 5 + s}", []).append(ST.rel_change(x["metrics"][key], y["metrics"][key]))
        return ST.hier2(g) if g else None

    pairs = [(pb, s) for pb in range(4) for s in range(5) if ok(runs.get(("shift", pb, s))) and ok(runs.get(("net", pb, s)))]
    r = {k: rel("net", "shift", k) for k in ("joint_raw", "path", "speed", "duration", "cmd_speed", "ee_raw")}
    succ = [(success(runs[("shift", pb, s)]["stage"]), success(runs[("net", pb, s)]["stage"])) for pb, s in pairs]
    n_sf, n_fs, good = ST.strict_count(succ)
    return {**r, "n_pairs": len(pairs), "attempts": {"interrupted": tries.count(False), "rerecorded": tries.count(True)}, "strict": {"success_to_fail": n_sf, "fail_to_success": n_fs, "ok": good},
            "verdict": ST.judge_closed(r, succ) if len(pairs) == 20 else f"只报告、未完成（完整对数 {len(pairs)}/20）",
            "raw_vs_shift": rel("raw", "shift", "joint_raw"), "raw_vs_net": rel("raw", "net", "joint_raw")}


def cmd_final() -> int:
    ctx, runs = C.make_ctx(), {}
    for cond in ("shift", "net", "raw"):
        repo = f"eval_pickcube_b4final_{cond}"
        stages = P.review_stages(repo)
        for i in range(P.n_recorded(repo)):
            m = None
            try:
                ep = C.load_closed(repo, i)
                if ep["mode"] != cond:
                    raise ValueError(f"{repo} 第 {i} 集的 B4 模式是 {ep['mode']}，应为 {cond}")
                m = C.closed_metrics(ep, ctx)
            except (OSError, ValueError) as e:
                print(f"  ⚠ {e}")
            tries = sorted((P.LR / repo / "high_rate").glob(f"episode_{i:06d}_b4.try*.npz"))   # 同一集号之前的尝试
            runs[(cond, i // 5, i % 5)] = {"metrics": m, "stage": stages.get(i),
                                           "tries": [bool(np.load(f)["complete"]) for f in tries]}
    out = final_results(runs)
    if out["attempts"]["interrupted"]:
        print(f"  ⚠ 有 {out['attempts']['interrupted']} 次被中断的尝试（*_b4.try*.npz）：那几对不进统计，先查是不是安全停机")
    rec = P.review_stages(P.RECORD_REPO)
    p_fail = sum(s < 3 for s in rec.values()) / max(len(rec), 1)
    out["chance_fail_prob"] = {"p_shift_fail": p_fail, "n_pairs": 20, "prob": ST.chance_fail_prob(p_fail, 20)}
    d = C.OUT / "final"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({**out, "runs": {f"{c}_{pb}_{s}": v for (c, pb, s), v in runs.items()}},
                                               ensure_ascii=False, indent=1))
    for k, name in (("joint_raw", "关节偏差（相对原始策略指令）"), ("path", "路径长度"), ("speed", "关节速度"),
                    ("duration", "任务时长"), ("cmd_speed", "策略指令自身速度"), ("ee_raw", "末端角速度偏差（只报告）")):
        if out[k]:
            print(f"Shift＋网络 相对 Shift 的{name}：{out[k]['effect']:+.1f}% [{out[k]['ci95'][0]:+.1f}, {out[k]['ci95'][1]:+.1f}]")
    s = out["strict"]
    print(f"成功率：成功→失败 {s['success_to_fail']} 对、失败→成功 {s['fail_to_success']} 对；完整配对 {out['n_pairs']}/20；"
          f"按 Shift 失败率 {p_fail:.1%}，偶然判失败的概率 {out['chance_fail_prob']['prob']:.3f}")
    for k, name in (("raw_vs_shift", "Shift"), ("raw_vs_net", "Shift＋网络")):
        if out[k]:
            print(f"（只报告）原始 ACT 相对 {name} 的关节偏差：{out[k]['effect']:+.1f}%")
    print(f"判定：{out['verdict']} → {d / 'summary.json'}")
    return 0


def cmd_check() -> int:
    ctx = C.make_ctx()
    sides, gaps, mism = {}, [], []
    for cond in ("raw", "shift"):
        repo, side = f"eval_pickcube_b4check_{cond}", {}
        stages = P.review_stages(repo)
        for i in range(10):
            m = None
            try:
                ep = C.load_closed(repo, i)
                if ep["mode"] != cond:
                    raise ValueError(f"{repo} 第 {i} 集的 B4 模式是 {ep['mode']}，应为 {cond}")
                m = C.closed_metrics(ep, ctx)
                mism.append(ep["act_mismatch"])
                if cond == "shift":               # 闭环 Shift（部分融合）与回放用的离线 Shift（最终指令提前 60 ms）之差
                    off = S.shifted(ep["a_pi"].astype(np.float32), 0.060)[:, :6].astype(float)
                    gaps.append(ep["cmd"] - off)  # 开头 1 s TE 只有几段预测、差得多，下面分开报
            except (OSError, ValueError) as e:
                print(f"  ⚠ {e}")
            side[f"c{i}"] = {"metrics": m, "stage": stages.get(i)}
        sides[cond] = side
    res = check_results(sides["raw"], sides["shift"])
    act = [0, 1, 2, 4, 5]
    g = np.concatenate([x[C.RR.HOLD:] for x in gaps])[:, act] * 1e3 if gaps else np.zeros((0, 5))
    g0 = np.concatenate([x[:C.RR.HOLD] for x in gaps])[:, act] * 1e3 if gaps else np.zeros((0, 5))
    res.update(shift_vs_offline_mrad={"max": float(np.abs(g).max()) if g.size else None,
                                      "rms": float(np.sqrt(np.mean(g ** 2))) if g.size else None,
                                      "first_1s_max": float(np.abs(g0).max()) if g0.size else None},
               act_mismatch_max=float(max(mism)) if mism else None, sides=sides)
    out = C.OUT / "check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k in sides["raw"]:
        a, b = sides["raw"][k], sides["shift"][k]
        ja = f"{a['metrics']['joint_raw']:.2f}" if a["metrics"] else "—"
        jb = f"{b['metrics']['joint_raw']:.2f}" if b["metrics"] else "—"
        print(f"  {k}：关节偏差 不提前 {ja} / Shift {jb} mrad；阶段 {a['stage']} / {b['stage']}")
    r, s = res["joint_raw"], res["strict"]
    if r:
        print(f"关节偏差（相对原始策略指令）Shift 相对不提前：{r['effect']:+.1f}% [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]"
              f"（回放实测 {REPLAY_SHIFT:+.1f}%）")
    print(f"成功率：成功→失败 {s['success_to_fail']} 对、失败→成功 {s['fail_to_success']} 对；完整配对 {res['n_pairs']}/10")
    if g.size:
        d = res["shift_vs_offline_mrad"]
        print(f"闭环 Shift 与离线 Shift 之差（第 1 s 之后）：最大 {d['max']:.3f}、RMS {d['rms']:.3f} mrad；"
              f"开头 1 s 最大 {d['first_1s_max']:.3f} mrad（TE 只有几段预测）")
    print(f"数据集动作与包装实发指令的最大差 {res['act_mismatch_max']}")
    print("✅ 通过：可以进第二步录制" if res["passed"] else "⛔ 没过：停下，和用户商量（不进第二步，也不自行改 Shift）")
    print(f"→ {out / 'summary.json'}")
    return 0


def main(argv=None) -> int:
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["check"]:
        return cmd_check()
    if argv == ["final"]:
        return cmd_final()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
