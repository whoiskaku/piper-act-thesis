#!/usr/bin/env python3
"""采完一档就核对 —— 把 RUNBOOK 的"跑的时候盯什么"变成可执行检查。

用法:  python scripts/check_probe_logs.py P1
       python scripts/check_probe_logs.py            # 核对 outputs/ab4 下所有档

退出码 0 = 全过; 1 = 有项不过(不要接着采下一档)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "_bc_probe", ROOT / "lerobot/common/policies/act/b_correction_deploy.py")
try:
    _bc = importlib.util.module_from_spec(_spec)
    sys.modules["_bc_probe"] = _bc
    _spec.loader.exec_module(_bc)
except Exception:                                   # noqa: BLE001
    sys.path.insert(0, str(ROOT))
    from lerobot.common.policies.act import b_correction_deploy as _bc  # type: ignore

# 预注册 §8 冻结的采集计划
PLAN = {
    "P0": {"mode": "off",   "mrad": None, "seeds": None, "n": 5},
    "P1": {"mode": "probe", "mrad": 1.5,
           "seeds": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007], "n": 8},
    "P2": {"mode": "probe", "mrad": 1.5,
           "seeds": [2000, 2000, 2001, 2001, 2002, 2002], "n": 6},
    "P3": {"mode": "probe", "mrad": 3.0, "seeds": [1000, 1001, 1002, 1003], "n": 4},
}
MIN_FRAMES = 550


def load(tier: str, root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob(f"{tier}_ep*.json"),
                    key=lambda p: int(p.stem.split("ep")[-1])):
        d = json.loads(f.read_text())
        out.append({"file": f.name, **d.get("summary", {})})
    return out


def check(tier: str, root: Path) -> bool:
    plan = PLAN[tier]
    eps = load(tier, root)
    print(f"\n=== {tier} ===")
    if not eps:
        print(f"  ✗ 没找到 {root}/{tier}_ep*.json")
        return False

    ok = True

    def bad(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  ✗ {msg}")

    if len(eps) != plan["n"]:
        bad(f"集数 {len(eps)} ≠ 计划 {plan['n']}")
    else:
        print(f"  ✓ 集数 {len(eps)}")

    modes = {e.get("mode") for e in eps}
    if modes != {plan["mode"]}:
        bad(f"mode={modes}, 应为 {{'{plan['mode']}'}}")
    else:
        print(f"  ✓ mode={plan['mode']}")

    if any(e.get("use_pi") for e in eps):
        bad("use_pi=True —— probe 档不该经过 π")
    if any(e.get("use_ff") for e in eps):
        bad("use_ff=True —— 本批不开 FF")
    if any(float(e.get("render_ms_max") or 0) > 0 for e in eps):
        bad("render_ms_max>0 —— 误开了 ideal 渲染")

    short = [(e["file"], e.get("frames")) for e in eps
             if int(e.get("frames") or 0) < MIN_FRAMES]
    if short:
        bad(f"帧数 < {MIN_FRAMES}: {short}  ⇒ 按 A 类技术故障整档重开")
    else:
        print(f"  ✓ 帧数全部 ≥ {MIN_FRAMES} "
              f"(min={min(int(e.get('frames') or 0) for e in eps)})")

    if any(int(e.get("stale") or 0) for e in eps):
        bad(f"stale≠0: {[(e['file'], e.get('stale')) for e in eps if e.get('stale')]}"
            "  ⇒ A 类技术故障, 整档重开")
    if any(int(e.get("nan") or 0) for e in eps):
        bad("nan≠0")

    if plan["mode"] == "off":
        print("  ✓ P0 无 probe 项可查")
        return ok

    # ---- probe 专项 ----
    if any(int(e.get("probe_overrun") or 0) for e in eps):
        bad("probe_overrun≠0 —— 跑的集数超过 seed 表")
    else:
        print("  ✓ probe_overrun=0")
    if any(int(e.get("probe_wrapped") or 0) for e in eps):
        bad("probe_wrapped≠0 —— 集长超出波形, 集内有重复激励")
    else:
        print("  ✓ probe_wrapped=0")

    mrads = {float(e.get("probe_mrad")) for e in eps if e.get("probe_mrad") is not None}
    if mrads != {plan["mrad"]}:
        bad(f"probe_mrad={mrads}, 应为 {{{plan['mrad']}}}")
    else:
        print(f"  ✓ probe_mrad={plan['mrad']}")

    clip = [float(e.get("probe_clip_pct") or 0) for e in eps]
    lim = 0.02 if plan["mrad"] <= 1.5 else 0.5
    if max(clip) > lim:
        bad(f"probe_clip_pct max={max(clip):.3f}% > {lim}%")
    else:
        print(f"  ✓ probe_clip_pct max={max(clip):.3f}%")

    got_seeds = [e.get("probe_seed") for e in eps]
    if got_seeds != plan["seeds"]:
        bad(f"实际 seed 序列 {got_seeds}\n      ≠ 计划 {plan['seeds']}")
    else:
        print(f"  ✓ seed 序列与计划一致")

    # 波形 hash: 落盘值必须等于该 seed 应有的 hash, 且分组结构正确
    print("  --- 逐集波形 hash ---")
    bad_hash = False
    for e in eps:
        s = e.get("probe_seed")
        want = _bc.probe_hash(int(s)) if s is not None else None
        got = e.get("probe_hash")
        flag = "✓" if got == want else "✗"
        if got != want:
            bad_hash = True
        print(f"    {e['file']:<16} seed={s} hash={got} {flag}")
    if bad_hash:
        bad("落盘 hash 与该 seed 应有的波形不符")

    uniq = len({e.get("probe_hash") for e in eps})
    want_uniq = len(set(plan["seeds"]))
    if uniq != want_uniq:
        bad(f"不同波形 {uniq} 种, 应为 {want_uniq} 种")
    else:
        print(f"  ✓ 不同波形 {uniq} 种(= 不同 seed 数)")

    if tier == "P2":
        pairs = [(eps[i].get("probe_hash"), eps[i + 1].get("probe_hash"))
                 for i in range(0, len(eps) - 1, 2)]
        if all(a == b for a, b in pairs):
            print("  ✓ P2 成对: ep0=ep1, ep2=ep3, ep4=ep5")
        else:
            bad(f"P2 未成对: {pairs}")

    if tier == "P3":
        p1 = load("P1", root)
        if p1:
            want = [e.get("probe_hash") for e in p1[:4]]
            got = [e.get("probe_hash") for e in eps]
            if want == got:
                print("  ✓ P3 波形与 P1 前 4 集相同(线性检验前提成立)")
            else:
                bad("P3 波形 ≠ P1 前 4 集 —— 幅值变化与激励变化混在一起, 线性检验失效")
    return ok


def main() -> int:
    root = ROOT / "outputs/ab4"
    tiers = [sys.argv[1]] if len(sys.argv) > 1 else \
        [t for t in PLAN if list(root.glob(f"{t}_ep*.json"))]
    if not tiers:
        print(f"没找到任何 {root}/P*_ep*.json")
        return 1
    allok = all(check(t, root) for t in tiers)
    print("\n" + ("全部通过 ✓" if allok else "有项不过 ✗ —— 不要接着采下一档"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
