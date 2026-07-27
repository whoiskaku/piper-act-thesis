#!/usr/bin/env python3
"""实验版本冻结 / 校验（2026-07-28，B′ ep0 4 mrad pilot）。

冻结：把 commit hash、工作树是否干净、关键源文件哈希、探针序列哈希（outputs/ 被
gitignore，只能靠哈希）、配置与环境版本写进 reports/FREEZE_<tag>.json。

校验：**每条复放前**跑一次 `--check`，任何一项与冻结时不符即停机 ——
六条复放期间代码或序列漂移会让 A1/A2 与 B± 不再可比，整批数据作废。

用法:
  python sim/freeze_manifest.py --tag b2_ep0_pilot           # 冻结
  python sim/freeze_manifest.py --tag b2_ep0_pilot --check   # 校验
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REPORTS = ROOT / "reports"

# 本实验链路上的源文件 —— 任一变动都会改变复放或判读
SOURCES = [
    "sim/analyze_probe_b2.py",
    "sim/make_probe_b2.py",
    "sim/probe_b2_gate.py",
    "sim/replay_probe.py",
    "sim/freeze_manifest.py",
    "lerobot/common/robot_devices/motors/piper.py",
    "lerobot/common/robot_devices/robots/piper.py",
    "lerobot/common/robot_devices/robots/configs.py",
    "scripts/tsnd_real/tsnd151_reader.py",
    "lerobot/common/policies/act/b_runtime.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as e:
        return f"<git failed: {e}>"


def build(ep, amp):
    src = {}
    for rel in SOURCES:
        p = ROOT / rel
        src[rel] = sha(p) if p.exists() else None
    seqs = {p.name: sha(p) for p in
            sorted((ROOT / "outputs/probe_b2").glob(f"seq_ep{ep}_*.npz"))}
    wj = ROOT / f"outputs/probe_b2/windows_ep{ep}.json"
    import numpy as np
    return {
        "tag": None, "ep": ep, "amp": amp,
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": git("status", "--porcelain") != "",
                "describe": git("log", "-1", "--format=%h %ad %s", "--date=short")},
        "sources": src,
        "sequences": seqs,
        "windows": json.loads(wj.read_text()) if wj.exists() else None,
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b2_ep0_pilot")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--amp", default="4mrad")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = REPORTS / f"FREEZE_{a.tag}.json"
    cur = build(a.ep, a.amp)
    cur["tag"] = a.tag

    if not a.check:
        REPORTS.mkdir(exist_ok=True)
        out.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
        print(f"=== 冻结 {a.tag} → {out.relative_to(REPO)} ===")
        print(f"  commit  {cur['git']['describe']}")
        print(f"  分支    {cur['git']['branch']}   工作树"
              f"{'  ⚠ 有未提交改动' if cur['git']['dirty'] else 'clean ✓'}")
        print(f"  源文件  {len(cur['sources'])} 个已记哈希")
        print(f"  探针序列 {len(cur['sequences'])} 个 (outputs/ 被 gitignore, 只能靠哈希)")
        wj = cur["windows"]
        if wj:
            print(f"  注入窗  {wj.get('windows_by_joint')}  跟踪源 {wj.get('src_rollout')}")
        print(f"\n  每条复放前跑: python sim/freeze_manifest.py --tag {a.tag} --check")
        return 0

    if not out.exists():
        print(f"✗ 找不到冻结记录 {out} —— 先冻结再上机"); return 1
    ref = json.loads(out.read_text())
    bad = []
    if ref["git"]["commit"] != cur["git"]["commit"]:
        bad.append(f"commit 变了: {ref['git']['commit'][:9]} → {cur['git']['commit'][:9]}")
    if cur["git"]["dirty"]:
        bad.append("工作树有未提交改动")
    for k, v in ref["sources"].items():
        if cur["sources"].get(k) != v:
            bad.append(f"源文件变了: {k}")
    for k, v in ref["sequences"].items():
        if cur["sequences"].get(k) != v:
            bad.append(f"探针序列变了: {k}")
    for k in set(cur["sequences"]) - set(ref["sequences"]):
        bad.append(f"多出探针序列: {k}")
    if ref.get("windows") != cur.get("windows"):
        bad.append("注入窗定义变了")
    print(f"=== 冻结校验 {a.tag} (冻结于 {ref['git']['describe']}) ===")
    if bad:
        for b in bad:
            print(f"  ✗ {b}")
        print("❌ 版本已漂移 —— 停机。六条复放期间代码/序列变动会使 A 与 B 不可比。")
        return 1
    print(f"  ✓ commit / {len(ref['sources'])} 源文件 / "
          f"{len(ref['sequences'])} 探针序列 / 注入窗  全部一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
