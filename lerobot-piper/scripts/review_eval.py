#!/usr/bin/env python3
"""逐条回放评测数据集, 标 分阶段成功 + 备注。ACT 口径分阶段成功率 + "为什么慢"。

阶段(pick-place): 0=没抓到  1=抓住未提起  2=提起未放好  3=放到目标位(成功)
备注自由填: 如 超时 / 慢 / 掉落 / 重试 / 完成后空动 ...

在带 GUI 的终端运行(ffplay 弹窗)。结果写 CSV(可中断续评)。
用法: python scripts/review_eval.py enolab-guo/eval_phase1_noimu_no_te

ffplay 窗口内: 空格暂停  ←/→快退进10s  s逐帧  q关窗
每条终端输入: "<阶段0-3> [备注]"  例:  3   |   3 慢   |   1 掉落   |   0 超时未抓到
              r=重看本条   q=存盘退出(下次续)
结束打印: 分阶段成功率 + 成功率(#阶段3/N) + 备注汇总。
"""
from __future__ import annotations
import csv
import subprocess
import sys
from pathlib import Path

STAGES = {0: "没抓到", 1: "抓住未提起", 2: "提起未放好", 3: "放到目标(成功)"}


def summarize(verdict):
    if not verdict:
        return
    stages = [s for s, _ in verdict.values()]
    n = len(stages)
    print(f"\n{'='*52}\n分阶段成功率 (n={n})\n{'='*52}")
    for thr, lab in [(1, "抓住"), (2, "提起"), (3, "放到目标=成功")]:
        c = sum(1 for s in stages if s >= thr)
        print(f"  ≥阶段{thr} ({lab:<10}): {c:>2}/{n} = {100*c/n:>3.0f}%")
    notes = [(ep, s, note) for ep, (s, note) in sorted(verdict.items()) if note]
    if notes:
        print(f"\n备注 ({len(notes)} 条):")
        for ep, s, note in notes:
            print(f"   ep{ep:03d}  阶段{s} | {note}")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/review_eval.py <repo_id>")
        sys.exit(1)
    repo = sys.argv[1]
    root = Path.home() / ".cache/huggingface/lerobot" / repo
    keys = sorted({p.parent.name for p in root.glob("videos/chunk-*/*/episode_*.mp4")})
    if not keys:
        print(f"没找到视频: {root}/videos/...")
        sys.exit(1)
    key = keys[0]
    vids = sorted(root.glob(f"videos/chunk-*/{key}/episode_*.mp4"),
                  key=lambda p: int(p.stem.split("_")[1]))

    out = Path(__file__).resolve().parents[1] / "outputs/review"
    out.mkdir(parents=True, exist_ok=True)
    log = out / (repo.replace("/", "__") + "_eval.csv")
    verdict = {}
    if log.exists():
        for r in csv.reader(log.open()):
            if len(r) >= 2:
                verdict[int(r[0])] = (int(r[1]), r[2] if len(r) > 2 else "")

    def save():
        with log.open("w", newline="") as f:
            w = csv.writer(f)
            for i in sorted(verdict):
                w.writerow([i, verdict[i][0], verdict[i][1]])

    def play(p):
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet",
                        "-x", "800", "-y", "600", "-window_title", p.name, str(p)])

    print(f"{repo}: {len(vids)} 条; 已评 {len(verdict)}。相机键={key}\n"
          f"阶段: 0=没抓到  1=抓住未提起  2=提起未放好  3=放到目标(成功)\n")
    for v in vids:
        idx = int(v.stem.split("_")[1])
        if idx in verdict:
            continue
        while True:
            play(v)
            ans = input(f"ep{idx:03d}  <阶段0-3> [备注]  (r重看/q退出): ").strip()
            if ans.lower() == "r":
                continue
            if ans.lower() == "q":
                save(); print(f"已存盘 {log}"); summarize(verdict); return
            parts = ans.split(maxsplit=1)
            try:
                stage = int(parts[0])
                assert stage in STAGES
            except (ValueError, AssertionError):
                print("  阶段需 0-3, 重输"); continue
            verdict[idx] = (stage, parts[1] if len(parts) > 1 else "")
            break
        save()
    print(f"\n全部评完。结果 {log}")
    summarize(verdict)


if __name__ == "__main__":
    main()
