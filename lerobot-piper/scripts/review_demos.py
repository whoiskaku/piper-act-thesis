#!/usr/bin/env python3
"""逐条回放某 LeRobot 数据集的每条 episode 视频, 人工标记 成功/失败。

在带 GUI 的终端里运行(用 ffplay 弹窗播放)。结果写 CSV, 可中断续评。
用于 Phase1 干净基线: 剔除抓取/放置失败的 demo, 只留成功集训 ACT。

用法:
  python scripts/review_demos.py enolab-guo/pickplace_clean50

ffplay 窗口内热键: 空格=暂停/继续  ←/→=快退/进10s  s=逐帧  q 或关窗=结束本条播放
每条播完在终端:    Enter 或 p=成功   f=失败   r=重看   q=存盘退出(下次续评)
结束后打印 成功/失败 编号 + 直接可贴的 --dataset.episodes 列表。
"""
from __future__ import annotations
import csv
import subprocess
import sys
from pathlib import Path


def _summary(verdict):
    fails = sorted(i for i, v in verdict.items() if v == "fail")
    passes = sorted(i for i, v in verdict.items() if v == "pass")
    print(f"\n成功 {len(passes)} 条: {passes}")
    print(f"失败 {len(fails)} 条: {fails}")
    print(f"\n训练用成功子集(贴给训练命令):\n  --dataset.episodes=\"{passes}\"")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/review_demos.py <repo_id>  例如 enolab-guo/pickplace_clean50")
        sys.exit(1)
    repo = sys.argv[1]
    root = Path.home() / ".cache/huggingface/lerobot" / repo
    if not root.exists():
        print(f"找不到数据集: {root}")
        sys.exit(1)

    # 自动找图像视频键(可能多相机, 取第一个)
    keys = sorted({p.parent.name for p in root.glob("videos/chunk-*/*/episode_*.mp4")})
    if not keys:
        print(f"没找到视频: {root}/videos/...")
        sys.exit(1)
    key = keys[0]
    if len(keys) > 1:
        print(f"多相机 {keys}, 用第一个: {key}")
    vids = sorted(root.glob(f"videos/chunk-*/{key}/episode_*.mp4"),
                  key=lambda p: int(p.stem.split("_")[1]))
    n = len(vids)

    out = Path(__file__).resolve().parents[1] / "outputs/review"
    out.mkdir(parents=True, exist_ok=True)
    log = out / (repo.replace("/", "__") + ".csv")

    verdict, note = {}, {}
    if log.exists():
        for r in csv.reader(log.open()):
            if len(r) >= 2:
                verdict[int(r[0])] = r[1]
                if len(r) >= 3:
                    note[int(r[0])] = r[2]

    def save():
        with log.open("w", newline="") as f:
            w = csv.writer(f)
            for i in sorted(verdict):
                w.writerow([i, verdict[i], note.get(i, "")])

    def play(p):
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet",
                        "-x", "800", "-y", "600", "-window_title", p.name, str(p)])

    print(f"数据集 {repo}: {n} 条; 已评 {len(verdict)} 条。相机键={key}\n"
          f"(按任务标准判定; 插入任务: 成功=电池完全插到位, 失败=没夹到/对不准/没插到底)\n")
    done = sum(1 for v in vids if int(v.stem.split("_")[1]) in verdict)
    for v in vids:
        idx = int(v.stem.split("_")[1])
        if idx in verdict:
            continue
        while True:
            play(v)
            ans = input(f"[{done+1}/{n}] ep{idx:03d}  Enter/p=成功  f=失败  r=重看  q=存盘退出: ").strip().lower()
            if ans == "r":
                continue
            if ans == "q":
                save()
                print(f"\n已存盘 {log}; 下次运行从 ep{idx:03d} 继续。")
                _summary(verdict)
                return
            verdict[idx] = "fail" if ans in ("f", "fail") else "pass"
            break
        done += 1
        save()
    print(f"\n全部 {n} 条评完。结果存 {log}")
    _summary(verdict)


if __name__ == "__main__":
    main()
