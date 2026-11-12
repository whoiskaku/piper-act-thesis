#!/usr/bin/env python3
"""从 LeRobot v2.1 数据集物理剔除 observation.imu_real 列, 生成无 IMU 副本。

动机: 上游 lerobot 会把所有 observation.* 列自动纳入 input_features;
imu_real 训练时被模型忽略但部署时(无 IMU 的机器人栈)会缺 key。
做法: parquet 去列(pyarrow), info.json/episodes_stats.jsonl 去条目,
episodes/tasks 原样, 视频硬链接(省磁盘, 失败回退复制)。

用法: python scripts/strip_imu_dataset.py --src enolab-guo/pickplace2_train --dst enolab-guo/pickplace2_noimu
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

LR = Path.home() / ".cache/huggingface/lerobot"
DROP = "observation.imu_real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = LR / args.src, LR / args.dst
    if not src.exists():
        raise SystemExit(f"源不存在: {src}")
    if dst.exists():
        raise SystemExit(f"目标已存在: {dst}")

    # --- parquet: 去列 ---
    n_ep = 0
    for p in sorted(src.glob("data/chunk-*/episode_*.parquet")):
        t = pq.read_table(p)
        if DROP in t.column_names:
            t = t.drop([DROP])
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        n_ep += 1

    # --- videos: 硬链接 ---
    n_v = 0
    for p in sorted(src.glob("videos/chunk-*/*/episode_*.mp4")):
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, out)
        except OSError:
            shutil.copy2(p, out)
        n_v += 1

    # --- meta ---
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "meta/info.json").read_text())
    info["features"].pop(DROP, None)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    with (dst / "meta/episodes_stats.jsonl").open("w") as f:
        for line in (src / "meta/episodes_stats.jsonl").read_text().splitlines():
            d = json.loads(line)
            d["stats"].pop(DROP, None)
            f.write(json.dumps(d) + "\n")
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(src / f"meta/{name}", dst / f"meta/{name}")

    print(f"完成: {dst}")
    print(f"  parquet {n_ep} 集(已去 {DROP}), 视频 {n_v} 个(硬链接), meta 已更新")
    print(f"  features: {list(info['features'].keys())}")


if __name__ == "__main__":
    main()
