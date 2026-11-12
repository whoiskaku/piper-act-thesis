#!/usr/bin/env python3
"""把某 LeRobot v2.1 数据集的一个 episode 子集, 重新编号为连续 0..N-1, 写成新数据集。

动机: lerobot 的 `--dataset.episodes` 子集选择在编号有跳号/超过子集长度时是坏的
  (episode_data_index 按位置建 size=N, 但 episode_index 列是全局值 → 基类
   _get_query_indices 用全局编号索引位置表 → 越界崩溃 / 静默错位)。
对策: 物理重建连续编号的数据集, 用 episodes=None 整集训练(标准路径)绕开。

做法(零风险保 schema):
  · parquet: 用 pyarrow 只替换 episode_index(→新号) 与 index(→全局连续) 两列, 其余原样拷;
  · 视频: 直接复制改名 episode_{new:06d}.mp4;
  · meta: info.json 改 totals; episodes/episodes_stats.jsonl 重映射编号; tasks.jsonl 原样。

成功子集默认从 outputs/review/<src>.csv 读 pass 行(review_demos.py 产出), 或 --episodes 覆盖。

用法:
  python scripts/rebuild_clean_dataset.py --src enolab-guo/pickplace_clean50 --dst enolab-guo/pickplace_clean
  python scripts/rebuild_clean_dataset.py --src ... --dst ... --episodes 0,1,2,5,8
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

LR = Path.home() / ".cache/huggingface/lerobot"
REPO = Path(__file__).resolve().parents[1]


def read_success_from_csv(src_repo: str) -> list[int]:
    csv_path = REPO / "outputs/review" / (src_repo.replace("/", "__") + ".csv")
    if not csv_path.exists():
        raise SystemExit(f"找不到评审 CSV: {csv_path}; 请用 --episodes 显式指定。")
    passes = []
    for r in csv.reader(csv_path.open()):
        if len(r) >= 2 and r[1].strip() == "pass":
            passes.append(int(r[0]))
    return sorted(passes)


def _video_nframes(path: Path) -> int:
    """精确帧数(count_packets); 用于把 parquet 裁到视频长度, 避免末尾时间戳超出视频触发容差错。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return int(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 repo_id, 如 enolab-guo/pickplace_clean50")
    ap.add_argument("--dst", required=True, help="目标 repo_id, 如 enolab-guo/pickplace_clean")
    ap.add_argument("--episodes", default=None, help="逗号分隔的成功 episode 全局编号; 缺省从 review CSV 读 pass")
    args = ap.parse_args()

    src = LR / args.src
    dst = LR / args.dst
    if not src.exists():
        raise SystemExit(f"源不存在: {src}")
    if dst.exists():
        raise SystemExit(f"目标已存在, 先删除或换名: {dst}")

    eps = ([int(x) for x in args.episodes.split(",")] if args.episodes
           else read_success_from_csv(args.src))
    eps = sorted(set(eps))
    print(f"重建 {args.src} → {args.dst}: {len(eps)} 条成功集 重新编号 0..{len(eps)-1}")
    print(f"  原全局编号: {eps}")

    info = json.loads((src / "meta/info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    # 旧 episodes_stats: old_ep -> stats dict
    old_stats = {}
    for line in (src / "meta/episodes_stats.jsonl").read_text().splitlines():
        d = json.loads(line)
        old_stats[int(d["episode_index"])] = d["stats"]

    # 旧 episodes.jsonl: old_ep -> 行(含 tasks/length); 继承任务名, 不再硬编码
    old_eps_meta = {}
    for line in (src / "meta/episodes.jsonl").read_text().splitlines():
        d = json.loads(line)
        old_eps_meta[int(d["episode_index"])] = d

    (dst / "meta").mkdir(parents=True)
    new_episodes, new_ep_stats = [], []
    running = 0  # 全局帧计数 → index 列
    for new_idx, old in enumerate(eps):
        chunk = new_idx // chunks_size
        # --- parquet: 替换 episode_index / index 两列, 其余原样 ---
        src_pq = src / data_tmpl.format(episode_chunk=old // chunks_size, episode_index=old)
        t = pq.read_table(src_pq)
        n = t.num_rows
        # 裁 parquet 尾巴到「双相机视频最短帧数」: 编码跳过坏帧会让视频比 parquet 短 1-2 帧,
        # 末尾时间戳超出视频 → lerobot 训练触发 timestamp 容差错。丢的是末尾静止帧, 无害。
        vframes = [_video_nframes(src / video_tmpl.format(
            episode_chunk=old // chunks_size, video_key=vk, episode_index=old)) for vk in video_keys]
        k = min([n] + vframes)
        if k < n:
            print(f"  ep{old}: parquet {n} → 裁到 {k} (视频最短 {min(vframes)}, 丢末尾 {n-k} 帧)")
            t = t.slice(0, k)
            n = k
        ep_type = t.schema.field("episode_index").type
        idx_type = t.schema.field("index").type
        t = t.set_column(t.schema.get_field_index("episode_index"),
                         "episode_index", pa.array([new_idx] * n, type=ep_type))
        t = t.set_column(t.schema.get_field_index("index"),
                         "index", pa.array(list(range(running, running + n)), type=idx_type))
        dst_pq = dst / data_tmpl.format(episode_chunk=chunk, episode_index=new_idx)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst_pq)
        running += n

        # --- videos: 复制改名 ---
        for vk in video_keys:
            s = src / video_tmpl.format(episode_chunk=old // chunks_size, video_key=vk, episode_index=old)
            d = dst / video_tmpl.format(episode_chunk=chunk, video_key=vk, episode_index=new_idx)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

        # --- meta 行 ---
        new_episodes.append({"episode_index": new_idx,
                             "tasks": old_eps_meta[old]["tasks"], "length": n})
        if old in old_stats:
            new_ep_stats.append({"episode_index": new_idx, "stats": old_stats[old]})
        else:
            print(f"  ⚠ 旧集 {old} 无 episodes_stats, 跳过其 stats 行")

    # --- meta 文件 ---
    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")
    with (dst / "meta/episodes.jsonl").open("w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")
    with (dst / "meta/episodes_stats.jsonl").open("w") as f:
        for e in new_ep_stats:
            f.write(json.dumps(e) + "\n")
    info["total_episodes"] = len(eps)
    info["total_frames"] = running
    info["total_videos"] = len(eps) * len(video_keys)
    info["total_chunks"] = (len(eps) - 1) // chunks_size + 1
    info["splits"] = {"train": f"0:{len(eps)}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    print(f"完成: {dst}")
    print(f"  total_episodes={len(eps)}  total_frames={running}  video_keys={video_keys}")
    print(f"  训练用: --dataset.repo_id={args.dst}  (不要再加 --dataset.episodes, 整集即 {len(eps)} 条成功集)")


if __name__ == "__main__":
    main()
