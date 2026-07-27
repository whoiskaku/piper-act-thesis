#!/usr/bin/env python3
"""重建全部标签(2026-07-22): ideal 改逐 chunk 渲染, 与部署一致。四个数据集。"""
import sys
import build_dagger_labels as B
SETS = [("enolab-guo/eval_pickplace2_100k", "pp2-deploy", 10),
        ("enolab-guo/eval_insert_100k", "ins-deploy", 10),
        ("enolab-guo/eval_pp2_fork_100k", "pp2f-deploy", 7),
        ("enolab-guo/eval_pp2_fork_60k", "pp2f60-deploy", 3)]
for repo, tag, n in SETS:
    B.DATASETS = [(repo, tag)]
    sys.argv = ["build_dagger_labels", "--eps", str(n)]
    print(f"\n=== {tag} ({n} 集) ===", flush=True)
    B.main()
