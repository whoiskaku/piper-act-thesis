#!/usr/bin/env bash
# 正式基线：裸 ACT「不开 TE」(A) vs「开 TE」(B)，两边都开 CUDA graph（否则开 TE 时真机只有约 13 Hz）。
# ABBA 交替分块抵消会话漂移：A B B A A B B A，每块 5 条（一个进程），每个条件共 20 条。
# 抓放：每块按 1→5 号标记位置依次摆方块（各位置在两个条件下各用 4 次）；插入：固定起始位置。
# 第一次出现的数据集新建，之后自动加 --control.resume=true；出错后可从第 N 块接着跑。
#
# 用法（conda activate lerobot 后）：
#   bash scripts/run_baseline.sh pickcube        # 从第 1 块开始
#   bash scripts/run_baseline.sh pickcube 4      # 从第 4 块接着跑
#   DRY=1 bash scripts/run_baseline.sh insert    # 只打印命令，不连机器
set -eu
cd ~/claude-test-2/lerobot-piper
TASK=${1:?用法: $0 pickcube|insert [起始块号]}
START=${2:-1}
case $TASK in
  pickcube) DESC="pick the cube and place it in the box"; SETUP="每条按 1→5 号标记位置依次摆方块" ;;
  insert)   DESC="insert it in the box"; SETUP="每条把销子摆在固定起始位置" ;;
  *) echo "任务只能是 pickcube 或 insert"; exit 1 ;;
esac
CKPT=outputs/train/2026-10-17/ACT_${TASK}_noimu_seed42/checkpoints/100000/pretrained_model
LR=$HOME/.cache/huggingface/lerobot/enolab-guo
unset REFINE_CKPT BCORR_CKPT

i=0
for cond in A B B A A B B A; do
  i=$((i + 1))
  [ "$i" -lt "$START" ] && continue
  if [ "$cond" = A ]; then
    repo=eval_${TASK}_base_noTE; name="不开 TE"; extra=()
  else
    repo=eval_${TASK}_base_TE; name="开 TE"
    extra=(--control.policy.temporal_ensemble_coeff=0.01 --control.policy.n_action_steps=1)
  fi
  [ -d "$LR/$repo" ] && extra+=(--control.resume=true)
  cmd=(python lerobot/scripts/control_robot.py
       --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
       --control.type=record --control.fps=20 "--control.single_task=$DESC"
       --control.repo_id=enolab-guo/$repo --control.num_episodes=5 --control.warmup_time_s=5
       --control.episode_time_s=25 --control.reset_time_s=15
       --control.push_to_hub=false --control.device=cuda --control.use_amp=false
       --control.policy.path=$CKPT --control.policy.cuda_graph=true "${extra[@]}")
  echo "=== 第 $i/8 块：$name（5 条，$SETUP）→ $repo"
  if [ "${DRY:-0}" = 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    read -rp "摆好后按回车开始（Ctrl+C 退出，之后用 '$0 $TASK $i' 从这一块接着跑）…"
    "${cmd[@]}"
  fi
done
echo "=== $TASK 完成：不开 TE、开 TE 各 20 条"
