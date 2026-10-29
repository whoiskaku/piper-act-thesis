#!/usr/bin/env bash
# B4（reports/PLAN_b4.md）的闭环录制：ACT 100K，开 TE（系数 0.01、每帧推理）、CUDA graph、20 Hz；B4_MODE 选包装模式。
# 每块 ≤ 5 条一个进程；开录前打印这一块各条方块摆在哪（相对桌上 1–5 号标记 = −4、−2、0、+2、+4 cm）。按数据集里已录的集数自动接着录：
# 中途断了，重跑同一条命令即可，位置不会和集号错开。
#   bash scripts/run_b4.sh check          # 第一步：闭环不提前（A，B4_MODE=raw）与闭环 Shift（B，B4_MODE=shift），ABBA 各 10 条
#   bash scripts/run_b4.sh record         # 第二步：闭环 Shift 录教师用的轨迹（positions.json 的顺序；失败的位置标完成败后排到队尾重录）
#   bash scripts/run_b4.sh final          # 第五步：闭环 Shift / Shift＋网络（B4_NET=outputs/b4/net/frozen.json）/ 原始 ACT，12 块轮换，各 20 条
#   DRY=1 bash scripts/run_b4.sh check    # 只打印命令和位置，不连机器
# 先 conda activate lerobot；第一次用前：python -u sim/b4_common.py cfg、python -u sim/b4_positions.py gen
set -eu
cd ~/claude-test-2/lerobot-piper
export LD_PRELOAD=$HOME/miniconda3/envs/lerobot/lib/libstdc++.so.6   # 先 torch 后 cv2 会报 CXXABI_1.3.15（2026-10-29）
MODE=${1:?用法: $0 check|record|final}
CKPT=outputs/train/2026-10-17/ACT_pickcube_noimu_seed42/checkpoints/100000/pretrained_model
LR=$HOME/.cache/huggingface/lerobot/enolab-guo
unset REFINE_CKPT BCORR_CKPT
case $MODE in
  check) conds=(raw shift shift raw) ;;
  record|final) conds=() ;;
  *) echo "模式只能是 check、record 或 final"; exit 1 ;;
esac

run_block() {   # 开录前一刻才看数据集：存过集就接着录；只有空壳（上次一集都没存就停了）就删掉新建。参数 --dry：只打印
  local saved
  saved=$(python -c "import sys; sys.path.insert(0, 'sim'); import b4_positions as P; print(P.n_recorded('$repo'))")
  extra=()
  if [ "$saved" -gt 0 ]; then
    extra+=(--control.resume=true)
  elif [ -d "$LR/$repo" ]; then
    echo "    $repo 只是空壳（一集都没存）：删掉后新建"
    if [ "${1:-}" != --dry ]; then rm -rf -- "${LR:?}/$repo"; fi
  fi
  local net=()
  [ "$b4" = net ] && net=(B4_NET=outputs/b4/net/frozen.json)
  cmd=(env B4_MODE=$b4 "${net[@]}" python lerobot/scripts/control_robot.py
       --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
       --control.type=record --control.fps=20 "--control.single_task=pick the cube and place it in the box"
       --control.repo_id=enolab-guo/$repo --control.num_episodes=$n --control.warmup_time_s=5
       --control.episode_time_s=25 --control.reset_time_s=15
       --control.push_to_hub=false --control.device=cuda --control.use_amp=false
       --control.policy.path=$CKPT --control.policy.cuda_graph=true
       --control.policy.temporal_ensemble_coeff=0.01 --control.policy.n_action_steps=1 "${extra[@]}")
  if [ "${1:-}" = --dry ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
}

b=0
while :; do
  b=$((b + 1))
  if [ "$MODE" = check ]; then
    [ "$b" -gt 4 ] && break
    b4=${conds[$((b - 1))]}
    repo=eval_pickcube_b4check_$b4
  elif [ "$MODE" = final ]; then
    [ "$b" -gt 12 ] && break
    b4=$(python -u sim/b4_positions.py cond "$b")
    repo=eval_pickcube_b4final_$b4
  else
    b4=shift
    repo=eval_pickcube_b4rec
  fi
  if ! pos=$(python -u sim/b4_positions.py block "$MODE" "$b"); then break; fi   # 要先标成败：提示已打印
  if [ -z "$pos" ]; then
    if [ "$MODE" = record ]; then break; else continue; fi                       # check / final：这一块已录满
  fi
  n=$(grep -c . <<< "$pos")
  echo "=== 第 $b 块：B4_MODE=$b4，$n 条 → $repo"
  k=0
  while read -r id cm desc; do
    k=$((k + 1))
    echo "    第 $k 条：$desc（$cm cm，$id）"
  done <<< "$pos"
  if [ "${DRY:-0}" = 1 ]; then
    run_block --dry
    [ "$MODE" = record ] && break                  # 不录就不会接着往下排
    continue
  fi
  read -rp "方块摆到第 1 条的位置后按回车（之后每条在复位的 15 s 里换到下一个位置；Ctrl+C 退出）…"
  if [ "$(python -u sim/b4_positions.py block "$MODE" "$b")" != "$pos" ]; then   # 等待期间数据集变了：重新排这一块
    echo "⚠ 等待期间数据集有变化，重新排这一块"
    b=$((b - 1))
    continue
  fi
  run_block
done
echo "=== $MODE：这一轮能录的都录完了"
