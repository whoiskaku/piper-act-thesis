#!/usr/bin/env bash
# pickplace2 从 100K 续训到 200K —— 跨任务确认"连续毛刺随训练量下降"(insert 已证)。
# 弹性重试(绕偶发启动段错误), nw 可调(NW=2 默认)。
# 用法(你终端, conda activate lerobot 后):
#   NW=2 nohup bash scripts/extend_pickplace2_200k.sh > outputs/train/2026-07-01/extend_pickplace2.log 2>&1 &
#   tail -f outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42.log   # 看 step 从 100K 往 200K 爬
set -u
cd ~/claude-test-2/lerobot-piper || exit 1
NW="${NW:-2}"
PB=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42

echo "[$(date)] 开始续训 pickplace2 100K→200K (nw=$NW)"
for try in $(seq 1 120); do
  if [ -d "$PB/checkpoints/200000" ]; then echo "[$(date)] OK pickplace2 200K 完成"; exit 0; fi
  st=$(tr -d ' \n' < "$PB/checkpoints/last/training_state/training_step.json" 2>/dev/null)
  echo "[$(date)] try#$try RESUME pickplace2 ($st) → 目标 200K"
  PYTHONFAULTHANDLER=1 python -u lerobot/scripts/train.py \
    --config_path="$PB/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --num_workers=$NW --steps=200000 --log_freq=500 >> "${PB}.log" 2>&1
  echo "[$(date)] 退出码 $? — 3s 后检查/重试"
  sleep 3
done
echo "[$(date)] FAIL pickplace2 试 120 次仍未到 200K"; exit 1
