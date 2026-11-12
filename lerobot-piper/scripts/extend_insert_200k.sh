#!/usr/bin/env bash
# 等 pickplace2 到 100K 且训练进程退出 → 从 100K 续训 insert 到 200K (再 +100K, 防 100K 欠训)。
# 弹性重试(绕偶发启动段错误), nw 可调(NW=2 默认)。
# 用法(你终端, conda activate lerobot 后):
#   NW=2 nohup bash scripts/extend_insert_200k.sh > outputs/train/2026-07-01/extend_insert.log 2>&1 &
#   tail -f outputs/train/2026-07-01/ACT_insert_noimu_seed42.log   # 看 step 从 100K 往 200K 爬
set -u
cd ~/claude-test-2/lerobot-piper || exit 1
NW="${NW:-2}"
IB=outputs/train/2026-07-01/ACT_insert_noimu_seed42
PB=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42

echo "[$(date)] 等 pickplace2 到 100K..."
while [ ! -d "$PB/checkpoints/100000" ]; do sleep 30; done
echo "[$(date)] pickplace2 已 100K; 等其训练进程退出(避免抢 GPU)..."
while pgrep -f "train.py.*pickplace2_train" >/dev/null 2>&1; do sleep 10; done
sleep 5

echo "[$(date)] 开始续训 insert 100K→200K (nw=$NW)"
for try in $(seq 1 120); do
  if [ -d "$IB/checkpoints/200000" ]; then echo "[$(date)] OK insert 200K 完成"; exit 0; fi
  st=$(tr -d ' \n' < "$IB/checkpoints/last/training_state/training_step.json" 2>/dev/null)
  echo "[$(date)] try#$try RESUME insert ($st) → 目标 200K"
  PYTHONFAULTHANDLER=1 python -u lerobot/scripts/train.py \
    --config_path="$IB/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --num_workers=$NW --steps=200000 --log_freq=500 >> "${IB}.log" 2>&1
  echo "[$(date)] 退出码 $? — 3s 后检查/重试"
  sleep 3
done
echo "[$(date)] FAIL insert 试 120 次仍未到 200K"; exit 1
