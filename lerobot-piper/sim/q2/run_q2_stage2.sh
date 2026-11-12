#!/usr/bin/env bash
# 阶段二（Q2 开发检验）正式运行 —— **只在 reports/PREREG_Q2_STAGE2.md 冻结后**，在用户自己的终端里运行。
# 五个阶段依次进行，每个阶段全部完成才进入下一个：
#   refs（G-2 在留出裸跑上，Q1 复查的背景）→ nom-inner → nom-final（选步、冻结名义 2a、零响应与固定 H 的分数）
#   → inc-inner（固定名义模型下的增量选步验证）→ inc-final（候选的分数）；然后汇总（q2_report.py）与作图（fig_q2.py）。
# 可重入：配置指纹与输入指纹都一致（终训还须 .pt / .npz 的哈希一致）的作业会被跳过；中断后直接重跑本脚本即可。
# 每次运行是一次「尝试」：日志写进新建的 $OUT/logs/attempt_NNN/（逐作业日志；status.tsv 记录时间、阶段、参数、退出码），
# 续跑不覆盖以前的失败证据。
# 墙钟上限：第一次 prepare 记录截止时间（+24 小时），续跑不重置；到点后作业拒绝/中止并写 INCOMPLETE.json，
# 汇总只写「未完成」、不作通过判决。
# 修复若涉及冻结代码（sim/q2 的 q2_*.py 与名单、只读复用的 sim/q1 代码）：须重新冻结预注册，并换新的 OUT 从头运行 ——
# 旧目录的截止时间记录绑定旧指纹，prepare 与作业都会拒绝。本脚本与作图脚本不在代码指纹里（不改变任何作业的计算）。
#   bash sim/q2/run_q2_stage2.sh            # 默认 OUT=outputs/q2_stage2 JOBS=15
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/home/enolab-guo/miniconda3/envs/lerobot/bin/python}
OUT=${OUT:-outputs/q2_stage2}
JOBS=${JOBS:-15}
mkdir -p "$OUT/logs"
n=1
while ! mkdir "$OUT/logs/attempt_$(printf %03d "$n")" 2>/dev/null; do n=$((n + 1)); done
LOG="$OUT/logs/attempt_$(printf %03d "$n")"
echo "本次尝试的日志：$LOG"

step() {  # step <名字> <命令…>：输出同时写进本次尝试的 <名字>.log，退出码记进 status.tsv
  local name=$1 rc=0
  shift
  "$@" 2>&1 | tee "$LOG/$name.log" || rc=$?
  printf '%s\t%s\t\t%s\n' "$(date -Iseconds)" "$name" "$rc" >> "$LOG/status.tsv"
  return "$rc"
}

run_phase() {
  local ph=$1
  "$PY" -u sim/q2/q2_run.py list-jobs --out "$OUT" --phase "$ph" > "$LOG/todo_$ph.txt"
  echo "[$ph] 待跑作业 $(wc -l < "$LOG/todo_$ph.txt") 个，并行 $JOBS"
  # -r：没有待跑作业时不调用；每行是该阶段作业的位置参数；逐作业日志与退出码留在本次尝试的目录里
  xargs -r -P "$JOBS" -L 1 sh -c \
    'py=$0; ph=$1; out=$2; log=$3; shift 3; name="${ph}_$(echo "$*" | tr " " _)"; rc=0
     "$py" -u sim/q2/q2_run.py "$ph" --out "$out" "$@" > "$log/$name.log" 2>&1 || rc=$?
     printf "%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$ph" "$*" "$rc" >> "$log/status.tsv"
     exit "$rc"' \
    "$PY" "$ph" "$OUT" "$LOG" < "$LOG/todo_$ph.txt" \
    || { echo "[$ph] 有作业失败或已到本轮截止时间（见 $LOG/${ph}_*.log 与 $LOG/status.tsv；超时会写 $OUT/INCOMPLETE.json）"; exit 1; }
  "$PY" -u sim/q2/q2_run.py list-jobs --out "$OUT" --phase "$ph" > "$LOG/todo_${ph}_after.txt"
  if [ -s "$LOG/todo_${ph}_after.txt" ]; then
    echo "[$ph] 仍有未完成的作业（见 $LOG/todo_${ph}_after.txt 与 ${ph}_*.log）—— 修复后重跑本脚本"; exit 1
  fi
}

step prepare "$PY" -u sim/q2/q2_run.py prepare --out "$OUT"
for ph in refs nom-inner nom-final inc-inner inc-final; do
  run_phase "$ph"
done
step report "$PY" -u sim/q2/q2_report.py --out "$OUT"
step fig "$PY" -u sim/q2/fig_q2.py --out "$OUT"
echo "完成：$OUT/summary.json、$OUT/summary.md、$OUT/fig/（本次尝试的日志：$LOG）"
