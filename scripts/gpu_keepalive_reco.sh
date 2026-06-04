#!/bin/bash
# gpu_keepalive.sh — keep H100 GPUs busy between experiments to prevent pod auto-cleanup.
# - Watches for python torchrun/train_gpt processes via /proc/[pid]/cmdline (no false positives).
# - When train_gpt absent: ensure 8 run-gpu.py dummies running.
# - When train_gpt present: kill dummies (yield).
# - Hang detection: train_gpt alive but ALL GPUs < 5% util for 600s → kill train_gpt
#   (NCCL deadlock / compile hang protection; pod release threshold is 1h peak<21%).

EXTRA=/workspace/izlley/extra
RUN_GPU="${EXTRA}/run-gpu.py"
LOG=/tmp/gpu_keepalive.log
PIDFILE=/tmp/gpu_keepalive.pid
NUM_GPUS=4

POLL_SEC=10
HANG_THRESHOLD_PCT=5         # any GPU > 5% util counts as alive
HANG_WINDOW_SEC=600          # 10 min sustained idle = hang
HANG_TICK_LIMIT=$(( HANG_WINDOW_SEC / POLL_SEC ))

[ -f "$RUN_GPU" ] || { echo "ERROR: $RUN_GPU missing" >&2; exit 1; }

echo $$ > "$PIDFILE"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "gpu_keepalive started pid=$$ (hang_window=${HANG_WINDOW_SEC}s, hang_threshold=${HANG_THRESHOLD_PCT}%)"

is_experiment_running() {
  for pid in $(pgrep -x 'python|python3' 2>/dev/null); do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmdline" in
      *train_gpt.py*|*reco_bench.pipelines.train*|*reco_bench.pipelines.evaluate*|*reco_bench.pipelines.build_index*) return 0 ;;
    esac
  done
  return 1
}

dummy_count() {
  local cnt=0
  for pid in $(pgrep -x 'python|python3' 2>/dev/null); do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmdline" in
      *run-gpu.py*) cnt=$((cnt+1)) ;;
    esac
  done
  echo "$cnt"
}

max_gpu_util() {
  # Returns highest util across all GPUs (0-100). 0 if nvidia-smi fails.
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{m=0}{if($1>m)m=$1}END{print m+0}'
}

start_dummies() {
  log "start_dummies (no train_gpt detected)"
  for gpu in $(seq 0 $((NUM_GPUS-1))); do
    nohup python "$RUN_GPU" "$gpu" > /dev/null 2>&1 &
    disown
  done
}

stop_dummies() {
  log "stop_dummies"
  for pid in $(pgrep -x 'python|python3' 2>/dev/null); do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmdline" in
      *run-gpu.py*) kill -9 "$pid" 2>/dev/null ;;
    esac
  done
  sleep 1
}

kill_hung_experiment() {
  log "HANG DETECTED: train_gpt alive but all GPUs <${HANG_THRESHOLD_PCT}% util for ${HANG_WINDOW_SEC}s — killing torchrun + train_gpt"
  pkill -9 -f "torchrun" 2>/dev/null
  for pid in $(pgrep -x 'python|python3' 2>/dev/null); do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmdline" in
      *train_gpt.py*|*reco_bench.pipelines.train*) kill -9 "$pid" 2>/dev/null ;;
    esac
  done
  sleep 3
}

trap 'log "watcher exiting"; rm -f "$PIDFILE"; exit 0' INT TERM EXIT

hang_ticks=0
prev_state="unknown"

while true; do
  if is_experiment_running; then
    if [ "$prev_state" != "experiment" ]; then
      log "state: experiment detected"
      prev_state="experiment"
    fi
    # Kill any leftover dummies
    if [ "$(dummy_count)" -gt 0 ]; then
      stop_dummies
    fi
    # Hang detection
    util=$(max_gpu_util)
    if [ "$util" -gt "$HANG_THRESHOLD_PCT" ]; then
      if [ "$hang_ticks" -gt 0 ]; then
        log "GPU activity resumed (max_util=${util}%) after ${hang_ticks} idle ticks — reset"
      fi
      hang_ticks=0
    else
      hang_ticks=$((hang_ticks + 1))
      if [ $((hang_ticks % 6)) -eq 0 ]; then
        log "GPU idle while train_gpt alive: ${hang_ticks}/${HANG_TICK_LIMIT} ticks (max_util=${util}%)"
      fi
      if [ "$hang_ticks" -ge "$HANG_TICK_LIMIT" ]; then
        kill_hung_experiment
        hang_ticks=0
        prev_state="hung"
      fi
    fi
  else
    if [ "$prev_state" != "idle" ]; then
      log "state: idle (no train_gpt)"
      prev_state="idle"
    fi
    hang_ticks=0
    cnt=$(dummy_count)
    if [ "$cnt" -lt "$NUM_GPUS" ]; then
      log "dummy_count=$cnt < $NUM_GPUS — restarting dummies"
      stop_dummies
      start_dummies
    fi
  fi
  sleep "$POLL_SEC"
done
