#!/usr/bin/env bash

# ReLoRA 60M HP sweep (online data)
# Grid:
#   weight_decay: 0, 0.1, 0.01
#   grad_clipping: 0, 0.5, 1
#   lr: 0.01, 0.005, 0.001
#
# Behavior:
# - Creates one run per combination
# - No offline data flags
# - save_every=21000 (effectively no intermediate checkpoints for 10k-step run)
# - Early-stop each run if NaN/Inf appears or loss exceeds threshold
#
# Usage:
#   bash training/training_bash_scripts/6_relora/sweep_relora_hp_60m.sh
# Optional env vars:
#   LOSS_HIGH_THRESHOLD=20            # default threshold for high loss
#   SPIKE_THRESHOLD=3                 # stop if loss jumps by this much between checks
#   SWEEP_SAVE_ROOT=checkpoints/relora-sweep-60m
#   SWEEP_LOG_ROOT=training/smoke_logs/relora_sweep

set -u

LOSS_HIGH_THRESHOLD="${LOSS_HIGH_THRESHOLD:-20}"
SPIKE_THRESHOLD="${SPIKE_THRESHOLD:-3}"
SWEEP_SAVE_ROOT="${SWEEP_SAVE_ROOT:-checkpoints/relora-sweep-60m}"
SWEEP_LOG_ROOT="${SWEEP_LOG_ROOT:-training/smoke_logs/relora_sweep}"

mkdir -p "$SWEEP_SAVE_ROOT" "$SWEEP_LOG_ROOT"

weight_decays=(0 0.1 0.01)
grad_clips=(0 0.5 1)
lrs=(0.01)

total_runs=0
stopped_early=0
completed=0
failed=0

extract_latest_loss() {
  local logfile="$1"
  python - "$logfile" << 'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)

text = path.read_text(errors="ignore")

patterns = [
    r"Eval loss at step\s+\d+\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
  r"final_eval_loss[^\d+-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
  r"\bloss\b[^\d+-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
]

latest = None
for pat in patterns:
    matches = re.findall(pat, text)
    if matches:
        try:
            latest = float(matches[-1])
            break
        except Exception:
            pass

print("" if latest is None else latest)
PY
}

is_loss_too_high() {
  local loss_value="$1"
  local threshold="$2"
  python - "$loss_value" "$threshold" << 'PY'
import sys
try:
    x = float(sys.argv[1])
    t = float(sys.argv[2])
    print("1" if x > t else "0")
except Exception:
    print("0")
PY
}

for wd in "${weight_decays[@]}"; do
  for gc in "${grad_clips[@]}"; do
    for lr in "${lrs[@]}"; do
      total_runs=$((total_runs + 1))

      tag="wd${wd}_gc${gc}_lr${lr}"
      tag_safe="$(echo "$tag" | sed 's/\./p/g')"
      run_name="relora_60m_${tag_safe}"
      save_dir="${SWEEP_SAVE_ROOT}/${run_name}"
      log_file="${SWEEP_LOG_ROOT}/${run_name}.log"

      echo "============================================================"
      echo "[RUN ${total_runs}] ${run_name}"
      echo "  weight_decay=${wd}  grad_clipping=${gc}  lr=${lr}"
      echo "  save_dir=${save_dir}"
      echo "  log_file=${log_file}"
      echo "============================================================"

      CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
        --method relora \
        --model_config training/configs/llama_configs/llama_60m.json \
        --lr "$lr" \
        --lora_r 128 \
        --relora 2000 \
        --restart_warmup_steps 50 \
        --batch_size 128 \
        --total_batch_size 512 \
        --num_training_steps 10000 \
        --warmup_steps 1000 \
        --weight_decay "$wd" \
        --grad_clipping "$gc" \
        --dtype bfloat16 \
        --eval_every 1000 \
        --save_every 21000 \
        --optimizer adamw \
        --scheduler cosine_restarts \
        --run_name "$run_name" \
        --save_dir "$save_dir" \
        > "$log_file" 2>&1 &

      pid=$!
      early_stop_reason=""
      prev_loss=""

      while kill -0 "$pid" 2>/dev/null; do
        sleep 15

        if grep -Eqi "\bnan\b|\binf\b|non-finite|overflow" "$log_file"; then
          early_stop_reason="nan_or_inf"
          break
        fi

        latest_loss="$(extract_latest_loss "$log_file")"
        if [[ -n "$latest_loss" ]]; then
          too_high="$(is_loss_too_high "$latest_loss" "$LOSS_HIGH_THRESHOLD")"
          if [[ "$too_high" == "1" ]]; then
            early_stop_reason="high_loss(${latest_loss})"
            break
          fi

          if [[ -n "$prev_loss" ]]; then
            spiked="$(python - "$latest_loss" "$prev_loss" "$SPIKE_THRESHOLD" << 'PY'
import sys
try:
    cur = float(sys.argv[1]); prev = float(sys.argv[2]); thr = float(sys.argv[3])
    print("1" if cur - prev > thr else "0")
except Exception:
    print("0")
PY
)"
            if [[ "$spiked" == "1" ]]; then
              early_stop_reason="spike(${prev_loss}->${latest_loss})"
              break
            fi
          fi
          prev_loss="$latest_loss"
        fi
      done

      if [[ -n "$early_stop_reason" ]]; then
        echo "[EARLY STOP] ${run_name} reason=${early_stop_reason}"
        stopped_early=$((stopped_early + 1))
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
      fi

      wait "$pid"
      rc=$?

      if [[ "$rc" -eq 0 ]]; then
        completed=$((completed + 1))
        echo "[DONE] ${run_name}"
      else
        failed=$((failed + 1))
        echo "[FAILED rc=${rc}] ${run_name}"
      fi
    done
  done
done

echo ""
echo "================ SWEEP SUMMARY ================"
echo "total_runs    : ${total_runs}"
echo "completed     : ${completed}"
echo "failed        : ${failed}"
echo "stopped_early : ${stopped_early}"
echo "logs          : ${SWEEP_LOG_ROOT}"
echo "checkpoints   : ${SWEEP_SAVE_ROOT}"
echo "==============================================="
