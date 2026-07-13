#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for 350m final checkpoints on up to 7 GPUs.
# Usage:
#   bash evaluation/eval_350m.sh
#   bash evaluation/eval_350m.sh "llama galore fira cola sltrain relora" "0,1,2,3,4,5,6"

METHODS="${1:-cola}"
GPU_LIST_CSV="${2:-6}"
CONFIG="${3:-evaluation/configs/full_eval_all_tasks.yaml}"

MODEL_SIZE="350m"
CHECKPOINT_STEP="60000"
CHECKPOINT_BASE="CHECKPOINTS/models-350m"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="evaluation/output/zero_shot_last_ckpt/${MODEL_SIZE}/run_${TS}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

IFS=' ' read -r -a METHODS_ARR <<< "$METHODS"
IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST_CSV"

if [[ ${#GPUS_ARR[@]} -eq 0 ]]; then
    echo "ERROR: no GPUs provided"
    exit 1
fi

MANIFEST="${RUN_ROOT}/manifest.tsv"
echo -e "method\tsize\tstep\tgpu\tcheckpoint\toutput_root\tlog" > "$MANIFEST"

echo "Running ${MODEL_SIZE} final checkpoints"
echo "Methods: $METHODS"
echo "GPUs: $GPU_LIST_CSV"
echo "Config: $CONFIG"
echo "Run root: $RUN_ROOT"
echo

PIDS=()
GPU_TAGS=()

for i in "${!METHODS_ARR[@]}"; do
    method="${METHODS_ARR[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"

    ckpt="${CHECKPOINT_BASE}/${method}-${MODEL_SIZE}/model_${CHECKPOINT_STEP}"
    if [[ ! -d "$ckpt" ]]; then
        echo "[SKIP] Missing checkpoint: $ckpt"
        continue
    fi

    method_out="${RUN_ROOT}/${method}"
    mkdir -p "$method_out"
    log_file="${LOG_DIR}/${method}.log"

    echo -e "${method}\t${MODEL_SIZE}\t${CHECKPOINT_STEP}\t${gpu}\t${ckpt}\t${method_out}\t${log_file}" >> "$MANIFEST"
    echo "[LAUNCH] method=${method} gpu=${gpu}"

    CUDA_VISIBLE_DEVICES="$gpu" python evaluation/run_lm_eval.py \
        --config "$CONFIG" \
        --output-root "$method_out" \
        --checkpoint "$ckpt" \
        > "$log_file" 2>&1 &

    PIDS+=("$!")
    GPU_TAGS+=("$gpu:$method")
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "ERROR: no jobs launched"
    exit 1
fi

echo
echo "Launched ${#PIDS[@]} jobs. Waiting..."
FAIL=0
for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then
        echo "[DONE] ${GPU_TAGS[$j]}"
    else
        echo "[FAIL] ${GPU_TAGS[$j]}"
        FAIL=1
    fi
done

echo
echo "Manifest: $MANIFEST"
echo "Logs: $LOG_DIR"

if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi
