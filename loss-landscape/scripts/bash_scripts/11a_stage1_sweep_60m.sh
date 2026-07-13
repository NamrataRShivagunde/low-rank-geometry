#!/usr/bin/env bash
# Stage 1 (local PL estimate) over all 60m checkpoints for ONE method.
# Usage:  METHOD=galore GPU=3 bash 11a_stage1_sweep_60m.sh
# Optional:
#   MAX_EXAMPLES, NUM_BATCHES, BATCH_SIZE, MAX_LENGTH, OUT_CSV, OUT_DIR
set -euo pipefail

METHOD="${METHOD:?set METHOD=llama|cola|fira|galore|relora|sltrain}"
GPU="${GPU:-7}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Eval-set params (override via env for re-sweeps). Defaults target 10k
# C4 examples with gradient accumulation so the population ||grad L||^2
# estimate is not dominated by minibatch noise.
MAX_EXAMPLES="${MAX_EXAMPLES:-10000}"
NUM_BATCHES="${NUM_BATCHES:-0}"      # 0 = use all batches
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"

CKPT_ROOT="CHECKPOINTS/models-60m/${METHOD}-60m"
OUT_DIR="${OUT_DIR:-results/pl-metric/stage1}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/${METHOD}-60m.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
rm -f "$OUT_CSV"

STEPS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

echo "[11a_sweep] method=$METHOD out=$OUT_CSV gpu=$GPU steps=${STEPS[*]}"
echo "[11a_sweep]   max_examples=$MAX_EXAMPLES num_batches=$NUM_BATCHES bs=$BATCH_SIZE"

for step in "${STEPS[@]}"; do
  ckpt="${CKPT_ROOT}/model_${step}"
  if [[ ! -d "$ckpt" ]]; then
    echo "  [skip] $ckpt not found"
    continue
  fi
  echo "=== method=$METHOD step=$step ==="
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n =ll-training --no-capture-output \
    python loss-landscape/scripts/single_checkpoint_scripts/11a_pl_metric_local.py \
      --model "$METHOD" \
      --checkpoint "$ckpt" \
      --output_csv "$OUT_CSV" \
      --max_examples "$MAX_EXAMPLES" \
      --num_batches "$NUM_BATCHES" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      --seed 42
done

echo "[11a_sweep] done method=$METHOD -> $OUT_CSV"
