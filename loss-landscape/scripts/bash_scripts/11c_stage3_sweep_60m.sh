#!/usr/bin/env bash
# Stage 3 (perturbation region study) on the final 60m checkpoint for ONE method.
# Usage:  METHOD=cola GPU=0 bash 11c_stage3_sweep_60m.sh
# Optional env:
#   STEP, RADII (space-separated), N_SAMPLES_PER_RADIUS, NUM_BATCHES,
#   BATCH_SIZE, MAX_LENGTH, MAX_EXAMPLES, PL_THRESHOLD, OUT_CSV, OUT_DIR
set -euo pipefail

METHOD="${METHOD:?set METHOD=llama|cola|fira|galore|relora|sltrain}"
GPU="${GPU:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STEP="${STEP:-10000}"
RADII_STR="${RADII:-0.01 0.05 0.1 0.5 1.0 5.0}"
read -r -a RADII <<< "$RADII_STR"

N_SAMPLES_PER_RADIUS="${N_SAMPLES_PER_RADIUS:-20}"
NUM_BATCHES="${NUM_BATCHES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
PL_THRESHOLD="${PL_THRESHOLD:-1e-4}"

CKPT_ROOT="CHECKPOINTS/models-60m/${METHOD}-60m"
ckpt="${CKPT_ROOT}/model_${STEP}"

OUT_DIR="${OUT_DIR:-results/pl-metric/stage3}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/${METHOD}-60m.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
rm -f "$OUT_CSV"

if [[ ! -d "$ckpt" ]]; then
  echo "[11c_sweep] checkpoint $ckpt not found, aborting"
  exit 1
fi

echo "[11c_sweep] method=$METHOD step=$STEP gpu=$GPU out=$OUT_CSV"
echo "[11c_sweep]   radii=${RADII[*]} n_samples=$N_SAMPLES_PER_RADIUS num_batches=$NUM_BATCHES"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n =ll-training --no-capture-output \
  python loss-landscape/scripts/single_checkpoint_scripts/11c_pl_metric_region.py \
    --model "$METHOD" \
    --checkpoint "$ckpt" \
    --output_csv "$OUT_CSV" \
    --max_examples "$MAX_EXAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --num_batches "$NUM_BATCHES" \
    --radii ${RADII[@]} \
    --n_samples_per_radius "$N_SAMPLES_PER_RADIUS" \
    --pl_threshold "$PL_THRESHOLD" \
    --seed 42

echo "[11c_sweep] done method=$METHOD -> $OUT_CSV"
