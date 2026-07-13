#!/usr/bin/env bash
# Stage 1 (local PL estimate) over all checkpoints for ONE method at ONE model size.
# Usage:  SIZE=130 METHOD=galore GPU=3 bash 11a_stage1_sweep.sh
# Optional env: MAX_EXAMPLES, NUM_BATCHES, BATCH_SIZE, MAX_LENGTH, OUT_CSV, OUT_DIR
#
# Size-specific checkpoint spacing (from CLAUDE.md section 12):
#   60  : model_1000 .. model_10000  (step 1000)
#   130 : model_2000 .. model_20000  (step 2000)
#   350 : model_6000 .. model_60000  (step 6000)
set -euo pipefail

SIZE="${SIZE:?set SIZE=60|130|350}"
METHOD="${METHOD:?set METHOD=llama|cola|fira|galore|relora|sltrain}"
GPU="${GPU:-7}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Eval-set params: 10k C4 examples with bs=16 -> 625 batches grad-accumulated.
MAX_EXAMPLES="${MAX_EXAMPLES:-10000}"
NUM_BATCHES="${NUM_BATCHES:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"

case "$SIZE" in
  60)  STEPS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000) ;;
  130) STEPS=(2000 4000 6000 8000 10000 12000 14000 16000 18000 20000) ;;
  350) STEPS=(6000 12000 18000 24000 30000 36000 42000 48000 54000 60000) ;;
  *) echo "Unknown SIZE=$SIZE"; exit 1 ;;
esac

CKPT_ROOT="CHECKPOINTS/models-${SIZE}m/${METHOD}-${SIZE}m"
OUT_DIR="${OUT_DIR:-results/pl-metric/stage1}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/${METHOD}-${SIZE}m.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
rm -f "$OUT_CSV"

echo "[11a_sweep] size=${SIZE}m method=$METHOD out=$OUT_CSV gpu=$GPU steps=${STEPS[*]}"
echo "[11a_sweep]   max_examples=$MAX_EXAMPLES num_batches=$NUM_BATCHES bs=$BATCH_SIZE"

for step in "${STEPS[@]}"; do
  ckpt="${CKPT_ROOT}/model_${step}"
  if [[ ! -d "$ckpt" ]]; then
    echo "  [skip] $ckpt not found"
    continue
  fi
  echo "=== size=${SIZE}m method=$METHOD step=$step ==="
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

echo "[11a_sweep] done size=${SIZE}m method=$METHOD -> $OUT_CSV"
