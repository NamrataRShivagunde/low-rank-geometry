#!/bin/bash
# Run cross-method 1D interpolation (script 9) across all training steps for a
# given model size. Each step's outputs go into its own folder.
#
# Usage:
#   bash loss-landscape/scripts/bash_scripts/9_1d_interpolation_cross_method_multi_step.sh <size> <gpu>
# Example:
#   bash loss-landscape/scripts/bash_scripts/9_1d_interpolation_cross_method_multi_step.sh 60m 7

set -e

SIZE="${1:-60m}"
GPU="${2:-7}"

export PYTHONPATH="$PWD/training:${PYTHONPATH:-}"

case "$SIZE" in
  60m)
    STEPS="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000"
    ;;
  130m)
    STEPS="2000 4000 6000 8000 10000 12000 14000 16000 18000 20000"
    ;;
  350m)
    STEPS="6000 12000 18000 24000 30000 36000 42000 48000 54000 60000"
    ;;
  *)
    echo "Unknown size: $SIZE (use 60m, 130m, 350m)"; exit 1;;
esac

CHECKPOINT_ROOT="CHECKPOINTS/models-${SIZE}"
OUTPUT_ROOT="results/cross_method_interpolation/models-${SIZE}"

for STEP in $STEPS; do
  STEP_DIR="${OUTPUT_ROOT}/step_${STEP}"
  RAW_JSON="${STEP_DIR}/cross_method_interp_${SIZE}_step${STEP}_raw.json"

  # Skip if already done
  if [ -f "$RAW_JSON" ]; then
    echo "[skip] ${SIZE} step ${STEP} already has raw JSON"
    continue
  fi

  echo "=================================================================="
  echo "Running ${SIZE} step ${STEP} (GPU ${GPU})"
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$GPU python loss-landscape/scripts/single_checkpoint_scripts/9_1d_interpolation_cross_method.py \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --model_size "$SIZE" \
    --step "$STEP" \
    --output_dir "$STEP_DIR" \
    --methods llama fira galore relora sltrain \
    --alpha_min 0.0 \
    --alpha_max 1.0 \
    --alpha_points 21 \
    --max_examples 500 \
    --max_length 256 \
    --batch_size 128 \
    --max_batches 20 \
    --device cuda
done

echo "=================================================================="
echo "All steps complete for ${SIZE}"
echo "=================================================================="
