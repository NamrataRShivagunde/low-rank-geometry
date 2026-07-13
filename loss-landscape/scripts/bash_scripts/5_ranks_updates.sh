#!/bin/bash
set -e

# Run update-rank computation for 60m, 130m, and 350m.
#
# Usage:
#   bash loss-landscape/scripts/bash_scripts/5_ranks_updates.sh [methods] [gpu_60m] [gpu_130m] [gpu_350m] [pair_limit] [output_root] [checkpoints_root] [svd_device] [model_device] [max_matrices] [checkpoint_template]
#
# Example:
#   bash loss-landscape/scripts/bash_scripts/5_ranks_updates.sh "llama cola fira galore relora sltrain" 0 1 2 0 results/rank-updates CHECKPOINTS cuda cuda 0 size_default_steps

METHODS_STR="${1:-llama cola fira galore relora sltrain}"
GPU_60M="${2:-0}"
GPU_130M="${3:-0}"
GPU_350M="${4:-0}"
PAIR_LIMIT="${5:-0}"
OUTPUT_ROOT="${6:-results/rank-updates}"
CHECKPOINTS_ROOT="${7:-CHECKPOINTS}"
SVD_DEVICE="${8:-cuda}"
MODEL_DEVICE="${9:-cuda}"
MAX_MATRICES="${10:-0}"
CHECKPOINT_TEMPLATE="${11:-size_default_steps}"

read -r -a METHODS <<< "$METHODS_STR"

echo "Running rank updates for methods: $METHODS_STR"
echo "GPUs: 60m=$GPU_60M, 130m=$GPU_130M, 350m=$GPU_350M"
echo "Output root: $OUTPUT_ROOT"
echo "Checkpoints root: $CHECKPOINTS_ROOT"
echo "Pair limit: $PAIR_LIMIT"
echo "Checkpoint template: $CHECKPOINT_TEMPLATE"

run_for_size() {
  local size="$1"
  local gpu="$2"
  local size_output_root="$OUTPUT_ROOT/models-${size}"

  echo ""
  echo "=== Running update-rank for ${size} on GPU ${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" python loss-landscape/scripts/single_checkpoint_scripts/5_ranks_updates.py \
    --checkpoints_root "$CHECKPOINTS_ROOT" \
    --output_root "$size_output_root" \
    --model_sizes "$size" \
    --methods "${METHODS[@]}" \
    --checkpoint_template "$CHECKPOINT_TEMPLATE" \
    --pair_limit "$PAIR_LIMIT" \
    --max_matrices "$MAX_MATRICES" \
    --max_matrix_elements 4000000 \
    --min_matrix_dim 2 \
    --svd_device "$SVD_DEVICE" \
    --model_device "$MODEL_DEVICE" \
    --seed 42
}

run_for_size "60m" "$GPU_60M"
run_for_size "130m" "$GPU_130M"
run_for_size "350m" "$GPU_350M"

echo ""
echo "All rank update runs completed."
