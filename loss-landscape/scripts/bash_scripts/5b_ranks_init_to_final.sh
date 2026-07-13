#!/bin/bash
set -e

# Compute and plot singular values of DeltaW = W_final - W_warm_start, grouped by
# projection type (q/k/v/o/gate/up/down). Mirrors ReLoRA paper Figures 3 and 4.
#
# Usage:
#   bash loss-landscape/scripts/bash_scripts/5b_ranks_init_to_final.sh \
#       [methods] [gpu_60m] [gpu_130m] [gpu_350m] \
#       [output_root] [checkpoints_root] [svd_device] [model_device]
#
# Example (default args):
#   bash loss-landscape/scripts/bash_scripts/5b_ranks_init_to_final.sh

METHODS_STR="${1:-llama galore fira sltrain relora}"
GPU_60M="${2:-0}"
GPU_130M="${3:-0}"
GPU_350M="${4:-0}"
OUTPUT_ROOT="${5:-results/rank-init-to-final}"
CHECKPOINTS_ROOT="${6:-CHECKPOINTS}"
SVD_DEVICE="${7:-cuda}"
MODEL_DEVICE="${8:-cuda}"

read -r -a METHODS <<< "$METHODS_STR"

echo "Methods:           $METHODS_STR"
echo "GPUs:              60m=$GPU_60M, 130m=$GPU_130M, 350m=$GPU_350M"
echo "Output root:       $OUTPUT_ROOT"
echo "Checkpoints root:  $CHECKPOINTS_ROOT"

run_for_size() {
  local size="$1"
  local gpu="$2"

  echo ""
  echo "=== Computing init->final SVs for ${size} on GPU ${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" python loss-landscape/scripts/single_checkpoint_scripts/5b_ranks_init_to_final.py \
    --checkpoints_root "$CHECKPOINTS_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model_sizes "$size" \
    --methods "${METHODS[@]}" \
    --svd_device "$SVD_DEVICE" \
    --model_device "$MODEL_DEVICE" \
    --seed 42
}

run_for_size "60m" "$GPU_60M"
run_for_size "130m" "$GPU_130M"
run_for_size "350m" "$GPU_350M"

echo ""
echo "=== Plotting SV spectra (Figure 3 style) ==="
python loss-landscape/scripts/multi_checkpoint_scripts/5b_plot_sv_spectra.py \
  --results_root "$OUTPUT_ROOT" \
  --methods "${METHODS[@]}"

echo ""
echo "=== Plotting SV>threshold counts (Figure 4 style) ==="
python loss-landscape/scripts/multi_checkpoint_scripts/5b_plot_sv_count_bars.py \
  --results_root "$OUTPUT_ROOT" \
  --methods "${METHODS[@]}"

echo ""
echo "All init->final SV runs completed."
