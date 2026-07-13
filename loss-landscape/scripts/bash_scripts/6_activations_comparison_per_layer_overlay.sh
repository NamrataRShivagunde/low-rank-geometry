#!/bin/bash
# Generate per-layer activation-comparison overlays for 60m, 130m, 350m.

set -e

export PYTHONPATH="$PWD/training:${PYTHONPATH:-}"

# 60M
python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_per_layer_overlay.py \
  --output_root results/activation-comparison \
  --model_size 60m \
  --output_dir results/activation-comparison/overlays/per_layer_60m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka baseline_mean_norm target_mean_norm

# 130M
python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_per_layer_overlay.py \
  --output_root results/activation-comparison \
  --model_size 130m \
  --output_dir results/activation-comparison/overlays/per_layer_130m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka baseline_mean_norm target_mean_norm

# 350M
python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_per_layer_overlay.py \
  --output_root results/activation-comparison \
  --model_size 350m \
  --output_dir results/activation-comparison/overlays/per_layer_350m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka baseline_mean_norm target_mean_norm

echo "All per-layer overlays created successfully"


# # All layers, all metrics
# python 6_activations_comparison_per_layer_overlay.py \
#   --output_root results/activation-comparison \
#   --model_size 60m \
#   --output_dir results/activation-comparison/overlays/per_layer_60m

# # Specific layers only
# python 6_activations_comparison_per_layer_overlay.py \
#   --output_root results/activation-comparison \
#   --model_size 60m \
#   --output_dir results/activation-comparison/overlays/per_layer_60m \
#   --layers layer_000 layer_001 layer_012

# # Run all sizes
# bash 6_activations_comparison_per_layer_overlay.sh