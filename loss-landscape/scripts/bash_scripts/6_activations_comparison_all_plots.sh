#!/bin/bash
# Generate all activation comparison plots after multi-checkpoint runs complete.

set -e

export PYTHONPATH="$PWD/training:${PYTHONPATH:-}"

# Per-layer overlay plots (metric vs step, one line per method, one plot per layer)
echo "=== Per-layer overlay plots ==="
python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_per_layer_overlay.py \
  --output_root results/activation-comparison \
  --model_size 60m \
  --output_dir results/activation-comparison/overlays/per_layer_60m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka

python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_per_layer_overlay.py \
  --output_root results/activation-comparison \
  --model_size 130m \
  --output_dir results/activation-comparison/overlays/per_layer_130m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka

python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_per_layer_overlay.py \
  --output_root results/activation-comparison \
  --model_size 350m \
  --output_dir results/activation-comparison/overlays/per_layer_350m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka

# Summary trend, last-layer, heatmap, and cross-size plots
echo "=== Summary and comparison plots ==="
python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_plots.py \
  --output_root results/activation-comparison \
  --plot_dir results/activation-comparison/plots \
  --sizes 60m 130m 350m \
  --metrics mean_l2_distance relative_l2_distance cosine_similarity cka

echo "All plots generated."
