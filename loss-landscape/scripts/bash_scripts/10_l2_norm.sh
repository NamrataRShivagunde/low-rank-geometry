#!/bin/bash
# Compute per-layer per-matrix L2 norms for llama & cola across 60m/130m/350m,
# then generate overlay + layerwise-overlay plots.
# Usage: bash loss-landscape/scripts/bash_scripts/10_l2_norm.sh

set -e

MULTI_CHKPT=loss-landscape/scripts/multi_checkpoint_scripts
OVERLAY=$MULTI_CHKPT/10_l2_norm_overlay.py
LAYERWISE=$MULTI_CHKPT/10_l2_norm_layerwise_overlay.py

for size in 60m 130m 350m; do
  echo "========== L2 norm: ${size} =========="
  python "$MULTI_CHKPT/10_l2_norm_multi_chkpt_${size}.py"

  echo "========== Overlay plots: ${size} =========="
  python "$OVERLAY" \
    --trend_csv "results/l2-norm/models-${size}/summary/l2_norm_trends_${size}.csv" \
    --output_dir "results/l2-norm/models-${size}/overlay" \
    --size_label "${size}"

  echo "========== Layerwise overlay plots: ${size} =========="
  python "$LAYERWISE" \
    --trend_csv "results/l2-norm/models-${size}/summary/l2_norm_trends_${size}.csv" \
    --output_dir "results/l2-norm/models-${size}/overlay/layerwise" \
    --size_label "${size}"
done

echo "========== All done =========="
