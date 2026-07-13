#!/bin/bash
set -e

# Run update-rank overlay plots for 60m, 130m, and 350m.
#
# Usage:
#   bash loss-landscape/scripts/bash_scripts/5_ranks_updates_overlay.sh [overlay_root] [methods] [dpi]
#
# Example:
#   bash loss-landscape/scripts/bash_scripts/5_ranks_updates_overlay.sh results/rank-updates "llama cola fira galore relora sltrain" 220

OVERLAY_ROOT="${1:-results/rank-updates}"
METHODS_STR="${2:-llama cola fira galore relora sltrain}"
DPI="${3:-220}"

python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_updates_overlay.py \
  --summary_csv "$OVERLAY_ROOT/models-60m/rank_updates_all_methods_sizes.csv" \
  --output_dir "$OVERLAY_ROOT/models-60m/overlay" \
  --output_name rank_updates_60m_overlay \
  --model_size 60m \
  --methods $METHODS_STR \
  --dpi "$DPI"

python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_updates_overlay.py \
  --summary_csv "$OVERLAY_ROOT/models-130m/rank_updates_all_methods_sizes.csv" \
  --output_dir "$OVERLAY_ROOT/models-130m/overlay" \
  --output_name rank_updates_130m_overlay \
  --model_size 130m \
  --methods $METHODS_STR \
  --dpi "$DPI"

python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_updates_overlay.py \
  --summary_csv "$OVERLAY_ROOT/models-350m/rank_updates_all_methods_sizes.csv" \
  --output_dir "$OVERLAY_ROOT/models-350m/overlay" \
  --output_name rank_updates_350m_overlay \
  --model_size 350m \
  --methods $METHODS_STR \
  --dpi "$DPI"
