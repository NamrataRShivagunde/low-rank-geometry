#!/bin/bash
# Regenerate interpolation plots from existing raw JSON files (no loss recomputation).

set -e

# 60m
python loss-landscape/scripts/single_checkpoint_scripts/7_1d_interpolation.py \
  --model_size 60m \
  --output_dir results/interpolation \
  --plot_only \
  --raw_json results/interpolation/interpolation_1d_60m_raw.json

# 130m
python loss-landscape/scripts/single_checkpoint_scripts/7_1d_interpolation.py \
  --model_size 130m \
  --output_dir results/interpolation \
  --plot_only \
  --raw_json results/interpolation/interpolation_1d_130m_raw.json

# 350m
python loss-landscape/scripts/single_checkpoint_scripts/7_1d_interpolation.py \
  --model_size 350m \
  --output_dir results/interpolation \
  --plot_only \
  --raw_json results/interpolation/interpolation_1d_350m_raw.json

echo "Plot-only regeneration complete"
