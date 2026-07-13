#!/bin/bash
# 1-D interpolation (consecutive checkpoints) for 60m, 130m, 350m models.

set -e

export PYTHONPATH="$PWD/training:${PYTHONPATH:-}"

# 60m
CUDA_VISIBLE_DEVICES=1 python loss-landscape/scripts/single_checkpoint_scripts/7_1d_interpolation.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --model_size 60m \
  --output_dir results/interpolation \
  --alpha_points 21 \
  --max_batches 50 \
  --max_examples 1000 \
  --max_length 256 \
  --batch_size 128

echo "Done: 60m interpolation"

# 130m
CUDA_VISIBLE_DEVICES=2 python loss-landscape/scripts/single_checkpoint_scripts/7_1d_interpolation.py \
  --checkpoint_root CHECKPOINTS/models-130m \
  --model_size 130m \
  --output_dir results/interpolation \
  --alpha_points 21 \
  --max_batches 50 \
  --max_examples 1000 \
  --max_length 256 \
  --batch_size 128

echo "Done: 130m interpolation"

# 350m
CUDA_VISIBLE_DEVICES=4 python loss-landscape/scripts/single_checkpoint_scripts/7_1d_interpolation.py \
  --checkpoint_root CHECKPOINTS/models-350m \
  --model_size 350m \
  --output_dir results/interpolation \
  --alpha_points 21 \
  --max_batches 50 \
  --max_examples 1000 \
  --max_length 256 \
  --batch_size 128

echo "Done: 350m interpolation"
echo "All interpolation runs complete"
