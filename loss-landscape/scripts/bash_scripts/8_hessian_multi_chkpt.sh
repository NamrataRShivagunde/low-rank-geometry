#!/bin/bash
# Run Hessian metrics for all checkpoints across all model sizes.
# Usage: bash loss-landscape/scripts/bash_scripts/8_hessian_multi_chkpt.sh

set -e

METHODS="llama cola fira sltrain galore relora"

# 60M
echo "========== Running Hessian metrics for 60m =========="
CUDA_VISIBLE_DEVICES=5 python loss-landscape/scripts/multi_checkpoint_scripts/8_hessian_multi_chkpt_60m.py \
  --output_root results/hessian/models-60m \
  --only_models $METHODS \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000 \
  --compute_min_eigenvalue

# 130M
echo "========== Running Hessian metrics for 130m =========="
CUDA_VISIBLE_DEVICES=5 python loss-landscape/scripts/multi_checkpoint_scripts/8_hessian_multi_chkpt_130m.py \
  --output_root results/hessian/models-130m \
  --only_models $METHODS \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000 \
  --compute_min_eigenvalue

# 350M
echo "========== Running Hessian metrics for 350m =========="
CUDA_VISIBLE_DEVICES=5 python loss-landscape/scripts/multi_checkpoint_scripts/8_hessian_multi_chkpt_350m.py \
  --output_root results/hessian/models-350m \
  --only_models $METHODS \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --compute_min_eigenvalue

echo "========== All Hessian batch runs complete =========="

# Generate overlay plots
echo "========== Generating overlay plots =========="

python loss-landscape/scripts/multi_checkpoint_scripts/8_hessian_overlay.py \
  --summary_csv results/hessian/models-60m/summary/hessian_trends_60m.csv \
  --output_dir results/hessian/plots/models-60m \
  --output_prefix hessian_60m

python loss-landscape/scripts/multi_checkpoint_scripts/8_hessian_overlay.py \
  --summary_csv results/hessian/models-130m/summary/hessian_trends_130m.csv \
  --output_dir results/hessian/plots/models-130m \
  --output_prefix hessian_130m

python loss-landscape/scripts/multi_checkpoint_scripts/8_hessian_overlay.py \
  --summary_csv results/hessian/models-350m/summary/hessian_trends_350m.csv \
  --output_dir results/hessian/plots/models-350m \
  --output_prefix hessian_350m

echo "========== All done =========="
