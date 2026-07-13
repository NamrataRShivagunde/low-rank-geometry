#!/bin/bash
# Batch activation-comparison metrics (target method vs llama) for 60m, 130m, 350m.

set -e

# Ensure custom method packages under training/ are importable (ReLoRA/CoLA/etc).
export PYTHONPATH="$PWD/training:${PYTHONPATH:-}"

GPU=7

# 60M
CUDA_VISIBLE_DEVICES=$GPU python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_multi_chkpt_60m.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --output_root results/activation-comparison \
  --tokenizer_checkpoint t5-base \
  --max_examples 1000 \
  --max_length 256 \
  --batch_size 128 \
  --max_batches 1000 \
  --only_models cola fira sltrain galore relora \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000

# 130M
CUDA_VISIBLE_DEVICES=$GPU python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_multi_chkpt_130m.py \
  --checkpoint_root CHECKPOINTS/models-130m \
  --output_root results/activation-comparison \
  --tokenizer_checkpoint t5-base \
  --max_examples 1000 \
  --max_length 256 \
  --batch_size 128 \
  --max_batches 1000 \
  --only_models cola fira sltrain galore relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000

# 350M
CUDA_VISIBLE_DEVICES=$GPU python loss-landscape/scripts/multi_checkpoint_scripts/6_activations_comparison_multi_chkpt_350m.py \
  --checkpoint_root CHECKPOINTS/models-350m \
  --output_root results/activation-comparison \
  --tokenizer_checkpoint t5-base \
  --max_examples 1000 \
  --max_length 256 \
  --batch_size 128 \
  --max_batches 1000 \
  --only_models cola fira sltrain galore relora \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000
