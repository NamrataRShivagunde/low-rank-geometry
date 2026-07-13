# 60M rank-related metrics for checkpoints
CUDA_VISIBLE_DEVICES=7 python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_multi_chkpt_60m.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --output_root results/rank-metrics/models-60m \
  --max_matrices 0 \
  --max_matrix_elements 4000000 \
  --min_matrix_dim 2 \
  --svd_device cuda \
  --seed 42 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000

# 130m
# 130M rank-related metrics for checkpoints
CUDA_VISIBLE_DEVICES=7 python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_multi_chkpt_130m.py \
  --checkpoint_root CHECKPOINTS/models-130m \
  --output_root results/rank-metrics/models-130m \
  --max_matrices 0 \
  --max_matrix_elements 4000000 \
  --min_matrix_dim 2 \
  --svd_device cuda \
  --seed 42 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000


# 350M rank-related metrics for checkpoints
CUDA_VISIBLE_DEVICES=7 python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_multi_chkpt_350m.py \
  --checkpoint_root CHECKPOINTS/models-350m \
  --output_root results/rank-metrics/models-350m \
  --max_matrices 0 \
  --max_matrix_elements 4000000 \
  --min_matrix_dim 2 \
  --svd_device cuda \
  --seed 42 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000
