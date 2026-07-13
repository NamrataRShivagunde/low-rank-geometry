# 60M: compute both expected sharpness and expected variance band
CUDA_VISIBLE_DEVICES=6 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_60m.py \
  --landscape_root results/loss-landscape-2d/models-60m \
  --output_root results/expected-sharpness-2d \
  --metric_mode both \
  --only_models llama cola fira sltrain galore relora \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000


# 130M: compute both expected sharpness and expected variance band
CUDA_VISIBLE_DEVICES=2 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_130m.py \
  --landscape_root results/loss-landscape-2d/models-130m \
  --output_root results/expected-sharpness-2d \
  --metric_mode both \
  --only_models llama cola fira sltrain galore relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000


# 350M: compute both expected sharpness and expected variance band
CUDA_VISIBLE_DEVICES=3 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_350m.py \
  --landscape_root results/loss-landscape-2d/models-350m \
  --output_root results/expected-sharpness-2d \
  --metric_mode both \
  --only_models llama cola fira sltrain galore relora \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000
