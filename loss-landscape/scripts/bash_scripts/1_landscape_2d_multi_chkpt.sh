
# 60M
CUDA_VISIBLE_DEVICES=0 python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_multi_chkpt_60m.py \
  --num_directions 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.003 --x_max 0.0031 --x_interval 0.00025 \
  --y_min -0.003 --y_max 0.003 --y_interval 0.00025 \
  --log_every 20 \
  --output_root results/loss-landscape-2d \
  --band_std_mult 1.0 \
  --only_models relora \
  --only_checkpoints model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000


# 130m
CUDA_VISIBLE_DEVICES=4 python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_multi_chkpt_130m.py \
  --num_directions 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 128 \
  --c4_max_length 256 \
  --x_min -0.003 --x_max 0.0031 --x_interval 0.00025 \
  --y_min -0.003 --y_max 0.003 --y_interval 0.00025 \
  --log_every 20 \
  --output_root results/loss-landscape-2d \
  --band_std_mult 1.0 \
  --only_models relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000



# 350m
CUDA_VISIBLE_DEVICES=1 python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_multi_chkpt_350m.py \
  --num_directions 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.003 --x_max 0.0031 --x_interval 0.00025 \
  --y_min -0.003 --y_max 0.003 --y_interval 0.00025 \
  --log_every 20 \
  --output_root results/loss-landscape-2d \
  --band_std_mult 1.0 \
  --only_models cola \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000



# running
# 350M - llama, cola 