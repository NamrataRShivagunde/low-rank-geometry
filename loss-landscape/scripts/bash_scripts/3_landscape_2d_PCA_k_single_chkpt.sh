# Single-checkpoint PCA top-k component averaging
CUDA_VISIBLE_DEVICES=6 python loss-landscape/scripts/single_checkpoint_scripts/3_landscape_2d_pca_topk_components.py \
  --model llama \
  --checkpoint CHECKPOINTS/models-60m/llama-60m/model_10000 \
  --task c4-val \
  --tokenizer t5-base \
  --top_k_components 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --output_dir results/loss-landscape-2d-PCA-k/models-60m/llama-60m/model_10000 \
  --log_every 10 \
  --band_std_mult 1.0
