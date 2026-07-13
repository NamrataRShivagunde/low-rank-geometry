#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/training:${PYTHONPATH:-}"

# Runs PCA-k landscapes for top-5 and top-10 components, computes expected sharpness,
# then creates overlays (per-k and k5-vs-k10 comparisons) for 60m/130m/350m.

# -----------------------------
# 1) PCA-k landscapes (top-5)
# -----------------------------
CUDA_VISIBLE_DEVICES=0 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_60m.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --output_root results/loss-landscape-2d-PCA-5 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 5 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000 \
  --mode 2D_PCA

CUDA_VISIBLE_DEVICES=1 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_130m.py \
  --checkpoint_root CHECKPOINTS/models-130m \
  --output_root results/loss-landscape-2d-PCA-5 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 5 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000 \
  --mode 2D_PCA

CUDA_VISIBLE_DEVICES=4 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_350m.py \
  --checkpoint_root CHECKPOINTS/models-350m \
  --output_root results/loss-landscape-2d-PCA-5 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 5 \
  --c4_max_examples 1000 \
  --c4_batch_size 128 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models sltrain relora \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --mode 2D_PCA

# ------------------------------
# 2) PCA-k landscapes (top-10)
# ------------------------------
CUDA_VISIBLE_DEVICES=7 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_60m.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --output_root results/loss-landscape-2d-PCA-10 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 10 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000 \
  --mode 2D_PCA

CUDA_VISIBLE_DEVICES=4 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_130m.py \
  --checkpoint_root CHECKPOINTS/models-130m \
  --output_root results/loss-landscape-2d-PCA-10 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 10 \
  --c4_max_examples 1000 \
  --c4_batch_size 256 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_20000 \
  --mode 2D_PCA

CUDA_VISIBLE_DEVICES=1 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_350m.py \
  --checkpoint_root CHECKPOINTS/models-350m \
  --output_root results/loss-landscape-2d-PCA-10 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 10 \
  --c4_max_examples 1000 \
  --c4_batch_size 128 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.005 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.005 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models cola \
  --only_checkpoints model_60000 \
  --mode 2D_PCA

# -----------------------------------------
# 3) Expected sharpness from PCA-5 outputs
# -----------------------------------------
CUDA_VISIBLE_DEVICES=6 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_60m.py \
  --landscape_root results/loss-landscape-2d-PCA-5/models-60m \
  --output_root results/expected-sharpness-2d-PCA-5 \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000

CUDA_VISIBLE_DEVICES=2 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_130m.py \
  --landscape_root results/loss-landscape-2d-PCA-5/models-130m \
  --output_root results/expected-sharpness-2d-PCA-5 \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000

CUDA_VISIBLE_DEVICES=3 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_350m.py \
  --landscape_root results/loss-landscape-2d-PCA-5/models-350m \
  --output_root results/expected-sharpness-2d-PCA-5 \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000

# ------------------------------------------
# 4) Expected sharpness from PCA-10 outputs
# ------------------------------------------
CUDA_VISIBLE_DEVICES=6 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_60m.py \
  --landscape_root results/loss-landscape-2d-PCA-10/models-60m \
  --output_root results/expected-sharpness-2d-PCA-10 \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000

CUDA_VISIBLE_DEVICES=2 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_130m.py \
  --landscape_root results/loss-landscape-2d-PCA-10/models-130m \
  --output_root results/expected-sharpness-2d-PCA-10 \
  --metric_mode both \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000

CUDA_VISIBLE_DEVICES=3 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_350m.py \
  --landscape_root results/loss-landscape-2d-PCA-10/models-350m \
  --output_root results/expected-sharpness-2d-PCA-10 \
  --metric_mode both \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000

# --------------------------------------
# 5)  overlays (PCA-5)
# --------------------------------------
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-5/summary/expected_sharpness_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-5/summary \
  --output_name expected_sharpness_60m_pca5_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-5/summary/expected_sharpness_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-5/summary \
  --output_name expected_sharpness_130m_pca5_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-5/summary/expected_sharpness_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-5/summary \
  --output_name expected_sharpness_350m_pca5_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

  # PCA-5 variance overlays
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-5/summary/average_variance_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-5/summary \
  --output_name average_variance_60m_pca5_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-5/summary/average_variance_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-5/summary \
  --output_name average_variance_130m_pca5_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-5/summary/average_variance_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-5/summary \
  --output_name average_variance_350m_pca5_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora


# ---------------------------------------
# 6) overlays (PCA-10)
# ---------------------------------------
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-10/summary/expected_sharpness_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-10/summary \
  --output_name expected_sharpness_60m_pca10_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-10/summary/expected_sharpness_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-10/summary \
  --output_name expected_sharpness_130m_pca10_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-10/summary/expected_sharpness_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-10/summary \
  --output_name expected_sharpness_350m_pca10_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

# varaince
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-10/summary/average_variance_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-10/summary \
  --output_name average_variance_60m_pca10_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-10/summary/average_variance_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-10/summary \
  --output_name average_variance_130m_pca10_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-10/summary/average_variance_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-10/summary \
  --output_name average_variance_350m_pca10_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora




  # top 100 components
  CUDA_VISIBLE_DEVICES=5 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_60m.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --output_root results/loss-landscape-2d-PCA-100 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 128 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.05 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.05 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_10000 \
  --mode 2D_PCA

  CUDA_VISIBLE_DEVICES=4 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_130m.py \
  --checkpoint_root CHECKPOINTS/models-130m \
  --output_root results/loss-landscape-2d-PCA-100 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 128 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.05 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.05 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models sltrain relora \
  --only_checkpoints model_20000 \
  --mode 2D_PCA


CUDA_VISIBLE_DEVICES=1 python loss-landscape/scripts/multi_checkpoint_scripts/3_landscape_2d_PCA_k_dir_multi_chkpt_350m.py \
  --checkpoint_root CHECKPOINTS/models-350m \
  --output_root results/loss-landscape-2d-PCA-100 \
  --task c4-val \
  --tokenizer t5-base \
  --num_directions 1 \
  --top_k_components 100 \
  --c4_max_examples 1000 \
  --c4_batch_size 32 \
  --c4_max_length 256 \
  --x_min -0.25 --x_max 0.251 --x_interval 0.1 \
  --y_min -0.25 --y_max 0.251 --y_interval 0.1 \
  --log_every 20 \
  --band_std_mult 1.0 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_60000 \
  --mode 2D_PCA

  CUDA_VISIBLE_DEVICES=6 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_60m.py \
  --landscape_root results/loss-landscape-2d-PCA-100/models-60m \
  --output_root results/expected-sharpness-2d-PCA-100 \
  --metric_mode sharpness \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_10000

CUDA_VISIBLE_DEVICES=2 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_130m.py \
  --landscape_root results/loss-landscape-2d-PCA-100/models-130m \
  --output_root results/expected-sharpness-2d-PCA-100 \
  --metric_mode both \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_20000

CUDA_VISIBLE_DEVICES=3 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_350m.py \
  --landscape_root results/loss-landscape-2d-PCA-100/models-350m \
  --output_root results/expected-sharpness-2d-PCA-100 \
  --metric_mode sharpness \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_60000

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-100/summary/expected_sharpness_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-100/summary \
  --output_name expected_sharpness_60m_pca100_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-100/summary/expected_sharpness_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-100/summary \
  --output_name expected_sharpness_130m_pca100_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-100/summary/expected_sharpness_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-100/summary \
  --output_name expected_sharpness_350m_pca100_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

  # variance overlays
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-100/summary/average_variance_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-100/summary \
  --output_name average_variance_60m_pca100_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora 

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-100/summary/average_variance_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-100/summary \
  --output_name average_variance_130m_pca100_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora 

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA-100/summary/average_variance_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA-100/summary \
  --output_name average_variance_350m_pca100_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora