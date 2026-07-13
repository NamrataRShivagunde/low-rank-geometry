# 60m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA/models-60m \
  --methods  llama galore fira cola sltrain relora \
  --checkpoints 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d-PCA/overlays \
  --output_name 60m_all_overlay \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.5,4.5 3.3,4.3 \
  --fig_width 4.0 \
  --fig_height 4.0

# 130m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA/models-130m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d-PCA/overlays \
  --output_name 130m_all_overlay \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.2,4.4 3,4.2 \
  --fig_width 4.0 \
  --fig_height 4.0

# 350m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA/models-350m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d-PCA/overlays \
  --output_name 350m_all_overlay \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 2.9,3.7 2.8,3.6 \
  --fig_width 4.0 \
  --fig_height 4.0


# sharpness or variance or both


# 60M (PCA)
CUDA_VISIBLE_DEVICES=6 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_60m.py \
  --landscape_root results/loss-landscape-2d-PCA/models-60m \
  --output_root results/expected-sharpness-2d-PCA \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000

# 130M (PCA)
CUDA_VISIBLE_DEVICES=2 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_130m.py \
  --landscape_root results/loss-landscape-2d-PCA/models-130m \
  --output_root results/expected-sharpness-2d-PCA \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000

# 350M (PCA)
CUDA_VISIBLE_DEVICES=3 python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_multi_chkpt_350m.py \
  --landscape_root results/loss-landscape-2d-PCA/models-350m \
  --output_root results/expected-sharpness-2d-PCA \
  --metric_mode variance \
  --only_models llama galore fira cola sltrain relora \
  --only_checkpoints model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000


# overlay graph

# 60M PCA sharpness overlay
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA/summary/expected_sharpness_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA/summary \
  --output_name expected_sharpness_60m_pca_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

# 130M PCA sharpness overlay
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA/summary/expected_sharpness_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA/summary \
  --output_name expected_sharpness_130m_pca_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora

# 350M PCA sharpness overlay
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA/summary/expected_sharpness_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA/summary \
  --output_name expected_sharpness_350m_pca_overlay \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain relora


# 60M PCA variance overlay
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA/summary/average_variance_60m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA/summary \
  --output_name average_variance_60m_pca_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA/summary/average_variance_130m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA/summary \
  --output_name average_variance_130m_pca_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora

python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d-PCA/summary/average_variance_350m_summary.csv \
  --output_dir results/expected-sharpness-2d-PCA/summary \
  --output_name average_variance_350m_pca_overlay \
  --metric_mode variance \
  --methods llama galore fira cola sltrain relora
