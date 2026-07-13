#!/bin/bash
# PCA-5 and PCA-10 2D landscape overlays and variance overlays
# Add these to the main 2_landscape_2d_PCA_overlay.sh script or run separately

# ========================================
# PCA-5 2D LANDSCAPE OVERLAYS
# ========================================

# 60m PCA-5 2D landscape overlay
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA-5/models-60m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d-PCA-5/overlays \
  --output_name 60m_all_overlay_pca5 \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.5,4.5 3.3,4.3 \
  --fig_width 4.0 \
  --fig_height 4.0

# 130m PCA-5 2D landscape overlay
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA-5/models-130m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d-PCA-5/overlays \
  --output_name 130m_all_overlay_pca5 \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.2,4.4 3,4.2 \
  --fig_width 4.0 \
  --fig_height 4.0

# 350m PCA-5 2D landscape overlay
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA-5/models-350m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d-PCA-5/overlays \
  --output_name 350m_all_overlay_pca5 \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 2.9,3.7 2.8,3.6 \
  --fig_width 4.0 \
  --fig_height 4.0

# PCA-5 sharpness overlays
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


# ========================================
# PCA-10 2D LANDSCAPE OVERLAYS
# ========================================

# 60m PCA-10 2D landscape overlay
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA-10/models-60m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d-PCA-10/overlays \
  --output_name 60m_all_overlay_pca10 \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.5,4.5 3.3,4.3 \
  --fig_width 4.0 \
  --fig_height 4.0

# 130m PCA-10 2D landscape overlay
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA-10/models-130m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d-PCA-10/overlays \
  --output_name 130m_all_overlay_pca10 \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.2,4.4 3,4.2 \
  --fig_width 4.0 \
  --fig_height 4.0

# 350m PCA-10 2D landscape overlay
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d-PCA-10/models-350m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d-PCA-10/overlays \
  --output_name 350m_all_overlay_pca10 \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 2.9,3.7 2.8,3.6 \
  --fig_width 4.0 \
  --fig_height 4.0

# PCA-10 sharpness overlays
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

# PCA-10 variance overlays
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
