
# 60m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d/overlays/all-v2 \
  --output_name 60m_all_overlay \
  --grid_cols 4 \
  --band_std_mult 3.0 \
  --row_ylim 3.5,4.1 3.4,4 \
  --fig_width 4.0 \
  --fig_height 6.0

python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d/overlays/all-v2 \
  --output_name 130m_all_overlay \
  --grid_cols 4 \
  --band_std_mult 3.0 \
  --row_ylim 3.2,4.4 3,4.2 \
  --fig_width 4.0 \
  --fig_height 6.0

# 350m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/all-v2 \
  --output_name 350m_all_overlay \
  --grid_cols 4 \
  --band_std_mult 3.0 \
  --row_ylim 2.9,3.7 2.8,3.6 \
  --fig_width 4.0 \
  --fig_height 6.0



# Without relora as like to have a zoomed in version for other emthods
# 60m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods llama galore fira cola sltrain \
  --checkpoints 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d/overlays/all-v2 \
  --output_name 60m_all_overlay_wo_relora \
  --grid_cols 4 \
  --band_std_mult 3.0 \
  --row_ylim 3.5,3.85 3.4,3.75 \
  --fig_width 4.0 \
  --fig_height 6.0


python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods llama galore fira cola sltrain \
  --checkpoints 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d/overlays/all-v2 \
  --output_name 130m_all_overlay_wo_relora \
  --grid_cols 4 \
  --band_std_mult 3.0 \
  --row_ylim 3.2,3.5 3.1,3.4 \
  --fig_width 4.0 \
  --fig_height 6.0



# 350m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods llama galore fira cola sltrain \
  --checkpoints model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/all-v2 \
  --output_name 350m_all_overlay_wo_relora \
  --grid_cols 4 \
  --band_std_mult 3.0 \
  --row_ylim 2.9,3.3 2.8,3.2 \
  --fig_width 4.0 \
  --fig_height 6.0


############### big graph with overlay, not visually visible that well ############

# 60m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d/overlays/all \
  --output_name 60m_all_overlay \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.5,4.5 3.5,4.5


python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d/overlays/all \
  --output_name 130m_all_overlay \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.2,5.2 2.5,4.5

# 350m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods llama galore fira cola sltrain relora \
  --checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/all \
  --output_name 350m_all_overlay \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 2.95,4.75 2.7,4.5



# Without relora as like to have a zoomed in version for other emthods
# 60m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods llama galore fira cola sltrain \
  --checkpoints 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --output_dir results/loss-landscape-2d/overlays/all \
  --output_name 60m_all_overlay_wo_relora \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.55,4.45 3.1,4


python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods llama galore fira cola sltrain \
  --checkpoints 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --output_dir results/loss-landscape-2d/overlays/all \
  --output_name 130m_all_overlay_wo_relora \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 3.25,4 3.15,3.9


# 350m
python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods llama galore fira cola sltrain \
  --checkpoints model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/all \
  --output_name 350m_all_overlay_wo_relora \
  --grid_cols 5 \
  --band_std_mult 3.0 \
  --row_ylim 2.9,3.6 2.7,3.4


  ####################
 #Individual method as axis scaling if same do not reflect the curves well

 # 60M

  # This helps to see curves for llama and sltrain
  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods llama \
  --checkpoints 1000 5000 10000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 60m_llama \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 4.3,4.5 3.6,3.7 3.5,3.6 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods galore \
  --checkpoints 1000 5000 10000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 60m_galore \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 4.18,4.24 3.65,3.71 3.54,3.6 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods cola  \
  --checkpoints 1000 5000 10000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 60m_cola \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 4.18,4.24 3.66,3.72 3.5,3.56 \
  --fig_width 4.0 \
  --fig_height 6.0


  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods fira  \
  --checkpoints 1000 5000 10000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 60m_fira \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 4.15,4.25 3.55,3.65 3.43,3.53 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods sltrain  \
  --checkpoints 1000 5000 10000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 60m_sltrain \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 4.32,4.38 3.675,3.735 3.56,3.62 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-60m \
  --methods relora  \
  --checkpoints 1000 5000 10000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 60m_relora \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 5.4,6 3.6,4.2 3.6,4.2 \
  --fig_width 4.0 \
  --fig_height 6.0

  # 130m individual

python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods llama \
  --checkpoints 2000 10000 20000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 130m_llama \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.85,3.95 3.3,3.4 3.2,3.3 \
  --fig_width 4.0 \
  --fig_height 6.0

 python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods galore \
  --checkpoints 2000 10000 20000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 130m_galore \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.8,3.9 3.3,3.4 3.2,3.3 \
  --fig_width 4.0 \
  --fig_height 6.0

python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods fira \
  --checkpoints 2000 10000 20000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 130m_fira \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.8,3.85 3.25,3.30 3.13,3.18 \
  --fig_width 4.0 \
  --fig_height 6.0

   python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods cola \
  --checkpoints 2000 10000 20000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 130m_cola \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.83,3.88 3.35,3.4 3.23,3.28 \
  --fig_width 4.0 \
  --fig_height 6.0

    python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods sltrain \
  --checkpoints 2000 10000 20000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 130m_sltrain \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.88,3.98 3.35,3.45 3.25,3.35 \
  --fig_width 4.0 \
  --fig_height 6.0

   python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-130m \
  --methods relora \
  --checkpoints 2000 10000 20000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 130m_relora \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 5,6 3.3,4.3 3.3,4.3 \
  --fig_width 4.0 \
  --fig_height 6.0



  # 350m individual

python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods llama \
  --checkpoints model_6000 model_30000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 350m_llama \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.45,3.55 3,3.1 2.93,3.03 \
  --fig_width 4.0 \
  --fig_height 6.0

python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods galore \
  --checkpoints model_6000 model_30000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 350m_galore \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.43,3.53 3.03,3.13 2.96,3.06 \
  --fig_width 4.0 \
  --fig_height 6.0

python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods fira \
  --checkpoints model_6000 model_30000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 350m_fira \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.4,3.5 2.9,3.0 2.8,2.9 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods cola \
  --checkpoints model_6000 model_30000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 350m_cola \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.43,3.53 3.05,3.15 2.95,3.05 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods sltrain \
  --checkpoints model_6000 model_30000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 350m_sltrain \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 3.53,3.63 3.08,3.18 3,3.1 \
  --fig_width 4.0 \
  --fig_height 6.0

  python loss-landscape/scripts/multi_checkpoint_scripts/1_landscape_2d_overlay.py \
  --root results/loss-landscape-2d/models-350m \
  --methods relora \
  --checkpoints model_6000 model_30000 model_60000 \
  --output_dir results/loss-landscape-2d/overlays/individual \
  --output_name 350m_relora \
  --grid_cols 3 \
  --band_std_mult 3.0 \
  --col_ylim 4.2,5 3.2,4 3.2,4 \
  --fig_width 4.0 \
  --fig_height 6.0
