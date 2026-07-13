# 60M expected sharpness overlay plot (methods on same graph)
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d/summary/expected_sharpness_60m_summary.csv \
  --output_dir results/expected-sharpness-2d/summary \
  --output_name expected_sharpness_60m_overlay_worelora \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain


# 60M expected variance band overlay plot (methods on same graph)
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d/summary/average_variance_60m_summary.csv \
  --output_dir results/expected-sharpness-2d/summary \
  --output_name average_variance_60m_overlay_worelora \
  --metric_mode variance \
  --methods llama galore fira cola sltrain



# 130M expected sharpness overlay plot (methods on same graph)
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d/summary/expected_sharpness_130m_summary.csv \
  --output_dir results/expected-sharpness-2d/summary \
  --output_name expected_sharpness_130m_overlay_worelora \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain

# 130M expected variance band overlay plot (methods on same graph)
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d/summary/average_variance_130m_summary.csv \
  --output_dir results/expected-sharpness-2d/summary \
  --output_name average_variance_130m_overlay_worelora \
  --metric_mode variance \
  --methods llama galore fira cola sltrain


# 350M expected sharpness overlay plot (methods on same graph)
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d/summary/expected_sharpness_350m_summary.csv \
  --output_dir results/expected-sharpness-2d/summary \
  --output_name expected_sharpness_350m_overlay_worelora \
  --metric_mode sharpness \
  --methods llama galore fira cola sltrain

# 350M expected variance band overlay plot (methods on same graph)
python loss-landscape/scripts/multi_checkpoint_scripts/4_expected_sharpness_overlay.py \
  --summary_csv results/expected-sharpness-2d/summary/average_variance_350m_summary.csv \
  --output_dir results/expected-sharpness-2d/summary \
  --output_name average_variance_350m_overlay_worelora \
  --metric_mode variance \
  --methods llama galore fira cola sltrain