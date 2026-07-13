#!/bin/bash
# Overlay rank metrics across methods vs training steps (60m models)

set -e

METRICS=(
    rank
    effective_rank
    stable_rank
    spectral_gap
    num_singular_gt_threshold
    singular_gt_threshold_ratio
)

SUMMARY_CSV="${1:-results/rank-metrics/models-60m/summary/rank_metrics_60m_summary.csv}"
OUTPUT_DIR="${2:-results/rank-metrics/models-60m/overlay}"

python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_overlay.py \
    --summary_csv "$SUMMARY_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "rank_metrics_60m_overlay" \
    --metrics "${METRICS[@]}" \
    --methods llama galore fira cola sltrain relora \
    --dpi 220

echo "✓ Rank metrics overlay plots saved to: $OUTPUT_DIR"


#130m
set -e

SUMMARY_CSV="${1:-results/rank-metrics/models-130m/summary/rank_metrics_130m_summary.csv}"
OUTPUT_DIR="${2:-results/rank-metrics/models-130m/overlay}"

python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_overlay.py \
    --summary_csv "$SUMMARY_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "rank_metrics_130m_overlay" \
    --metrics "${METRICS[@]}" \
    --methods llama galore fira cola sltrain relora \
    --dpi 220

echo "✓ Rank metrics overlay plots saved to: $OUTPUT_DIR"


#350m
set -e

SUMMARY_CSV="${1:-results/rank-metrics/models-350m/summary/rank_metrics_350m_summary.csv}"
OUTPUT_DIR="${2:-results/rank-metrics/models-350m/overlay}"

python loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_overlay.py \
    --summary_csv "$SUMMARY_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "rank_metrics_350m_overlay" \
    --metrics "${METRICS[@]}" \
    --methods llama galore fira cola sltrain relora \
    --dpi 220

echo "✓ Rank metrics overlay plots saved to: $OUTPUT_DIR"
