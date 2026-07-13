#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$REPO_ROOT/loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_layerwise_overlay.py"

SIZES=(60m 130m 350m)
METHODS=(llama cola galore fira relora sltrain)

for size in "${SIZES[@]}"; do
  # ----- rank-metrics (per-checkpoint weight rank) -----
  base="$REPO_ROOT/results/rank-metrics/models-${size}/models-${size}"
  out="$REPO_ROOT/results/rank-metrics/models-${size}/overlay/layerwise"
  for method in "${METHODS[@]}"; do
    md="$base/${method}-${size}"
    [ -d "$md" ] || continue
    python "$SCRIPT" \
      --method_dir "$md" \
      --per_matrix_csv_name rank_metrics_per_matrix.csv \
      --metric_kind ranks \
      --method_name "$method" \
      --size_label "$size" \
      --output_dir "$out"
  done

  # ----- rank-updates (DeltaW between consecutive checkpoints) -----
  base="$REPO_ROOT/results/rank-updates/models-${size}"
  out="$REPO_ROOT/results/rank-updates/models-${size}/overlay/layerwise"
  for method in "${METHODS[@]}"; do
    md="$base/${method}-${size}"
    [ -d "$md" ] || continue
    python "$SCRIPT" \
      --method_dir "$md" \
      --per_matrix_csv_name rank_update_per_matrix.csv \
      --metric_kind updates \
      --method_name "$method" \
      --size_label "$size" \
      --output_dir "$out"
  done
done

echo "All layerwise overlays generated."
