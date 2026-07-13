#!/usr/bin/env bash
# Stage 1 (local PL estimate) over all llama-60m checkpoints.
# Uses the 1000-step spacing convention from CLAUDE.md §12.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

METHOD=llama
CKPT_ROOT="CHECKPOINTS/models-60m/${METHOD}-60m"
OUT_CSV="results/pl-metric/stage1/${METHOD}-60m.csv"
mkdir -p "$(dirname "$OUT_CSV")"

# Fresh sweep — remove any stale CSV so the new run starts with a header row.
rm -f "$OUT_CSV"

STEPS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
GPU=${CUDA_VISIBLE_DEVICES:-7}

echo "[11a_sweep] method=$METHOD out=$OUT_CSV gpu=$GPU steps=${STEPS[*]}"

for step in "${STEPS[@]}"; do
  ckpt="${CKPT_ROOT}/model_${step}"
  if [[ ! -d "$ckpt" ]]; then
    echo "  [skip] $ckpt not found"
    continue
  fi
  echo "=== step=$step ==="
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n =ll-training --no-capture-output \
    python loss-landscape/scripts/single_checkpoint_scripts/11a_pl_metric_local.py \
      --model "$METHOD" \
      --checkpoint "$ckpt" \
      --output_csv "$OUT_CSV" \
      --max_examples 1000 \
      --num_batches 30 \
      --batch_size 4 \
      --max_length 128 \
      --seed 42
done

echo "[11a_sweep] done -> $OUT_CSV"
