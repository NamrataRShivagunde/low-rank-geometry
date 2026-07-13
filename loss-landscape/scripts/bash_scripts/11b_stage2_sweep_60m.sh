#!/usr/bin/env bash
# Stage 2 (NTK lambda_min + Hessian trace) over 60m checkpoints for ONE method.
# Usage:  METHOD=cola GPU=0 bash 11b_stage2_sweep_60m.sh
# Optional env overrides:
#   STEPS (space-separated), NTK_PROBES, LANCZOS_ITERS, HUTCHINSON_SAMPLES,
#   HESSIAN_BATCHES, BATCH_SIZE, MAX_LENGTH, NTK_BATCH, NTK_SEQ_LEN,
#   OUT_CSV, OUT_DIR
set -euo pipefail

METHOD="${METHOD:?set METHOD=llama|cola|fira|galore|relora|sltrain}"
GPU="${GPU:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# All 10 strategic checkpoints at 1k-step spacing (matches what other metrics use).
STEPS_STR="${STEPS:-1000 2000 3000 4000 5000 6000 7000 8000 9000 10000}"
read -r -a STEPS <<< "$STEPS_STR"

NTK_PROBES="${NTK_PROBES:-32}"
LANCZOS_ITERS="${LANCZOS_ITERS:-30}"
HUTCHINSON_SAMPLES="${HUTCHINSON_SAMPLES:-30}"
HESSIAN_BATCHES="${HESSIAN_BATCHES:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
NTK_BATCH="${NTK_BATCH:-1}"
NTK_SEQ_LEN="${NTK_SEQ_LEN:-32}"

CKPT_ROOT="CHECKPOINTS/models-60m/${METHOD}-60m"
OUT_DIR="${OUT_DIR:-results/pl-metric/stage2}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/${METHOD}-60m.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
# Fresh sweep; start with empty CSV so re-runs are clean.
rm -f "$OUT_CSV"

echo "[11b_sweep] method=$METHOD out=$OUT_CSV gpu=$GPU steps=${STEPS[*]}"
echo "[11b_sweep]   ntk_probes=$NTK_PROBES lanczos_iters=$LANCZOS_ITERS hutch=$HUTCHINSON_SAMPLES"

for step in "${STEPS[@]}"; do
  ckpt="${CKPT_ROOT}/model_${step}"
  if [[ ! -d "$ckpt" ]]; then
    echo "  [skip] $ckpt not found"
    continue
  fi
  echo "=== method=$METHOD step=$step ==="
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n =ll-training --no-capture-output \
    python loss-landscape/scripts/single_checkpoint_scripts/11b_pl_metric_ntk_hessian.py \
      --model "$METHOD" \
      --checkpoint "$ckpt" \
      --output_csv "$OUT_CSV" \
      --max_examples "$MAX_EXAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      --ntk_batch "$NTK_BATCH" \
      --ntk_seq_len "$NTK_SEQ_LEN" \
      --ntk_probes "$NTK_PROBES" \
      --lanczos_iters "$LANCZOS_ITERS" \
      --hessian_batches "$HESSIAN_BATCHES" \
      --hutchinson_samples "$HUTCHINSON_SAMPLES" \
      --seed 42
done

echo "[11b_sweep] done method=$METHOD -> $OUT_CSV"
