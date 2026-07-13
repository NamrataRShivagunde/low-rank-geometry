#!/usr/bin/env bash
# Recovery driver:
#   (1) Stage 2 for sltrain with --skip_ntk (NTK incompatible with torch.func due
#       to sltrain's custom autograd.Function; we still record hessian_trace).
#   (2) Stage 3 for all 6 methods (final 60m checkpoint, perturbation region).
set -uo pipefail

GPU="${GPU:-0}"
METHODS=(llama cola fira galore relora sltrain)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/pl-metric

# -----------------------------------------------------------------------------
# Stage 2 recovery for sltrain — Hessian trace only; NTK left as NaN.
# -----------------------------------------------------------------------------
SLTRAIN_CSV="results/pl-metric/stage2/sltrain-60m.csv"
mkdir -p "$(dirname "$SLTRAIN_CSV")"
rm -f "$SLTRAIN_CSV"

STEPS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
echo "======== Stage 2 recovery: sltrain (--skip_ntk), GPU=$GPU ========"
for step in "${STEPS[@]}"; do
  ckpt="CHECKPOINTS/models-60m/sltrain-60m/model_${step}"
  if [[ ! -d "$ckpt" ]]; then
    echo "  [skip] $ckpt not found"; continue
  fi
  echo "=== sltrain step=$step (NTK skipped, Hessian trace only) ==="
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n =ll-training --no-capture-output \
    python loss-landscape/scripts/single_checkpoint_scripts/11b_pl_metric_ntk_hessian.py \
      --model sltrain --checkpoint "$ckpt" \
      --output_csv "$SLTRAIN_CSV" \
      --max_examples 1000 --batch_size 4 --max_length 128 \
      --hessian_batches 8 --hutchinson_samples 30 \
      --skip_ntk --seed 42 \
    2>&1 | tee -a logs/pl-metric/stage2_sltrain_skipntk_60m.log
done

# -----------------------------------------------------------------------------
# Stage 3 — all 6 methods, final checkpoint (model_10000).
# -----------------------------------------------------------------------------
echo "======== Stage 3 sweep, GPU=$GPU, methods: ${METHODS[*]} ========"
for m in "${METHODS[@]}"; do
  echo ">>>>>> stage3 method=$m <<<<<<"
  METHOD="$m" GPU="$GPU" \
    bash loss-landscape/scripts/bash_scripts/11c_stage3_sweep_60m.sh \
    2>&1 | tee "logs/pl-metric/stage3_${m}_60m.log"
done

echo "======== Recovery sweeps finished ========"
