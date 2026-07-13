#!/usr/bin/env bash
# Driver: run Stage 2 + Stage 3 on all 60m methods sequentially on a single GPU.
# Usage:  GPU=0 bash 11bc_stage2_stage3_all_60m.sh
set -euo pipefail

GPU="${GPU:-0}"
METHODS=(llama cola fira galore relora sltrain)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/pl-metric

echo "======== Stage 2 sweep, GPU=$GPU, methods: ${METHODS[*]} ========"
for m in "${METHODS[@]}"; do
  echo ">>>>>> stage2 method=$m <<<<<<"
  METHOD="$m" GPU="$GPU" \
    bash loss-landscape/scripts/bash_scripts/11b_stage2_sweep_60m.sh \
    2>&1 | tee "logs/pl-metric/stage2_${m}_60m.log"
done

echo "======== Stage 3 sweep, GPU=$GPU, methods: ${METHODS[*]} ========"
for m in "${METHODS[@]}"; do
  echo ">>>>>> stage3 method=$m <<<<<<"
  METHOD="$m" GPU="$GPU" \
    bash loss-landscape/scripts/bash_scripts/11c_stage3_sweep_60m.sh \
    2>&1 | tee "logs/pl-metric/stage3_${m}_60m.log"
done

echo "======== All sweeps finished ========"
