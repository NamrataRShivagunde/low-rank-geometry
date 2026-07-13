#!/usr/bin/env bash
# Sweep ||theta||^2 over all checkpoints for ONE method at ONE size.
# Usage:  SIZE=60 METHOD=llama GPU=7 bash 11a_param_norm_sweep.sh
set -euo pipefail

SIZE="${SIZE:?set SIZE=60|130|350}"
METHOD="${METHOD:?set METHOD=llama|cola|fira|galore|relora|sltrain}"
GPU="${GPU:-7}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

case "$SIZE" in
  60)  STEPS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000) ;;
  130) STEPS=(2000 4000 6000 8000 10000 12000 14000 16000 18000 20000) ;;
  350) STEPS=(6000 12000 18000 24000 30000 36000 42000 48000 54000 60000) ;;
  *) echo "Unknown SIZE=$SIZE"; exit 1 ;;
esac

CKPT_ROOT="CHECKPOINTS/models-${SIZE}m/${METHOD}-${SIZE}m"
OUT_CSV="${OUT_CSV:-results/pl-metric/stage1_param_norm/${METHOD}-${SIZE}m.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
rm -f "$OUT_CSV"

echo "[param_norm_sweep] size=${SIZE}m method=$METHOD out=$OUT_CSV gpu=$GPU"

for step in "${STEPS[@]}"; do
  ckpt="${CKPT_ROOT}/model_${step}"
  if [[ ! -d "$ckpt" ]]; then
    echo "  [skip] $ckpt not found"
    continue
  fi
  echo "=== size=${SIZE}m method=$METHOD step=$step ==="
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n =ll-training --no-capture-output \
    python loss-landscape/scripts/single_checkpoint_scripts/11a_param_norm.py \
      --model "$METHOD" --checkpoint "$ckpt" --output_csv "$OUT_CSV" --seed 42
done

echo "[param_norm_sweep] done size=${SIZE}m method=$METHOD -> $OUT_CSV"
