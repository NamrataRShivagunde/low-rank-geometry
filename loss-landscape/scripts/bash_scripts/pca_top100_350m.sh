#!/bin/bash
# PCA top-100 directions for 350m.
# Usage: bash loss-landscape/scripts/bash_scripts/pca_top100_350m.sh <GPU_ID> [method1 method2 ...]
set -euo pipefail
GPU_ID=${1:-3}
shift || true
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"
SCRIPT="${REPO_ROOT}/loss-landscape/scripts/single_checkpoint_scripts/3b_landscape_2d_pca_topk_components_variants.py"
AGG_SCRIPT="${REPO_ROOT}/loss-landscape/scripts/single_checkpoint_scripts/4b_pca_aggregate_subsets.py"
COMMON="--bias_direction_mode zero --svd_weight_mode sigma --c4_max_examples 1000 --c4_batch_size 128 --c4_max_length 256 --x_min -0.05 --x_max 0.0501 --x_interval 0.004"

K_VALUES=$(seq 1 100 | tr '\n' ' ')

if [ $# -gt 0 ]; then
    METHODS="$@"
else
    METHODS="llama cola fira galore relora sltrain"
fi

CKPTS="model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000"
RESULT_ROOT="results/pca-verification-k100"

for m in ${METHODS}; do
    for c in ${CKPTS}; do
        out="${RESULT_ROOT}/350m/${m}/${c}"
        if [ -f "${out}/stats.json" ]; then
            echo "SKIP sweep: ${out}"
        else
            echo "$(date '+%H:%M:%S') k=1..100 -> ${out}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n =ll-training --no-capture-output python ${SCRIPT} \
                --model ${m} --checkpoint "CHECKPOINTS/models-350m/${m}-350m/${c}" --tokenizer t5-base \
                --k_values ${K_VALUES} ${COMMON} --output_dir "${out}"
        fi
        if [ ! -f "${out}/subset_k100/sharpness/sharpness_summary.json" ]; then
            conda run -n =ll-training --no-capture-output python ${AGG_SCRIPT} --run_dir "${out}" --subsets 5 10 100
        else
            echo "SKIP subsets: ${out}"
        fi
    done
done
echo "$(date '+%H:%M:%S') 350m done!"
