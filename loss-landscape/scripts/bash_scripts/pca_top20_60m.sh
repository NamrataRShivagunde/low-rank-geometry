#!/bin/bash
# PCA top-20 directions for 60m: sweep k=1..20, then aggregate subsets (top-5, top-10, top-20).
# ~4 hrs on one GPU for all 6 methods x 10 checkpoints.
# Split by method: bash pca_top20_60m.sh <GPU_ID> method1 method2 ...
#
# Usage:
#   bash loss-landscape/scripts/bash_scripts/pca_top20_60m.sh 1 llama cola fira
#   bash loss-landscape/scripts/bash_scripts/pca_top20_60m.sh 2 galore relora sltrain
set -euo pipefail
GPU_ID=${1:-1}
shift || true
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/training"
SCRIPT="${REPO_ROOT}/loss-landscape/scripts/single_checkpoint_scripts/3b_landscape_2d_pca_topk_components_variants.py"
AGG_SCRIPT="${REPO_ROOT}/loss-landscape/scripts/single_checkpoint_scripts/4b_pca_aggregate_subsets.py"
COMMON="--bias_direction_mode zero --svd_weight_mode sigma --c4_max_examples 1000 --c4_batch_size 128 --c4_max_length 256 --x_min -0.05 --x_max 0.0501 --x_interval 0.004"

K_VALUES=$(seq 1 20 | tr '\n' ' ')

if [ $# -gt 0 ]; then
    METHODS="$@"
else
    METHODS="llama cola fira galore relora sltrain"
fi

CKPTS="model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000"
RESULT_ROOT="results/pca-verification-k20"

for m in ${METHODS}; do
    for c in ${CKPTS}; do
        out="${RESULT_ROOT}/60m/${m}/${c}"
        if [ -f "${out}/stats.json" ]; then
            echo "SKIP sweep: ${out}"
        else
            echo "$(date '+%H:%M:%S') k=1..20 -> ${out}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n =ll-training --no-capture-output python ${SCRIPT} \
                --model ${m} --checkpoint "CHECKPOINTS/models-60m/${m}-60m/${c}" --tokenizer t5-base \
                --k_values ${K_VALUES} ${COMMON} --output_dir "${out}"
        fi
        # Post-process: build top-5, top-10, top-20 subsets with sharpness
        if [ ! -f "${out}/subset_k020/sharpness/sharpness_summary.json" ]; then
            conda run -n =ll-training --no-capture-output python ${AGG_SCRIPT} --run_dir "${out}" --subsets 5 10 20
        else
            echo "SKIP subsets: ${out}"
        fi
    done
done
echo "$(date '+%H:%M:%S') 60m done!"
