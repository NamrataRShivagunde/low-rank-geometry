#!/bin/bash
# PCA k=5,10,100 for 350m only. ~3.3 hrs on one GPU.
# Usage: bash loss-landscape/scripts/bash_scripts/pca_k5_k10_k100_350m.sh <GPU_ID>
set -euo pipefail
GPU_ID=${1:-3}
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"
SCRIPT="loss-landscape/scripts/single_checkpoint_scripts/3b_landscape_2d_pca_topk_components_variants.py"
COMMON="--bias_direction_mode zero --svd_weight_mode sigma --c4_max_examples 1000 --c4_batch_size 128 --c4_max_length 256 --x_min -0.05 --x_max 0.0501 --x_interval 0.004"

run_single_k() {
    local k=$1 model=$2 ckpt_path=$3 out=$4
    if [ -f "${out}/stats.json" ]; then echo "SKIP: ${out}"; return; fi
    echo "k=${k} -> ${out}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n =ll-training --no-capture-output python ${SCRIPT} \
        --model ${model} --checkpoint ${ckpt_path} --tokenizer t5-base --k_values ${k} ${COMMON} --output_dir ${out}
    # build aggregate structure + compute sharpness
    local agg="${out}/aggregate/npy"; mkdir -p "${agg}"
    cp "${out}/k_$(printf '%03d' ${k})/loss_1d.npy" "${agg}/loss_mean.npy"
    conda run -n =ll-training --no-capture-output python -c "import numpy as np; m=np.load('${agg}/loss_mean.npy'); np.save('${agg}/loss_variance.npy', np.zeros_like(m))"
    conda run -n =ll-training --no-capture-output python loss-landscape/scripts/single_checkpoint_scripts/4_expected_sharpness.py \
        --landscape_dir "${out}" --output_dir "${out}/sharpness" --metric sharpness
}

METHODS="llama cola fira galore relora sltrain"
CKPTS="model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000"

for K in 5 10 100; do
    echo "===== 350m k=${K} ====="
    for m in ${METHODS}; do
        for c in ${CKPTS}; do
            run_single_k ${K} ${m} "CHECKPOINTS/models-350m/${m}-350m/${c}" "results/pca-verification-k${K}/350m/${m}/${c}"
        done
    done
done
echo "350m done!"
