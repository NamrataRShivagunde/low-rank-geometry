#!/bin/bash
# Run PCA landscape sweep for k=5,10,100 across all methods, checkpoints, and sizes.
# Results saved to results/pca-verification-k5, results/pca-verification-k10, results/pca-verification-k100
#
# Usage:
#   bash loss-landscape/scripts/bash_scripts/pca_k5_k10_k100_all.sh <GPU_ID>
#   e.g. bash loss-landscape/scripts/bash_scripts/pca_k5_k10_k100_all.sh 1
#
# To parallelize across GPUs, run separate sizes on different GPUs:
#   GPU 1: 60m     (~40 min)
#   GPU 2: 130m    (~75 min)
#   GPU 3-4: 350m  (~1.5 hrs each half)

set -euo pipefail

GPU_ID=${1:-1}
SCRIPT="loss-landscape/scripts/single_checkpoint_scripts/3b_landscape_2d_pca_topk_components_variants.py"
COMMON_ARGS="--bias_direction_mode zero --svd_weight_mode sigma --c4_max_examples 1000 --c4_batch_size 128 --c4_max_length 256 --x_min -0.05 --x_max 0.0501 --x_interval 0.004"

run_single_k() {
    local k=$1
    local model=$2
    local checkpoint_path=$3
    local tokenizer=$4
    local output_dir=$5

    if [ -f "${output_dir}/stats.json" ]; then
        echo "SKIP (already done): ${output_dir}"
        return
    fi

    echo "Running k=${k} -> ${output_dir}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n =ll-training python ${SCRIPT} \
        --model ${model} \
        --checkpoint ${checkpoint_path} \
        --tokenizer ${tokenizer} \
        --k_values ${k} \
        ${COMMON_ARGS} \
        --output_dir ${output_dir}
}

compute_sharpness() {
    local landscape_dir=$1
    local output_dir="${landscape_dir}/sharpness"

    if [ -f "${output_dir}/sharpness_summary.json" ]; then
        echo "SKIP sharpness (already done): ${output_dir}"
        return
    fi

    # Create aggregate structure from k_XXX/loss_1d.npy so sharpness script works
    local k_dir
    k_dir=$(ls -d "${landscape_dir}"/k_* 2>/dev/null | head -1)
    if [ -z "${k_dir}" ]; then
        echo "WARN: no k_XXX dir in ${landscape_dir}"
        return
    fi

    local agg_npy="${landscape_dir}/aggregate/npy"
    mkdir -p "${agg_npy}"
    # Copy loss_1d.npy as loss_mean.npy (single direction = no averaging needed)
    cp "${k_dir}/loss_1d.npy" "${agg_npy}/loss_mean.npy"
    # Create zero variance (single direction)
    conda run -n =ll-training python -c "
import numpy as np
m = np.load('${agg_npy}/loss_mean.npy')
np.save('${agg_npy}/loss_variance.npy', np.zeros_like(m))
"

    echo "Computing sharpness: ${output_dir}"
    conda run -n =ll-training python loss-landscape/scripts/single_checkpoint_scripts/4_expected_sharpness.py \
        --landscape_dir "${landscape_dir}" \
        --output_dir "${output_dir}" \
        --metric sharpness
}

# ---- 60m ----
METHODS_60="llama cola fira galore relora sltrain"
CKPTS_60="model_1000 model_2000 model_3000 model_4000 model_5000 model_6000 model_7000 model_8000 model_9000 model_10000"

for K in 5 10 100; do
    echo ""
    echo "========================================="
    echo "  60m — k=${K}"
    echo "========================================="
    for method in ${METHODS_60}; do
        for ckpt in ${CKPTS_60}; do
            step=${ckpt#model_}
            checkpoint_path="CHECKPOINTS/models-60m/${method}-60m/${ckpt}"
            # llama uses llama-60m but the model arg is llama
            if [ "${method}" = "llama" ]; then
                model_arg="llama"
            else
                model_arg="${method}"
            fi
            output_dir="results/pca-verification-k${K}/60m/${method}/${ckpt}"
            run_single_k ${K} ${model_arg} ${checkpoint_path} t5-base "${output_dir}"
            compute_sharpness "${output_dir}"
        done
    done
done

# ---- 130m ----
METHODS_130="llama cola fira galore relora sltrain"
CKPTS_130="model_2000 model_4000 model_6000 model_8000 model_10000 model_12000 model_14000 model_16000 model_18000 model_20000"

for K in 5 10 100; do
    echo ""
    echo "========================================="
    echo "  130m — k=${K}"
    echo "========================================="
    for method in ${METHODS_130}; do
        for ckpt in ${CKPTS_130}; do
            step=${ckpt#model_}
            checkpoint_path="CHECKPOINTS/models-130m/${method}-130m/${ckpt}"
            if [ "${method}" = "llama" ]; then
                model_arg="llama"
            else
                model_arg="${method}"
            fi
            output_dir="results/pca-verification-k${K}/130m/${method}/${ckpt}"
            run_single_k ${K} ${model_arg} ${checkpoint_path} t5-base "${output_dir}"
            compute_sharpness "${output_dir}"
        done
    done
done

# ---- 350m ----
METHODS_350="llama cola fira galore relora sltrain"
CKPTS_350="model_6000 model_12000 model_18000 model_24000 model_30000 model_36000 model_42000 model_48000 model_54000 model_60000"

for K in 5 10 100; do
    echo ""
    echo "========================================="
    echo "  350m — k=${K}"
    echo "========================================="
    for method in ${METHODS_350}; do
        for ckpt in ${CKPTS_350}; do
            step=${ckpt#model_}
            checkpoint_path="CHECKPOINTS/models-350m/${method}-350m/${ckpt}"
            if [ "${method}" = "llama" ]; then
                model_arg="llama"
            else
                model_arg="${method}"
            fi
            output_dir="results/pca-verification-k${K}/350m/${method}/${ckpt}"
            run_single_k ${K} ${model_arg} ${checkpoint_path} t5-base "${output_dir}"
            compute_sharpness "${output_dir}"
        done
    done
done

echo ""
echo "All done!"
