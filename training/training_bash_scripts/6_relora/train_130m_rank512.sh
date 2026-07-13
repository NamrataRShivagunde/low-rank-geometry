#!/usr/bin/env bash

# 130M, ReLoRA, rank 512, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --method relora \
    --model_config training/configs/llama_configs/llama_130m.json \
    --lr 0.001 \
    --lora_r 512 \
    --relora 5000 \
    --reset_optimizer_on_relora False \
    --optimizer_random_pruning 0.99 \
    --optimizer_magnitude_pruning 0.0 \
    --restart_warmup_steps 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --save_every 2000 \
    --optimizer adamw \
    --method relora \
    --scheduler cosine_restarts \
    --run_name "relora_130m-rank512" \
    --save_dir checkpoints/relora-130m-rank512