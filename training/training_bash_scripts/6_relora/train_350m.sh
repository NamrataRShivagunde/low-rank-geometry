#!/usr/bin/env bash

# 350M, ReLoRA, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --method relora \
    --model_config training/configs/llama_configs/llama_350m.json \
    --lr 0.0005 \
    --lora_r 256 \
    --relora 5000 \
    --restart_warmup_steps 200 \
    --reset_optimizer_on_relora False \
    --optimizer_random_pruning 0.99 \
    --optimizer_magnitude_pruning 0.0 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 3000 \
    --save_every 3000 \
    --optimizer adamw \
    --method relora \
    --scheduler cosine_restarts \
    --offline_mode \
    --offline_data_path training/datasets-7b/c4/tokenized \
    --run_name "relora_350m_lr5e-4" \
    --save_dir checkpoints/relora-350m-lr5e-4
