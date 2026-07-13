#!/usr/bin/env bash

# 60M, SwitchLoRA, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --method switchlora \
    --model_config training/configs/llama_configs/llama_60m.json \
    --lr 0.001 \
    --lora_rank 128 \
    --switch_lora_interval 40 \
    --switch_lora_descent_rate 0.995 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100000 \
    --optimizer adamw \
    --method switchlora \
    --run_name "switchlora_60m" \
    --save_dir checkpoints/switchlora-60m
