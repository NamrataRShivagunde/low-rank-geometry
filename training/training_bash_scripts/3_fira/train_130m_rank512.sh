#!/usr/bin/env bash

# 130M, Fira-AdamW, rank 512, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_130m.json \
    --lr 0.01 \
    --alpha 0.25 \
    --rank 512 \
    --update_proj_gap 200 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --optimizer fira_adamw \
    --save_every 2000 \
    --method fira \
    --run_name "fira_130m-rank512" \
    --save_dir checkpoints/fira-130m-rank512