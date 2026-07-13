#!/usr/bin/env bash
# configs taken from https://github.com/xichen-fy/Fira/tree/main/pre_training_c4/scriptss

# 60M, Fira-AdamW, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_60m.json \
    --lr 0.01 \
    --alpha 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer fira_adamw \
    --save_every 1000 \
    --method fira \
    --save_dir checkpoints/fira-60m-rank256 \
    --run_name "fira_60m-rank256" \
    --save_dir checkpoints/fira-60m-rank256
