#!/usr/bin/env bash

# 60M, Full-rank, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_60m.json \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --save_every 1000 \
    --method fullrank \
    --offline_mode \
    --offline_data_path /datasets-7b/c4/tokenized \
    --run_name "llama_60m" \
    --save_dir checkpoints/llama-60m
    #--continue_from checkpoints/llama_60m-2026-03-10-13-57-07/model_10/
