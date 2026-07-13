#!/usr/bin/env bash

# 1B, CoLA, 8 GPU, 1 Node
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 training/torchrun_main_common.py \
#     --model_config training/configs/cola_configs/cola_1b.json \
#     --lr 0.002 \
#     --optimizer adamw \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 100000 \
#     --warmup_steps 10000 \
#     --weight_decay 0.01 \
#     --dtype bfloat16 \
#     --eval_every 5000 \
#     --grad_clipping 0.5 \
#     --save_every 5000 \
#     --method cola \
#     --offline_mode \
#     --offline_data_path /datasets-7b/c4/tokenized \
#     --run_name "cola_1b" \
#     --save_dir checkpoints/cola-1b


CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node=1 training/torchrun_main_common.py \
    --model_config training/configs/cola_configs/cola_1b.json \
    --lr 0.002 \
    --optimizer adamw \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --save_every 1000 \
    --method cola \
    --offline_mode \
    --offline_data_path training/datasets-7b/c4/tokenized \
    --run_name "cola_1b" \
    --save_dir checkpoints/cola-1b