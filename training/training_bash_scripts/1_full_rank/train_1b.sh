#!/usr/bin/env bash

# 1B, Full-rank, 1 GPU, 1 Node, trained on 13.1B
# CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
#     --model_config training/configs/llama_configs/llama_1b.json \
#     --lr 0.0005 \
#     --batch_size 32 \
#     --total_batch_size 512 \
#     --num_training_steps 100000 \
#     --warmup_steps 10000 \
#     --weight_decay 0.0 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --optimizer adamw \
#     --save_every 1000 \
#     --method fullrank \
#     --offline_mode \
#     --offline_data_path training/datasets-7b/c4/tokenized \
#     --run_name "llama_1b" \
#     --save_dir checkpoints/llama-1b


CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_1b.json \
    --lr 0.0004 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --save_every 1000 \
    --method fullrank \
    --offline_mode \
    --offline_data_path training/datasets-7b/c4/tokenized \
    --run_name "llama_1b_lr_0.0004" \
    --save_dir checkpoints/llama-1b-lr-0.0004