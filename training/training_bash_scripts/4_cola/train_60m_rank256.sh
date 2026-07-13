#!/usr/bin/env bash

# 60M, CoLA, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 training/torchrun_main_common.py \
    --model_config training/configs/cola_configs/cola_60m_rank256.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --save_every 1000 \
    --method cola \
    --run_name "cola_60m-rank256" \
    --save_dir checkpoints/cola-60m-rank256
