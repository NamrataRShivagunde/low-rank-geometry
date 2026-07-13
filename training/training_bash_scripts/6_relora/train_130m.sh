#!/usr/bin/env bash

# 130M, ReLoRA, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --method relora \
    --model_config training/configs/llama_configs/llama_130m.json \
    --lr 0.001 \
    --lora_r 256 \
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
    --run_name "relora_130m_lr1e-3" \
    --save_dir checkpoints/relora-130m-lr1e-3


# Total params: 178.93M
# 2026-03-15 16:20:52.819 | INFO     | __main__:main:576 - Trainable params: 94.00M