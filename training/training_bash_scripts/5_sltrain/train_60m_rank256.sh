#!/usr/bin/env bash
# taken from https://github.com/andyjm3/SLTrain/tree/main/scripts/llm_pretrain

export PYTHONPATH="${PWD}/training:${PWD}/training/SLTrain:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_60m.json \
    --lr 0.002 \
    --peft_model sltrain \
    --optimizer adamw \
    --rank 256 \
    --sp_ratio 0.03 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 32 \
    --save_every 1000 \
    --method sltrain \
    --run_name "sltrain_60m-rank256" \
    --save_dir checkpoints/sltrain-60m-lr-0.002-rank-256