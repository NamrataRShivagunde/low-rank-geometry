#!/usr/bin/env bash
# taken from https://github.com/andyjm3/SLTrain/tree/main/scripts/llm_pretrain

export PYTHONPATH="${PWD}/training:${PWD}/training/SLTrain:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_130m.json \
    --lr 0.003 \
    --peft_model sltrain \
    --optimizer adamw \
    --rank 256 \
    --sp_ratio 0.03 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --lora_alpha 16 \
    --save_every 2000 \
    --method sltrain \
    --run_name "sltrain_130m" \
    --save_dir checkpoints/sltrain-130m