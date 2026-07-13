#!/usr/bin/env bash
# taken from https://github.com/andyjm3/SLTrain/tree/main/scripts/llm_pretrain

export PYTHONPATH="${PWD}/training:${PWD}/training/SLTrain:${PYTHONPATH}"

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
#     --model_config training/configs/llama_configs/llama_1b.json \
#     --lr 0.003 \
#     --peft_model sltrain \
#     --optimizer adamw \
#     --rank 256 \
#     --sp_ratio 0.03 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 100000 \
#     --warmup_steps 10000 \
#     --weight_decay 0 \
#     --dtype bfloat16 \
#     --eval_every 5000 \
#     --lora_alpha 8 \
#     --save_every 5000 \
#     --method sltrain \
#     --offline_mode \
#     --offline_data_path /datasets-7b/c4/tokenized \
#     --run_name "sltrain_1b" \
#     --save_dir checkpoints/sltrain-1b


CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_1b.json \
    --lr 0.003 \
    --peft_model sltrain \
    --optimizer adamw \
    --rank 256 \
    --sp_ratio 0.03 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 8 \
    --save_every 1000 \
    --method sltrain \
    --offline_mode \
    --offline_data_path training/datasets-7b/c4/tokenized \
    --run_name "sltrain_1b" \
    --save_dir checkpoints/sltrain-1b