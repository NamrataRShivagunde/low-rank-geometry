#!/usr/bin/env bash
# taken from https://github.com/andyjm3/SLTrain/tree/main/scripts/llm_pretrain

export PYTHONPATH="${PWD}/training:${PWD}/training/SLTrain:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_350m.json \
    --lr 0.002 \
    --peft_model sltrain \
    --optimizer adamw \
    --rank 256 \
    --sp_ratio 0.03 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 3000 \
    --lora_alpha 16 \
    --save_every 3000 \
    --method sltrain \
    --offline_mode \
    --offline_data_path training/datasets-7b/c4/tokenized \
    --run_name "sltrain_350m-lr2e-3" \
    --save_dir checkpoints/sltrain-350m-lr2e-3

#      - Total params: 194.29M                                                                                                                                             ││···························································
# │ 6:01:47.648 | INFO     | __main__:main:576 - Trainable params: 194.29M                                                                                                                                         ││···························································
# │ 6:01:47.648 | INFO     | __main__:main:579 - Saving model to checkpoints/sltrain-350m every 3000 update steps   