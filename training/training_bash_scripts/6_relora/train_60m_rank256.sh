#!/usr/bin/env bash

# 60M, ReLoRA, 1 GPU, 1 Node
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --method relora \
    --model_config training/configs/llama_configs/llama_60m.json \
    --lr 0.001 \
    --lora_r 256 \
    --relora 1000 \
    --reset_optimizer_on_relora False \
    --optimizer_random_pruning 0.99 \
    --optimizer_magnitude_pruning 0.0 \
    --restart_warmup_steps 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 1000 \
    --optimizer adamw \
    --method relora \
    --scheduler cosine_restarts \
    --run_name "relora_60m-rank256" \
    --save_dir checkpoints/relora-60m-rank256

#      6:19:35.394 | INFO     | __main__:main:575 - Total params: 68.07M                                                                                                                                              ││···························································
# │ 6:19:35.395 | INFO     | __main__:main:576 - Trainable params: 42.77M                                                                                                                                          ││···························································
# │ 6:19:35.395 | INFO     | __main__:main:579 - Saving model to checkpoints/relora-60m every 1000 update steps 