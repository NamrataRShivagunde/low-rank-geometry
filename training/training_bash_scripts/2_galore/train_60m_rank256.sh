# !/bin/bash
# optimizer change, model config will be from llama configs
# configs takens from https://github.com/jiaweizzhao/GaLore/tree/master/scripts/benchmark_c4

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --save_every 1000 \
    --method galore \
    --run_name "galore_60m-rank256" \
    --save_dir checkpoints/galore-60m-rank256