# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
#     --model_config training/configs/llama_configs/llama_1b.json \
#     --lr 0.01 \
#     --alpha 0.0625 \
#     --rank 512 \
#     --update_proj_gap 200 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 100000 \
#     --warmup_steps 10000 \
#     --weight_decay 0 \
#     --dtype bfloat16 \
#     --eval_every 5000 \
#     --optimizer fira_adamw \
#     --save_every 5000 \
#     --method fira \
#     --offline_mode \
#     --offline_data_path /datasets-7b/c4/tokenized \
#     --run_name "fira_1b" \
#     --save_dir checkpoints/fira-1b

w