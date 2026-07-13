CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 training/torchrun_main_common.py \
    --model_config training/configs/llama_configs/llama_350m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 3000 \
    --optimizer galore_adamw \
    --save_every 3000 \
    --method galore \
    --run_name "galore_350m-ol" \
    --save_dir checkpoints/galore-350m-ol \
    --offline_mode \
    --offline_data_path training/datasets-7b/c4/tokenized \
    --continue_from checkpoints/galore-350m-ol/model_42000

#     Total params: 367.97M                                                                                                                                     ││···························································
# │ -03-15 15:59:34.529 | INFO     | __main__:main:576 - Trainable params: 367.97M
# Total params with GaLore enabled: 302.38M   