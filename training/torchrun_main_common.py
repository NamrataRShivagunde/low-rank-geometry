import os
import time
import json
import random
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from transformers import default_data_collator

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from GaLore.peft_pretraining import training_utils, args_utils
from GaLore.peft_pretraining.dataloader import PreprocessedIterableDataset
from GaLore.peft_pretraining.modeling_llama import LlamaForCausalLM
from CoLA.cola import ColaConfig, ColaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from fira import FiraAdamW


transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="fullrank", choices=["fullrank", 'fira', 'galore', "cola", "sltrain", "relora", "switchlora"])
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts", "warm_stable_decay"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--stable_steps", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    parser.add_argument("--tags", type=str, default=None)
    # offline data loading from disk, if set to False, will load data from HuggingFace datasets with streaming mode
    parser.add_argument("--offline_mode", default=False, action="store_true")
    parser.add_argument("--offline_data_path", type=str, default="/datasets/c4/tokenized")
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # Fira parameters
    parser.add_argument("--alpha", type=float, default=1.0)

    # SLTrain parameters
    parser.add_argument("--peft_model", type=str, default="full", choices=["full", "sltrain"])
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--sp_ratio", type=float, default=0.01)

    # ReLoRA parameters
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--relora", type=int, default=None)
    parser.add_argument("--restart_warmup_steps", type=int, default=0)
    parser.add_argument("--force_keep_original", default=False, action="store_true")
    parser.add_argument("--reset_optimizer_on_relora", type=bool, default=False)
    parser.add_argument("--optimizer_random_pruning", type=float, default=0.0, help="Randomly prune this ratio of weights in the optimizer state when restarting optimizer")
    parser.add_argument("--optimizer_magnitude_pruning", type=float, default=0.0, help="Prune this ratio of weights with smallest magnitude in the optimizer state when restarting optimizer")

    # SwitchLoRA parameters
    parser.add_argument("--use_lora", default=False, action="store_true")
    parser.add_argument("--switch_lora", default=False, action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--switch_lora_interval", type=int, default=40)
    parser.add_argument("--switch_lora_descent_rate", type=float, default=0.995)
    parser.add_argument("--quantize", type=str, default=None, choices=[None, "4bit", "8bit"])
    parser.add_argument("--use_double_quant", default=False, action="store_true")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, eval_dataloader=None):
    _time = time.time()

    if eval_dataloader is None:
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True) 
        val_data = val_data.shuffle(seed=42)

        if is_main_process():
            logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)
    


    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1

    if is_main_process():
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in (
        val_data_mapped.batch(batch_size=batch_size)
        if eval_dataloader is None
        else eval_dataloader
    ):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)

            if is_main_process():
                logger.info(
                    f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}"
                )

            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        model_name = Path(args.model_config).stem
        run_name = args.run_name or (None if args.name == "test" else args.name) or f"{args.method}-{model_name}"
        wandb.init(project="loss-landscape-low-rank", name=run_name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # For offline resume, read update_step early so we can skip consumed samples
    # before dataloader creation (avoids replaying old batches in Python loop).
    resume_update_step_for_data = 0
    if args.offline_mode and args.continue_from is not None:
        resume_state_path = os.path.join(args.continue_from, "training_state.json")
        if os.path.exists(resume_state_path):
            with open(resume_state_path) as f:
                _resume_state = json.load(f)
            resume_update_step_for_data = int(_resume_state.get("update_step", 0))
            if resume_update_step_for_data < 0:
                raise ValueError(f"Invalid update_step in {resume_state_path}: {resume_update_step_for_data}")
            logger.info(
                f"Offline resume pre-read found update_step={resume_update_step_for_data}; "
                "will skip consumed training samples before building dataloader"
            )
        else:
            logger.warning(
                f"Offline resume requested but missing {resume_state_path}; "
                "will fall back to loop-based skipping"
            )

    #data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
    # to support offline data mode
    if args.offline_mode:
        logger.info("Loading tokenized data from disk") # we are loading data with was downloaded usng shuffleseed 42, so it is already shuffled
        data = datasets.load_from_disk(args.offline_data_path)
        logger.info("Finished loading from disk")
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

        seed_for_shuffle = 42
        logger.info(f"Shuffling data with seed {seed_for_shuffle}")
        data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)


    # if not args.single_gpu:
    #     data = datasets.distributed.split_dataset_by_node(
    #         data, rank=global_rank, world_size=world_size,
    #     )

    if args.offline_mode:
        train_data: datasets.Dataset = data["train"]
        eval_data: datasets.Dataset = data["validation"]

    if not args.single_gpu:
        if args.offline_mode:
            train_data = datasets.distributed.split_dataset_by_node(
                train_data,
                rank=global_rank,
                world_size=world_size,
            )
            eval_data = datasets.distributed.split_dataset_by_node(
                eval_data,
                rank=global_rank,
                world_size=world_size,
            )
        else:
            data = datasets.distributed.split_dataset_by_node(
                data,
                rank=global_rank,
                world_size=world_size,
            )

    if args.offline_mode and resume_update_step_for_data > 0:
        consumed_samples_per_rank = (
            resume_update_step_for_data * args.gradient_accumulation * args.batch_size
        )
        train_len = len(train_data)
        if consumed_samples_per_rank >= train_len:
            raise ValueError(
                "Resume would skip the entire rank-local training dataset: "
                f"consumed={consumed_samples_per_rank}, train_len={train_len}. "
                "Check continue_from checkpoint and offline dataset consistency."
            )
        train_data = train_data.select(range(consumed_samples_per_rank, train_len))
        logger.info(
            "Offline resume fast-forward: "
            f"skipped {consumed_samples_per_rank} rank-local samples, "
            f"remaining {len(train_data)}"
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch


    # dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    if args.offline_mode:
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=default_data_collator,
            shuffle=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=default_data_collator,
            shuffle=True,
        )
    else:
        # it doesn't matter which tokenizer we use, because we train from scratch
        # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice

        dataset = PreprocessedIterableDataset(
            data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=args.workers
        )
        eval_dataloader = None


    if args.method.lower() == "cola":
        model_config = ColaConfig.from_pretrained(args.model_config)
        model = ColaForCausalLM(model_config)
        model.generation_config.pad_token_id=0
    else:
        model_config = AutoConfig.from_pretrained(args.model_config)
        if args.use_hf_model:
            model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
        else:
            model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.method.lower() == "switchlora":
        from SwitchLoRA.llama.switchlora import switch_lora
        from SwitchLoRA.llama.switchlora import switch_lora, lora_utils
        from SwitchLoRA.llama.switchlora.optimizer import StepAdamW

        args.use_lora = True
        args.switch_lora = True
        switch_lora.set_hyper_args(args)

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    resume_optimizer_state = None

    def _move_optimizer_state_to_device(obj, target_device):
        if torch.is_tensor(obj):
            return obj.to(device=target_device, non_blocking=True)
        if isinstance(obj, dict):
            return {key: _move_optimizer_state_to_device(value, target_device) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_move_optimizer_state_to_device(value, target_device) for value in obj]
        if isinstance(obj, tuple):
            return tuple(_move_optimizer_state_to_device(value, target_device) for value in obj)
        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in vars(obj).items():
                setattr(obj, attr_name, _move_optimizer_state_to_device(attr_value, target_device))
            return obj
        return obj

    def _load_model_state(path):
        from safetensors.torch import load_file as safe_load_file
        return safe_load_file(path, device="cpu")
   

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)


    if args.method.lower() == "sltrain":
        from SLTrain.train_utils import build_model as build_sltrain_model

        args.peft_model = "sltrain"
        args.target_modules = ["attn", "mlp", "attention"]
        model = build_sltrain_model(model, args)
    elif args.method.lower() == "relora":
        from ReLoRA.peft_pretraining.relora import ReLoRaModel

        need_linear_weight = (
            args.relora is not None
            or args.force_keep_original
            or args.continue_from is not None
        )
        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["attn", "attention", "mlp"],
            trainable_scaling=args.train_scaling,
            keep_original_weights=True,
            lora_only=not need_linear_weight,
        )
    elif args.method.lower() == "switchlora":
        from SwitchLoRA.llama.switchlora import switch_lora

        model = switch_lora.SwitchLoRAModel(
            model,
            ["attn", "attention", "mlp"],
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            quantize=args.quantize,
            use_double_quant=args.use_double_quant,
        )

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "model.safetensors")
        model_state = _load_model_state(checkpoint_path)

        if hasattr(model, "wrapped_model") or args.method.lower() in ["sltrain", "relora", "switchlora"]:
            model.wrapped_model.load_state_dict(model_state, strict=True)
        elif hasattr(model, "origin_model"):
            model.origin_model.load_state_dict(model_state, strict=True)
        else:
            model.load_state_dict(model_state, strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        optimizer_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            resume_optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded optimizer state from {optimizer_path}")
        else:
            raise FileNotFoundError(
                f"--continue_from requires optimizer state, but not found: {optimizer_path}"
            )

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
            if args.offline_mode and resume_update_step_for_data not in [0, update_step]:
                logger.warning(
                    "Mismatch between pre-read update_step for offline skipping "
                    f"({resume_update_step_for_data}) and loaded checkpoint update_step ({update_step})"
                )
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    if any(word in args.optimizer.lower() for word in ['fira', 'galore']):
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"enable {args.method} for weights in module: ", module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw

        if args.method.lower() == "galore":
            param_groups = [
                {"params": regular_params},
                {
                    "params": galore_params,
                    "rank": args.rank,
                    "update_proj_gap": args.update_proj_gap,
                    "scale": args.galore_scale,
                    "proj_type": args.proj_type,
                },
            ]
        elif args.method.lower() == "fira":
            param_groups = [
                {"params": regular_params},
                {
                    "params": galore_params,
                    "rank": args.rank,
                    "update_proj_gap": args.update_proj_gap,
                    "alpha": args.alpha,
                    "proj_type": args.proj_type,
                },
            ]

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if any(word in args.optimizer.lower() for word in ['fira', 'galore']):
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "fira_adamw":
        optimizer = FiraAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    stable_steps=args.stable_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        if args.method.lower() == "relora":
            scheduler = training_utils.get_scheculer(
                optimizer=optimizer,
                scheduler_type=args.scheduler,
                num_training_steps=args.num_training_steps,
                warmup_steps=args.warmup_steps,
                min_lr_ratio=args.min_lr_ratio,
                cycle_length=args.relora,
                restart_warmup_steps=args.restart_warmup_steps,
            )
        else:
            scheduler = training_utils.get_scheculer(
                optimizer=optimizer,
                scheduler_type=args.scheduler,
                num_training_steps=args.num_training_steps,
                warmup_steps=args.warmup_steps,
                stable_steps=args.stable_steps,
                min_lr_ratio=args.min_lr_ratio,
            )

    if resume_optimizer_state is not None:
        if "optimizer" in resume_optimizer_state:
            optimizer.load_state_dict(resume_optimizer_state["optimizer"])
            for _, state in optimizer.state.items():
                for state_key, state_value in list(state.items()):
                    state[state_key] = _move_optimizer_state_to_device(state_value, device)
            logger.info("Optimizer state restored and moved to device")
        else:
            logger.warning("Key 'optimizer' missing in optimizer checkpoint")

        if not layer_wise_flag and "scheduler" in resume_optimizer_state:
            scheduler.load_state_dict(resume_optimizer_state["scheduler"])
            logger.info("Scheduler state restored")

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ReLoRA-specific state — used only when args.method == "relora"
    if args.method.lower() == "relora":
        n_lora_restarts = 0
        n_optimizer_resets = 0
        scheduler_start_step = update_step
        if args.method.lower() == "relora":
            lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
            optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
        else:
            lora_params = []
            optimizer_state_keys = None

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()

    use_loop_skip_for_resume = not (
        args.offline_mode and args.continue_from is not None and resume_update_step_for_data > 0
    )

    for batch_idx, batch in enumerate(train_dataloader):
        if use_loop_skip_for_resume and batch_idx // args.gradient_accumulation < update_step:
            # Skipping data that are already seen in previous steps
            continue

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue


        # The below code is only executed during the update step
        
        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        grad_norm = sum(
            [
                torch.norm(p.grad.clone().detach().cpu())
                for p in model.parameters()
                if p.grad is not None
            ]
        )

        if global_rank == 0: pbar.update(1)
        
        if not layer_wise_flag:
            if args.method.lower() == "switchlora" and args.switch_lora:
                switch_lora.switch_lora(
                    model,
                    optimizer,
                    update_step,
                    args.switch_lora_descent_rate * args.num_training_steps,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(current_model_directory)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
                
            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # ##############################
        # RELORA: merge LoRA adapters into base weights, then reinit adapters.
        # This block is intentionally placed AFTER the periodic save so that
        # when save_every and the relora frequency coincide the checkpoint
        # captures the model state *before* the merge.
        if args.method.lower() == "relora" and args.relora is not None:
           
            if  ((update_step - scheduler_start_step) % args.relora == 1) and update_step>1:
                logger.info(
                    f"Performing lora reset at update step {update_step}. "
                    f"Current lr is {optimizer.param_groups[0]['lr']}"
                )
                n_lora_restarts += 1
                model.module.merge_and_reinit() # MERGE and REINIT Step

                from ReLoRA.peft_pretraining import training_utils as relora_training_utils
                logger.info(
                    f"Performing optimizer reset at update step {update_step}. "
                    f"Current lr is {optimizer.param_groups[0]['lr']}"
                )
                n_optimizer_resets += 1
                relora_training_utils.optimizer_reset(
                    optimizer,
                    reset_params=lora_params,
                    optimizer_state_keys=optimizer_state_keys,
                    reset_optimizer_on_relora=args.reset_optimizer_on_relora,
                    optimizer_random_pruning=args.optimizer_random_pruning,
                    optimizer_magnitude_pruning=args.optimizer_magnitude_pruning,
                )
        # ##############################

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            
            model.eval()

            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, eval_dataloader
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_perplexity": np.exp(total_loss),
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

            model.train()

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size
        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "gradnorm": grad_norm,
                "max_memory": max_memory,
            },
                step=global_step,
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, eval_dataloader
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_perplexity": np.exp(total_loss),
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )
    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
