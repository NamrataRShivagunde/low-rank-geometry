import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def compute_nll_loss(model, dataloader, pad_idx):
    """
    Mean of batch losses computed over dataloader.
    All tensors are moved to the model's device (GPU).
    """
    model.eval()
    total_loss = 0.0
    model_device = next(model.parameters()).device
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move all batch tensors to model device
            input_ids = batch[0].to(model_device)
            attention_mask = batch[1].to(model_device)

            labels = input_ids.clone()
            labels[labels == pad_idx] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            batch_loss = outputs.loss.item()

            total_loss += batch_loss
            num_batches += 1

        total_loss = total_loss / num_batches

        return total_loss


def get_c4_dataloader(tokenizer: AutoTokenizer, max_examples: int = 2000, max_length: int = 128, batch_size: int = 4):
    """Load a sampled subset of the C4 validation split in streaming mode and return a PyTorch DataLoader.

    Safe/small defaults are used to make smoke tests practical.
    """
    print(f"Loading C4 validation split (up to {max_examples} examples) in streaming mode...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    dataset = dataset.shuffle(seed=42)

    streamed_samples = []
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        streamed_samples.append(example)

    tokenized = tokenizer([ex["text"] for ex in streamed_samples], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    dataset_torch = torch.utils.data.TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)
    print(f"C4 DataLoader ready with {len(dataset_torch)} examples.")
    return dataloader


def load_model_from_args(args):
    """Build a model from args.model and args.checkpoint using the shared project logic."""
    requested_device = getattr(args, "device", "cuda")
    use_cuda = torch.cuda.is_available() and requested_device != "cpu"
    target_device = torch.device(requested_device if use_cuda else "cpu")

    if args.model == "llama" or args.model == "galore" or args.model == "fira":  # this will work for full rank, galore, fira
        AutoConfig.from_pretrained(args.checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    elif args.model == "cola":
        from training.CoLA.cola import ColaConfig, ColaForCausalLM

        ColaConfig.from_pretrained(args.checkpoint)
        model = ColaForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    elif args.model == "sltrain":
        from safetensors.torch import load_file as safe_load_file
        from training.SLTrain.train_utils import build_model as build_sltrain_model

        model_config = AutoConfig.from_pretrained(args.checkpoint)
        model = AutoModelForCausalLM.from_config(model_config)

        splora_cfg_path = os.path.join(args.checkpoint, "splora_config.json")
        if os.path.exists(splora_cfg_path):
            with open(splora_cfg_path, "r") as f:
                splora_cfg = json.load(f)
        else:
            splora_cfg = {}

        args.rank = splora_cfg.get("r", getattr(args, "rank", 128))
        args.lora_alpha = splora_cfg.get("lora_alpha", getattr(args, "lora_alpha", 32))
        args.lora_dropout = splora_cfg.get("lora_dropout", getattr(args, "lora_dropout", 0.0))
        args.sp_ratio = splora_cfg.get("sp_ratio", getattr(args, "sp_ratio", 0.01))
        args.train_scaling = splora_cfg.get("trainable_scaling", getattr(args, "train_scaling", False))
        args.target_modules = splora_cfg.get("target_modules", getattr(args, "target_modules", "attn,mlp,attention"))
        args.peft_model = "sltrain"

        model = build_sltrain_model(model, args)
        model = model.to(dtype=torch.bfloat16)

        checkpoint_state_path = os.path.join(args.checkpoint, "model.safetensors")
        checkpoint_state = safe_load_file(checkpoint_state_path, device="cpu")
        model.wrapped_model.load_state_dict(checkpoint_state, strict=False)
        model = model.to(target_device)

    elif args.model == "relora":
        from ReLoRA.peft_pretraining.relora import ReLoRaModel
        from safetensors.torch import load_file as safe_load_file

        model_config = AutoConfig.from_pretrained(args.checkpoint)
        model = AutoModelForCausalLM.from_config(model_config)

        relora_config_path = os.path.join(args.checkpoint, "relora_config.json")
        if os.path.exists(relora_config_path):
            with open(relora_config_path, "r") as f:
                relora_cfg = json.load(f)
        else:
            relora_cfg = {}

        args.lora_r = relora_cfg.get("r", getattr(args, "rank", 128))
        args.lora_alpha = relora_cfg.get("lora_alpha", getattr(args, "lora_alpha", 32))
        args.lora_dropout = relora_cfg.get("lora_dropout", getattr(args, "lora_dropout", 0.0))
        args.train_scaling = relora_cfg.get("trainable_scaling", getattr(args, "train_scaling", False))
        args.target_modules = relora_cfg.get("target_modules", getattr(args, "target_modules", "attn,mlp,attention"))

        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["attn", "attention", "mlp"],
            trainable_scaling=args.train_scaling,
            keep_original_weights=True,
            lora_only=False,
        )
        model = model.to(dtype=torch.bfloat16)

        checkpoint_path = os.path.join(args.checkpoint, "model.safetensors")
        model_state = safe_load_file(checkpoint_path, device="cpu")
        model.wrapped_model.load_state_dict(model_state, strict=False)
        model = model.to(target_device)

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    return model
