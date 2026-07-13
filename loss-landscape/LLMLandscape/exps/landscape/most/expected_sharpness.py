import argparse
import os
import json
import csv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from utils.plot import Landscape4Model, Landscape4ModelPCA
from tester.LMEval import lm_eval_c4


def compute_nll_loss(model, dataloader, pad_idx):
    """
    Token-normalized NLL computed over dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(model_device)
            attention_mask = batch[1].to(model_device)

            labels = input_ids.clone()
            labels[labels == pad_idx] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            batch_loss = outputs.loss.item()
            batch_tokens = (labels != -100).sum().item()

            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens

        if total_tokens == 0:
            return float("nan")

        return total_loss / total_tokens


def get_c4_dataloader(tokenizer, max_examples=1000, max_length=256, batch_size=128):
    print(f"Loading C4 validation split (up to {max_examples} examples) in streaming mode...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    dataset = dataset.shuffle(seed=42)

    streamed_samples = []
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        streamed_samples.append(example)

    tokenized = tokenizer(
        [ex["text"] for ex in streamed_samples],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    dataset_torch = torch.utils.data.TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)
    print(f"C4 DataLoader ready with {len(dataset_torch)} examples.")
    return dataloader


def load_model(checkpoint_path, model_type):
    if model_type == "llama":
        AutoConfig.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif model_type == "cola":
        from training.CoLA.cola import ColaConfig, ColaForCausalLM
        ColaConfig.from_pretrained(checkpoint_path)
        model = ColaForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Compute expected sharpness around a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["llama", "cola"], required=True)
    parser.add_argument("--task", type=str, choices=["c4-val"], default="c4-val")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--c4_max_examples", type=int, default=1000)
    parser.add_argument("--c4_max_length", type=int, default=256)
    parser.add_argument("--c4_batch_size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--num_directions", type=int, default=10)
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--use_pca", action="store_true", help="Use PCA/SVD principal components for perturbation directions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="output/expected_sharpness")
    parser.add_argument("--output_name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, args.model)
    model.to(device)

    if args.task == "c4-val":
        if args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                raise ValueError("Tokenizer has no pad/eos/unk token. Provide a tokenizer with a pad token.")

        pad_idx = tokenizer.pad_token_id
        dataloader = get_c4_dataloader(
            tokenizer,
            max_examples=args.c4_max_examples,
            max_length=args.c4_max_length,
            batch_size=args.c4_batch_size,
        )
        benchmark = lambda m: compute_nll_loss(m, dataloader, pad_idx)
    else:
        raise ValueError("Only c4-val is supported in this script.")

    if args.use_pca:
        drawer = Landscape4ModelPCA(
            model,
            benchmark,
            mode="2D",
            device=device,
        )
    else:
        drawer = Landscape4Model(
            model,
            benchmark,
            mode="2D",
            direction="WeightNormGaussian",
            device=device,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        ckpt_name = os.path.basename(os.path.normpath(args.checkpoint))
        output_name = f"{args.model}_{ckpt_name}_{args.task}"

    repeats = []
    component_deltas = []
    if args.use_pca:
        base_loss = float(benchmark(model))
        for direction_index in range(args.num_directions):
            if hasattr(drawer, "set_pca_component_index"):
                drawer.set_pca_component_index(direction_index)
            drawer.find_direction()
            with torch.no_grad():
                drawer.add_perturbation_to_params(args.epsilon, 0.0)
            perturbed_loss = float(benchmark(model))
            with torch.no_grad():
                drawer._recover_params(args.epsilon, 0.0)
            component_deltas.append(perturbed_loss - base_loss)

        expected_delta = float(np.mean(component_deltas)) if component_deltas else float("nan")
        expected_delta_std = float(np.std(component_deltas)) if component_deltas else float("nan")
        repeats.append(expected_delta)
        print(
            f"PCA components: n={args.num_directions}, base={base_loss:.6f}, "
            f"mean_delta={expected_delta:.6f}, std_delta={expected_delta_std:.6f}"
        )
    else:
        for r in range(args.num_repeats):
            base_loss = float(benchmark(model))
            deltas = []
            for direction_index in range(args.num_directions):
                drawer.find_direction(seed=None)
                with torch.no_grad():
                    drawer.add_perturbation_to_params(args.epsilon, 0.0)
                perturbed_loss = float(benchmark(model))
                with torch.no_grad():
                    drawer._recover_params(args.epsilon, 0.0)
                deltas.append(perturbed_loss - base_loss)
            repeat_delta = float(np.mean(deltas))
            repeats.append(repeat_delta)
            print(f"Repeat {r + 1}: base={base_loss:.6f}, delta={repeat_delta:.6f}")

        expected_delta = float(np.mean(repeats))
        expected_delta_std = float(np.std(repeats))

    if args.use_pca:
        print(f"Expected sharpness (mean ΔL over PCA components): {expected_delta:.6f} ± {expected_delta_std:.6f}")
    else:
        print(f"Expected sharpness (mean ΔL over repeats): {expected_delta:.6f} ± {expected_delta_std:.6f}")

    # Save outputs
    json_path = os.path.join(args.output_dir, f"{output_name}.json")
    csv_path = os.path.join(args.output_dir, f"{output_name}.csv")

    payload = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "task": args.task,
        "tokenizer": args.tokenizer,
        "c4_max_examples": args.c4_max_examples,
        "c4_max_length": args.c4_max_length,
        "c4_batch_size": args.c4_batch_size,
        "epsilon": args.epsilon,
        "num_directions": args.num_directions,
        "num_repeats": args.num_repeats,
        "num_repeats_effective": 1 if args.use_pca else args.num_repeats,
        "use_pca": bool(args.use_pca),
        "component_deltas": component_deltas if args.use_pca else None,
        "repeat_deltas": repeats,
        "expected_delta": expected_delta,
        "expected_delta_std": expected_delta_std,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "delta"])
        for i, d in enumerate(repeats, start=1):
            writer.writerow([i, d])

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
