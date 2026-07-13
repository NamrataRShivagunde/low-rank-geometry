#!/usr/bin/env python3
"""Compare activation statistics between a baseline checkpoint and a target checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_LANDSCAPE_ROOT = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "LLMLandscape").is_dir()),
    SCRIPT_DIR.parent,
)
LLM_ROOT = LOSS_LANDSCAPE_ROOT / "LLMLandscape"
sys.path.insert(0, str(LLM_ROOT))

# Allow landscape_eval_utils to import training.* modules.
REPO_ROOT = next(
    (p for p in [LOSS_LANDSCAPE_ROOT, *LOSS_LANDSCAPE_ROOT.parents] if (p / "training").is_dir()),
    LOSS_LANDSCAPE_ROOT.parent,
)
sys.path.insert(0, str(REPO_ROOT))

from exps.landscape.most.landscape_eval_utils import get_c4_dataloader, load_model_from_args


_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)blocks?\.(\d+)(?:\.|$)"),
]

_TARGET_MODULE_TOKENS = (
    "self_attn",
    "attn",
    "attention",
    "mlp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare checkpoint activations against a baseline model")
    parser.add_argument("--baseline_model", type=str, choices=["llama", "cola", "fira", "galore", "relora", "sltrain"], default="llama")
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--target_model", type=str, choices=["llama", "cola", "fira", "galore", "relora", "sltrain"], required=True)
    parser.add_argument("--target_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--tokenizer_checkpoint", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _extract_layer_key(name: str) -> str:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return f"layer_{int(match.group(1)):03d}"
    return "non_layer"


def _is_target_module(name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    lname = name.lower()
    return any(token in lname for token in _TARGET_MODULE_TOKENS)


def _get_activation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    pad_idx: int,
    max_batches: int,
) -> tuple | list:
    """Aggregate hidden states from all layers across all batches.
    
    Returns tuple/list of tensors, one per layer, each shape [total_positions, H].
    """
    model.eval()
    device = _model_device(model)
    _ = pad_idx  # Intentional: no masking, use all positions.

    layer_tensors: list[list[torch.Tensor]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            # Request hidden states from all layers.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
                raise RuntimeError("Model output did not include hidden_states")

            # Initialize layer accumulator on first batch
            if batch_idx == 0:
                layer_tensors = [[] for _ in range(len(outputs.hidden_states))]

            # Accumulate each layer's hidden states
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                # Reshape [B, s, H] -> [B*s, H]
                reshaped = hidden_state.float().cpu().reshape(-1, hidden_state.shape[-1])
                layer_tensors[layer_idx].append(reshaped)

    if not layer_tensors or not layer_tensors[0]:
        raise RuntimeError("No hidden states collected")

    # Concatenate all batches for each layer
    result = []
    for layer_idx in range(len(layer_tensors)):
        concatenated = torch.cat(layer_tensors[layer_idx], dim=0)
        result.append(concatenated)

    return tuple(result) 



def _compute_metrics(baseline_hidden_states: tuple | list, target_hidden_states: tuple | list) -> list[dict[str, Any]]:
    """Compute activation metrics per layer comparing baseline vs target.

    Each hidden state tensor has shape [N, H] where N = total positions across
    all batches and H = hidden dimension.  Metrics are computed **per position**
    (i.e. along dim=-1) and then averaged so that results are comparable across
    different token counts.

    Metrics returned per layer
    --------------------------
    baseline_mean_norm   : mean L2 norm of baseline activation vectors
    target_mean_norm     : mean L2 norm of target activation vectors
    mean_l2_distance     : mean per-position L2 distance  (lower = more similar)
    relative_l2_distance : mean_l2_distance / baseline_mean_norm  (scale-free)
    cosine_similarity    : mean per-position cosine similarity  (1 = identical direction)
    cka                  : linear CKA between the two activation matrices (1 = identical subspace)
    """
    eps = 1e-12
    metrics_per_layer: list[dict[str, Any]] = []

    if len(baseline_hidden_states) != len(target_hidden_states):
        raise RuntimeError(f"Layer count mismatch: {len(baseline_hidden_states)} vs {len(target_hidden_states)}")

    for layer_idx, (baseline_hidden, target_hidden) in enumerate(zip(baseline_hidden_states, target_hidden_states)):
        assert baseline_hidden.shape == target_hidden.shape, (
            f"Shape mismatch at layer {layer_idx}: {baseline_hidden.shape} vs {target_hidden.shape}"
        )

        # --- per-position norms ---
        baseline_norms = baseline_hidden.norm(dim=-1)          # [N]
        target_norms = target_hidden.norm(dim=-1)              # [N]
        baseline_mean_norm = float(baseline_norms.mean().item())
        target_mean_norm = float(target_norms.mean().item())

        # --- per-position L2 distance, then average ---
        per_pos_l2 = (baseline_hidden - target_hidden).norm(dim=-1)  # [N]
        mean_l2_distance = float(per_pos_l2.mean().item())

        # --- relative L2 (scale-free) ---
        relative_l2 = mean_l2_distance / (baseline_mean_norm + eps)

        # --- per-position cosine similarity, then average ---
        cos_sim = float(
            torch.nn.functional.cosine_similarity(baseline_hidden, target_hidden, dim=-1).mean().item()
        )

        # --- linear CKA (Kornblith et al., 2019) ---
        cka = float(_linear_cka(baseline_hidden, target_hidden, eps))

        metrics_per_layer.append(
            {
                "layer": f"layer_{layer_idx:03d}",
                "baseline_mean_norm": baseline_mean_norm,
                "target_mean_norm": target_mean_norm,
                "mean_l2_distance": mean_l2_distance,
                "relative_l2_distance": relative_l2,
                "cosine_similarity": cos_sim,
                "cka": cka,
            }
        )

    return metrics_per_layer


def _linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute linear CKA between two activation matrices.

    Args:
        x, y: [N, H] activation matrices (N positions, H features).

    Returns:
        Scalar CKA value in [0, 1].
    """
    # Center columns (features)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Gram matrices via dot products  (N x N would be huge; use the trace trick)
    # HSIC(X,Y) = ||Y^T X||_F^2 / (N-1)^2  for centered data
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    yx = y.T @ x                        # [Hy, Hx]
    xx = x.T @ x                        # [Hx, Hx]
    yy = y.T @ y                        # [Hy, Hy]

    hsic_xy = float((yx * yx).sum().item())
    hsic_xx = float((xx * xx).sum().item())
    hsic_yy = float((yy * yy).sum().item())

    denom = (hsic_xx * hsic_yy) ** 0.5
    if denom < eps:
        return 0.0
    return hsic_xy / denom


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))

    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False, trust_remote_code=True)
    
    dataloader = get_c4_dataloader(
        tokenizer=tokenizer,
        max_examples=int(args.max_examples),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
    )

    baseline_args = argparse.Namespace(
        model=args.baseline_model,
        checkpoint=args.baseline_checkpoint,
        device=args.device,
    )
    target_args = argparse.Namespace(
        model=args.target_model,
        checkpoint=args.target_checkpoint,
        device=args.device,
    )

    print(f"Loading baseline model: {args.baseline_model} from {args.baseline_checkpoint}")
    baseline_model = load_model_from_args(baseline_args)

    print(f"Loading target model: {args.target_model} from {args.target_checkpoint}")
    target_model = load_model_from_args(target_args)

    print("Computing baseline activations (all layers)...")
    baseline_hidden_states = _get_activation(
        model=baseline_model,
        dataloader=dataloader,
        pad_idx=int(tokenizer.pad_token_id),
        max_batches=int(args.max_batches),
    )

    print("Computing target activations (all layers)...")
    target_hidden_states = _get_activation(
        model=target_model,
        dataloader=dataloader,
        pad_idx=int(tokenizer.pad_token_id),
        max_batches=int(args.max_batches),
    )

    baseline_num_layers = len(baseline_hidden_states)
    target_num_layers = len(target_hidden_states)

    if baseline_num_layers != target_num_layers:
        raise RuntimeError(
            f"Layer count mismatch: baseline={baseline_num_layers} target={target_num_layers}"
        )

    # Compute metrics per layer
    print("Computing per-layer activation metrics...")
    per_layer_metrics = _compute_metrics(baseline_hidden_states, target_hidden_states)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-layer metrics
    per_layer_json = output_dir / "activation_metrics_per_layer.json"
    with open(per_layer_json, "w") as f:
        json.dump(per_layer_metrics, f, indent=2)

    per_layer_csv = output_dir / "activation_metrics_per_layer.csv"
    if per_layer_metrics:
        with open(per_layer_csv, "w", newline="") as f:
            fieldnames = list(per_layer_metrics[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_layer_metrics)

    # Create overall summary
    summary = {
        "baseline_model": args.baseline_model,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "target_model": args.target_model,
        "target_checkpoint": str(args.target_checkpoint),
        "max_examples": int(args.max_examples),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches),
        "num_layers": baseline_num_layers,
    }

    summary_json = output_dir / "activation_comparison_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    summary_csv = output_dir / "activation_comparison_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved overall summary JSON: {summary_json}")
    print(f"Saved overall summary CSV : {summary_csv}")
    print(f"Saved per-layer metrics JSON: {per_layer_json}")
    if per_layer_metrics:
        print(f"Saved per-layer metrics CSV : {per_layer_csv}")
    

if __name__ == "__main__":
    main()
