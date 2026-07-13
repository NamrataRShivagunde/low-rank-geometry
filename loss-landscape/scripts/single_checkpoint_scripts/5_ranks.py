#!/usr/bin/env python3
"""Compute rank and related spectral metrics for a single checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_LANDSCAPE_ROOT = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "LLMLandscape").is_dir()),
    SCRIPT_DIR.parent,
)
LLM_ROOT = LOSS_LANDSCAPE_ROOT / "LLMLandscape"
sys.path.insert(0, str(LLM_ROOT))

# Allow landscape_eval_utils to import training.* modules (e.g., training.CoLA.cola)
REPO_ROOT = next(
    (p for p in [LOSS_LANDSCAPE_ROOT, *LOSS_LANDSCAPE_ROOT.parents] if (p / "training").is_dir()),
    LOSS_LANDSCAPE_ROOT.parent,
)
sys.path.insert(0, str(REPO_ROOT))

from exps.landscape.most.landscape_eval_utils import load_model_from_args


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


def _extract_layer_key(param_name: str) -> str | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(param_name)
        if match:
            return f"layer_{int(match.group(1)):03d}"
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(np.mean(np.asarray(values, dtype=float)))


def _is_target_rank_matrix(param_name: str) -> bool:
    """Keep only matrices from attention/MLP blocks where low-rank is typically applied."""
    name = str(param_name).lower()
    return any(token in name for token in _TARGET_MODULE_TOKENS)


def _metrics_from_singular_values(
    singular_values: torch.Tensor,
    rows: int,
    cols: int,
    rank_tol: float | None,
    sv_threshold: float,
) -> dict[str, float]:
    eps = 1e-12
    min_dim = max(1, min(rows, cols))
    tol = float(rank_tol) if rank_tol is not None else eps
    threshold = float(sv_threshold)

    num_singular_gt_threshold = float((singular_values > threshold).sum().item())
    singular_gt_threshold_ratio = num_singular_gt_threshold / float(min_dim)

    s = singular_values[singular_values > tol]
    if s.numel() == 0:
        return {
            "rank": 0.0,
            "rank_ratio": 0.0,
            "effective_rank": 0.0,
            "effective_rank_score": 0.0,
            "stable_rank": 0.0,
            "spectral_gap": 0.0,
            "singular_value_ratio": 0.0,
            "num_singular_gt_threshold": num_singular_gt_threshold,
            "singular_gt_threshold_ratio": singular_gt_threshold_ratio,
        }

    rank = float(s.numel())
    rank_ratio = rank / float(min_dim)

    p = s / (s.sum())
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = float(torch.exp(entropy).item())
    effective_rank_score = effective_rank / float(min_dim)

    stable_rank = float((torch.sum(s**2) / (s[0] ** 2)).item())

    top = s[0]
    second = s[1] if s.numel() > 1 else torch.zeros((), dtype=s.dtype, device=s.device)
    spectral_gap = float(((top - second) / (top)).item())
    singular_value_ratio = float((second / (top)).item())

    return {
        "rank": rank,
        "rank_ratio": rank_ratio,
        "effective_rank": effective_rank,
        "effective_rank_score": effective_rank_score,
        "stable_rank": stable_rank,
        "spectral_gap": spectral_gap,
        "singular_value_ratio": singular_value_ratio,
        "num_singular_gt_threshold": num_singular_gt_threshold,
        "singular_gt_threshold_ratio": singular_gt_threshold_ratio,
    }


def _resolve_svd_device(svd_device: str) -> torch.device:
    mode = str(svd_device).strip().lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--svd_device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported svd device mode: {svd_device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute rank and spectral metrics for a single checkpoint")
    parser.add_argument("--model", type=str, choices=["llama", "cola", "fira", "galore", "relora", "sltrain"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--include_all_layer_matrices",
        action="store_true",
        help="If set, include all 2D layer matrices. Default computes only attention/MLP matrices.",
    )
    parser.add_argument("--include_non_layer", action="store_true")
    parser.add_argument("--max_matrices", type=int, default=128)
    parser.add_argument("--max_matrix_elements", type=int, default=4_000_000)
    parser.add_argument("--min_matrix_dim", type=int, default=2)
    parser.add_argument("--rank_tol", type=float, default=None)
    parser.add_argument(
        "--sv_threshold",
        type=float,
        default=0.1,
        help="Threshold for counting singular values (default: 0.1)",
    )
    parser.add_argument(
        "--svd_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for SVD/rank computation. 'auto' uses CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svd_device = _resolve_svd_device(args.svd_device)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = load_model_from_args(args)
    model.eval()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "rank",
        "rank_ratio",
        "effective_rank",
        "effective_rank_score",
        "stable_rank",
        "spectral_gap",
        "singular_value_ratio",
        "num_singular_gt_threshold",
        "singular_gt_threshold_ratio",
    ]

    processed = 0
    skipped = 0
    records: list[dict[str, Any]] = []

    for param_name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue

        rows = int(parameter.shape[0])
        cols = int(parameter.shape[1])

        if rows < int(args.min_matrix_dim) or cols < int(args.min_matrix_dim):
            skipped += 1
            continue

        numel = rows * cols
        if numel > int(args.max_matrix_elements):
            skipped += 1
            continue

        layer_key = _extract_layer_key(param_name)
        if not args.include_non_layer and layer_key is None:
            skipped += 1
            continue

        if not args.include_all_layer_matrices and not _is_target_rank_matrix(param_name):
            skipped += 1
            continue

        matrix = parameter.detach().float().to(svd_device)
        try:
            singular_values = torch.linalg.svdvals(matrix)
        except RuntimeError as exc:
            if svd_device.type == "cuda":
                print(f"Warning: CUDA SVD failed for {param_name}, retrying on CPU ({exc})")
                singular_values = torch.linalg.svdvals(matrix.cpu())
            else:
                raise
        metrics = _metrics_from_singular_values(
            singular_values,
            rows,
            cols,
            rank_tol=args.rank_tol,
            sv_threshold=args.sv_threshold,
        )

        record = {
            "param_name": param_name,
            "layer": layer_key if layer_key is not None else "non_layer",
            "rows": rows,
            "cols": cols,
            "numel": numel,
            **metrics,
        }
        records.append(record)
        processed += 1

        if int(args.max_matrices) > 0 and processed >= int(args.max_matrices):
            break

    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "matrix_scope": "all_layer_matrices" if args.include_all_layer_matrices else "attention_mlp_only",
        "svd_device": str(svd_device),
        "sv_threshold": float(args.sv_threshold),
        "num_matrices_processed": processed,
        "num_matrices_skipped": skipped,
    }
    for key in metric_keys:
        summary[key] = _mean([float(r[key]) for r in records]) if records else math.nan

    summary_path = output_dir / "rank_metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    per_matrix_json = output_dir / "rank_metrics_per_matrix.json"
    with open(per_matrix_json, "w") as f:
        json.dump(records, f, indent=2)

    per_matrix_csv = output_dir / "rank_metrics_per_matrix.csv"
    with open(per_matrix_csv, "w", newline="") as f:
        fieldnames = [
            "param_name",
            "layer",
            "rows",
            "cols",
            "numel",
            *metric_keys,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved rank summary JSON: {summary_path}")
    print(f"Saved per-matrix JSON: {per_matrix_json}")
    print(f"Saved per-matrix CSV: {per_matrix_csv}")


if __name__ == "__main__":
    main()
