#!/usr/bin/env python3
"""Compute update magnitude and update rank between consecutive checkpoints.

For each method and model size, this script:
1) discovers available model_* checkpoints,
2) builds consecutive pairs (e.g., model_1000 -> model_2000),
3) computes per-matrix update metrics on DeltaW = W_t - W_{t-1},
4) writes pair-level outputs and a compact summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
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


_CHECKPOINT_STEP_TEMPLATES: dict[str, dict[str, list[int]]] = {
    "none": {},
    "size_default_steps": {
        "60m": list(range(1000, 10_001, 1000)),
        "130m": list(range(2000, 20_001, 2000)),
        "350m": list(range(6000, 60_001, 6000)),
    },
}


def _extract_layer_key(param_name: str) -> str | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(param_name)
        if match:
            return f"layer_{int(match.group(1)):03d}"
    return None


def _is_target_rank_matrix(param_name: str) -> bool:
    name = str(param_name).lower()
    return any(token in name for token in _TARGET_MODULE_TOKENS)


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(np.mean(np.asarray(values, dtype=float)))


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


def _step_from_checkpoint_dir(path: Path) -> int | None:
    match = re.fullmatch(r"model_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _parse_checkpoint_steps(raw_values: list[str]) -> list[int]:
    """Parse checkpoint-step inputs from space- or comma-separated CLI values."""
    parsed: list[int] = []
    for raw in raw_values:
        if raw is None:
            continue
        for piece in str(raw).split(","):
            token = piece.strip()
            if not token:
                continue
            parsed.append(int(token))
    return sorted(set(parsed))


def _template_steps_for_size(template_name: str, model_size: str) -> list[int]:
    template = _CHECKPOINT_STEP_TEMPLATES.get(str(template_name), {})
    return list(template.get(str(model_size), []))


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
            "update_rank": 0.0,
            "update_rank_ratio": 0.0,
            "update_effective_rank": 0.0,
            "update_effective_rank_score": 0.0,
            "update_stable_rank": 0.0,
            "update_spectral_gap": 0.0,
            "update_singular_value_ratio": 0.0,
            "update_num_singular_gt_threshold": num_singular_gt_threshold,
            "update_singular_gt_threshold_ratio": singular_gt_threshold_ratio,
        }

    rank = float(s.numel())
    rank_ratio = rank / float(min_dim)

    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = float(torch.exp(entropy).item())
    effective_rank_score = effective_rank / float(min_dim)

    stable_rank = float((torch.sum(s**2) / (s[0] ** 2)).item())

    top = s[0]
    second = s[1] if s.numel() > 1 else torch.zeros((), dtype=s.dtype, device=s.device)
    spectral_gap = float(((top - second) / top).item())
    singular_value_ratio = float((second / top).item())

    return {
        "update_rank": rank,
        "update_rank_ratio": rank_ratio,
        "update_effective_rank": effective_rank,
        "update_effective_rank_score": effective_rank_score,
        "update_stable_rank": stable_rank,
        "update_spectral_gap": spectral_gap,
        "update_singular_value_ratio": singular_value_ratio,
        "update_num_singular_gt_threshold": num_singular_gt_threshold,
        "update_singular_gt_threshold_ratio": singular_gt_threshold_ratio,
    }


def _load_model(model_name: str, checkpoint: Path, device: str) -> torch.nn.Module:
    args = SimpleNamespace(model=model_name, checkpoint=str(checkpoint), device=device)
    model = load_model_from_args(args)
    model.eval()
    return model


def _build_checkpoint_pairs(
    model_dir: Path,
    checkpoint_steps: list[int] | None = None,
) -> list[tuple[Path, Path, int, int]]:
    checkpoint_dirs: list[tuple[int, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        step = _step_from_checkpoint_dir(child)
        if step is None:
            continue
        checkpoint_dirs.append((step, child))

    checkpoint_dirs.sort(key=lambda x: x[0])
    if checkpoint_steps:
        requested = sorted(set(int(s) for s in checkpoint_steps))
        by_step = {step: path for step, path in checkpoint_dirs}
        missing = [step for step in requested if step not in by_step]
        if missing:
            print(
                f"Warning: {model_dir.name} missing requested checkpoints: "
                + ", ".join(str(s) for s in missing)
            )
        checkpoint_dirs = [(step, by_step[step]) for step in requested if step in by_step]

    pairs: list[tuple[Path, Path, int, int]] = []
    for idx in range(1, len(checkpoint_dirs)):
        prev_step, prev_path = checkpoint_dirs[idx - 1]
        curr_step, curr_path = checkpoint_dirs[idx]
        pairs.append((prev_path, curr_path, prev_step, curr_step))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rank/magnitude of weight updates between consecutive checkpoints",
    )
    parser.add_argument(
        "--checkpoints_root",
        type=Path,
        default=Path("CHECKPOINTS"),
        help="Root checkpoint directory that contains models-60m/models-130m/models-350m",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/rank-updates"),
        help="Output root directory",
    )
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=["60m", "130m", "350m"],
        help="Model sizes to process",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["llama", "galore", "fira", "cola", "sltrain", "relora"],
        help="Methods to process",
    )
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
        "--checkpoint_steps",
        nargs="*",
        default=[],
        help=(
            "Optional predefined checkpoint steps to pair (e.g. '1000 2000 3000' "
            "or '1000,2000,3000'). If provided, only these checkpoints are used."
        ),
    )
    parser.add_argument(
        "--checkpoint_template",
        type=str,
        choices=sorted(_CHECKPOINT_STEP_TEMPLATES.keys()),
        default="none",
        help=(
            "Predefined checkpoint-step template by model size. "
            "Use 'size_default_steps' for 60m/130m/350m default grids. "
            "Ignored when --checkpoint_steps is provided."
        ),
    )
    parser.add_argument(
        "--sv_threshold",
        type=float,
        default=0.1,
        help="Threshold for singular-value-count metric on DeltaW (default: 0.1)",
    )
    parser.add_argument(
        "--svd_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for SVD/rank computation. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda",
        help="Device hint passed to landscape loader (default: cuda)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_limit", type=int, default=0, help="Limit consecutive pairs per method-size (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svd_device = _resolve_svd_device(args.svd_device)
    cli_checkpoint_steps = _parse_checkpoint_steps(list(args.checkpoint_steps))
    if cli_checkpoint_steps:
        print(
            "Using CLI checkpoint steps: "
            + ", ".join(str(s) for s in cli_checkpoint_steps)
        )
    elif args.checkpoint_template != "none":
        print(f"Using checkpoint template: {args.checkpoint_template}")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    checkpoints_root = args.checkpoints_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_pair_summaries: list[dict[str, Any]] = []

    for size in args.model_sizes:
        size_dir = checkpoints_root / f"models-{size}"
        if not size_dir.exists():
            print(f"Warning: size directory missing: {size_dir}")
            continue

        for method in args.methods:
            model_folder = f"{method}-{size}"
            model_dir = size_dir / model_folder
            if not model_dir.exists():
                print(f"Warning: model directory missing: {model_dir}")
                continue

            requested_steps = (
                cli_checkpoint_steps if cli_checkpoint_steps else _template_steps_for_size(args.checkpoint_template, size)
            )
            if requested_steps and not cli_checkpoint_steps:
                print(
                    f"Using template steps for {model_folder}: "
                    + ", ".join(str(s) for s in requested_steps)
                )

            pairs = _build_checkpoint_pairs(model_dir, checkpoint_steps=requested_steps or None)
            if args.pair_limit > 0:
                pairs = pairs[: int(args.pair_limit)]
            if not pairs:
                print(f"Warning: no consecutive checkpoints found for {model_folder}")
                continue

            method_out_dir = output_root / f"models-{size}" / model_folder
            method_out_dir.mkdir(parents=True, exist_ok=True)

            method_summary_rows: list[dict[str, Any]] = []

            for pair_index, (prev_ckpt, curr_ckpt, prev_step, curr_step) in enumerate(pairs, start=1):
                pair_name = f"model_{prev_step}_to_model_{curr_step}"
                pair_out_dir = method_out_dir / pair_name
                pair_out_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"[{pair_index}/{len(pairs)}] {model_folder}: "
                    f"model_{prev_step} -> model_{curr_step}"
                )

                prev_model = _load_model(method, prev_ckpt, args.model_device)
                curr_model = _load_model(method, curr_ckpt, args.model_device)

                prev_named = dict(prev_model.named_parameters())
                curr_named = dict(curr_model.named_parameters())

                processed = 0
                skipped = 0
                records: list[dict[str, Any]] = []
                total_update_sq = 0.0
                total_prev_sq = 0.0

                for param_name, prev_param in prev_named.items():
                    curr_param = curr_named.get(param_name)

                    # some sanity checks
                    if curr_param is None:
                        skipped += 1
                        continue
                    if prev_param.shape != curr_param.shape:
                        skipped += 1
                        continue
                    if prev_param.ndim != 2:
                        continue

                    rows = int(prev_param.shape[0])
                    cols = int(prev_param.shape[1])
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

                    # only do for attn and mlp related matrics
                    if not args.include_all_layer_matrices and not _is_target_rank_matrix(param_name):
                        skipped += 1
                        continue

                    prev_w = prev_param.detach().float()
                    curr_w = curr_param.detach().float()
                    delta = (curr_w - prev_w).to(svd_device)

                    singular_values = torch.linalg.svdvals(delta)

                    update_norm = float(torch.linalg.matrix_norm(delta, ord="fro").item())
                    prev_norm = float(torch.linalg.matrix_norm(prev_w, ord="fro").item())
                    ratio = update_norm / prev_norm if prev_norm > 0 else math.nan

                    total_update_sq += update_norm * update_norm
                    total_prev_sq += prev_norm * prev_norm

                    sv_metrics = _metrics_from_singular_values(
                        singular_values=singular_values,
                        rows=rows,
                        cols=cols,
                        rank_tol=args.rank_tol,
                        sv_threshold=args.sv_threshold,
                    )

                    record = {
                        "param_name": param_name,
                        "layer": layer_key if layer_key is not None else "non_layer",
                        "rows": rows,
                        "cols": cols,
                        "numel": numel,
                        "update_fro_norm": update_norm,
                        "prev_fro_norm": prev_norm,
                        "update_to_prev_ratio": ratio,
                        **sv_metrics,
                    }
                    records.append(record)
                    processed += 1

                    if int(args.max_matrices) > 0 and processed >= int(args.max_matrices):
                        break

                del prev_model
                del curr_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pair_summary = {
                    "model_size": size,
                    "method": method,
                    "prev_checkpoint": str(prev_ckpt),
                    "curr_checkpoint": str(curr_ckpt),
                    "prev_step": prev_step,
                    "curr_step": curr_step,
                    "step_delta": curr_step - prev_step,
                    "matrix_scope": "all_layer_matrices" if args.include_all_layer_matrices else "attention_mlp_only",
                    "svd_device": str(svd_device),
                    "sv_threshold": float(args.sv_threshold),
                    "num_matrices_processed": processed,
                    "num_matrices_skipped": skipped,
                    "total_update_fro_norm": math.sqrt(total_update_sq),
                    "total_prev_fro_norm": math.sqrt(total_prev_sq),
                }
                pair_summary["total_update_to_prev_ratio"] = (
                    pair_summary["total_update_fro_norm"] / pair_summary["total_prev_fro_norm"]
                    if pair_summary["total_prev_fro_norm"] > 0
                    else math.nan
                )

                metric_keys = [
                    "update_fro_norm",
                    "update_to_prev_ratio",
                    "update_rank",
                    "update_rank_ratio",
                    "update_effective_rank",
                    "update_effective_rank_score",
                    "update_stable_rank",
                    "update_spectral_gap",
                    "update_singular_value_ratio",
                    "update_num_singular_gt_threshold",
                    "update_singular_gt_threshold_ratio",
                ]
                for key in metric_keys:
                    pair_summary[key] = _mean([float(r[key]) for r in records]) if records else math.nan

                pair_summary["total_num_singular_gt_threshold"] = (
                    float(sum(float(r["update_num_singular_gt_threshold"]) for r in records))
                    if records
                    else math.nan
                )

                method_summary_rows.append(pair_summary)
                all_pair_summaries.append(pair_summary)

                with open(pair_out_dir / "rank_update_summary.json", "w") as f:
                    json.dump(pair_summary, f, indent=2)
                with open(pair_out_dir / "rank_update_per_matrix.json", "w") as f:
                    json.dump(records, f, indent=2)
                with open(pair_out_dir / "rank_update_per_matrix.csv", "w", newline="") as f:
                    fieldnames = [
                        "param_name",
                        "layer",
                        "rows",
                        "cols",
                        "numel",
                        "update_fro_norm",
                        "prev_fro_norm",
                        "update_to_prev_ratio",
                        "update_rank",
                        "update_rank_ratio",
                        "update_effective_rank",
                        "update_effective_rank_score",
                        "update_stable_rank",
                        "update_spectral_gap",
                        "update_singular_value_ratio",
                        "update_num_singular_gt_threshold",
                        "update_singular_gt_threshold_ratio",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(records)

            method_summary_csv = method_out_dir / "rank_updates_summary.csv"
            method_summary_json = method_out_dir / "rank_updates_summary.json"
            with open(method_summary_json, "w") as f:
                json.dump(method_summary_rows, f, indent=2)

            if method_summary_rows:
                with open(method_summary_csv, "w", newline="") as f:
                    fieldnames = list(method_summary_rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(method_summary_rows)

    global_csv = output_root / "rank_updates_all_methods_sizes.csv"
    global_json = output_root / "rank_updates_all_methods_sizes.json"

    with open(global_json, "w") as f:
        json.dump(all_pair_summaries, f, indent=2)

    if all_pair_summaries:
        with open(global_csv, "w", newline="") as f:
            fieldnames = list(all_pair_summaries[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_pair_summaries)

    print(f"Saved global summary JSON: {global_json}")
    print(f"Saved global summary CSV: {global_csv}")


if __name__ == "__main__":
    main()
