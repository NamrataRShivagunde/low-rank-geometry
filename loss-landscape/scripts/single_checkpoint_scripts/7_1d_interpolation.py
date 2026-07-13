#!/usr/bin/env python3
"""1-D interpolation between two checkpoints for all methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
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

from exps.landscape.most.landscape_eval_utils import get_c4_dataloader, compute_nll_loss, load_model_from_args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1-D interpolation between two checkpoints")
    parser.add_argument("--checkpoint_root", type=Path, required=False, default=None, help="Checkpoint root directory (e.g., CHECKPOINTS/models-60m)")
    parser.add_argument("--model_size", type=str, required=True, choices=["60m", "130m", "350m"])
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        nargs="*",
        default=None,
        help="Checkpoint steps to use. Consecutive pairs are interpolated. If omitted, uses size defaults.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to plot (default: all)")
    parser.add_argument("--alpha_points", type=int, default=20, help="Number of interpolation points (default: 21)")
    parser.add_argument("--max_batches", type=int, default=1000, help="Max batches for loss computation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing raw JSON, skip interpolation/loss computation")
    parser.add_argument("--raw_json", type=Path, default=None, help="Path to existing raw JSON (default: output_dir/interpolation_1d_<size>_raw.json)")


    return parser.parse_args()


def _default_checkpoint_steps(model_size: str) -> list[int]:
    if model_size == "60m":
        return [x for x in range(1000, 10001, 1000)]
    if model_size == "130m":
        return [x for x in range(2000, 20001, 2000)]
    if model_size == "350m":
        return [x for x in range(6000, 60001, 6000)]
    raise ValueError(f"Unsupported model size: {model_size}")


def _checkpoint_name(step: int) -> str:
    return f"model_{int(step)}"


def _pair_sort_key(pair_label: str) -> int:
    return int(pair_label.split("->")[0])


def _build_sequential_curve(
    pair_map: dict[str, list[float]],
    alphas: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Stitch consecutive pair interpolation curves into one step-axis curve."""
    xs: list[float] = []
    ys: list[float] = []

    for idx, (pair_label, losses) in enumerate(sorted(pair_map.items(), key=lambda kv: _pair_sort_key(kv[0]))):
        start_step, end_step = pair_label.split("->")
        start_val = float(start_step)
        end_val = float(end_step)

        pair_x = np.linspace(start_val, end_val, len(losses)).tolist()
        pair_y = [float(v) for v in losses]

        # Drop duplicated boundary point for all but first pair.
        if idx > 0 and pair_x:
            pair_x = pair_x[1:]
            pair_y = pair_y[1:]

        xs.extend(pair_x)
        ys.extend(pair_y)

    return xs, ys



def _interpolate_and_eval(
    baseline_model: torch.nn.Module,
    target_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    alphas: np.ndarray,
    max_batches: int,
) -> list[float]:
    """Interpolate between baseline and target models and compute losses.
    
    alpha = 0 -> baseline
    alpha = 1 -> target
    """
    device = next(baseline_model.parameters()).device
    
    # Store original parameters
    baseline_params = {name: p.detach().clone() for name, p in baseline_model.named_parameters()}
    target_params = {name: p.detach().clone() for name, p in target_model.named_parameters()}
    
    losses = []
    tokenizer = AutoTokenizer.from_pretrained("t5-base", trust_remote_code=True)
    
    for alpha in alphas:
        # Interpolate: (1-alpha) * baseline + alpha * target
        with torch.no_grad():
            for name, p in baseline_model.named_parameters():
                if name in target_params:
                    p.copy_((1.0 - alpha) * baseline_params[name] + alpha * target_params[name])
        
        loss = compute_nll_loss(baseline_model, dataloader, pad_idx=tokenizer.pad_token_id)
        losses.append(loss)
        print(f"  α={alpha:.2f}: loss={loss:.4f}")
    
    # Restore baseline parameters
    with torch.no_grad():
        for name, p in baseline_model.named_parameters():
            p.copy_(baseline_params[name])
    
    return losses


def _plot_results(
    output_dir: Path,
    model_size: str,
    alphas: np.ndarray,
    results: dict[str, dict[str, list[float]]],
    dpi: int,
) -> None:
    # Plot one figure per method, each with all consecutive checkpoint-pair curves.
    print(f"\n{'='*80}")
    print("Plotting results...")
    print(f"{'='*80}")

    for method, pair_map in sorted(results.items()):
        fig, ax = plt.subplots(figsize=(12, 7))

        for pair_label, losses in sorted(
            pair_map.items(), key=lambda kv: _pair_sort_key(kv[0])
        ):
            ax.plot(alphas, losses, marker="o", linewidth=2.0, markersize=4.0, label=pair_label)

        ax.set_xlabel("Interpolation alpha (0=start checkpoint, 1=end checkpoint)", fontsize=13)
        ax.set_ylabel("Validation Loss", fontsize=13)
        ax.set_title(f"1-D Interpolation Consecutive Checkpoints: {method}-{model_size}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best", ncol=2)
        ax.tick_params(axis="both", labelsize=11)

        fig.tight_layout()

        png_path = output_dir / f"interpolation_1d_{method}_{model_size}_consecutive.png"
        pdf_path = output_dir / f"interpolation_1d_{method}_{model_size}_consecutive.pdf"

        fig.savefig(png_path, dpi=dpi)
        fig.savefig(pdf_path)
        plt.close(fig)

        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

    # Additional plot: one graph with all methods, stitched over training-step axis.
    method_order = ["llama", "galore", "fira", "cola", "sltrain", "relora"]
    method_colors = {
        "llama": "#1f77b4",
        "cola": "#d62728",
        "fira": "#2ca02c",
        "galore": "#ff7f0e",
        "relora": "#bf3b2c6b",
        "sltrain": "#e377c2",
    }

    fig, ax = plt.subplots(figsize=(13, 7))

    for method in method_order:
        pair_map = results.get(method)
        if not pair_map:
            continue
        seq_x, seq_y = _build_sequential_curve(pair_map, alphas)
        if not seq_x:
            continue
        ax.plot(
            seq_x,
            seq_y,
            marker="o",
            linewidth=2.2,
            markersize=3.0,
            label=method,
            color=method_colors.get(method, "#333333"),
        )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Validation Loss", fontsize=13)
    ax.set_title(f"Sequential 1-D Interpolation Overlay Across Methods ({model_size})", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")
    ax.tick_params(axis="both", labelsize=11)

    overlay_png = output_dir / f"interpolation_1d_all_methods_{model_size}_sequential.png"
    overlay_pdf = output_dir / f"interpolation_1d_all_methods_{model_size}_sequential.pdf"
    fig.tight_layout()
    fig.savefig(overlay_png, dpi=dpi)
    fig.savefig(overlay_pdf)
    plt.close(fig)

    print(f"Saved overlay PNG: {overlay_png}")
    print(f"Saved overlay PDF: {overlay_pdf}")


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_raw_json = args.output_dir / f"interpolation_1d_{args.model_size}_raw.json"
    raw_json = args.raw_json.resolve() if args.raw_json else default_raw_json

    if args.plot_only:
        if not raw_json.exists():
            raise FileNotFoundError(f"Raw JSON not found for plot-only mode: {raw_json}")
        payload = json.loads(raw_json.read_text())
        alphas = np.array(payload["alphas"], dtype=float)
        results = payload["results"]
        _plot_results(args.output_dir, args.model_size, alphas, results, args.dpi)
        print(f"Loaded raw interpolation JSON: {raw_json}")
        return

    if args.checkpoint_root is None:
        raise ValueError("--checkpoint_root is required unless --plot_only is set")

    checkpoint_root = args.checkpoint_root.resolve()
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", trust_remote_code=True)
    
    # Dataloader
    dataloader = get_c4_dataloader(
        tokenizer=tokenizer,
        max_examples=args.max_examples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    
    # Interpolation points
    alphas = np.linspace(0, 1, args.alpha_points)

    checkpoint_steps = list(args.checkpoint_steps) if args.checkpoint_steps else _default_checkpoint_steps(args.model_size)
    checkpoint_steps = sorted(set(int(x) for x in checkpoint_steps))
    if len(checkpoint_steps) < 2:
        raise ValueError("Need at least 2 checkpoint steps for interpolation")
    checkpoint_pairs = list(zip(checkpoint_steps[:-1], checkpoint_steps[1:]))
    
    # Model size tag -> method list
    METHOD_MAP = {
        "60m": ["llama", "cola", "fira", "galore", "relora", "sltrain"],
        "130m": ["llama", "cola", "fira", "galore", "relora", "sltrain"],
        "350m": ["llama", "cola", "fira", "galore", "relora", "sltrain"],
    }
    
    methods = args.methods if args.methods else METHOD_MAP.get(args.model_size, [])
    
    # results[method][pair_label] = list of losses over alpha
    results: dict[str, dict[str, list[float]]] = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Processing method: {method}")
        print(f"{'='*80}")
        
        method_folder = checkpoint_root / f"{method}-{args.model_size}"

        if not method_folder.exists():
            print(f"  Method folder not found: {method_folder}, skipping")
            continue

        method_results: dict[str, list[float]] = {}

        for start_step, end_step in checkpoint_pairs:
            start_name = _checkpoint_name(start_step)
            end_name = _checkpoint_name(end_step)
            pair_label = f"{start_step}->{end_step}"

            start_ckpt = method_folder / start_name
            end_ckpt = method_folder / end_name
            if not start_ckpt.exists() or not end_ckpt.exists():
                print(f"  Missing pair {pair_label} for {method}, skipping pair")
                continue

            print(f"\n  Pair: {pair_label}")
            print(f"  Loading start checkpoint: {start_ckpt}")
            start_args = argparse.Namespace(
                model=method,
                checkpoint=str(start_ckpt),
                device=args.device,
            )
            start_model = load_model_from_args(start_args)

            print(f"  Loading end checkpoint: {end_ckpt}")
            end_args = argparse.Namespace(
                model=method,
                checkpoint=str(end_ckpt),
                device=args.device,
            )
            end_model = load_model_from_args(end_args)

            print(f"  Interpolating and computing losses for {method} {pair_label}...")
            losses = _interpolate_and_eval(start_model, end_model, dataloader, alphas, args.max_batches)
            method_results[pair_label] = losses

            del start_model, end_model
            torch.cuda.empty_cache()

        if method_results:
            results[method] = method_results
    
    if not results:
        raise RuntimeError("No interpolation results were produced. Check method/checkpoint availability.")

    with open(raw_json, "w") as f:
        json.dump({"alphas": alphas.tolist(), "results": results}, f, indent=2)

    _plot_results(args.output_dir, args.model_size, alphas, results, args.dpi)

    print(f"Saved raw interpolation JSON: {raw_json}")


if __name__ == "__main__":
    main()
