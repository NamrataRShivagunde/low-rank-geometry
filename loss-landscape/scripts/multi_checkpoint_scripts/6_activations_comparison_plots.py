#!/usr/bin/env python3
"""Generate comparison plots for activation metrics across methods and model sizes.

Plot types:
  1. Per-layer overlay: metric vs training step, one line per method (existing)
  2. Last-layer comparison: metric at last layer across methods, grouped by size
  3. Cross-layer heatmap: methods vs layers for a given checkpoint step
  4. Summary trend: mean-across-layers metric vs training step, one line per method
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLOR_MAP = {
    "llama": "#1f77b4",
    "galore": "#ff7f0e",
    "fira": "#2ca02c",
    "cola": "#d62728",
    "relora": "#9467bd",
    "sltrain": "#e377c2",
}

METRICS = ["mean_l2_distance", "relative_l2_distance", "cosine_similarity", "cka"]

METRIC_LABELS = {
    "mean_l2_distance": "Mean L2 Distance",
    "relative_l2_distance": "Relative L2 Distance",
    "cosine_similarity": "Cosine Similarity",
    "cka": "Linear CKA",
    "baseline_mean_norm": "Baseline Mean Norm",
    "target_mean_norm": "Target Mean Norm",
}


def _method_color(method: str) -> str:
    key = method.strip().lower()
    return METHOD_COLOR_MAP.get(key, "#7f7f7f")


def _step_from_checkpoint(name: str) -> int | None:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def _method_from_folder(folder: str) -> str:
    return re.sub(r"[-_]\d+m$", "", folder.strip().lower())


def load_all_data(output_root: Path, model_size: str) -> dict[str, dict[str, list[dict]]]:
    """Load per-layer metrics: data[method][checkpoint_name] = list[dict]."""
    model_root = output_root / f"models-{model_size}"
    data: dict[str, dict[str, list[dict]]] = defaultdict(dict)

    if not model_root.exists():
        print(f"Warning: {model_root} not found")
        return data

    for method_folder in sorted(model_root.iterdir()):
        if not method_folder.is_dir():
            continue
        method = _method_from_folder(method_folder.name)
        if method == "llama":
            continue  # baseline, skip

        for ckpt_folder in sorted(method_folder.iterdir()):
            if not ckpt_folder.is_dir():
                continue
            pl_file = ckpt_folder / "activation_metrics_per_layer.json"
            if pl_file.exists():
                try:
                    data[method][ckpt_folder.name] = json.loads(pl_file.read_text())
                except Exception as e:
                    print(f"Warning: {pl_file}: {e}")

    return data


def plot_summary_trends(
    data: dict[str, dict[str, list[dict]]],
    model_size: str,
    output_dir: Path,
    metrics: list[str],
    dpi: int = 220,
) -> None:
    """Plot 4: mean-across-layers metric vs training step, one line per method."""
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        plotted = 0

        for method in sorted(data.keys()):
            points = []
            for ckpt_name, layers in sorted(data[method].items()):
                step = _step_from_checkpoint(ckpt_name)
                if step is None:
                    continue
                values = [ld.get(metric, 0.0) for ld in layers if ld.get(metric) is not None]
                if values:
                    points.append((step, float(np.mean(values))))

            if not points:
                continue
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=method, color=_method_color(method))
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        label = METRIC_LABELS.get(metric, metric)
        ax.set_xlabel("Training Step", fontsize=14)
        ax.set_ylabel(f"Mean {label} (across layers)", fontsize=14)
        ax.set_title(f"{model_size} — {label} vs Training Step (mean across layers)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"summary_trend_{model_size}_{metric}.png", dpi=dpi)
        fig.savefig(output_dir / f"summary_trend_{model_size}_{metric}.pdf")
        plt.close(fig)
        print(f"Saved summary trend: {metric} ({model_size})")


def plot_last_layer_comparison(
    data: dict[str, dict[str, list[dict]]],
    model_size: str,
    output_dir: Path,
    metrics: list[str],
    dpi: int = 220,
) -> None:
    """Plot 2: Last layer metric vs training step across methods."""
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        plotted = 0

        for method in sorted(data.keys()):
            points = []
            for ckpt_name, layers in sorted(data[method].items()):
                step = _step_from_checkpoint(ckpt_name)
                if step is None or not layers:
                    continue
                last_layer = layers[-1]
                val = last_layer.get(metric)
                if val is not None:
                    points.append((step, float(val)))

            if not points:
                continue
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="s", linewidth=2, markersize=5, label=method, color=_method_color(method))
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        label = METRIC_LABELS.get(metric, metric)
        ax.set_xlabel("Training Step", fontsize=14)
        ax.set_ylabel(label, fontsize=14)
        ax.set_title(f"{model_size} — Last Layer {label} vs Training Step", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"last_layer_{model_size}_{metric}.png", dpi=dpi)
        fig.savefig(output_dir / f"last_layer_{model_size}_{metric}.pdf")
        plt.close(fig)
        print(f"Saved last-layer plot: {metric} ({model_size})")


def plot_cross_layer_heatmap(
    data: dict[str, dict[str, list[dict]]],
    model_size: str,
    output_dir: Path,
    checkpoint_name: str,
    metrics: list[str],
    dpi: int = 220,
) -> None:
    """Plot 3: Heatmap of methods (rows) vs layers (cols) for one checkpoint."""
    for metric in metrics:
        methods_with_data = []
        layer_names: list[str] = []
        rows_data: list[list[float]] = []

        for method in sorted(data.keys()):
            ckpt_data = data[method].get(checkpoint_name)
            if not ckpt_data:
                continue
            if not layer_names:
                layer_names = [ld["layer"] for ld in ckpt_data]
            values = [ld.get(metric, float("nan")) for ld in ckpt_data]
            methods_with_data.append(method)
            rows_data.append(values)

        if not rows_data:
            continue

        mat = np.array(rows_data)
        fig, ax = plt.subplots(figsize=(max(10, len(layer_names) * 0.8), max(4, len(methods_with_data) * 0.6 + 2)))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn" if "cosine" in metric or "cka" in metric else "YlOrRd")
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels([l.replace("layer_", "L") for l in layer_names], rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(methods_with_data)))
        ax.set_yticklabels(methods_with_data, fontsize=12)

        # Annotate cells
        for i in range(len(methods_with_data)):
            for j in range(len(layer_names)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > np.nanmax(np.abs(mat)) * 0.6 else "black")

        label = METRIC_LABELS.get(metric, metric)
        ax.set_title(f"{model_size} — {label} — {checkpoint_name}", fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_{model_size}_{checkpoint_name}_{metric}.png", dpi=dpi)
        fig.savefig(output_dir / f"heatmap_{model_size}_{checkpoint_name}_{metric}.pdf")
        plt.close(fig)
        print(f"Saved heatmap: {metric} ({model_size}, {checkpoint_name})")


def plot_cross_size_last_layer(
    all_data: dict[str, dict[str, dict[str, list[dict]]]],
    output_dir: Path,
    metrics: list[str],
    dpi: int = 220,
) -> None:
    """Plot: Compare last-layer metric across model sizes for each method."""
    sizes = sorted(all_data.keys())
    for metric in metrics:
        fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 6), sharey=True)
        if len(sizes) == 1:
            axes = [axes]

        for ax, size in zip(axes, sizes):
            data = all_data[size]
            for method in sorted(data.keys()):
                points = []
                for ckpt_name, layers in sorted(data[method].items()):
                    step = _step_from_checkpoint(ckpt_name)
                    if step is None or not layers:
                        continue
                    val = layers[-1].get(metric)
                    if val is not None:
                        points.append((step, float(val)))
                if not points:
                    continue
                points.sort()
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=method, color=_method_color(method))

            ax.set_xlabel("Training Step", fontsize=12)
            ax.set_title(f"{size}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        label = METRIC_LABELS.get(metric, metric)
        axes[0].set_ylabel(f"Last Layer {label}", fontsize=12)
        fig.suptitle(f"Last Layer {label} — Cross-Size Comparison", fontsize=15, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"cross_size_last_layer_{metric}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / f"cross_size_last_layer_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved cross-size last-layer: {metric}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate activation comparison plots")
    parser.add_argument("--output_root", type=Path, required=True, help="Root output dir (results/activation-comparison)")
    parser.add_argument("--plot_dir", type=Path, required=True, help="Directory to save plots")
    parser.add_argument("--sizes", nargs="*", default=["60m", "130m", "350m"])
    parser.add_argument("--metrics", nargs="*", default=METRICS)
    parser.add_argument("--heatmap_checkpoints", nargs="*", default=None,
                        help="Checkpoint names for heatmaps (default: last checkpoint per size)")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    plot_dir = args.plot_dir.resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict[str, dict[str, list[dict]]]] = {}

    default_heatmap_ckpts = {
        "60m": "model_10000",
        "130m": "model_20000",
        "350m": "model_60000",
    }

    for size in args.sizes:
        print(f"\n{'='*60}")
        print(f"Processing {size}...")
        print(f"{'='*60}")

        data = load_all_data(output_root, size)
        if not data:
            print(f"No data for {size}, skipping")
            continue

        all_data[size] = data
        size_dir = plot_dir / size
        size_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Summary trends (mean across layers)
        plot_summary_trends(data, size, size_dir, args.metrics, args.dpi)

        # Plot 2: Last layer comparison
        plot_last_layer_comparison(data, size, size_dir, args.metrics, args.dpi)

        # Plot 3: Cross-layer heatmaps
        heatmap_ckpt = default_heatmap_ckpts.get(size, "model_10000")
        if args.heatmap_checkpoints:
            for hc in args.heatmap_checkpoints:
                plot_cross_layer_heatmap(data, size, size_dir, hc, args.metrics, args.dpi)
        else:
            plot_cross_layer_heatmap(data, size, size_dir, heatmap_ckpt, args.metrics, args.dpi)

    # Plot 4: Cross-size comparison
    if len(all_data) > 1:
        print(f"\n{'='*60}")
        print("Cross-size comparison plots...")
        print(f"{'='*60}")
        cross_dir = plot_dir / "cross_size"
        cross_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_size_last_layer(all_data, cross_dir, args.metrics, args.dpi)

    print(f"\nAll plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
