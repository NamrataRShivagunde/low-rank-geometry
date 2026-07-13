#!/usr/bin/env python3
"""Overlay per-layer activation metrics across methods vs training steps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_COLOR_MAP = {
	"full_rank": "#1f77b4",  # blue
	"full-rank": "#1f77b4",
	"fullrank": "#1f77b4",
	"llama": "#1f77b4",  # blue (baseline)
	"galore": "#ff7f0e",  # orange
	"fira": "#2ca02c",  # green
	"cola": "#d62728",  # red
	"relora": "#9467bd",  # purple
	"switchlora": "#8c564b",  # brown
	"sltrain": "#e377c2",  # pink
}

FALLBACK_METHOD_COLORS = [
	"#17becf",
	"#bcbd22",
	"#7f7f7f",
	"#8c564b",
	"#e377c2",
	"#d62728",
	"#2ca02c",
	"#ff7f0e",
]


def _method_color(method_name: str) -> str:
	"""Get color for a method, using consistent colormap across all visualizations."""
	key = str(method_name).strip().lower()
	if key in METHOD_COLOR_MAP:
		return METHOD_COLOR_MAP[key]
	digest = hashlib.md5(key.encode("utf-8")).hexdigest()
	idx = int(digest[:8], 16) % len(FALLBACK_METHOD_COLORS)
	return FALLBACK_METHOD_COLORS[idx]


def _step_from_checkpoint(name: str) -> int | None:
    text = str(name).strip()
    if text.startswith("model_"):
        text = text[len("model_"):]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _method_from_model_folder(folder: str) -> str:
    text = str(folder).strip().lower()
    text = re.sub(r"[-_]\d+m$", "", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay per-layer activation metrics across methods vs training steps")
    parser.add_argument("--output_root", type=Path, required=True, help="Root output directory (results/activation-comparison)")
    parser.add_argument("--model_size", type=str, required=True, choices=["60m", "130m", "350m"])
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save overlay plots")
    parser.add_argument("--output_name", type=str, default="activation_per_layer_overlay", help="Output filename stem")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Specific metrics to plot (default: l2_distance, dot_product, cosine_similarity)",
    )
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset/order of methods to plot")
    parser.add_argument("--layers", nargs="*", default=None, help="Specific layers to plot (default: all). Format: layer_000, layer_001, etc.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def discover_per_layer_data(output_root: Path, model_size: str) -> dict[str, dict[str, dict[int, list[dict]]]]:
    """Discover per-layer metrics for all methods and checkpoints.
    
    Returns:
        data[method][checkpoint_name] = list of per-layer dicts
    """
    model_root = output_root / f"models-{model_size}"
    data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    
    for method_folder in sorted(model_root.iterdir()):
        if not method_folder.is_dir() or method_folder.name == "llama-" + model_size:
            continue
        
        method = _method_from_model_folder(method_folder.name)
        
        for checkpoint_folder in sorted(method_folder.iterdir()):
            if not checkpoint_folder.is_dir():
                continue
            
            per_layer_file = checkpoint_folder / "activation_metrics_per_layer.json"
            if per_layer_file.exists():
                try:
                    per_layer_data = json.loads(per_layer_file.read_text())
                    data[method][checkpoint_folder.name] = per_layer_data
                except Exception as e:
                    print(f"Warning: could not load {per_layer_file}: {e}")
    
    return data


def main() -> None:
    args = parse_args()

    output_root = args.output_root.resolve()
    
    print(f"Discovering per-layer data from {output_root}/models-{args.model_size}...")
    data = discover_per_layer_data(output_root, args.model_size)

    if not data:
        raise ValueError("No per-layer metrics found")

    default_metrics = ["mean_l2_distance", "relative_l2_distance", "cosine_similarity", "cka"]
    metrics_to_plot = args.metrics if args.metrics else default_metrics

    # Extract unique layers from all data
    all_layers_set = set()
    for method_data in data.values():
        for checkpoint_data in method_data.values():
            for layer_dict in checkpoint_data:
                all_layers_set.add(layer_dict.get("layer"))
    
    all_layers = sorted(all_layers_set)
    layers_to_plot = args.layers if args.layers else all_layers
    
    print(f"Found {len(all_layers)} unique layers")
    print(f"Plotting {len(layers_to_plot)} layers")
    print(f"Methods: {sorted(data.keys())}")

    if args.methods:
        method_order = [m.strip().lower() for m in args.methods if m.strip()]
    else:
        method_order = sorted(data.keys())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # For each layer and metric, create an overlay plot
    for layer_name in layers_to_plot:
        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(11, 6.5))

            plotted = 0
            for method in method_order:
                method_data = data.get(method, {})
                if not method_data:
                    continue

                # Collect (step, value) pairs for this method
                points = []
                for checkpoint_name, per_layer_list in sorted(method_data.items()):
                    step = _step_from_checkpoint(checkpoint_name)
                    if step is None:
                        continue

                    # Find this layer in the per-layer data
                    for layer_dict in per_layer_list:
                        if layer_dict.get("layer") == layer_name:
                            value = layer_dict.get(metric)
                            if value is not None:
                                points.append((step, float(value)))
                            break

                if not points:
                    continue

                points.sort(key=lambda x: x[0])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                color = _method_color(method)
                ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=method, color=color)
                plotted += 1

            if plotted == 0:
                print(f"Warning: no data for layer {layer_name} metric {metric}, skipping")
                plt.close(fig)
                continue

            ax.set_xlabel("training step", fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_title(f"{layer_name} - {metric}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=12)
            ax.legend(fontsize=12)

            png_path = args.output_dir / f"{args.output_name}_{layer_name}_{metric}.png"
            pdf_path = args.output_dir / f"{args.output_name}_{layer_name}_{metric}.pdf"

            fig.tight_layout()
            fig.savefig(png_path, dpi=args.dpi)
            fig.savefig(pdf_path)
            plt.close(fig)

            print(f"Saved {layer_name} {metric} overlay: {png_path}")


if __name__ == "__main__":
    main()
