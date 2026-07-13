#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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
    # Example: cola-60m -> cola
    text = re.sub(r"[-_]\d+m$", "", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay rank metrics across methods vs training steps")
    parser.add_argument("--summary_csv", type=Path, required=True, help="Path to rank_metrics_*_summary.csv")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save overlay plots")
    parser.add_argument("--output_name", type=str, default="rank_metrics_overlay", help="Output filename stem")
    parser.add_argument("--metrics", nargs="*", default=None, help="Specific metrics to plot (default: rank, effective_rank, stable_rank, spectral_gap)")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset/order of methods to plot")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    # Metrics to plot by default
    default_metrics = ["rank", "effective_rank", "stable_rank", "spectral_gap"]
    metrics_to_plot = args.metrics if args.metrics else default_metrics

    # Read CSV and organize by metric and method
    data: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    with open(args.summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = _method_from_model_folder(row["model_folder"])
            step = _step_from_checkpoint(row["checkpoint"])
            if step is None:
                continue

            for metric in metrics_to_plot:
                if metric not in row:
                    continue
                try:
                    value = float(row[metric])
                    data[metric][method].append((step, value))
                except (ValueError, KeyError):
                    pass

    if not data:
        raise ValueError(f"No valid rows found in {args.summary_csv}")

    if args.methods:
        method_order = [m.strip().lower() for m in args.methods if m.strip()]
    else:
        method_order = sorted(set().union(*[set(data[m].keys()) for m in data.keys()]))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create one plot per metric
    for metric in metrics_to_plot:
        if metric not in data:
            print(f"Warning: metric '{metric}' not found in CSV, skipping")
            continue

        series = data[metric]
        if not series:
            print(f"Warning: no data for metric '{metric}', skipping")
            continue

        fig, ax = plt.subplots(figsize=(11, 6.5))

        plotted = 0
        for method in method_order:
            points = series.get(method, [])
            if not points:
                continue
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=method)
            plotted += 1

        if plotted == 0:
            print(f"Warning: no methods plotted for metric '{metric}', skipping")
            plt.close(fig)
            continue

        ax.set_xlabel("training step", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12)

        png_path = args.output_dir / f"{args.output_name}_{metric}.png"
        pdf_path = args.output_dir / f"{args.output_name}_{metric}.pdf"

        fig.tight_layout()
        fig.savefig(png_path, dpi=args.dpi)
        fig.savefig(pdf_path)
        plt.close(fig)

        print(f"Saved {metric} overlay PNG: {png_path}")
        print(f"Saved {metric} overlay PDF: {pdf_path}")


if __name__ == "__main__":
    main()
