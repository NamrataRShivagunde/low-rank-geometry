#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay update-rank metrics across methods vs training step")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        required=True,
        help="Path to rank_updates_all_methods_sizes.csv or rank_updates_summary.csv",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save overlay plots")
    parser.add_argument("--output_name", type=str, default="rank_updates_overlay", help="Output filename stem")
    parser.add_argument("--model_size", type=str, default=None, help="Optional model size filter (e.g., 60m, 130m, 350m)")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Metrics to plot (default: total_update_to_prev_ratio, update_rank, "
            "update_effective_rank, update_stable_rank, update_fro_norm, "
            "update_num_singular_gt_threshold, update_singular_gt_threshold_ratio, "
            "total_num_singular_gt_threshold)"
        ),
    )
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset/order of methods to plot")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    default_metrics = [
        "total_update_to_prev_ratio",
        "update_rank",
        "update_effective_rank",
        "update_stable_rank",
        "update_fro_norm",
        "update_num_singular_gt_threshold",
        "update_singular_gt_threshold_ratio",
        "total_num_singular_gt_threshold",
    ]
    metrics_to_plot = args.metrics if args.metrics else default_metrics

    data: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    with open(args.summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows_seen = 0
        rows_used = 0

        for row in reader:
            rows_seen += 1

            size = str(row.get("model_size", "")).strip()
            if args.model_size and size != str(args.model_size).strip():
                continue

            method = str(row.get("method", "")).strip().lower()
            if not method:
                continue

            try:
                step = int(float(row.get("curr_step", "nan")))
            except ValueError:
                continue

            for metric in metrics_to_plot:
                if metric not in row:
                    continue
                try:
                    value = float(row[metric])
                except ValueError:
                    continue
                data[metric][method].append((step, value))

            rows_used += 1

    if rows_seen == 0:
        raise ValueError(f"CSV is empty: {args.summary_csv}")
    if rows_used == 0:
        suffix = f" for model_size={args.model_size}" if args.model_size else ""
        print(f"Warning: No usable rows found in {args.summary_csv}{suffix}")
        return

    if args.methods:
        method_order = [m.strip().lower() for m in args.methods if m.strip()]
    else:
        method_order = sorted(set().union(*[set(data[m].keys()) for m in data.keys()]))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_label = "training step (current checkpoint)"

    for metric in metrics_to_plot:
        if metric not in data:
            print(f"Warning: metric '{metric}' not found, skipping")
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
            plt.close(fig)
            print(f"Warning: no methods plotted for metric '{metric}', skipping")
            continue

        subtitle = f"model_size={args.model_size}" if args.model_size else "all model sizes"
        ax.set_title(f"{metric} ({subtitle})", fontsize=14)
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11)

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
