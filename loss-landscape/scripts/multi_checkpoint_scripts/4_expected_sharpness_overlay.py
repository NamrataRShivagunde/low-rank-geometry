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
    parser = argparse.ArgumentParser(description="Overlay expected sharpness across methods vs training steps")
    parser.add_argument("--summary_csv", type=Path, required=True, help="Path to expected_sharpness_*_summary.csv")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save overlay plot")
    parser.add_argument("--output_name", type=str, default="expected_sharpness_overlay", help="Output filename stem")
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=["sharpness", "variance"],
        default="sharpness",
        help="Which metric to read from summary CSV and plot.",
    )
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset/order of methods to plot")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    series = defaultdict(list)

    metric_col = "expected_sharpness" if args.metric_mode == "sharpness" else "average_variance"
    y_label = "expected sharpness" if args.metric_mode == "sharpness" else "average variance"

    with open(args.summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            method = _method_from_model_folder(row["model_folder"])
            step = _step_from_checkpoint(row["checkpoint"])
            raw_metric_val = row.get(metric_col, "")
            if raw_metric_val is None or str(raw_metric_val).strip() == "":
                raise ValueError(
                    f"Missing metric '{metric_col}' at CSV row {row_idx} "
                    f"(model_folder={row.get('model_folder')}, checkpoint={row.get('checkpoint')})"
                )
            try:
                metric_val = float(raw_metric_val)
            except ValueError:
                raise ValueError(
                    f"Invalid metric '{metric_col}'='{raw_metric_val}' at CSV row {row_idx} "
                    f"(model_folder={row.get('model_folder')}, checkpoint={row.get('checkpoint')})"
                )
            if step is None:
                raise ValueError(
                    f"Invalid checkpoint value '{row.get('checkpoint')}' at CSV row {row_idx}"
                )
            series[method].append((step, metric_val))

    if not series:
        raise ValueError(f"No valid rows found in {args.summary_csv}")

    if args.methods:
        method_order = [m.strip().lower() for m in args.methods if m.strip()]
    else:
        method_order = sorted(series.keys())

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
        raise ValueError("No methods plotted. Check --methods against CSV content.")

    ax.set_xlabel("training step", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    #ax.set_title("Expected Sharpness vs Training Step", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.output_name}.png"
    pdf_path = args.output_dir / f"{args.output_name}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved expected sharpness overlay PNG: {png_path}")
    print(f"Saved expected sharpness overlay PDF: {pdf_path}")


if __name__ == "__main__":
    main()
