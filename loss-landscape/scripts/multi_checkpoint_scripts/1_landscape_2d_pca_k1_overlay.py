#!/usr/bin/env python3
"""Overlay PCA-k1 verification landscapes across methods and checkpoints.

Adapted from 1_landscape_2d_overlay.py for the pca-verification-k1 results
layout where stats.json lives at the checkpoint level (not inside aggregate/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLOR_MAP = {
    "full_rank": "#1f77b4",
    "full-rank": "#1f77b4",
    "fullrank": "#1f77b4",
    "llama": "#1f77b4",
    "galore": "#ff7f0e",
    "fira": "#2ca02c",
    "cola": "#d62728",
    "relora": "#9467bd",
    "switchlora": "#8c564b",
    "sltrain": "#e377c2",
}

FALLBACK_METHOD_COLORS = [
    "#17becf", "#bcbd22", "#7f7f7f", "#8c564b",
    "#e377c2", "#d62728", "#2ca02c", "#ff7f0e",
]


def _normalize_checkpoint_name(value: str) -> str:
    text = str(value).strip()
    if text.startswith("model_"):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return f"model_{int(digits)}"


def _checkpoint_step(checkpoint_name: str) -> int | None:
    text = str(checkpoint_name).strip()
    if text.startswith("model_"):
        text = text[len("model_"):]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _checkpoint_short_label(checkpoint_name: str) -> str:
    step = _checkpoint_step(checkpoint_name)
    if step is None:
        return checkpoint_name
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def _checkpoint_numeric_label(checkpoint_name: str) -> str:
    step = _checkpoint_step(checkpoint_name)
    if step is None:
        return checkpoint_name
    return str(step)


def _method_color(method_name: str) -> str:
    key = str(method_name).strip().lower()
    if key in METHOD_COLOR_MAP:
        return METHOD_COLOR_MAP[key]
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FALLBACK_METHOD_COLORS)
    return FALLBACK_METHOD_COLORS[idx]


def _extract_method_name(folder_name: str) -> str:
    name = folder_name.lower()
    name = re.sub(r"[-_]\d+m$", "", name)
    return name


def _extract_curve(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return np.asarray(arr[arr.shape[0] // 2], dtype=float)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _build_x_axis(checkpoint_dir: Path, curve_size: int) -> np.ndarray:
    """Build x-axis from stats.json at checkpoint level or aggregate level."""
    for candidate in [checkpoint_dir / "stats.json", checkpoint_dir / "aggregate" / "stats.json"]:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text())
                # Try top-level keys first, then command_args
                x_min = payload.get("x_min") or payload.get("command_args", {}).get("x_min")
                x_interval = payload.get("x_interval") or payload.get("command_args", {}).get("x_interval")
                if isinstance(x_min, (int, float)) and isinstance(x_interval, (int, float)):
                    return float(x_min) + np.arange(curve_size, dtype=float) * float(x_interval)
                # Try x_grid directly
                x_grid = payload.get("x_grid")
                if x_grid and len(x_grid) == curve_size:
                    return np.array(x_grid, dtype=float)
            except Exception:
                pass
    return np.arange(curve_size, dtype=float)


def _parse_row_ylim(values: list[str], rows: int) -> list[tuple[float, float]]:
    if not values:
        return []
    if len(values) != rows:
        raise ValueError(f"--row_ylim must provide exactly {rows} values; got {len(values)}")
    out: list[tuple[float, float]] = []
    for token in values:
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid --row_ylim token '{token}'. Expected min,max format.")
        y_min, y_max = float(parts[0].strip()), float(parts[1].strip())
        out.append((y_min, y_max))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay PCA-k1 landscapes across methods and checkpoints")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing method folders (e.g. results/pca-verification-k1/60m)")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--grid_cols", type=int, default=5)
    parser.add_argument("--band_std_mult", type=float, default=1.0)
    parser.add_argument("--row_ylim", nargs="*", default=[])
    parser.add_argument("--zero_center", action="store_true",
                        help="Subtract loss at alpha=0 so all curves start at 0 (compare curvature)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fig_width", type=float, default=4.0)
    parser.add_argument("--fig_height", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    ckpts = [_normalize_checkpoint_name(c) for c in args.checkpoints]
    methods = [m.strip() for m in args.methods]

    output_name = args.output_name.strip() if args.output_name else ""
    if not output_name:
        size_part = root.name.upper()
        output_name = f"pca_k1_{size_part}_overlay"

    rows = int(math.ceil(len(ckpts) / max(1, args.grid_cols)))
    cols = max(1, args.grid_cols)
    manual_row_ylims = _parse_row_ylim(args.row_ylim, rows) if args.row_ylim else []

    fig, axes = plt.subplots(rows, cols, figsize=(args.fig_width * cols, args.fig_height * rows), squeeze=False)

    row_y_values: list[list[float]] = [[] for _ in range(rows)]
    legend_handle_by_label: dict[str, object] = {}

    for idx, ckpt in enumerate(ckpts):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        for method in methods:
            method_dir = root / method
            if not method_dir.is_dir():
                print(f"Warning: method dir not found: {method_dir}")
                continue

            ckpt_dir = method_dir / ckpt
            if not ckpt_dir.is_dir():
                print(f"Warning: checkpoint dir not found: {ckpt_dir}")
                continue

            agg_npy = ckpt_dir / "aggregate" / "npy"
            mean_path = agg_npy / "loss_mean.npy"
            var_path = agg_npy / "loss_variance.npy"
            if not mean_path.exists() or not var_path.exists():
                print(f"Warning: npy files not found in {agg_npy}")
                continue

            mean_arr = np.load(mean_path)
            var_arr = np.load(var_path)

            mean_curve = _extract_curve(mean_arr)
            var_curve = _extract_curve(var_arr)
            n = min(mean_curve.size, var_curve.size)
            mean_curve = mean_curve[:n]
            std_curve = np.sqrt(np.maximum(var_curve[:n], 0.0))
            x = _build_x_axis(ckpt_dir, n)

            # Zero-center: subtract loss at alpha closest to 0
            if args.zero_center:
                zero_idx = int(np.argmin(np.abs(x)))
                mean_curve = mean_curve - mean_curve[zero_idx]

            band = args.band_std_mult * std_curve
            label = _extract_method_name(method)
            color = _method_color(label)

            line = ax.plot(x, mean_curve, linewidth=2.0, label=label, color=color)
            ax.fill_between(x, mean_curve - band, mean_curve + band, color=color, alpha=0.20)

            if label not in legend_handle_by_label:
                legend_handle_by_label[label] = line[0]

            row_y_values[r].extend((mean_curve - band).tolist())
            row_y_values[r].extend((mean_curve + band).tolist())

        ax.set_title(_checkpoint_numeric_label(ckpt), fontsize=24)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3)

    # Turn off unused subplots
    for idx in range(len(ckpts), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    # Apply y-limits
    if manual_row_ylims:
        for r in range(rows):
            y_min, y_max = manual_row_ylims[r]
            for c_idx in range(cols):
                axes[r][c_idx].set_ylim(bottom=y_min, top=y_max)
    else:
        for r in range(rows):
            if not row_y_values[r]:
                continue
            y_min = float(np.min(row_y_values[r]))
            y_max = float(np.max(row_y_values[r]))
            for c_idx in range(cols):
                axes[r][c_idx].set_ylim(bottom=y_min, top=y_max)

    fig.supxlabel("alpha (perturbation scale)", fontsize=20)
    fig.supylabel("loss - loss(0)" if args.zero_center else "val loss", fontsize=20)

    handles = list(legend_handle_by_label.values())
    labels = list(legend_handle_by_label.keys())
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, len(labels)),
            frameon=False,
            fontsize=20,
        )

    fig.tight_layout(rect=(0.03, 0.08, 1, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{output_name}.png"
    pdf_path = args.output_dir / f"{output_name}.pdf"
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved overlay PNG: {png_path}")
    print(f"Saved overlay PDF: {pdf_path}")


if __name__ == "__main__":
    main()
