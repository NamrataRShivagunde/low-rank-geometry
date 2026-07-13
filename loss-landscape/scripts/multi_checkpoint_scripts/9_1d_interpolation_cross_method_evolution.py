#!/usr/bin/env python3
"""Aggregate cross-method interpolation results across training steps and plot
how the inter-method barrier (IMIB) evolves.

Reads outputs of `9_1d_interpolation_cross_method.py` from per-step folders:
    results/cross_method_interpolation/models-<size>/step_<step>/cross_method_interp_*_barriers.json

Produces:
    evolution/barriers_vs_step.csv                 (flat table: pair, step, imib)
    evolution/imib_evolution_overlay.png           (one line per pair, imib vs step)
    evolution/imib_evolution_llama_pairs.png       (zoom: only llama<->method)
    evolution/imib_evolution_method_pairs.png      (zoom: only method<->method)
    evolution/imib_heatmap_pair_x_step.png         (heatmap pair × step)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PAIR_COLORS = {
    # llama pairs (use method color)
    "llama__fira": "#2ca02c",
    "llama__galore": "#ff7f0e",
    "llama__relora": "#9467bd",
    "llama__sltrain": "#e377c2",
    # method pairs (distinct colors)
    "fira__galore": "#17becf",
    "fira__relora": "#bcbd22",
    "fira__sltrain": "#8c564b",
    "galore__relora": "#e31a1c",
    "galore__sltrain": "#636363",
    "relora__sltrain": "#6a3d9a",
}


def _pair_label(a: str, b: str) -> str:
    return f"{a}__{b}"


def _pair_display(a: str, b: str) -> str:
    return f"{a}\u2194{b}"


def discover_steps(size_root: Path) -> list[int]:
    steps = []
    for entry in sorted(size_root.iterdir()):
        if entry.is_dir() and entry.name.startswith("step_"):
            try:
                steps.append(int(entry.name[len("step_"):]))
            except ValueError:
                continue
    return sorted(steps)


def load_barriers(size_root: Path, size: str, steps: list[int]) -> list[dict[str, Any]]:
    """Load barrier JSON files from all steps into a flat list of dicts."""
    rows: list[dict[str, Any]] = []
    for step in steps:
        jpath = size_root / f"step_{step}" / f"cross_method_interp_{size}_step{step}_barriers.json"
        if not jpath.exists():
            print(f"[warn] missing {jpath}")
            continue
        try:
            data = json.loads(jpath.read_text())
            rows.extend(data)
        except Exception as exc:
            print(f"[warn] failed to load {jpath}: {exc}")
    return rows


def write_barrier_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "pair", "method_a", "method_b", "step",
        "loss_at_alpha_0", "loss_at_alpha_1",
        "endpoint_mean_loss", "max_interior_loss", "max_interior_alpha",
        "imib_barrier", "imib_barrier_relative",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")


def plot_evolution_overlay(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    filter_fn=None,
    dpi: int = 220,
) -> None:
    """Line plot: IMIB vs training step, one line per pair."""
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if filter_fn is not None and not filter_fn(r):
            continue
        grouped[r["pair"]].append((int(r["step"]), float(r["imib_barrier"])))

    if not grouped:
        print(f"[skip] no data for {output_path}")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for pair, points in sorted(grouped.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs, ys,
            marker="o", linewidth=2.2, markersize=6,
            label=pair.replace("__", "\u2194"),
            color=PAIR_COLORS.get(pair, None),
        )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("IMIB barrier", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_heatmap(
    rows: list[dict[str, Any]],
    output_path: Path,
    size: str,
    dpi: int = 220,
) -> None:
    """Heatmap: rows=pair, cols=step, color=IMIB."""
    pairs = sorted(set(r["pair"] for r in rows))
    steps = sorted(set(int(r["step"]) for r in rows))
    if not pairs or not steps:
        return

    mat = np.full((len(pairs), len(steps)), np.nan)
    pair_to_i = {p: i for i, p in enumerate(pairs)}
    step_to_j = {s: j for j, s in enumerate(steps)}

    for r in rows:
        i = pair_to_i[r["pair"]]
        j = step_to_j[int(r["step"])]
        mat[i, j] = float(r["imib_barrier"])

    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.9), max(5, len(pairs) * 0.5 + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], fontsize=10, rotation=45)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p.replace("__", "\u2194") for p in pairs], fontsize=11)

    for i in range(len(pairs)):
        for j in range(len(steps)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > np.nanmax(mat) * 0.55 else "black")

    ax.set_title(f"IMIB evolution — {size}", fontsize=14)
    ax.set_xlabel("Training Step", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="IMIB barrier")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot IMIB evolution across training steps")
    p.add_argument("--size_root", type=Path, required=True,
                   help="Directory like results/cross_method_interpolation/models-60m")
    p.add_argument("--size", type=str, required=True, choices=["60m", "130m", "350m"])
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Default: <size_root>/evolution")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    size_root = args.size_root.resolve()
    if not size_root.exists():
        raise FileNotFoundError(f"Size root not found: {size_root}")

    output_dir = args.output_dir or (size_root / "evolution")
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = discover_steps(size_root)
    print(f"Discovered {len(steps)} steps under {size_root}: {steps}")

    rows = load_barriers(size_root, args.size, steps)
    if not rows:
        print("No barrier data found, nothing to plot")
        return

    write_barrier_csv(rows, output_dir / f"barriers_vs_step_{args.size}.csv")

    plot_evolution_overlay(
        rows,
        output_dir / f"imib_evolution_overlay_{args.size}.png",
        title=f"Inter-Method Interpolation Barrier vs Training Step ({args.size})",
        dpi=args.dpi,
    )
    plot_evolution_overlay(
        rows,
        output_dir / f"imib_evolution_llama_pairs_{args.size}.png",
        title=f"IMIB vs Training Step — llama\u2194method pairs ({args.size})",
        filter_fn=lambda r: "llama" in (r["method_a"], r["method_b"]),
        dpi=args.dpi,
    )
    plot_evolution_overlay(
        rows,
        output_dir / f"imib_evolution_method_pairs_{args.size}.png",
        title=f"IMIB vs Training Step — method\u2194method pairs ({args.size})",
        filter_fn=lambda r: "llama" not in (r["method_a"], r["method_b"]),
        dpi=args.dpi,
    )
    plot_heatmap(rows, output_dir / f"imib_heatmap_pair_x_step_{args.size}.png", args.size, args.dpi)

    print("\nAll evolution outputs saved.")


if __name__ == "__main__":
    main()
