#!/usr/bin/env python3
"""
Script to plot the evolution grid of U-curves from saved .npy files.

Usage:
    python plot_evolution_grid.py --curves_npy_dir output/landscape_evolution/llama_60m/curves_npy \
        --output_png evolution_grid.png --grid_cols 5 --task c4-val
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot evolution grid of U-curves from saved .npy files.")
    parser.add_argument("--curves_npy_dir", type=str, required=True,
                        help="Directory containing the saved .npy files (e.g., curves_npy) for the first model")
    parser.add_argument("--curves_npy_dir2", type=str, default=None,
                        help="Directory containing the saved .npy files for the second model (optional)")
    parser.add_argument("--labels", type=str, nargs='+', default=["Model1", "Model2"],
                        help="Labels for the models")
    parser.add_argument("--output_png", type=str, default="evolution_grid.png",
                        help="Output PNG filename")
    parser.add_argument("--grid_cols", type=int, default=5,
                        help="Number of columns in the grid")
    parser.add_argument("--task", type=str, default="c4-val",
                        help="Task name for ylabel")
    args = parser.parse_args()

    if not os.path.isdir(args.curves_npy_dir):
        raise ValueError(f"Directory {args.curves_npy_dir} does not exist")

    dirs = [args.curves_npy_dir]
    if args.curves_npy_dir2:
        if not os.path.isdir(args.curves_npy_dir2):
            raise ValueError(f"Directory {args.curves_npy_dir2} does not exist")
        dirs.append(args.curves_npy_dir2)

    all_curves = []
    for d_idx, d in enumerate(dirs):
        # Find all ys.npy files
        ys_files = [f for f in os.listdir(d) if f.endswith('_ys.npy')]
        if not ys_files:
            raise ValueError(f"No _ys.npy files found in directory {d}")

        curves = []
        for ys_file in ys_files:
            base = ys_file.replace('_ys.npy', '')
            ys_path = os.path.join(d, ys_file)
            xs_path = os.path.join(d, f"{base}_xs.npy")
            if not os.path.exists(xs_path):
                print(f"Warning: {xs_path} not found, skipping {base}")
                continue

            ys = np.load(ys_path)
            xs = np.load(xs_path)

            # Extract step
            m = re.search(r"(\d+)$", base)
            step = int(m.group(1)) if m else 0

            curves.append({"step": step, "name": base, "xs": xs, "ys": ys, "model": d_idx})

        all_curves.extend(curves)

    if not all_curves:
        raise ValueError("No valid curves found")

    # Group by step
    from collections import defaultdict
    step_curves = defaultdict(list)
    for c in all_curves:
        step_curves[c["step"]].append(c)

    # For plotting, use the steps that have at least one curve
    steps = sorted(step_curves.keys())
    curves_list = [step_curves[s] for s in steps]

    # Plot
    cols = max(1, int(args.grid_cols))
    rows = int(np.ceil(len(curves_list) / cols))
    fig_w, fig_h = cols * 5.2, rows * 4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    # Find global min and max for consistent y-limits
    all_ys = []
    for cl in curves_list:
        for c in cl:
            all_ys.extend(c["ys"])
    all_ys = np.array(all_ys)
    global_min, global_max = float(np.nanmin(all_ys)), float(np.nanmax(all_ys))

    legend_handles, legend_labels = None, None
    for i, cl in enumerate(curves_list):
        r, col = i // cols, i % cols
        ax = axes[r][col]
        step = steps[i]
        for c in cl:
            label = args.labels[c["model"]]
            ax.plot(c["xs"], c["ys"], linewidth=1.5, label=label)
        ax.set_title(f"Step {step}", fontsize=10, pad=6)
        # ax.set_xlabel("Perturbation magnitude")
        # ax.set_ylabel(f"{args.task} loss")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if r == 0:  # First row
            ax.set_ylim(3.75, 5.5)
        else:  # Other rows
            ax.set_ylim(3.4, 4)

    # Hide empty axes
    total = rows * cols
    for j in range(len(curves_list), total):
        r, col = j // cols, j % cols
        axes[r][col].axis('off')

    if legend_handles:
        fig.legend(
            legend_handles,
            ["Baseline", "CoLA"],
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False,
            fontsize=26,
            bbox_to_anchor=(0.5, 0.98),
        )

    fig.supxlabel("Perturbation", fontsize=20, y=0.04)
    fig.supylabel("Val Loss", fontsize=20, x=0.03)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.99, wspace=0.18, hspace=0.35)
    fig.savefig(args.output_png, dpi=250)
    fig.savefig(os.path.join(os.path.dirname(args.output_png), "evolution_grid.pdf"), dpi=250)
    plt.close(fig)
    print(f"Saved plot to {args.output_png}")

if __name__ == "__main__":
    main()