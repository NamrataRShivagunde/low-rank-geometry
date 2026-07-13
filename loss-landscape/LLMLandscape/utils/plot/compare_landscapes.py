"""
Overlay multiple 1D loss landscape arrays stored as .npy files on a single plot.

Examples:
  python loss-landscape/LLMLandscape/utils/plot/compare_landscapes.py \
    loss-landscape/LLMLandscape/output/out_c4_baseline_model10001/c4-val-llama-high.npy \
    loss-landscape/LLMLandscape/output/out_c4_model10001/c4-val-cola-high.npy \
    --labels fullrank cola \
    --output_dir loss-landscape/LLMLandscape/output/compare_A_vs_B \
    --fname A_vs_B

Optional x-axis control (defaults to index 0..N-1):
  python .../compare_landscapes.py fileA.npy fileB.npy --labels A B --x_min -0.005 --x_interval 0.00025

Plot metric:
  --plot ppl    # plots exp(loss) for token-level perplexity
  --plot loss   # plots the raw NLL loss (default)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim > 1:
        # Flatten 2D to 1D if passed a mesh; typical 2D save is (1, N)
        arr = arr.reshape(-1)
    return arr.astype(float)


def build_x_axis(n: int, x_min: float = None, x_interval: float = None, x_max: float = None) -> np.ndarray:
    if x_min is not None and x_interval is not None:
        if x_max is not None:
            xs = np.arange(x_min, x_max, x_interval)
            # If lengths mismatch due to floating error, resample to n
            if len(xs) != n:
                xs = x_min + np.arange(n) * x_interval
        else:
            xs = x_min + np.arange(n) * x_interval
    else:
        xs = np.arange(n)
    return xs


def main():
    parser = argparse.ArgumentParser(description="Overlay multiple .npy loss landscapes on one plot")
    parser.add_argument("files", nargs="+", help="Paths to .npy files to overlay")
    parser.add_argument("--labels", nargs="*", default=None, help="Labels for each .npy (same length as files)")
    parser.add_argument("--plot", choices=["loss", "ppl"], default="loss", help="Plot loss (NLL) or perplexity (exp(loss))")
    parser.add_argument("--x_min", type=float, default=None, help="Optional x-axis start (e.g., -0.005)")
    parser.add_argument("--x_interval", type=float, default=None, help="Optional x-axis step (e.g., 0.00025)")
    parser.add_argument("--x_max", type=float, default=None, help="Optional x-axis end (used with x_min/x_interval)")
    parser.add_argument("--output_dir", type=str, default="loss-landscape/LLMLandscape/output/compare", help="Directory to save the plot")
    parser.add_argument("--fname", type=str, default="compare", help="Filename stem for the output PNG")
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    parser.add_argument("--width", type=float, default=6.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.0, help="Figure height in inches")
    parser.add_argument("--grid", action="store_true", help="Show background grid")
    parser.add_argument("--y_min", type=float, default=None, help="Optional y-axis min")
    parser.add_argument("--y_max", type=float, default=None, help="Optional y-axis max")
    parser.add_argument("--y_interval", type=float, default=None, help="Optional y-axis step (e.g., 0.00025)")
    args = parser.parse_args()

    arrays = [load_array(p) for p in args.files]
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1 and args.x_min is None:
        # Allow different lengths only if user supplies explicit x-axis
        raise SystemExit("All arrays must have the same length unless you provide --x_min/--x_interval")

    n = lengths[0]
    xs = build_x_axis(n, args.x_min, args.x_interval, args.x_max)

    labels = args.labels
    if labels is None or len(labels) != len(arrays):
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.files]

    # Compute plotted values
    plotted = []
    for arr in arrays:
        if args.plot == "ppl":
            plotted.append(np.exp(arr))
        else:
            plotted.append(arr)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.fname}.png")

    plt.figure(figsize=(args.width, args.height))
    for y, lab in zip(plotted, labels):
        plt.plot(xs, y, label=lab, linewidth=1.5)
    plt.xlabel("perturbation" if args.x_min is not None else "Index")
    plt.ylabel("val loss")
    if args.grid:
        plt.grid(True, alpha=0.3)
    if args.y_min is not None or args.y_max is not None:
        plt.ylim(args.y_min, args.y_max)
    if args.y_interval is not None:
        plt.yticks(np.arange(args.y_min, args.y_max, args.y_interval))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved overlay plot to {out_path}")


if __name__ == "__main__":
    main()
