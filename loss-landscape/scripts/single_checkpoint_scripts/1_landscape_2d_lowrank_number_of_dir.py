#!/usr/bin/env python3
import argparse
import csv
import io
import json
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_LANDSCAPE_ROOT = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "LLMLandscape").is_dir()),
    SCRIPT_DIR.parent,
)
LLM_ROOT = LOSS_LANDSCAPE_ROOT / "LLMLandscape"
sys.path.insert(0, str(LLM_ROOT))

from utils.plot import Landscape4Model, Landscape4ModelPCA 
from exps.landscape.most.landscape_eval_utils import compute_nll_loss, get_c4_dataloader, load_model_from_args  # noqa: E402


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute multiple random-direction landscapes in one Python process, "
            "then save per-direction arrays and aggregated mean/variance plots."
        )
    )
    parser.add_argument("--model", type=str, choices=["llama", "cola", "fira", "galore", "relora", "sltrain"], default="llama")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["c4-val"], default="c4-val")
    parser.add_argument("--tokenizer", type=str, default="t5-base")

    parser.add_argument("--num_directions", type=int, default=20)
    parser.add_argument("--mode", type=str, choices=["2D", "3D", "2D_PCA"], default="2D")
    parser.add_argument("--component", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=None, help="Optional base seed; direction i uses seed_base + i")

    parser.add_argument("--c4_max_examples", type=int, default=1000)
    parser.add_argument("--c4_max_length", type=int, default=256)
    parser.add_argument("--c4_batch_size", type=int, default=128)

    parser.add_argument("--x_min", type=float, default=-0.005)
    parser.add_argument("--x_max", type=float, default=0.005)
    parser.add_argument("--x_interval", type=float, default=0.25e-3)
    parser.add_argument("--y_min", type=float, default=-5e-3)
    parser.add_argument("--y_max", type=float, default=5e-3)
    parser.add_argument("--y_interval", type=float, default=1e-3)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_per_direction_plots", action="store_true")
    parser.add_argument("--band_std_mult", type=float, default=1.0, help="Scale for std band in 1D mean plot")
    parser.add_argument("--log_every", type=int, default=10, help="Print concise timing log every N directions")
    parser.add_argument("--show_inner_progress", action="store_true", help="Show per-direction internal tqdm output")
    parser.add_argument("--plot_only", action="store_true", help="Skip compute and aggregate from existing direction .npy files")
    parser.add_argument("--directions_dir", type=str, default=None, help="Optional directory containing direction_*.npy for --plot_only")

    args = parser.parse_args()

    run_start = time.time()

    # set up directory paths for saving results
    output_dir = Path(args.output_dir)
    direction_npy_dir = output_dir / "directions" / "npy"
    direction_plot_dir = output_dir / "directions" / "plots"
    agg_npy_dir = output_dir / "aggregate" / "npy"
    agg_plot_dir = output_dir / "aggregate" / "plots"

    direction_npy_dir.mkdir(parents=True, exist_ok=True)
    if args.save_per_direction_plots:
        direction_plot_dir.mkdir(parents=True, exist_ok=True)
    agg_npy_dir.mkdir(parents=True, exist_ok=True)
    agg_plot_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    direction_times = []
    timeline_rows = []

    # sometimes we only need to plot the already generated .npy files.
    if args.plot_only:
        source_dir = Path(args.directions_dir) if args.directions_dir else direction_npy_dir
        direction_files = sorted(source_dir.glob("direction_*.npy"))
        if not direction_files:
            raise FileNotFoundError(f"No direction_*.npy files found in {source_dir}")

        print(f"Plot-only mode: loading {len(direction_files)} direction files from {source_dir}")
        for idx, file_path in enumerate(direction_files, start=1):
            all_results.append(np.load(file_path))
            if idx == 1 or idx == len(direction_files) or idx % max(1, args.log_every) == 0:
                print(f"[{idx:>4}/{len(direction_files)}] loaded {file_path.name}")
        effective_num_directions = len(all_results)
        drawer = None
    else:
        model = load_model_from_args(args) # loads model from eval utils

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        pad_idx = tokenizer.pad_token_id

        dataloader = get_c4_dataloader(
            tokenizer,
            max_examples=args.c4_max_examples,
            max_length=args.c4_max_length,
            batch_size=args.c4_batch_size,
        )
        benchmark = lambda m, verbose=False: compute_nll_loss(m, dataloader, pad_idx)

        drawer_mode = "2D" if args.mode == "2D_PCA" else args.mode
        if args.mode == "2D_PCA":
            drawer = Landscape4ModelPCA(
                model,
                benchmark,
                device=torch.device("cuda"),
                save_path=str(direction_plot_dir if args.save_per_direction_plots else output_dir),
                mode=drawer_mode,
                k_components=args.component,
            )
        else:
            drawer = Landscape4Model(
                model,
                benchmark,
                direction="Gaussian",
                device=torch.device("cuda"),
                save_path=str(direction_plot_dir if args.save_per_direction_plots else output_dir),
                mode=drawer_mode,
            )

        drawer.synthesize_coordinates(
            x_min=args.x_min,
            x_max=args.x_max,
            x_interval=args.x_interval,
            y_min=args.y_min,
            y_max=args.y_max,
            y_interval=args.y_interval,
        )

        print(f"Starting run: {args.num_directions} directions")
        for idx in range(1, args.num_directions + 1):
            direction_start = time.time()
            run_tag = f"direction_{idx:02d}"
            seed = None if args.seed_base is None else args.seed_base + idx
            if idx == 1:
                print(f"Direction seed mode: {'random' if seed is None else 'seed_base+i'}")

            drawer.find_direction(seed=seed)

            result = drawer.compute_for_draw()
      
            all_results.append(result)

            run_npy_path = direction_npy_dir / f"{run_tag}.npy"
            np.save(run_npy_path, result)

            direction_elapsed = time.time() - direction_start
            direction_times.append(direction_elapsed)
            avg_per_direction = sum(direction_times) / len(direction_times)
            remaining = args.num_directions - idx
            eta_seconds = avg_per_direction * remaining
            total_elapsed = time.time() - run_start
            timeline_rows.append(
                {
                    "direction": idx,
                    "direction_seconds": direction_elapsed,
                    "avg_direction_seconds": avg_per_direction,
                    "elapsed_seconds": total_elapsed,
                    "eta_seconds": eta_seconds,
                }
            )

            if idx == 1 or idx == args.num_directions or idx % max(1, args.log_every) == 0:
                print(
                    f"[{idx:>4}/{args.num_directions}] "
                    f"step={direction_elapsed:>5.2f}s  "
                    f"avg={avg_per_direction:>5.2f}s  "
                    f"elapsed={_format_hms(total_elapsed)}  "
                    f"eta={_format_hms(eta_seconds)}"
                )

            if args.save_per_direction_plots:
                drawer.draw_figure(drawer.mesh_x, drawer.mesh_y, result, saving_name=f"{run_tag}.png")

        effective_num_directions = args.num_directions

    stack = np.stack(all_results, axis=0)
    mean_arr = np.mean(stack, axis=0)
    var_arr = np.var(stack, axis=0) ## sigma ** 2

    mean_npy = agg_npy_dir / "loss_mean.npy"
    var_npy = agg_npy_dir / "loss_variance.npy"
    np.save(mean_npy, mean_arr)
    np.save(var_npy, var_arr)
    print(f"Saved mean array: {mean_npy}")
    print(f"Saved variance array: {var_npy}")

    #Keep only aggregate artifacts after successful save.
    direction_parent = direction_npy_dir.parent  # directions/ folder
    if direction_parent.exists():
        shutil.rmtree(direction_parent)
        print(f"Removed per-direction directory: {direction_parent}")

    metadata = {
        "command_args": vars(args),
        "command_string": " ".join(sys.argv),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "num_directions": effective_num_directions,
        "mode": args.mode,
        "seed_base": args.seed_base,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "x_interval": args.x_interval,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "y_interval": args.y_interval,
        "c4_max_examples": args.c4_max_examples,
        "c4_max_length": args.c4_max_length,
        "c4_batch_size": args.c4_batch_size,
        "band_std_mult": args.band_std_mult,
        "mean_shape": list(mean_arr.shape),
        "std_min": float(np.sqrt(np.maximum(var_arr, 0.0)).min()),
        "std_max": float(np.sqrt(np.maximum(var_arr, 0.0)).max()),
        "std_mean": float(np.sqrt(np.maximum(var_arr, 0.0)).mean()),
        "total_runtime_seconds": time.time() - run_start,
        "avg_direction_seconds": float(np.mean(direction_times)) if direction_times else 0.0,
        "plot_only": args.plot_only,
    }
    metadata_path = output_dir / "aggregate" / "stats.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved stats metadata: {metadata_path}")

    if timeline_rows:
        timeline_path = output_dir / "aggregate" / "timeline.csv"
        with open(timeline_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "direction",
                    "direction_seconds",
                    "avg_direction_seconds",
                    "elapsed_seconds",
                    "eta_seconds",
                ],
            )
            writer.writeheader()
            writer.writerows(timeline_rows)
        print(f"Saved timeline: {timeline_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib not available, skipping plots: {exc}")
        return

    if mean_arr.ndim == 1: # one direction and not 2 directions
        if drawer is not None and hasattr(drawer, "mesh_x"):
            x = drawer.mesh_x[0]
        else:
            # Plot-only mode fallback: reconstruct perturbation axis from configured range.
            x = np.linspace(args.x_min, args.x_max, mean_arr.shape[0], endpoint=False)
        std_arr = np.sqrt(np.maximum(var_arr, 0.0))
        band = args.band_std_mult * std_arr # band_std_mult is 1, unless its very hard to see the var around mean

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, mean_arr, linewidth=2, label="Mean")
        ax.fill_between(
            x,
            mean_arr - band,
            mean_arr + band,
            alpha=0.35,
            #label=f"Mean +/- {args.band_std_mult:g} std",
        )
        #ax.set_title(f"Mean with variance band over {effective_num_directions} directions")
        ax.set_xlabel("perturbation")
        ax.set_ylabel("val loss")
        ax.grid(True, alpha=0.3)
        #ax.legend()

        png_path = agg_plot_dir / "loss_mean_with_variance_band.png"
        pdf_path = agg_plot_dir / "loss_mean_with_variance_band.pdf"
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved combined plot: {png_path}")
        print(f"Saved combined plot: {pdf_path}")
    else:
        raise ValueError(f"Unsupported array shape for plotting: {mean_arr.shape}")

    total_runtime = time.time() - run_start
    print(f"Completed {effective_num_directions} directions in {total_runtime/60:.2f} minutes")


if __name__ == "__main__":
    main()
