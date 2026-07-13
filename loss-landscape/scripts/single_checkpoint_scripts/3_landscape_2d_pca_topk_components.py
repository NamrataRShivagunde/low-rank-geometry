#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
import time
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

from utils.plot import Landscape4ModelPCA  # noqa: E402
from exps.landscape.most.landscape_eval_utils import compute_nll_loss, get_c4_dataloader, load_model_from_args  # noqa: E402


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute landscape over PCA components (top-k or component range) for a single checkpoint, "
            "then save per-component arrays and aggregated mean/variance plots."
        )
    )

    parser.add_argument("--model", type=str, choices=["llama", "cola", "fira", "galore", "relora", "sltrain"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["c4-val"], default="c4-val")
    parser.add_argument("--tokenizer", type=str, default="t5-base")

    parser.add_argument("--top_k_components", type=int, default=1, help="Number of PCA components to average")
   
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
    parser.add_argument("--save_per_component_plots", action="store_true")
    parser.add_argument("--band_std_mult", type=float, default=1.0, help="Scale for std band in 1D mean plot")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--plot_only", action="store_true", help="Skip compute and aggregate from existing component .npy files")
    parser.add_argument("--components_dir", type=str, default=None, help="Optional directory containing component_*.npy files")

    args = parser.parse_args()
    run_start = time.time()

    if args.top_k_components < 1:
        raise ValueError("--top_k_components must be >= 1")
    component_list = list(range(1, args.top_k_components + 1))

    output_dir = Path(args.output_dir)
    component_npy_dir = output_dir / "components" / "npy"
    component_plot_dir = output_dir / "components" / "plots"
    agg_npy_dir = output_dir / "aggregate" / "npy"
    agg_plot_dir = output_dir / "aggregate" / "plots"

    component_npy_dir.mkdir(parents=True, exist_ok=True)
    if args.save_per_component_plots:
        component_plot_dir.mkdir(parents=True, exist_ok=True)
    agg_npy_dir.mkdir(parents=True, exist_ok=True)
    agg_plot_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[np.ndarray] = []
    run_times: list[float] = []
    timeline_rows: list[dict] = []

    if args.plot_only:
        source_dir = Path(args.components_dir) if args.components_dir else component_npy_dir
        component_files = sorted(source_dir.glob("component_*.npy"))
        if not component_files:
            raise FileNotFoundError(f"No component_*.npy files found in {source_dir}")

        print(f"Plot-only mode: loading {len(component_files)} component files from {source_dir}")
        for idx, file_path in enumerate(component_files, start=1):
            all_results.append(np.load(file_path))
            if idx == 1 or idx == len(component_files) or idx % max(1, args.log_every) == 0:
                print(f"[{idx:>4}/{len(component_files)}] loaded {file_path.name}")
        effective_runs = len(all_results)
        drawer = None
    else:
        model = load_model_from_args(args)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        pad_idx = tokenizer.pad_token_id

        dataloader = get_c4_dataloader(
            tokenizer,
            max_examples=args.c4_max_examples,
            max_length=args.c4_max_length,
            batch_size=args.c4_batch_size,
        )
        benchmark = lambda m, verbose=False: compute_nll_loss(m, dataloader, pad_idx)

        drawer = Landscape4ModelPCA(
            model,
            benchmark,
            device=torch.device("cuda"),
            save_path=str(component_plot_dir if args.save_per_component_plots else output_dir),
            mode="2D",
            k_components=component_list[0],
        )

        print("Precomputing PCA cache once for all components...")
        drawer.warmup_cache()

        drawer.synthesize_coordinates(
            x_min=args.x_min,
            x_max=args.x_max,
            x_interval=args.x_interval,
            y_min=args.y_min,
            y_max=args.y_max,
            y_interval=args.y_interval,
        )

        print(f"Starting PCA-component run: {len(component_list)} components ({component_list[0]}..{component_list[-1]})")
        total_runs = len(component_list)
        for idx, component in enumerate(component_list, start=1):
            run_start_one = time.time()
            run_tag = f"component_{component:03d}"

            drawer.k_components = component
            drawer.find_direction(seed=None)
            result = drawer.compute_for_draw()
            all_results.append(result)

            run_npy_path = component_npy_dir / f"{run_tag}.npy"
            np.save(run_npy_path, result)

            elapsed_one = time.time() - run_start_one
            run_times.append(elapsed_one)
            avg_elapsed = sum(run_times) / len(run_times)
            remaining = total_runs - idx
            eta_seconds = avg_elapsed * remaining
            total_elapsed = time.time() - run_start

            timeline_rows.append(
                {
                    "run_index": idx,
                    "component": component,
                    "run_tag": run_tag,
                    "run_seconds": elapsed_one,
                    "avg_run_seconds": avg_elapsed,
                    "elapsed_seconds": total_elapsed,
                    "eta_seconds": eta_seconds,
                }
            )

            if idx == 1 or idx == total_runs or idx % max(1, args.log_every) == 0:
                print(
                    f"[{idx:>4}/{total_runs}] comp={component:>3d}  "
                    f"step={elapsed_one:>5.2f}s  "
                    f"avg={avg_elapsed:>5.2f}s  "
                    f"elapsed={_format_hms(total_elapsed)}  "
                    f"eta={_format_hms(eta_seconds)}"
                )

            if args.save_per_component_plots:
                drawer.draw_figure(drawer.mesh_x, drawer.mesh_y, result, saving_name=f"{run_tag}.png")

        effective_runs = total_runs

    if len(all_results) == 1:
        mean_arr = np.asarray(all_results[0], dtype=float)
        var_arr = np.zeros_like(mean_arr, dtype=float)
    else:
        stack = np.stack(all_results, axis=0)
        mean_arr = np.mean(stack, axis=0)
        var_arr = np.var(stack, axis=0)

    mean_npy = agg_npy_dir / "loss_mean.npy"
    var_npy = agg_npy_dir / "loss_variance.npy"
    np.save(mean_npy, mean_arr)
    np.save(var_npy, var_arr)
    print(f"Saved mean array: {mean_npy}")
    print(f"Saved variance array: {var_npy}")

    # Keep only aggregate artifacts after successful save.
    component_parent = component_npy_dir.parent
    if component_parent.exists():
        shutil.rmtree(component_parent)
        print(f"Removed per-component directory: {component_parent}")

    metadata = {
        "command_args": vars(args),
        "command_string": " ".join(sys.argv),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "mode": "2D_PCA",
        "effective_runs": effective_runs,
        "component_list": component_list,
        "top_k_components": args.top_k_components,
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
        "avg_run_seconds": float(np.mean(run_times)) if run_times else 0.0,
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
                    "run_index",
                    "component",
                    "run_tag",
                    "run_seconds",
                    "avg_run_seconds",
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

    if mean_arr.ndim == 1:
        if drawer is not None and hasattr(drawer, "mesh_x"):
            x = drawer.mesh_x[0]
        else:
            x = np.linspace(args.x_min, args.x_max, mean_arr.shape[0], endpoint=False)
        std_arr = np.sqrt(np.maximum(var_arr, 0.0))
        band = args.band_std_mult * std_arr

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, mean_arr, linewidth=2, label="Mean")
        ax.fill_between(
            x,
            mean_arr - band,
            mean_arr + band,
            alpha=0.35,
        )
        ax.set_xlabel("perturbation")
        ax.set_ylabel("val loss")
        ax.grid(True, alpha=0.3)

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
    print(f"Completed {effective_runs} component runs in {total_runtime/60:.2f} minutes")


if __name__ == "__main__":
    main()
