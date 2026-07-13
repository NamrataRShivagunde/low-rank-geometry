#!/usr/bin/env python3
"""
Batch script to run landscape analysis on all checkpoints in CHECKPOINTS/models-60m/

Structure:
  CHECKPOINTS/models-60m/
    ├── llama-60m/
    │   ├── model_1000/
    │   ├── model_2000/
    │   └── ...
    ├── cola-60m/
    └── ...

Results are saved to:
  results/loss-landscape-2d/models-60m/
    └── (same structure as above)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import time
from typing import List, Tuple


# Mapping from checkpoint folder name to model argument
MODEL_TYPE_MAP = {
    "llama-60m": "llama",
    "cola-60m": "cola",
    "fira-60m": "fira",
    "galore-60m": "galore",
    "relora-60m": "relora",
    "sltrain-60m": "sltrain",
}


def discover_checkpoints(checkpoint_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Discover all checkpoints in checkpoint_root.
    
    Returns:
        List of (model_type, checkpoint_name, checkpoint_path) tuples
        where model_type is from MODEL_TYPE_MAP and checkpoint_name is e.g., 'model_1000'
    """
    checkpoints = []
    
    for model_folder in sorted(checkpoint_root.iterdir()):
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        if model_name not in MODEL_TYPE_MAP:
            print(f"⚠️  Skipping unknown model type: {model_name}")
            continue
        
        model_type = MODEL_TYPE_MAP[model_name]
        
        # Find all checkpoint subdirectories (e.g., model_1000, model_2000, ...)
        for checkpoint_dir in sorted(model_folder.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("model_"):
                checkpoints.append((model_type, checkpoint_dir.name, checkpoint_dir))
    
    return checkpoints


def build_output_dir(base_output: Path, model_folder: str, checkpoint_name: str) -> Path:
    """Build the output directory following the same structure as checkpoints."""
    return base_output / "models-60m" / model_folder / checkpoint_name


def run_landscape_for_checkpoint(
    model_type: str,
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint_index: int,
    total_checkpoints: int
) -> bool:
    """
    Run the landscape script for a single checkpoint.
    
    Returns:
        True if successful, False otherwise
    """
    # Build the command
    cmd = [
        "python",
            "loss-landscape/scripts/single_checkpoint_scripts/1_landscape_2d_lowrank_number_of_dir.py",
        "--model", model_type,
        "--checkpoint", str(checkpoint_path),
        "--task", args.task,
        "--tokenizer", args.tokenizer,
        "--num_directions", str(args.num_directions),
        "--c4_max_examples", str(args.c4_max_examples),
        "--c4_batch_size", str(args.c4_batch_size),
        "--c4_max_length", str(args.c4_max_length),
        "--x_min", str(args.x_min),
        "--x_max", str(args.x_max),
        "--x_interval", str(args.x_interval),
        "--y_min", str(args.y_min),
        "--y_max", str(args.y_max),
        "--y_interval", str(args.y_interval),
        "--output_dir", str(output_dir),
        "--log_every", str(args.log_every),
    ]
    
    if args.plot_only:
        cmd.append("--plot_only")
    
    if args.show_inner_progress:
        cmd.append("--show_inner_progress")
    
    if args.band_std_mult != 1.0:
        cmd.extend(["--band_std_mult", str(args.band_std_mult)])
    
    if args.save_per_direction_plots:
        cmd.append("--save_per_direction_plots")
    
    # Print progress header
    print("\n" + "=" * 80)
    print(f"[{checkpoint_index}/{total_checkpoints}] Running landscape for: {checkpoint_path.parent.name}/{checkpoint_path.name}")
    print(f"  Model type: {model_type}")
    print(f"  Output: {output_dir}")
    print("=" * 80)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ Completed: {checkpoint_path.parent.name}/{checkpoint_path.name}")
            return True
        else:
            print(f"❌ Failed: {checkpoint_path.parent.name}/{checkpoint_path.name} (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ Error running command for {checkpoint_path.parent.name}/{checkpoint_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch run landscape analysis on all checkpoints in CHECKPOINTS/models-60m/"
    )
    
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("CHECKPOINTS/models-60m"),
        help="Root directory containing all checkpoint folders (default: CHECKPOINTS/models-60m)"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/loss-landscape-2d"),
        help="Root output directory for results (default: results/loss-landscape-2d)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="c4-val",
        help="Task name (default: c4-val)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="t5-base",
        help="HF tokenizer ID (default: t5-base)"
    )
    parser.add_argument(
        "--num_directions",
        type=int,
        default=100,
        help="Number of random directions (default: 200)"
    )
    parser.add_argument(
        "--c4_max_examples",
        type=int,
        default=1000,
        help="Max C4 examples (default: 1000)"
    )
    parser.add_argument(
        "--c4_batch_size",
        type=int,
        default=128,
        help="C4 batch size (default: 128)"
    )
    parser.add_argument(
        "--c4_max_length",
        type=int,
        default=256,
        help="C4 max length (default: 256)"
    )
    parser.add_argument(
        "--x_min",
        type=float,
        default=-0.003,
        help="Min x-axis perturbation (default: -0.003)"
    )
    parser.add_argument(
        "--x_max",
        type=float,
        default=0.0031,
        help="Max x-axis perturbation (default: 0.0031)"
    )
    parser.add_argument(
        "--x_interval",
        type=float,
        default=0.0005,
        help="X-axis interval (default: 0.0005)"
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=-0.003,
        help="Min y-axis perturbation (default: -0.003)"
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=0.003,
        help="Max y-axis perturbation (default: 0.003)"
    )
    parser.add_argument(
        "--y_interval",
        type=float,
        default=0.0005,
        help="Y-axis interval (default: 0.0005)"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Log progress every N directions (default: 20)"
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip computation, only re-plot from existing direction files"
    )
    parser.add_argument(
        "--show_inner_progress",
        action="store_true",
        help="Show per-direction progress bars (default: off for cleaner output)"
    )
    parser.add_argument(
        "--band_std_mult",
        type=float,
        default=1.0,
        help="Variance band scaling factor (default: 1.0)"
    )
    parser.add_argument(
        "--save_per_direction_plots",
        action="store_true",
        help="Save individual plots for each direction"
    )
    parser.add_argument(
        "--skip_checkpoints",
        type=str,
        nargs="*",
        default=[],
        help="Checkpoint names to skip (e.g., model_1000 model_2000)"
    )
    parser.add_argument(
        "--only_checkpoints",
        type=str,
        nargs="*",
        default=[],
        help="If specified, only run these checkpoint names (e.g., model_1000 model_2000)"
    )
    parser.add_argument(
        "--only_models",
        type=str,
        nargs="*",
        default=[],
        help="If specified, only run these model types (e.g., llama cola fira)"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = Path(args.output_root).resolve()
    
    # Validate checkpoint root exists
    if not checkpoint_root.exists():
        print(f"❌ Error: Checkpoint root does not exist: {checkpoint_root}")
        sys.exit(1)
    
    # Discover all checkpoints
    print(f"🔍 Discovering checkpoints in {checkpoint_root}...")
    checkpoints = discover_checkpoints(checkpoint_root)
    
    # Filter by model type if specified
    if args.only_models:
        checkpoints = [
            (mt, cn, cp) for mt, cn, cp in checkpoints 
            if mt in args.only_models
        ]
    
    # Filter by checkpoint name
    if args.skip_checkpoints:
        checkpoints = [
            (mt, cn, cp) for mt, cn, cp in checkpoints 
            if cn not in args.skip_checkpoints
        ]
    
    if args.only_checkpoints:
        checkpoints = [
            (mt, cn, cp) for mt, cn, cp in checkpoints 
            if cn in args.only_checkpoints
        ]
    
    if not checkpoints:
        print("❌ No checkpoints found matching the filters!")
        sys.exit(1)
    
    print(f"✅ Found {len(checkpoints)} checkpoints to process")
    print()
    
    # Run landscape analysis for each checkpoint
    successful = 0
    failed = 0
    failed_checkpoints = []
    
    start_time = time.time()
    
    for idx, (model_type, checkpoint_name, checkpoint_path) in enumerate(checkpoints, 1):
        # Build output directory structure
        model_folder_name = next(
            k for k, v in MODEL_TYPE_MAP.items() if v == model_type
        )
        output_dir = build_output_dir(output_root, model_folder_name, checkpoint_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run landscape analysis
        if run_landscape_for_checkpoint(
            model_type,
            checkpoint_path,
            output_dir,
            args,
            idx,
            len(checkpoints)
        ):
            successful += 1
        else:
            failed += 1
            failed_checkpoints.append(f"{model_folder_name}/{checkpoint_name}")
    
    # Print summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 80)
    print("BATCH LANDSCAPE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total     : {len(checkpoints)} checkpoints")
    print(f"✅ Success : {successful}")
    print(f"❌ Failed  : {failed}")
    print(f"⏱️  Time    : {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("=" * 80)
    
    if failed_checkpoints:
        print("\nFailed checkpoints:")
        for cp in failed_checkpoints:
            print(f"  - {cp}")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
