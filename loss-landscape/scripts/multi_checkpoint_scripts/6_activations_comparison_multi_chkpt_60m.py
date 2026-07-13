#!/usr/bin/env python3
"""Batch compute activation-comparison metrics for 60m checkpoints vs llama baseline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


SIZE_TAG = "60m"
MODEL_TYPE_MAP = {
    "llama-60m": "llama",
    "cola-60m": "cola",
    "fira-60m": "fira",
    "galore-60m": "galore",
    "relora-60m": "relora",
    "sltrain-60m": "sltrain",
}
BASELINE_FOLDER = f"llama-{SIZE_TAG}"


def discover_pairs(checkpoint_root: Path) -> List[Tuple[str, str, str, Path, Path]]:
    pairs: List[Tuple[str, str, str, Path, Path]] = []

    baseline_root = checkpoint_root / BASELINE_FOLDER
    if not baseline_root.exists():
        raise FileNotFoundError(f"Baseline folder not found: {baseline_root}")

    for model_folder_name, model_type in sorted(MODEL_TYPE_MAP.items()):
        if model_folder_name == BASELINE_FOLDER:
            continue

        target_root = checkpoint_root / model_folder_name
        if not target_root.exists() or not target_root.is_dir():
            print(f"Skipping missing model folder: {target_root}")
            continue

        for checkpoint_dir in sorted(target_root.iterdir()):
            if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith("model_"):
                continue

            baseline_ckpt = baseline_root / checkpoint_dir.name
            if not baseline_ckpt.exists():
                print(
                    f"Skipping {model_folder_name}/{checkpoint_dir.name}: "
                    f"missing baseline {BASELINE_FOLDER}/{checkpoint_dir.name}"
                )
                continue

            pairs.append((model_type, model_folder_name, checkpoint_dir.name, checkpoint_dir, baseline_ckpt))

    return pairs


def build_output_dir(base_output: Path, model_folder: str, checkpoint_name: str) -> Path:
    return base_output / f"models-{SIZE_TAG}" / model_folder / checkpoint_name


def run_single(
    target_model: str,
    target_checkpoint: Path,
    baseline_checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
    index: int,
    total: int,
) -> bool:
    cmd = [
        "python",
        "loss-landscape/scripts/single_checkpoint_scripts/6_activations_comparison.py",
        "--baseline_model",
        "llama",
        "--baseline_checkpoint",
        str(baseline_checkpoint),
        "--target_model",
        target_model,
        "--target_checkpoint",
        str(target_checkpoint),
        "--output_dir",
        str(output_dir),
        "--max_examples",
        str(args.max_examples),
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--max_batches",
        str(args.max_batches),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]

    if args.tokenizer_checkpoint:
        cmd.extend(["--tokenizer_checkpoint", args.tokenizer_checkpoint])

    print("\n" + "=" * 80)
    print(f"[{index}/{total}] Activation comparison: {target_checkpoint.parent.name}/{target_checkpoint.name}")
    print(f"  Baseline: {baseline_checkpoint}")
    print(f"  Target  : {target_checkpoint}")
    print(f"  Output  : {output_dir}")
    print("=" * 80)

    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"Completed: {target_checkpoint.parent.name}/{target_checkpoint.name}")
        return True

    print(f"Failed: {target_checkpoint.parent.name}/{target_checkpoint.name} (exit code {result.returncode})")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch compute activation comparison metrics for 60m")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("CHECKPOINTS/models-60m"))
    parser.add_argument("--output_root", type=Path, default=Path("results/activation-comparison"))
    parser.add_argument("--tokenizer_checkpoint", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--skip_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--only_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--only_models", type=str, nargs="*", default=[])
    args = parser.parse_args()

    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()

    if not checkpoint_root.exists():
        print(f"Error: checkpoint root not found: {checkpoint_root}")
        sys.exit(1)

    print(f"Discovering checkpoint pairs in {checkpoint_root}...")
    pairs = discover_pairs(checkpoint_root)

    if args.only_models:
        pairs = [p for p in pairs if p[0] in args.only_models]
    if args.skip_checkpoints:
        pairs = [p for p in pairs if p[2] not in args.skip_checkpoints]
    if args.only_checkpoints:
        pairs = [p for p in pairs if p[2] in args.only_checkpoints]

    if not pairs:
        print("No checkpoint pairs matched filters")
        sys.exit(1)

    print(f"Found {len(pairs)} target-vs-baseline pairs")

    rows = []
    failed_list: List[str] = []
    successful = 0
    failed = 0

    start = time.time()
    for idx, (target_model, model_folder_name, checkpoint_name, target_ckpt, baseline_ckpt) in enumerate(pairs, start=1):
        out_dir = build_output_dir(output_root, model_folder_name, checkpoint_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = run_single(
            target_model=target_model,
            target_checkpoint=target_ckpt,
            baseline_checkpoint=baseline_ckpt,
            output_dir=out_dir,
            args=args,
            index=idx,
            total=len(pairs),
        )

        if ok:
            successful += 1
            # Aggregate per-layer metrics
            per_layer_path = out_dir / "activation_metrics_per_layer.json"
            if per_layer_path.exists():
                per_layer_metrics = json.loads(per_layer_path.read_text())

                # Metric keys produced by the current activation script
                _METRIC_KEYS = [
                    "baseline_mean_norm",
                    "target_mean_norm",
                    "mean_l2_distance",
                    "relative_l2_distance",
                    "cosine_similarity",
                    "cka",
                ]

                # Aggregate across all layers (compute mean)
                aggregated: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
                for layer_data in per_layer_metrics:
                    for k in _METRIC_KEYS:
                        aggregated[k].append(layer_data.get(k, 0.0))

                final_metrics = {
                    k: (sum(v) / len(v) if v else 0.0)
                    for k, v in aggregated.items()
                }

                rows.append(
                    {
                        "model_folder": model_folder_name,
                        "checkpoint": checkpoint_name,
                        "baseline_checkpoint": str(baseline_ckpt),
                        "target_checkpoint": str(target_ckpt),
                        **final_metrics,
                    }
                )
        else:
            failed += 1
            failed_list.append(f"{model_folder_name}/{checkpoint_name}")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summary_dir / f"activation_metrics_{SIZE_TAG}_summary.json"
    summary_csv = summary_dir / f"activation_metrics_{SIZE_TAG}_summary.csv"

    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    if rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    elapsed = int(time.time() - start)
    hh = elapsed // 3600
    mm = (elapsed % 3600) // 60
    ss = elapsed % 60

    print("\n" + "=" * 80)
    print("ACTIVATION COMPARISON BATCH COMPLETE")
    print("=" * 80)
    print(f"Total     : {len(pairs)}")
    print(f"Success   : {successful}")
    print(f"Failed    : {failed}")
    print(f"Runtime   : {hh:02d}:{mm:02d}:{ss:02d}")
    print(f"Summary   : {summary_json}")
    if rows:
        print(f"SummaryCSV: {summary_csv}")

    if failed_list:
        print("Failed checkpoints:")
        for item in failed_list:
            print(f"  - {item}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
