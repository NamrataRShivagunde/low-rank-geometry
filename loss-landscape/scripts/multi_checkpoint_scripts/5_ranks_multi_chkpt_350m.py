#!/usr/bin/env python3
"""Batch compute rank-related metrics for 350m checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


MODEL_TYPE_MAP = {
    "llama-350m": "llama",
    "cola-350m": "cola",
    "fira-350m": "fira",
    "galore-350m": "galore",
    "relora-350m": "relora",
    "sltrain-350m": "sltrain",
}


def discover_checkpoints(checkpoint_root: Path) -> List[Tuple[str, str, Path]]:
    checkpoints: List[Tuple[str, str, Path]] = []

    for model_folder in sorted(checkpoint_root.iterdir()):
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        if model_name not in MODEL_TYPE_MAP:
            print(f"Skipping unknown model folder: {model_name}")
            continue

        model_type = MODEL_TYPE_MAP[model_name]
        for checkpoint_dir in sorted(model_folder.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("model_"):
                checkpoints.append((model_type, checkpoint_dir.name, checkpoint_dir))

    return checkpoints


def build_output_dir(base_output: Path, model_folder: str, checkpoint_name: str) -> Path:
    return base_output / "models-350m" / model_folder / checkpoint_name


def run_rank_for_checkpoint(
    model_type: str,
    checkpoint_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    index: int,
    total: int,
) -> bool:
    cmd = [
        "python",
        "loss-landscape/scripts/single_checkpoint_scripts/5_ranks.py",
        "--model",
        model_type,
        "--checkpoint",
        str(checkpoint_dir),
        "--output_dir",
        str(output_dir),
        "--max_matrices",
        str(args.max_matrices),
        "--max_matrix_elements",
        str(args.max_matrix_elements),
        "--min_matrix_dim",
        str(args.min_matrix_dim),
        "--svd_device",
        str(args.svd_device),
        "--seed",
        str(args.seed),
    ]

    if args.include_non_layer:
        cmd.append("--include_non_layer")
    if args.rank_tol is not None:
        cmd.extend(["--rank_tol", str(args.rank_tol)])

    print("\n" + "=" * 80)
    print(f"[{index}/{total}] Rank metrics: {checkpoint_dir.parent.name}/{checkpoint_dir.name}")
    print(f"  Model : {model_type}")
    print(f"  Input : {checkpoint_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"Completed: {checkpoint_dir.parent.name}/{checkpoint_dir.name}")
            return True
        print(f"Failed: {checkpoint_dir.parent.name}/{checkpoint_dir.name} (exit code {result.returncode})")
        return False
    except Exception as exc:
        print(f"Error running {checkpoint_dir}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch compute rank metrics for 350m checkpoints")
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("CHECKPOINTS/models-350m"),
        help="Root dir containing model/checkpoint folders",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/rank-metrics/models-350m"),
        help="Root dir to store rank metric outputs",
    )
    parser.add_argument("--include_non_layer", action="store_true")
    parser.add_argument("--max_matrices", type=int, default=128)
    parser.add_argument("--max_matrix_elements", type=int, default=4_000_000)
    parser.add_argument("--min_matrix_dim", type=int, default=2)
    parser.add_argument("--rank_tol", type=float, default=None)
    parser.add_argument("--svd_device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--only_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--only_models", type=str, nargs="*", default=[])
    args = parser.parse_args()

    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()

    if not checkpoint_root.exists():
        print(f"Error: checkpoint root not found: {checkpoint_root}")
        sys.exit(1)

    print(f"Discovering checkpoints in {checkpoint_root}...")
    checkpoints = discover_checkpoints(checkpoint_root)

    if args.only_models:
        checkpoints = [(mt, cn, cp) for mt, cn, cp in checkpoints if mt in args.only_models]
    if args.skip_checkpoints:
        checkpoints = [(mt, cn, cp) for mt, cn, cp in checkpoints if cn not in args.skip_checkpoints]
    if args.only_checkpoints:
        checkpoints = [(mt, cn, cp) for mt, cn, cp in checkpoints if cn in args.only_checkpoints]

    if not checkpoints:
        print("No checkpoints matched filters")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints")

    successful = 0
    failed = 0
    failed_list: List[str] = []
    aggregate_rows = []

    start = time.time()
    for idx, (model_type, checkpoint_name, checkpoint_path) in enumerate(checkpoints, start=1):
        model_folder_name = next(k for k, v in MODEL_TYPE_MAP.items() if v == model_type)
        out_dir = build_output_dir(output_root, model_folder_name, checkpoint_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = run_rank_for_checkpoint(
            model_type=model_type,
            checkpoint_dir=checkpoint_path,
            output_dir=out_dir,
            args=args,
            index=idx,
            total=len(checkpoints),
        )

        if ok:
            successful += 1
            summary_path = out_dir / "rank_metrics_summary.json"
            if summary_path.exists():
                payload = json.loads(summary_path.read_text())
                aggregate_rows.append(
                    {
                        "model_folder": model_folder_name,
                        "checkpoint": checkpoint_name,
                        "rank": payload.get("rank"),
                        "rank_ratio": payload.get("rank_ratio"),
                        "effective_rank": payload.get("effective_rank"),
                        "effective_rank_score": payload.get("effective_rank_score"),
                        "stable_rank": payload.get("stable_rank"),
                        "spectral_gap": payload.get("spectral_gap"),
                        "singular_value_ratio": payload.get("singular_value_ratio"),
                        "num_singular_gt_threshold": payload.get("num_singular_gt_threshold"),
                        "singular_gt_threshold_ratio": payload.get("singular_gt_threshold_ratio"),
                        "num_matrices_processed": payload.get("num_matrices_processed"),
                    }
                )
        else:
            failed += 1
            failed_list.append(f"{model_folder_name}/{checkpoint_name}")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summary_dir / "rank_metrics_350m_summary.json"
    summary_csv = summary_dir / "rank_metrics_350m_summary.csv"

    with open(summary_json, "w") as f:
        json.dump(aggregate_rows, f, indent=2)

    if aggregate_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model_folder",
                    "checkpoint",
                    "rank",
                    "rank_ratio",
                    "effective_rank",
                    "effective_rank_score",
                    "stable_rank",
                    "spectral_gap",
                    "singular_value_ratio",
                    "num_singular_gt_threshold",
                    "singular_gt_threshold_ratio",
                    "num_matrices_processed",
                ],
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)

    elapsed = int(time.time() - start)
    hh = elapsed // 3600
    mm = (elapsed % 3600) // 60
    ss = elapsed % 60

    print("\n" + "=" * 80)
    print("RANK METRICS BATCH COMPLETE")
    print("=" * 80)
    print(f"Total     : {len(checkpoints)}")
    print(f"Success   : {successful}")
    print(f"Failed    : {failed}")
    print(f"Runtime   : {hh:02d}:{mm:02d}:{ss:02d}")
    print(f"Summary   : {summary_json}")
    if aggregate_rows:
        print(f"SummaryCSV: {summary_csv}")

    if failed_list:
        print("Failed checkpoints:")
        for item in failed_list:
            print(f"  - {item}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
