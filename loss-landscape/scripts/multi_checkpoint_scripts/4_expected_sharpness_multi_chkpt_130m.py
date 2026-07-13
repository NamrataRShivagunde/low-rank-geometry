#!/usr/bin/env python3
"""
Batch script to compute expected sharpness for all checkpoints under
results/loss-landscape-2d-PCA/models-130m/ (or a custom root).

It reads each checkpoint's aggregate/npy/loss_mean.npy and computes
center-based sharpness deltas via the single-checkpoint sharpness script.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


MODEL_TYPE_MAP = {
    "llama-130m": "llama",
    "cola-130m": "cola",
    "fira-130m": "fira",
    "galore-130m": "galore",
    "relora-130m": "relora",
    "sltrain-130m": "sltrain",
}


def discover_checkpoints(landscape_root: Path) -> List[Tuple[str, str, Path]]:
    checkpoints: List[Tuple[str, str, Path]] = []

    for model_folder in sorted(landscape_root.iterdir()):
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name
        if model_name not in MODEL_TYPE_MAP:
            print(f"Skipping unknown model folder: {model_name}")
            continue

        model_type = MODEL_TYPE_MAP[model_name]
        for checkpoint_dir in sorted(model_folder.iterdir()):
            if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith("model_"):
                continue
            mean_path = checkpoint_dir / "aggregate" / "npy" / "loss_mean.npy"
            if mean_path.exists():
                checkpoints.append((model_type, checkpoint_dir.name, checkpoint_dir))

    return checkpoints


def build_output_dir(base_output: Path, model_folder: str, checkpoint_name: str) -> Path:
    return base_output / "models-130m" / model_folder / checkpoint_name


def run_sharpness_for_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    index: int,
    total: int,
) -> bool:
    cmd = [
        "python",
        "loss-landscape/scripts/single_checkpoint_scripts/4_expected_sharpness.py",
        "--landscape_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(output_dir),
    ]

    if args.center_index is not None:
        cmd.extend(["--center_index", str(args.center_index)])
    cmd.extend(["--metric", args.metric_mode])

    print("\n" + "=" * 80)
    print(f"[{index}/{total}] Sharpness: {checkpoint_dir.parent.name}/{checkpoint_dir.name}")
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
    parser = argparse.ArgumentParser(description="Batch compute expected sharpness for 130m checkpoints")
    parser.add_argument(
        "--landscape_root",
        type=Path,
        default=Path("results/loss-landscape-2d-PCA/models-130m"),
        help="Root dir with per-checkpoint landscape outputs",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/expected-sharpness-2d-PCA"),
        help="Root dir to store sharpness outputs",
    )
    parser.add_argument("--center_index", type=int, default=None)
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=["sharpness", "variance", "both"],
        default="both",
        help="Which metrics to compute per checkpoint and aggregate.",
    )
    parser.add_argument("--skip_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--only_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--only_models", type=str, nargs="*", default=[])
    args = parser.parse_args()

    landscape_root = args.landscape_root.resolve()
    output_root = args.output_root.resolve()

    if not landscape_root.exists():
        print(f"Error: landscape root not found: {landscape_root}")
        sys.exit(1)

    print(f"Discovering checkpoints in {landscape_root}...")
    checkpoints = discover_checkpoints(landscape_root)

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
    sharpness_rows = []
    variance_rows = []

    start = time.time()
    for idx, (model_type, checkpoint_name, checkpoint_path) in enumerate(checkpoints, start=1):
        model_folder_name = next(k for k, v in MODEL_TYPE_MAP.items() if v == model_type)
        out_dir = build_output_dir(output_root, model_folder_name, checkpoint_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = run_sharpness_for_checkpoint(
            checkpoint_dir=checkpoint_path,
            output_dir=out_dir,
            args=args,
            index=idx,
            total=len(checkpoints),
        )

        if ok:
            successful += 1
            if args.metric_mode in {"sharpness", "both"}:
                sharpness_path = out_dir / "sharpness_summary.json"
                if sharpness_path.exists():
                    payload = json.loads(sharpness_path.read_text())
                    sharpness_rows.append(
                        {
                            "model_folder": model_folder_name,
                            "checkpoint": checkpoint_name,
                            "expected_sharpness": payload.get("expected_sharpness"),
                            "center_loss": payload.get("center_loss"),
                            "min_loss": payload.get("min_loss"),
                            "center_to_min_delta": payload.get("center_to_min_delta"),
                        }
                    )
            if args.metric_mode in {"variance", "both"}:
                variance_path = out_dir / "variance_summary.json"
                if variance_path.exists():
                    payload = json.loads(variance_path.read_text())
                    variance_rows.append(
                        {
                            "model_folder": model_folder_name,
                            "checkpoint": checkpoint_name,
                            "average_variance": payload.get("average_variance"),
                        }
                    )
        else:
            failed += 1
            failed_list.append(f"{model_folder_name}/{checkpoint_name}")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    sharpness_summary_json = summary_dir / "expected_sharpness_130m_summary.json"
    sharpness_summary_csv = summary_dir / "expected_sharpness_130m_summary.csv"
    variance_summary_json = summary_dir / "average_variance_130m_summary.json"
    variance_summary_csv = summary_dir / "average_variance_130m_summary.csv"

    if args.metric_mode in {"sharpness", "both"}:
        with open(sharpness_summary_json, "w") as f:
            json.dump(sharpness_rows, f, indent=2)
        if sharpness_rows:
            with open(sharpness_summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_folder",
                        "checkpoint",
                        "expected_sharpness",
                        "center_loss",
                        "min_loss",
                        "center_to_min_delta",
                    ],
                )
                writer.writeheader()
                writer.writerows(sharpness_rows)

    if args.metric_mode in {"variance", "both"}:
        with open(variance_summary_json, "w") as f:
            json.dump(variance_rows, f, indent=2)
        if variance_rows:
            with open(variance_summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_folder",
                        "checkpoint",
                        "average_variance",
                    ],
                )
                writer.writeheader()
                writer.writerows(variance_rows)

    elapsed = int(time.time() - start)
    hh = elapsed // 3600
    mm = (elapsed % 3600) // 60
    ss = elapsed % 60

    print("\n" + "=" * 80)
    print("EXPECTED SHARPNESS BATCH COMPLETE")
    print("=" * 80)
    print(f"Total     : {len(checkpoints)}")
    print(f"Success   : {successful}")
    print(f"Failed    : {failed}")
    print(f"Runtime   : {hh:02d}:{mm:02d}:{ss:02d}")
    if args.metric_mode in {"sharpness", "both"}:
        print(f"Sharpness Summary   : {sharpness_summary_json}")
        if sharpness_rows:
            print(f"Sharpness SummaryCSV: {sharpness_summary_csv}")
    if args.metric_mode in {"variance", "both"}:
        print(f"Average Variance Summary    : {variance_summary_json}")
        if variance_rows:
            print(f"Average Variance SummaryCSV : {variance_summary_csv}")

    if failed_list:
        print("Failed checkpoints:")
        for item in failed_list:
            print(f"  - {item}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
