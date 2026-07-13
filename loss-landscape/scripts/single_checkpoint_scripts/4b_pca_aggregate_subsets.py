#!/usr/bin/env python3
"""Aggregate PCA landscape results into subsets (top-5, top-10, top-100).

Given a checkpoint run directory with k_001/, k_002/, ..., k_100/ subdirs,
each containing loss_1d.npy, this script:
  1. Builds aggregate mean/variance for top-K subsets (default: 5, 10, 100)
  2. Saves them under subset_k005/, subset_k010/, subset_k100/ with the same
     aggregate/npy/ structure used by the overlay and sharpness scripts
  3. Computes expected sharpness for each subset

Usage:
    python 4b_pca_aggregate_subsets.py --run_dir results/pca-verification-k100/60m/llama/model_1000
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_sharpness(curve: np.ndarray) -> dict:
    """Compute expected sharpness from a 1D loss curve."""
    center_idx = len(curve) // 2
    center_loss = float(curve[center_idx])
    min_idx = int(np.argmin(curve))
    min_loss = float(curve[min_idx])
    max_offset = min(center_idx, len(curve) - 1 - center_idx)

    sym_values = []
    rows = []
    for offset in range(1, max_offset + 1):
        plus_idx = min(len(curve) - 1, center_idx + offset)
        minus_idx = max(0, center_idx - offset)
        l_plus = float(curve[plus_idx])
        l_minus = float(curve[minus_idx])
        sym_delta = 0.5 * (l_plus + l_minus) - center_loss
        sym_values.append(sym_delta)
        rows.append({
            "offset": offset,
            "plus_index": plus_idx,
            "minus_index": minus_idx,
            "loss_plus": l_plus,
            "loss_minus": l_minus,
            "delta_plus": l_plus - center_loss,
            "delta_minus": l_minus - center_loss,
            "symmetric_delta": sym_delta,
        })

    return {
        "curve_length": int(len(curve)),
        "center_index": center_idx,
        "max_offset": max_offset,
        "num_offsets": len(sym_values),
        "center_loss": center_loss,
        "min_index": min_idx,
        "min_loss": min_loss,
        "center_to_min_delta": center_loss - min_loss,
        "expected_sharpness": float(np.mean(sym_values)) if sym_values else 0.0,
        "offsets": list(range(1, max_offset + 1)),
        "per_offset": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Checkpoint run dir containing k_001/, k_002/, etc.")
    parser.add_argument("--subsets", type=int, nargs="+", default=[5, 10, 100],
                        help="Top-K subsets to aggregate (default: 5 10 100)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    # Find all available k directories
    k_dirs = sorted(run_dir.glob("k_*"))
    available_k = []
    for kd in k_dirs:
        npy = kd / "loss_1d.npy"
        if npy.exists():
            k_num = int(kd.name.split("_")[1])
            available_k.append((k_num, npy))
    available_k.sort(key=lambda x: x[0])

    if not available_k:
        print(f"No k_*/loss_1d.npy found in {run_dir}")
        return

    print(f"Found {len(available_k)} k-directions: {available_k[0][0]}..{available_k[-1][0]}")

    for top_k in args.subsets:
        # Select the top-K components (k=1 through k=top_k)
        subset = [(k, p) for k, p in available_k if k <= top_k]
        if not subset:
            print(f"  top-{top_k}: no components available, skipping")
            continue

        actual_k = len(subset)
        print(f"  top-{top_k}: aggregating {actual_k} components (k={subset[0][0]}..{subset[-1][0]})")

        # Load and stack
        curves = np.stack([np.load(p) for _, p in subset], axis=0)
        mean_arr = curves.mean(axis=0)
        var_arr = curves.var(axis=0)

        # Save in the standard aggregate layout
        out_dir = run_dir / f"subset_k{top_k:03d}"
        agg_npy = out_dir / "aggregate" / "npy"
        agg_npy.mkdir(parents=True, exist_ok=True)
        np.save(agg_npy / "loss_mean.npy", mean_arr)
        np.save(agg_npy / "loss_variance.npy", var_arr)

        # Compute and save sharpness
        sharp = compute_sharpness(mean_arr)
        sharp["landscape_dir"] = str(out_dir)
        sharp["top_k"] = top_k
        sharp["actual_components"] = actual_k
        sharp["k_range"] = [subset[0][0], subset[-1][0]]

        sharp_dir = out_dir / "sharpness"
        sharp_dir.mkdir(parents=True, exist_ok=True)
        with open(sharp_dir / "sharpness_summary.json", "w") as f:
            json.dump(sharp, f, indent=2)

        print(f"    expected_sharpness={sharp['expected_sharpness']:.6f}, "
              f"var_mean={float(var_arr.mean()):.6f}")


if __name__ == "__main__":
    main()
