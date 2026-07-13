#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_curve(path: Path) -> np.ndarray:
	arr = np.load(path)
	arr = np.asarray(arr, dtype=float)
	return arr.flatten()


def _resolve_center_index(curve_size: int, requested_center_index: int | None) -> int:
	if requested_center_index is None:
		return int(curve_size // 2)
	if requested_center_index < 0 or requested_center_index >= curve_size:
		raise ValueError(
			f"--center_index out of range: {requested_center_index}. "
			f"Valid range is [0, {curve_size - 1}]"
		)
	return int(requested_center_index)


def _write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compute expected sharpness from checkpoint landscape .npy by measuring "
			"loss delta when moving away from center index in .npy curve."
		)
	)
	parser.add_argument("--landscape_dir", type=Path, required=True, help="Checkpoint landscape dir containing aggregate/npy/loss_mean.npy")
	parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save sharpness outputs")
	parser.add_argument("--center_index", type=int, default=None, help="Optional center index; default is midpoint")
	parser.add_argument(
		"--metric",
		type=str,
		choices=["sharpness", "variance", "both"],
		default="sharpness",
		help="Which metric to compute: sharpness (default), variance band, or both.",
	)
	parser.add_argument(
		"--compute_variance",
		action="store_true",
		help="Convenience flag: compute variance-band only (equivalent to --metric variance).",
	)
	args = parser.parse_args()

	landscape_dir = args.landscape_dir.resolve()
	mean_path = landscape_dir / "aggregate" / "npy" / "loss_mean.npy"
	var_path = landscape_dir / "aggregate" / "npy" / "loss_variance.npy"

	metric_mode = "variance" if args.compute_variance else args.metric

	curve = _load_curve(mean_path)
	center_idx = _resolve_center_index(curve.size, args.center_index)
	max_offset = min(center_idx, curve.size - 1 - center_idx)

	offsets = list(range(1, max_offset + 1))

	args.output_dir.mkdir(parents=True, exist_ok=True)

	if metric_mode in {"sharpness", "both"}:
		center_loss = float(curve[center_idx])
		min_idx = int(np.argmin(curve))
		min_loss = float(curve[min_idx])

		rows = []
		sym_sharpness_values = []
		for offset in offsets:
			plus_idx = min(curve.size - 1, center_idx + offset)
			minus_idx = max(0, center_idx - offset)
			l_plus = float(curve[plus_idx])
			l_minus = float(curve[minus_idx])
			delta_plus = l_plus - center_loss
			delta_minus = l_minus - center_loss
			symmetric_delta = 0.5 * (l_plus + l_minus) - center_loss
			sym_sharpness_values.append(symmetric_delta)
			rows.append(
				{
					"offset": offset,
					"plus_index": plus_idx,
					"minus_index": minus_idx,
					"loss_plus": l_plus,
					"loss_minus": l_minus,
					"delta_plus": delta_plus,
					"delta_minus": delta_minus,
					"symmetric_delta": symmetric_delta,
				}
			)

		expected_sharpness = float(np.mean(sym_sharpness_values)) if sym_sharpness_values else 0.0
		sharpness_summary = {
			"landscape_dir": str(landscape_dir),
			"curve_length": int(curve.size),
			"center_index": center_idx,
			"max_offset": int(max_offset),
			"num_offsets": int(len(offsets)),
			"center_loss": center_loss,
			"min_index": min_idx,
			"min_loss": min_loss,
			"center_to_min_delta": center_loss - min_loss,
			"expected_sharpness": expected_sharpness,
			"offsets": offsets,
			"per_offset": rows,
		}

		sharpness_json_path = args.output_dir / "sharpness_summary.json"
		sharpness_csv_path = args.output_dir / "sharpness_per_offset.csv"
		with open(sharpness_json_path, "w") as f:
			json.dump(sharpness_summary, f, indent=2)
		_write_rows_csv(
			sharpness_csv_path,
			rows,
			[
				"offset",
				"plus_index",
				"minus_index",
				"loss_plus",
				"loss_minus",
				"delta_plus",
				"delta_minus",
				"symmetric_delta",
			],
		)

		print(f"Saved sharpness JSON: {sharpness_json_path}")
		print(f"Saved sharpness CSV: {sharpness_csv_path}")
		print(
			"Sharpness summary: "
			f"center_loss={center_loss:.6f}, min_loss={min_loss:.6f}, "
			f"center_to_min_delta={center_loss - min_loss:.6f}, "
			f"expected_sharpness={expected_sharpness:.6f}"
		)

	if metric_mode in {"variance", "both"}:
		var_curve = _load_curve(var_path)

		# Compute average variance across the curve.
		avg_variance = float(np.mean(var_curve))
		variance_summary = {
			"landscape_dir": str(landscape_dir),
			"curve_length": int(var_curve.size),
			"average_variance": avg_variance,
		}

		# Save variance summary JSON
		variance_json_path = args.output_dir / "variance_summary.json"
		with open(variance_json_path, "w") as f:
			json.dump(variance_summary, f, indent=2)
		print(f"Saved variance JSON: {variance_json_path}")
		print(f"Average variance summary: average_variance={avg_variance:.6f}")

if __name__ == "__main__":
	main()
