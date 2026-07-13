#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _normalize_checkpoint_name(value: str) -> str:
	text = str(value).strip()
	if text.startswith("model_"):
		return text
	digits = "".join(ch for ch in text if ch.isdigit())
	if not digits:
		return text
	return f"model_{int(digits)}"


def _checkpoint_step(checkpoint_name: str) -> int | None:
	text = str(checkpoint_name).strip()
	if text.startswith("model_"):
		text = text[len("model_"):]
	digits = "".join(ch for ch in text if ch.isdigit())
	if not digits:
		return None
	return int(digits)


def _checkpoint_short_label(checkpoint_name: str) -> str:
	step = _checkpoint_step(checkpoint_name)
	if step is None:
		return checkpoint_name
	if step % 1000 == 0:
		return f"{step // 1000}k"
	return str(step)


def _checkpoint_numeric_label(checkpoint_name: str) -> str:
	step = _checkpoint_step(checkpoint_name)
	if step is None:
		return checkpoint_name
	return str(step)


def _model_size_from_root(root: Path) -> str | None:
	# Common roots: .../models-60m or .../models-350m
	for part in [root.name, *(p.name for p in root.parents)]:
		match = re.search(r"models[-_](\d+m)", part.lower())
		if match:
			return match.group(1)
	# Fallback: infer from a child dir like llama-60m
	for child in root.iterdir():
		if not child.is_dir():
			continue
		match = re.search(r"[-_](\d+m)$", child.name.lower())
		if match:
			return match.group(1)
	return None


def _checkpoint_tag_for_filename(checkpoints: list[str]) -> str:
	steps = sorted([s for s in (_checkpoint_step(c) for c in checkpoints) if s is not None])
	if steps:
		all_steps = list(range(1000, 10001, 1000))
		if steps == all_steps:
			return "all"
	short = [_checkpoint_short_label(c) for c in checkpoints]
	return "_".join(short)


def _auto_output_name(methods: list[str], checkpoints: list[str], model_size: str | None) -> str:
	if model_size is None:
		size_tag = "UNKNOWN"
	else:
		size_tag = model_size.upper()
	method_tag = "_".join(m.strip().lower() for m in methods)
	ckpt_tag = _checkpoint_tag_for_filename(checkpoints)
	return f"{size_tag}_{method_tag}_{ckpt_tag}_overlay"


def _extract_method_name(folder_name: str) -> str:
	# Extract method name from folder like "fira-60m" or "galore_350m" -> just "fira" or "galore"
	name = folder_name.lower()
	# Remove model size suffixes
	name = re.sub(r"[-_]\d+m$", "", name)
	return name


def _candidate_method_folders(method: str) -> list[str]:
	base = method.strip()
	base_lower = base.lower()
	out = [base, base_lower]
	suffixes = ["60m", "130m", "350m"]
	for s in suffixes:
		out.append(f"{base}-{s}")
		out.append(f"{base_lower}-{s}")
		out.append(f"{base}_{s}")
		out.append(f"{base_lower}_{s}")
	seen = set()
	ordered = []
	for item in out:
		if item in seen:
			continue
		seen.add(item)
		ordered.append(item)
	return ordered


def _find_method_dir(root: Path, method: str) -> Path:
	for name in _candidate_method_folders(method):
		candidate = root / name
		if candidate.is_dir():
			return candidate

	# Fallback to partial match (e.g., provided "fira" finds "fira-60m").
	matches = [p for p in root.iterdir() if p.is_dir() and method.lower() in p.name.lower()]
	if len(matches) == 1:
		return matches[0]
	if len(matches) > 1:
		match_names = ", ".join(sorted(p.name for p in matches))
		raise FileNotFoundError(f"Multiple folders match method '{method}': {match_names}. Use explicit folder-like method names.")
	raise FileNotFoundError(f"No folder found for method '{method}' under {root}")


def _find_checkpoint_dir(method_dir: Path, checkpoint_name: str) -> Path:
	direct = method_dir / checkpoint_name
	if direct.is_dir():
		return direct

	# Some runs keep an extra level like method_dir/<run_name>/model_1000
	nested = sorted(method_dir.glob(f"**/{checkpoint_name}"))
	nested = [p for p in nested if p.is_dir()]
	if len(nested) == 1:
		return nested[0]
	if len(nested) > 1:
		names = ", ".join(str(p) for p in nested[:5])
		raise FileNotFoundError(f"Multiple dirs found for {checkpoint_name} under {method_dir}: {names}")
	raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found under {method_dir}")


def _find_aggregate_dir(checkpoint_dir: Path) -> Path:
	candidates = [
		checkpoint_dir / "aggregate",
		checkpoint_dir / "landscape-2d" / "aggregate",
		checkpoint_dir / "artifacts" / "landscape_2d" / "aggregate",
	]
	for c in candidates:
		if (c / "npy" / "loss_mean.npy").exists() and (c / "npy" / "loss_variance.npy").exists():
			return c
	raise FileNotFoundError(f"Aggregate npy files not found under {checkpoint_dir}")


def _extract_curve(arr: np.ndarray) -> np.ndarray:
	if arr.ndim == 1:
		return np.asarray(arr, dtype=float)
	if arr.ndim == 2:
		return np.asarray(arr[arr.shape[0] // 2], dtype=float)
	raise ValueError(f"Unsupported array shape: {arr.shape}")


def _build_x_axis(agg_dir: Path, curve_size: int) -> np.ndarray:
	stats_path = agg_dir / "stats.json"
	if stats_path.exists():
		try:
			payload = json.loads(stats_path.read_text())
			x_min = payload.get("x_min")
			x_interval = payload.get("x_interval")
			if isinstance(x_min, (int, float)) and isinstance(x_interval, (int, float)):
				return float(x_min) + np.arange(curve_size, dtype=float) * float(x_interval)
		except Exception:
			pass
	return np.arange(curve_size, dtype=float)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Overlay mean +/- variance bands across methods and checkpoints")
	parser.add_argument("--root", type=Path, required=True, help="Root directory containing method folders")
	parser.add_argument("--methods", nargs="+", required=True, help="Methods/folders to overlay (e.g. llama cola fira galore)")
	parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint ids (e.g. 1000 2000 or model_1000 model_2000)")
	parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save overlay plot")
	parser.add_argument("--output_name", type=str, default="", help="Output file stem (optional; auto-generated if omitted)")
	parser.add_argument("--grid_cols", type=int, default=3, help="Columns for checkpoint subplot grid")
	parser.add_argument("--band_std_mult", type=float, default=1.0, help="Band multiplier for std shading")
	parser.add_argument("--dpi", type=int, default=200)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	root = args.root.resolve()
	if not root.is_dir():
		raise FileNotFoundError(f"Root directory not found: {root}")

	ckpts = [_normalize_checkpoint_name(c) for c in args.checkpoints]
	methods = [m.strip() for m in args.methods]
	model_size = _model_size_from_root(root)
	output_name = args.output_name.strip() if args.output_name else ""
	if not output_name:
		output_name = _auto_output_name(methods, ckpts, model_size)

	rows = int(math.ceil(len(ckpts) / max(1, args.grid_cols)))
	cols = max(1, args.grid_cols)
	fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.5 * rows), squeeze=False)

	# Track y-values per row for row-constant y-limits
	row_y_values = [[] for _ in range(rows)]
	legend_handle_by_label = {}

	for idx, ckpt in enumerate(ckpts):
		r, c = divmod(idx, cols)
		curve_payload = []

		for method in methods:

			method_dir = _find_method_dir(root, method)
			ckpt_dir = _find_checkpoint_dir(method_dir, ckpt)
			agg_dir = _find_aggregate_dir(ckpt_dir)

			mean_arr = np.load(agg_dir / "npy" / "loss_mean.npy")
			var_arr = np.load(agg_dir / "npy" / "loss_variance.npy")

			mean_curve = _extract_curve(mean_arr)
			var_curve = _extract_curve(var_arr)
			n = min(mean_curve.size, var_curve.size)
			mean_curve = mean_curve[:n]
			std_curve = np.sqrt(np.maximum(var_curve[:n], 0.0))
			x = _build_x_axis(agg_dir, n)

			band = args.band_std_mult * std_curve
			curve_payload.append(
				{
					"x": x,
					"mean": mean_curve,
					"band": band,
					"label": _extract_method_name(method_dir.name),
				}
			)

		ax = axes[r][c]
		for item in curve_payload:
			line = ax.plot(item["x"], item["mean"], linewidth=2.0, label=item["label"])
			ax.fill_between(item["x"], item["mean"] - item["band"], item["mean"] + item["band"], alpha=0.20)
			handle = line[0] if isinstance(line, list) else line
			if item["label"] not in legend_handle_by_label:
				legend_handle_by_label[item["label"]] = handle

			# Track y-values for this row
			row_y_values[r].extend(item["mean"] - item["band"])
			row_y_values[r].extend(item["mean"] + item["band"])

		ax.set_title(_checkpoint_numeric_label(ckpt), fontsize=24)
		ax.tick_params(axis="both", labelsize=16)
		ax.grid(True, alpha=0.3)

	for idx in range(len(ckpts), rows * cols):
		r, c = divmod(idx, cols)
		axes[r][c].axis("off")

	# Apply row-constant y-limits
	for r in range(rows):
		if row_y_values[r]:
			y_min = float(np.min(row_y_values[r]))
			y_max = float(np.max(row_y_values[r]))
			for c in range(cols):
				if hasattr(axes[r][c], "set_ylim"):
					axes[r][c].set_ylim(bottom=y_min, top=y_max)

	# Single global axis labels (avoid repeating per subplot)
	fig.supylabel("val loss", fontsize=24)

	handles = list(legend_handle_by_label.values())
	labels = list(legend_handle_by_label.keys())
	if handles:
		fig.legend(
			handles,
			labels,
			loc="lower center",
			bbox_to_anchor=(0.5, 0.01),
			ncol=min(6, len(labels)),
			frameon=False,
			fontsize=24,
		)

	fig.tight_layout(rect=(0.03, 0.1, 1, 1)) #rect=(left, bottom, right, top)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	png_path = args.output_dir / f"{output_name}.png"
	pdf_path = args.output_dir / f"{output_name}.pdf"
	fig.savefig(png_path, dpi=args.dpi)
	fig.savefig(pdf_path)
	plt.close(fig)

	print(f"Saved overlay PNG: {png_path}")
	print(f"Saved overlay PDF: {pdf_path}")


if __name__ == "__main__":
	main()
