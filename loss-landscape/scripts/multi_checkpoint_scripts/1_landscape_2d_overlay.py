#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLOR_MAP = {
	"full_rank": "#1f77b4",  # blue
	"full-rank": "#1f77b4",
	"fullrank": "#1f77b4",
	"llama": "#1f77b4",
	"galore": "#ff7f0e",
	"fira": "#2ca02c",
	"cola": "#d62728",
	"relora": "#9467bd",
	"switchlora": "#8c564b",
	"sltrain": "#e377c2",
}

FALLBACK_METHOD_COLORS = [
	"#17becf",
	"#bcbd22",
	"#7f7f7f",
	"#8c564b",
	"#e377c2",
	"#d62728",
	"#2ca02c",
	"#ff7f0e",
]


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


def _method_color(method_name: str) -> str:
	key = str(method_name).strip().lower()
	if key in METHOD_COLOR_MAP:
		return METHOD_COLOR_MAP[key]
	digest = hashlib.md5(key.encode("utf-8")).hexdigest()
	idx = int(digest[:8], 16) % len(FALLBACK_METHOD_COLORS)
	return FALLBACK_METHOD_COLORS[idx]


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


def _parse_row_ylim(values: list[str], rows: int) -> list[tuple[float, float]]:
	if not values:
		return []
	if len(values) != rows:
		raise ValueError(f"--row_ylim must provide exactly {rows} values (one min,max pair per row); got {len(values)}")
	out: list[tuple[float, float]] = []
	for token in values:
		parts = token.split(",")
		if len(parts) != 2:
			raise ValueError(f"Invalid --row_ylim token '{token}'. Expected min,max format.")
		y_min = float(parts[0].strip())
		y_max = float(parts[1].strip())
		if y_min >= y_max:
			raise ValueError(f"Invalid --row_ylim token '{token}': min must be < max")
		out.append((y_min, y_max))
	return out


def _parse_col_ylim(values: list[str], cols: int) -> list[tuple[float, float]]:
	if not values:
		return []
	if len(values) != cols:
		raise ValueError(f"--col_ylim must provide exactly {cols} values (one min,max pair per column); got {len(values)}")
	out: list[tuple[float, float]] = []
	for token in values:
		parts = token.split(",")
		if len(parts) != 2:
			raise ValueError(f"Invalid --col_ylim token '{token}'. Expected min,max format.")
		y_min = float(parts[0].strip())
		y_max = float(parts[1].strip())
		if y_min >= y_max:
			raise ValueError(f"Invalid --col_ylim token '{token}': min must be < max")
		out.append((y_min, y_max))
	return out


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Overlay mean +/- variance bands across methods and checkpoints")
	parser.add_argument("--root", type=Path, required=True, help="Root directory containing method folders")
	parser.add_argument("--methods", nargs="+", required=True, help="Methods/folders to overlay (e.g. llama cola fira galore)")
	parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint ids (e.g. 1000 2000 or model_1000 model_2000)")
	parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save overlay plot")
	parser.add_argument("--output_name", type=str, default="", help="Output file stem (optional; auto-generated if omitted)")
	parser.add_argument("--grid_cols", type=int, default=3, help="Columns for checkpoint subplot grid")
	parser.add_argument("--band_std_mult", type=float, default=1.0, help="Band multiplier for std shading")
	parser.add_argument(
		"--row_ylim",
		nargs="*",
		default=[],
		help="Optional manual y-limits per row as min,max pairs. Example for 2 rows: --row_ylim 3.50,4.20 3.40,3.90",
	)
	parser.add_argument(
		"--col_ylim",
		nargs="*",
		default=[],
		help="Optional manual y-limits per column as min,max pairs. Example for 3 columns: --col_ylim 3.50,4.20 3.40,3.90 3.30,3.80",
	)
	parser.add_argument("--dpi", type=int, default=200)
	# figsize
	parser.add_argument("--fig_width", type=float, default=6.0, help="Width of each subplot in inches")
	parser.add_argument("--fig_height", type=float, default=18.0, help="Height of each subplot in inches")
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
	manual_row_ylims = _parse_row_ylim(args.row_ylim, rows)
	manual_col_ylims = _parse_col_ylim(args.col_ylim, cols)
	if manual_row_ylims and manual_col_ylims:
		raise ValueError("Use either --row_ylim or --col_ylim, not both.")
	fig, axes = plt.subplots(rows, cols, figsize=(args.fig_width * cols, args.fig_height * rows), squeeze=False)

	# Track y-values per row for row-constant y-limits
	row_y_values = [[] for _ in range(rows)]
	col_y_values = [[] for _ in range(cols)]
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
			color = _method_color(item["label"])
			line = ax.plot(item["x"], item["mean"], linewidth=2.0, label=item["label"], color=color)
			ax.fill_between(item["x"], item["mean"] - item["band"], item["mean"] + item["band"], color=color, alpha=0.20)
			handle = line[0] if isinstance(line, list) else line
			if item["label"] not in legend_handle_by_label:
				legend_handle_by_label[item["label"]] = handle

			# Track y-values for this row
			row_y_values[r].extend(item["mean"] - item["band"])
			row_y_values[r].extend(item["mean"] + item["band"])
			col_y_values[c].extend(item["mean"] - item["band"])
			col_y_values[c].extend(item["mean"] + item["band"])

		ax.set_title(_checkpoint_numeric_label(ckpt), fontsize=24)
		ax.tick_params(axis="both", labelsize=16)
		ax.grid(True, alpha=0.3)

	for idx in range(len(ckpts), rows * cols):
		r, c = divmod(idx, cols)
		axes[r][c].axis("off")

	# Apply y-limits:
	# 1) manual row limits if provided
	# 2) manual column limits if provided
	# 3) default automatic row limits (existing behavior)
	if manual_row_ylims:
		for r in range(rows):
			y_min, y_max = manual_row_ylims[r]
			for c in range(cols):
				if hasattr(axes[r][c], "set_ylim"):
					axes[r][c].set_ylim(bottom=y_min, top=y_max)
	elif manual_col_ylims:
		for c in range(cols):
			y_min, y_max = manual_col_ylims[c]
			for r in range(rows):
				if hasattr(axes[r][c], "set_ylim"):
					axes[r][c].set_ylim(bottom=y_min, top=y_max)
	else:
		for r in range(rows):
			if not row_y_values[r]:
				continue
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
