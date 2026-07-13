#!/usr/bin/env python3
"""PCA top-k landscape with 1D-tensor direction variants.

Wraps Landscape4ModelPCA to:
- remove the max_elements_for_pca skip
- drop the redundant per-component find_direction call (cache is already warm)
- support --bias_direction_mode {current, zero, unit, gaussian} for 1D tensors
- accept an explicit list of k values via --k_values
- run a 1D sweep (mode="2D" inside the engine, y is ignored)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

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

from utils.plot.Landscape4Model import Landscape4ModelPCA  # noqa: E402
from exps.landscape.most.landscape_eval_utils import (  # noqa: E402
    compute_nll_loss,
    get_c4_dataloader,
    load_model_from_args,
)


BIAS_MODES = ("current", "zero", "unit", "gaussian")
SVD_WEIGHT_MODES = ("none", "sigma")


class PCAVariantsDrawer(Landscape4ModelPCA):
    """PCA drawer with configurable 1D-tensor rule, optional σ_k scaling, and no redundant find_direction."""

    def __init__(
        self,
        bias_direction_mode: str = "current",
        svd_weight_mode: str = "none",
        gaussian_seed: int = 1234,
        **kwargs,
    ):
        if bias_direction_mode not in BIAS_MODES:
            raise ValueError(
                f"bias_direction_mode must be one of {BIAS_MODES}, got {bias_direction_mode}"
            )
        if svd_weight_mode not in SVD_WEIGHT_MODES:
            raise ValueError(
                f"svd_weight_mode must be one of {SVD_WEIGHT_MODES}, got {svd_weight_mode}"
            )
        super().__init__(**kwargs)
        self.bias_direction_mode = bias_direction_mode
        self.svd_weight_mode = svd_weight_mode
        self.gaussian_seed = gaussian_seed
        self._gaussian_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _direction_for_1d(self, name: str, param: torch.Tensor) -> torch.Tensor:
        mode = self.bias_direction_mode
        param_fp = param.detach().float().reshape(-1)

        if mode == "current":
            return param_fp.to(param.dtype).reshape_as(param)

        if mode == "zero":
            return torch.zeros_like(param)

        if mode == "unit":
            norm = param_fp.norm()
            if norm.item() == 0:
                return torch.zeros_like(param)
            return (param_fp / norm).to(param.dtype).reshape_as(param)

        if mode == "gaussian":
            cached = self._gaussian_cache.get(name)
            if cached is not None:
                return cached
            seed = (self.gaussian_seed + hash(name)) & 0xFFFFFFFF
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            g = torch.randn(param.shape, generator=gen, dtype=torch.float32).to(param.device)
            gn = g.norm()
            if gn.item() == 0:
                direction = torch.zeros_like(param)
            else:
                scale = param_fp.norm()
                direction = (g / gn * scale).to(param.dtype)
            self._gaussian_cache[name] = direction
            return direction

        raise AssertionError(f"unknown bias mode {mode}")

    @torch.no_grad()
    def _rank_k_directions_for_name(
        self, param_name: str, param: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if param.ndim == 1:
            x_dir = self._direction_for_1d(param_name, param)
            y_dir = torch.zeros_like(param)
            return x_dir, y_dir

        if param.ndim == 2:
            U, S, Vh = self._cached_svd(param, param_name)
            comp_idx = max(0, min(self.k_components - 1, S.numel() - 1))
            slice_dir = torch.outer(U[:, comp_idx], Vh[comp_idx, :])
            if self.svd_weight_mode == "sigma":
                slice_dir = S[comp_idx] * slice_dir
            y_rank = torch.zeros_like(slice_dir)
            return slice_dir.to(param.dtype), y_rank.to(param.dtype)

        raise ValueError(f"Unsupported parameter shape: {param.shape}")

    @torch.no_grad()
    def build_directions_for_k(self, k: int) -> None:
        """Populate x_unit_vector / y_unit_vector directly from the cache, no find_direction."""
        self.k_components = k
        self.x_unit_vector = {}
        self.y_unit_vector = {}
        for name, param in self.model.named_parameters():
            x_dir, y_dir = self._rank_k_directions_for_name(name, param.data)
            self.x_unit_vector[name] = x_dir.to(param.device)
            self.y_unit_vector[name] = y_dir.to(param.device)


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA top-k landscape with configurable 1D-tensor direction rule"
    )

    parser.add_argument(
        "--model", type=str, choices=["llama", "cola", "fira", "galore", "relora", "sltrain"], required=True
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["c4-val"], default="c4-val")
    parser.add_argument("--tokenizer", type=str, default="t5-base")

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 100],
        help="Explicit list of PCA component indices to evaluate (1-based)",
    )
    parser.add_argument(
        "--bias_direction_mode",
        type=str,
        choices=BIAS_MODES,
        default="current",
    )
    parser.add_argument(
        "--svd_weight_mode",
        type=str,
        choices=SVD_WEIGHT_MODES,
        default="none",
        help="'none' = unit rank-1 slice (legacy); 'sigma' = σ_k * outer(U_k, V_k)",
    )
    parser.add_argument("--gaussian_seed", type=int, default=1234)

    parser.add_argument("--c4_max_examples", type=int, default=1000)
    parser.add_argument("--c4_max_length", type=int, default=256)
    parser.add_argument("--c4_batch_size", type=int, default=128)

    parser.add_argument("--x_min", type=float, default=-0.25)
    parser.add_argument("--x_max", type=float, default=0.251)
    parser.add_argument("--x_interval", type=float, default=0.02)
    # y is ignored in mode="2D" (see Landscape4Model.compute_for_draw) but required by the engine API.
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=1.0)
    parser.add_argument("--y_interval", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=5)

    args = parser.parse_args()
    run_start = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    drawer = PCAVariantsDrawer(
        bias_direction_mode=args.bias_direction_mode,
        svd_weight_mode=args.svd_weight_mode,
        gaussian_seed=args.gaussian_seed,
        model=model,
        loss=benchmark,
        device=torch.device("cuda"),
        save_path=str(output_dir),
        mode="2D",
        max_elements_for_pca=int(1e12),
        k_components=args.k_values[0],
    )

    print(f"[bias_mode={args.bias_direction_mode} svd_weight={args.svd_weight_mode}] warmup_cache...")
    t0 = time.time()
    drawer.warmup_cache()
    print(f"warmup_cache: {time.time()-t0:.2f}s over {len(drawer._pca_cache)} 2D params")

    drawer.synthesize_coordinates(
        x_min=args.x_min,
        x_max=args.x_max,
        x_interval=args.x_interval,
        y_min=args.y_min,
        y_max=args.y_max,
        y_interval=args.y_interval,
    )

    timeline_rows = []
    per_k_results = []

    for idx, k in enumerate(args.k_values, start=1):
        k_start = time.time()
        drawer.build_directions_for_k(k)
        result = drawer.compute_for_draw()

        k_dir = output_dir / f"k_{k:03d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        np.save(k_dir / "loss_1d.npy", result)

        elapsed = time.time() - k_start
        total_elapsed = time.time() - run_start
        print(
            f"[{idx}/{len(args.k_values)}] k={k:>3d}  "
            f"elapsed={elapsed:.1f}s  total={_format_hms(total_elapsed)}"
        )
        timeline_rows.append(
            {
                "k": k,
                "seconds": elapsed,
                "total_elapsed": total_elapsed,
            }
        )
        per_k_results.append((k, result))

    # Aggregate mean/std across the requested k values.
    stack = np.stack([r for _, r in per_k_results], axis=0)
    mean_arr = stack.mean(axis=0)
    var_arr = stack.var(axis=0)
    np.save(output_dir / "loss_mean.npy", mean_arr)
    np.save(output_dir / "loss_variance.npy", var_arr)

    # Metadata
    mesh_x = drawer.mesh_x[0]
    metadata = {
        "command_args": vars(args),
        "command_string": " ".join(sys.argv),
        "bias_direction_mode": args.bias_direction_mode,
        "k_values": args.k_values,
        "x_grid": mesh_x.tolist(),
        "num_x_points": int(mesh_x.shape[0]),
        "total_runtime_seconds": time.time() - run_start,
        "warmup_seconds": float(t0),
        "cached_2d_params": len(drawer._pca_cache),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "timeline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "seconds", "total_elapsed"])
        writer.writeheader()
        writer.writerows(timeline_rows)

    print(f"Completed in {_format_hms(time.time()-run_start)} — outputs in {output_dir}")


if __name__ == "__main__":
    main()
