#!/usr/bin/env python3
"""Cross-method 1-D interpolation between llama (fullrank) and low-rank methods.

Unlike 7_1d_interpolation.py which interpolates between consecutive checkpoints
of the *same* method, this script interpolates between *different* methods at
the same training step. The two endpoints live in different parameter
parameterizations, so we materialize each method's weights to a llama-compatible
dense state_dict, load both into a plain LlamaForCausalLM, and interpolate.

Supported pairs (cola is excluded — its `cola_b @ silu(cola_a)` factorization has
a nonlinear SiLU between factors, so no dense llama-equivalent weight exists):

    llama <-> {fira, galore, relora, sltrain}
    fira  <-> {galore, relora, sltrain}
    galore <-> {relora, sltrain}
    relora <-> sltrain

Metric: Inter-Method Interpolation Barrier (IMIB)
    barrier = max(loss[alpha in (0,1)]) - 0.5*(loss[alpha=0] + loss[alpha=1])
    Higher values mean the two minima are in more separated basins.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_LANDSCAPE_ROOT = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "LLMLandscape").is_dir()),
    SCRIPT_DIR.parent,
)
LLM_ROOT = LOSS_LANDSCAPE_ROOT / "LLMLandscape"
sys.path.insert(0, str(LLM_ROOT))

REPO_ROOT = next(
    (p for p in [LOSS_LANDSCAPE_ROOT, *LOSS_LANDSCAPE_ROOT.parents] if (p / "training").is_dir()),
    LOSS_LANDSCAPE_ROOT.parent,
)
sys.path.insert(0, str(REPO_ROOT))

from exps.landscape.most.landscape_eval_utils import (  # noqa: E402
    compute_nll_loss,
    get_c4_dataloader,
)


SUPPORTED_METHODS = ("llama", "fira", "galore", "relora", "sltrain")


METHOD_COLORS = {
    "llama": "#1f77b4",
    "galore": "#ff7f0e",
    "fira": "#2ca02c",
    "relora": "#9a4213",
    "sltrain": "#e377c2",
}


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------

def _load_raw_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    sd_path = checkpoint_path / "model.safetensors"
    if not sd_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")
    return load_file(str(sd_path), device="cpu")


def _load_method_config(checkpoint_path: Path, config_name: str) -> dict[str, Any]:
    p = checkpoint_path / config_name
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _materialize_relora(raw_sd: dict[str, torch.Tensor], scaling: float) -> dict[str, torch.Tensor]:
    """Collapse relora's base `weight` + `lora_A.weight` + `lora_B.weight` into a
    single dense `weight` matching the llama parameter name.

    W_eff = weight + scaling * (lora_B @ lora_A)
    """
    out: dict[str, torch.Tensor] = {}
    lora_A: dict[str, torch.Tensor] = {}
    lora_B: dict[str, torch.Tensor] = {}

    for k, v in raw_sd.items():
        if k.endswith(".lora_A.weight"):
            lora_A[k[: -len(".lora_A.weight")]] = v
        elif k.endswith(".lora_B.weight"):
            lora_B[k[: -len(".lora_B.weight")]] = v
        else:
            out[k] = v

    merged = 0
    for module_name, A in lora_A.items():
        B = lora_B.get(module_name)
        if B is None:
            continue
        base_key = module_name + ".weight"
        base = out.get(base_key)
        if base is None:
            continue
        # A: (r, in), B: (out, r), W: (out, in)
        delta = (B.float() @ A.float()) * float(scaling)
        out[base_key] = (base.float() + delta).to(base.dtype)
        merged += 1

    if merged == 0:
        raise RuntimeError("relora materialization merged 0 layers — check state dict structure")
    print(f"  [relora] merged {merged} LoRA adapters into base weights (scaling={scaling})")
    return out


def _materialize_sltrain(raw_sd: dict[str, torch.Tensor], scaling: float) -> dict[str, torch.Tensor]:
    """Turn sltrain's lora_A/lora_B/sparse_index/sparse_value into a dense `weight`.

    W_eff = scaling * (lora_B @ lora_A) + scatter(sparse_index, sparse_value)

    Sparse indices are flat indices into the (out_features, in_features) matrix.
    """
    out: dict[str, torch.Tensor] = {}
    lora_A: dict[str, torch.Tensor] = {}
    lora_B: dict[str, torch.Tensor] = {}
    sp_idx: dict[str, torch.Tensor] = {}
    sp_val: dict[str, torch.Tensor] = {}

    for k, v in raw_sd.items():
        if k.endswith(".lora_A"):
            lora_A[k[: -len(".lora_A")]] = v
        elif k.endswith(".lora_B"):
            lora_B[k[: -len(".lora_B")]] = v
        elif k.endswith(".sparse_index"):
            sp_idx[k[: -len(".sparse_index")]] = v
        elif k.endswith(".sparse_value"):
            sp_val[k[: -len(".sparse_value")]] = v
        else:
            out[k] = v

    materialized = 0
    for module_name, A in lora_A.items():
        B = lora_B.get(module_name)
        idx = sp_idx.get(module_name)
        val = sp_val.get(module_name)
        if B is None or idx is None or val is None:
            continue
        out_features, r = B.shape
        r2, in_features = A.shape
        assert r == r2, f"Rank mismatch for {module_name}: {r} vs {r2}"

        lowrank = (B.float() @ A.float()) * float(scaling)  # (out, in)

        sparse = torch.zeros(out_features * in_features, dtype=torch.float32)
        sparse[idx.long()] = val.float()
        sparse = sparse.view(out_features, in_features)

        W_eff = (lowrank + sparse).to(torch.float32)
        out[module_name + ".weight"] = W_eff
        materialized += 1

    if materialized == 0:
        raise RuntimeError("sltrain materialization produced 0 layers — check state dict structure")
    print(f"  [sltrain] materialized {materialized} SpLoRA layers (scaling={scaling})")
    return out


def materialize_to_llama_state_dict(
    method: str,
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """Return a state_dict that can be loaded into a plain LlamaForCausalLM."""
    raw = _load_raw_state_dict(checkpoint_path)

    if method in {"llama", "fira", "galore"}:
        return raw

    if method == "relora":
        cfg = _load_method_config(checkpoint_path, "relora_config.json")
        r = int(cfg.get("r", 128))
        alpha = float(cfg.get("lora_alpha", 32))
        scaling = alpha / r
        return _materialize_relora(raw, scaling)

    if method == "sltrain":
        cfg = _load_method_config(checkpoint_path, "splora_config.json")
        r = int(cfg.get("r", 128))
        alpha = float(cfg.get("lora_alpha", 32))
        scaling = alpha / r
        return _materialize_sltrain(raw, scaling)

    raise ValueError(f"Method {method} cannot be materialized to llama state_dict (cola has nonlinear factorization)")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_llama_from_checkpoint(llama_checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Build a plain LlamaForCausalLM from a llama checkpoint's config. This is
    the target architecture we load interpolated weights into."""
    cfg = AutoConfig.from_pretrained(str(llama_checkpoint))
    model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(dtype=torch.bfloat16).to(device)
    return model


def set_model_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """Load state_dict into model, casting dtype as needed."""
    target_dtype = next(model.parameters()).dtype
    typed_sd = {k: v.to(dtype=target_dtype) for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(typed_sd, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch. Missing: {missing[:5]}... Unexpected: {unexpected[:5]}...")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_and_evaluate(
    llama_shell: torch.nn.Module,
    state_A: dict[str, torch.Tensor],
    state_B: dict[str, torch.Tensor],
    alphas: np.ndarray,
    dataloader: torch.utils.data.DataLoader,
    pad_idx: int,
) -> list[float]:
    """For each alpha, set llama weights to (1-alpha)*A + alpha*B and compute loss."""
    device = next(llama_shell.parameters()).device
    dtype = next(llama_shell.parameters()).dtype

    # Cast once to target dtype so interpolation does not re-cast every time.
    A_cast = {k: v.to(dtype=dtype, device=device) for k, v in state_A.items() if k in state_B}
    B_cast = {k: v.to(dtype=dtype, device=device) for k, v in state_B.items() if k in state_A}

    param_names = set(dict(llama_shell.named_parameters()).keys())
    # Only interpolate params the shell actually owns
    shared_keys = [k for k in A_cast.keys() if k in B_cast and k in param_names]
    missing_from_shell = [k for k in A_cast.keys() if k not in param_names]
    if missing_from_shell:
        print(f"  [interp] {len(missing_from_shell)} keys in state_dict not in llama shell (will be ignored, e.g. rotary_emb buffers)")

    losses: list[float] = []
    for alpha in alphas:
        with torch.no_grad():
            for name, param in llama_shell.named_parameters():
                if name not in A_cast or name not in B_cast:
                    continue
                param.copy_((1.0 - float(alpha)) * A_cast[name] + float(alpha) * B_cast[name])

        loss = compute_nll_loss(llama_shell, dataloader, pad_idx=pad_idx)
        losses.append(float(loss))
        print(f"    alpha={alpha:+.3f}  loss={loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Metric: barrier
# ---------------------------------------------------------------------------

def compute_barrier(alphas: np.ndarray, losses: list[float]) -> dict[str, float]:
    """Compute IMIB (Inter-Method Interpolation Barrier) and supporting stats.

    Barrier = max(interior_losses) - 0.5 * (loss[0] + loss[-1])
    where "interior" is strictly between alpha=0 and alpha=1 (alpha values
    <=0 or >=1 are endpoints or extrapolations).
    """
    alphas = np.asarray(alphas, dtype=float)
    losses_arr = np.asarray(losses, dtype=float)

    # Endpoint losses via interpolation onto alpha=0 and alpha=1
    # (use nearest idx since we always include 0 and 1 as sample points)
    idx0 = int(np.argmin(np.abs(alphas - 0.0)))
    idx1 = int(np.argmin(np.abs(alphas - 1.0)))
    loss0 = float(losses_arr[idx0])
    loss1 = float(losses_arr[idx1])

    interior_mask = (alphas > 0.0) & (alphas < 1.0)
    interior = losses_arr[interior_mask]
    if interior.size == 0:
        max_interior = max(loss0, loss1)
        max_alpha = 0.5
    else:
        max_interior = float(np.max(interior))
        max_alpha = float(alphas[interior_mask][int(np.argmax(interior))])

    endpoint_mean = 0.5 * (loss0 + loss1)
    barrier = float(max_interior - endpoint_mean)

    return {
        "loss_at_alpha_0": loss0,
        "loss_at_alpha_1": loss1,
        "endpoint_mean_loss": endpoint_mean,
        "max_interior_loss": max_interior,
        "max_interior_alpha": max_alpha,
        "imib_barrier": barrier,
        "imib_barrier_relative": barrier / max(endpoint_mean, 1e-12),
    }



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-method 1D interpolation (Inter-Method Interpolation Barrier)")
    p.add_argument("--checkpoint_root", type=Path, required=True)
    p.add_argument("--model_size", type=str, required=True, choices=["60m", "130m", "350m"])
    p.add_argument("--step", type=int, required=True, help="Training step to compare across methods")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["llama", "fira", "galore", "relora", "sltrain"],
        help="Methods to include in pairwise interpolation",
    )
    p.add_argument(
        "--pairs",
        type=str,
        nargs="*",
        default=None,
        help="Explicit pair list 'a:b' (default: all unique pairs among --methods)",
    )
    p.add_argument("--alpha_min", type=float, default=-0.25)
    p.add_argument("--alpha_max", type=float, default=1.25)
    p.add_argument("--alpha_points", type=int, default=31)
    p.add_argument("--max_examples", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_batches", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def _all_pairs(methods: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            pairs.append((methods[i], methods[j]))
    return pairs


def _parse_explicit_pairs(pair_strs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in pair_strs:
        a, b = s.split(":")
        out.append((a.strip(), b.strip()))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer + dataloader (shared across all pairs) ---
    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False, trust_remote_code=True)
    dataloader = get_c4_dataloader(
        tokenizer=tokenizer,
        max_examples=args.max_examples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    pad_idx = int(tokenizer.pad_token_id)

    # --- Model shell (plain llama architecture) ---
    llama_ckpt = args.checkpoint_root / f"llama-{args.model_size}" / f"model_{args.step}"
    if not llama_ckpt.exists():
        raise FileNotFoundError(f"Llama checkpoint not found: {llama_ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Building llama shell from {llama_ckpt} on {device}")
    llama_shell = build_llama_from_checkpoint(llama_ckpt, device)

    # --- Pre-materialize each method's state_dict once ---
    state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    for method in args.methods:
        method_ckpt = args.checkpoint_root / f"{method}-{args.model_size}" / f"model_{args.step}"
        if not method_ckpt.exists():
            print(f"Warning: {method_ckpt} missing, skipping method")
            continue
        print(f"Materializing {method} @ step {args.step} ...")
        state_dicts[method] = materialize_to_llama_state_dict(method, method_ckpt)

        # Sanity: load into shell and compute endpoint loss
        set_model_state(llama_shell, state_dicts[method], strict=False)
        endpoint_loss = compute_nll_loss(llama_shell, dataloader, pad_idx=pad_idx)
        print(f"  {method} endpoint loss (via llama shell): {endpoint_loss:.4f}")

    # --- Pairs to run ---
    if args.pairs:
        pairs = _parse_explicit_pairs(args.pairs)
    else:
        pairs = _all_pairs([m for m in args.methods if m in state_dicts])

    # --- Interpolation loop ---
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    results: dict[str, dict[str, Any]] = {}
    all_barriers: list[dict[str, Any]] = []

    for a, b in pairs:
        if a not in state_dicts or b not in state_dicts:
            print(f"Skipping {a}<->{b}: missing state dict")
            continue
        pair_label = f"{a}__{b}"
        print(f"\n{'='*70}\nInterpolation: {a} <-> {b}\n{'='*70}")
        losses = interpolate_and_evaluate(
            llama_shell,
            state_dicts[a],
            state_dicts[b],
            alphas,
            dataloader,
            pad_idx,
        )
        barrier_stats = compute_barrier(alphas, losses)
        print(f"  IMIB barrier: {barrier_stats['imib_barrier']:.4f} "
              f"(relative: {barrier_stats['imib_barrier_relative']:.4f})")

        results[pair_label] = {
            "method_a": a,
            "method_b": b,
            "step": args.step,
            "alphas": alphas.tolist(),
            "losses": losses,
            **barrier_stats,
        }
        all_barriers.append({
            "pair": pair_label,
            "method_a": a,
            "method_b": b,
            "step": args.step,
            **barrier_stats,
        })

    # --- Save raw JSON ---
    raw_path = args.output_dir / f"cross_method_interp_{args.model_size}_step{args.step}_raw.json"
    raw_path.write_text(json.dumps({
        "model_size": args.model_size,
        "step": args.step,
        "alphas": alphas.tolist(),
        "results": results,
    }, indent=2))
    print(f"\nSaved raw JSON: {raw_path}")

    # --- Save barrier summary CSV + JSON ---
    if all_barriers:
        barr_csv = args.output_dir / f"cross_method_interp_{args.model_size}_step{args.step}_barriers.csv"
        with open(barr_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_barriers[0].keys()))
            writer.writeheader()
            writer.writerows(all_barriers)
        print(f"Saved barriers CSV: {barr_csv}")

        barr_json = args.output_dir / f"cross_method_interp_{args.model_size}_step{args.step}_barriers.json"
        barr_json.write_text(json.dumps(all_barriers, indent=2))

    # --- Plots ---
    _plot_per_pair(results, alphas, args.output_dir, args.model_size, args.step, args.dpi)
    _plot_overlay(results, alphas, args.output_dir, args.model_size, args.step, args.dpi)
    _plot_barrier_bar(all_barriers, args.output_dir, args.model_size, args.step, args.dpi)

    print("\nAll cross-method interpolation outputs saved.")


def _plot_per_pair(
    results: dict[str, dict[str, Any]],
    alphas: np.ndarray,
    output_dir: Path,
    model_size: str,
    step: int,
    dpi: int,
) -> None:
    for pair_label, entry in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(alphas, entry["losses"], marker="o", linewidth=2, markersize=5, color="#333")
        ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(entry["endpoint_mean_loss"], color="blue", linestyle=":", alpha=0.6, label="endpoint mean")
        ax.scatter([entry["max_interior_alpha"]], [entry["max_interior_loss"]],
                   color="red", zorder=5, label=f"barrier peak (IMIB={entry['imib_barrier']:.3f})")
        ax.set_xlabel(rf"Interpolation $\alpha$ (0={entry['method_a']}, 1={entry['method_b']})", fontsize=12)
        ax.set_ylabel("Validation NLL Loss", fontsize=12)
        ax.set_title(f"{model_size} step={step}: {entry['method_a']} <-> {entry['method_b']}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        png = output_dir / f"cross_method_interp_{model_size}_step{step}_{pair_label}.png"
        fig.savefig(png, dpi=dpi)
        fig.savefig(png.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  saved: {png}")


def _plot_overlay(
    results: dict[str, dict[str, Any]],
    alphas: np.ndarray,
    output_dir: Path,
    model_size: str,
    step: int,
    dpi: int,
) -> None:
    # One subplot group: pairs that include llama, and pairs that don't
    llama_pairs = [e for e in results.values() if "llama" in (e["method_a"], e["method_b"])]
    other_pairs = [e for e in results.values() if "llama" not in (e["method_a"], e["method_b"])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, group, title in [
        (axes[0], llama_pairs, f"llama <-> method ({model_size} step={step})"),
        (axes[1], other_pairs, f"method <-> method ({model_size} step={step})"),
    ]:
        for entry in sorted(group, key=lambda e: e["imib_barrier"], reverse=True):
            other = entry["method_b"] if entry["method_a"] == "llama" else entry["method_a"]
            label = f"{entry['method_a']}<->{entry['method_b']} (IMIB={entry['imib_barrier']:.2f})"
            color = METHOD_COLORS.get(other, None) if "llama" in (entry["method_a"], entry["method_b"]) else None
            ax.plot(alphas, entry["losses"], marker="o", linewidth=2, markersize=4, label=label, color=color)
        ax.axvline(0.0, color="gray", linestyle="--", alpha=0.4)
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel(r"Interpolation $\alpha$", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Validation NLL Loss", fontsize=12)
    fig.tight_layout()
    png = output_dir / f"cross_method_interp_{model_size}_step{step}_overlay.png"
    fig.savefig(png, dpi=dpi)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved overlay: {png}")


def _plot_barrier_bar(
    barriers: list[dict[str, Any]],
    output_dir: Path,
    model_size: str,
    step: int,
    dpi: int,
) -> None:
    if not barriers:
        return
    sorted_b = sorted(barriers, key=lambda x: x["imib_barrier"], reverse=True)
    labels = [f"{b['method_a']}<->{b['method_b']}" for b in sorted_b]
    values = [b["imib_barrier"] for b in sorted_b]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels) + 3), 6))
    bars = ax.barh(labels, values, color="#d62728", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.3f}",
                va="center", ha="left", fontsize=10)
    ax.set_xlabel("IMIB barrier (max interior loss − endpoint mean)", fontsize=12)
    ax.set_title(f"Inter-Method Interpolation Barrier ({model_size}, step={step})", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    png = output_dir / f"cross_method_interp_{model_size}_step{step}_barrier_chart.png"
    fig.savefig(png, dpi=dpi)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved barrier chart: {png}")


if __name__ == "__main__":
    main()
