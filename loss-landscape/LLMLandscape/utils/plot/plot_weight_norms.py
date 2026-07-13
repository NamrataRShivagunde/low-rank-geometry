import argparse
import os
import re
import csv
from typing import Dict, List, Tuple

import torch
import matplotlib.pyplot as plt

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None


def find_checkpoints(base_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(base_dir):
        parent = os.path.dirname(base_dir)
        hint = ""
        if os.path.isdir(parent):
            candidates = ", ".join(sorted(os.listdir(parent)))
            hint = f" Available under {parent}: {candidates}"
        raise FileNotFoundError(f"Checkpoint base_dir not found: {base_dir}.{hint}")

    def has_model_subdirs(path: str) -> bool:
        for n in os.listdir(path):
            if re.match(r"model_\d+", n) and os.path.isdir(os.path.join(path, n)):
                return True
        return False

    if not has_model_subdirs(base_dir):
        basename = os.path.basename(base_dir)
        nested = os.path.join(base_dir, basename)
        if os.path.isdir(nested) and has_model_subdirs(nested):
            base_dir = nested
        else:
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if len(subdirs) == 1 and has_model_subdirs(subdirs[0]):
                base_dir = subdirs[0]

    ckpts = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and re.match(r"model_\d+", name):
            m = re.search(r"(\d+)", name)
            if m:
                ckpts.append((int(m.group(1)), path))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def find_weight_file(ckpt_dir: str) -> str:
    if os.path.isdir(ckpt_dir):
        subdirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
        if subdirs:
            model_dirs = [d for d in subdirs if re.match(r"model_\d+", d)]
            if model_dirs:
                raise FileNotFoundError(
                    f"No weight file found directly in {ckpt_dir}. It looks like a checkpoint base dir. "
                    f"Point --llama_dir/--cola_dir to the parent that contains model_* folders, not inside it."
                )

    candidates = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.pt",
        "model.pth",
        "checkpoint.pt",
    ]
    for c in candidates:
        p = os.path.join(ckpt_dir, c)
        if os.path.isfile(p):
            return p

    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith((".bin", ".pt", ".pth", ".safetensors")):
                return os.path.join(root, f)

    raise FileNotFoundError(f"No weight file found in {ckpt_dir}")


def load_state_dict(weight_file: str) -> Dict[str, torch.Tensor]:
    if weight_file.endswith(".safetensors"):
        if load_safetensors is None:
            raise ImportError("safetensors not installed. pip install safetensors")
        return load_safetensors(weight_file)

    obj = torch.load(weight_file, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
    return obj


def layer_key(param_name: str, depth: int) -> str:
    parts = param_name.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else param_name


def compute_mean_layer_norm(
    state_dict: Dict[str, torch.Tensor],
    depth: int,
    include_regex: str = None,
    exclude_regex: str = None,
) -> float:
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    layer_sumsq: Dict[str, float] = {}
    for name, tensor in state_dict.items():
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        if not torch.is_tensor(tensor):
            continue

        key = layer_key(name, depth)
        if key not in layer_sumsq:
            layer_sumsq[key] = 0.0
        layer_sumsq[key] += float(torch.sum(tensor.float().pow(2)).item())

    if not layer_sumsq:
        return float("nan")

    layer_norms = [s ** 0.5 for s in layer_sumsq.values()]
    return sum(layer_norms) / len(layer_norms)


def compute_layer_norms(
    state_dict: Dict[str, torch.Tensor],
    depth: int,
    include_regex: str = None,
    exclude_regex: str = None,
) -> Dict[str, float]:
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    layer_sumsq: Dict[str, float] = {}
    for name, tensor in state_dict.items():
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        if not torch.is_tensor(tensor):
            continue

        key = layer_key(name, depth)
        if key not in layer_sumsq:
            layer_sumsq[key] = 0.0
        layer_sumsq[key] += float(torch.sum(tensor.float().pow(2)).item())

    return {k: v ** 0.5 for k, v in layer_sumsq.items()}


def compute_curve(
    base_dir: str,
    depth: int,
    include_regex: str = None,
    exclude_regex: str = None,
) -> List[Tuple[int, float]]:
    points = []
    ckpts = find_checkpoints(base_dir)
    for step, ckpt_dir in ckpts:
        weight_file = find_weight_file(ckpt_dir)
        state_dict = load_state_dict(weight_file)
        mean_norm = compute_mean_layer_norm(
            state_dict, depth, include_regex, exclude_regex
        )
        points.append((step, mean_norm))
    return points


def compute_curve_per_layer(
    base_dir: str,
    depth: int,
    include_regex: str = None,
    exclude_regex: str = None,
) -> Tuple[List[int], Dict[str, List[float]]]:
    ckpts = find_checkpoints(base_dir)
    steps: List[int] = []
    layer_series: Dict[str, List[float]] = {}

    for step, ckpt_dir in ckpts:
        weight_file = find_weight_file(ckpt_dir)
        state_dict = load_state_dict(weight_file)
        layer_norms = compute_layer_norms(
            state_dict, depth, include_regex, exclude_regex
        )
        steps.append(step)
        for layer in layer_series.keys():
            layer_series[layer].append(layer_norms.get(layer, float("nan")))
        for layer, val in layer_norms.items():
            if layer not in layer_series:
                layer_series[layer] = [float("nan")] * (len(steps) - 1)
                layer_series[layer].append(val)
    return steps, layer_series


def write_csv(path: str, points: List[Tuple[int, float]], label: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", f"mean_weight_norm_{label}"])
        for step, val in points:
            writer.writerow([step, val])


def write_per_layer_csv(
    path: str, steps: List[int], layer_series: Dict[str, List[float]], label: str
):
    layers = sorted(layer_series.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"{label}:{layer}" for layer in layers])
        for i, step in enumerate(steps):
            row = [step] + [layer_series[layer][i] for layer in layers]
            writer.writerow(row)


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_dir", required=True)
    parser.add_argument("--cola_dir", required=True)
    parser.add_argument("--layer_prefix_depth", type=int, default=3)
    parser.add_argument("--include_regex", type=str, default=None)
    parser.add_argument("--exclude_regex", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/weight_norms")
    parser.add_argument("--plot_name", type=str, default="weight_norms.png")
    parser.add_argument("--per_layer", action="store_true")
    parser.add_argument("--per_layer_dir", type=str, default="per_layer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    llama_points = compute_curve(
        args.llama_dir, args.layer_prefix_depth, args.include_regex, args.exclude_regex
    )
    cola_points = compute_curve(
        args.cola_dir, args.layer_prefix_depth, args.include_regex, args.exclude_regex
    )

    write_csv(os.path.join(args.output_dir, "llama_weight_norms.csv"), llama_points, "llama")
    write_csv(os.path.join(args.output_dir, "cola_weight_norms.csv"), cola_points, "cola")

    plt.figure(figsize=(8, 5))
    plt.plot([s for s, _ in llama_points], [v for _, v in llama_points], label="llama")
    plt.plot([s for s, _ in cola_points], [v for _, v in cola_points], label="cola")
    plt.xlabel("Steps")
    plt.ylabel("Mean Weight Norm (mean over layers)")
    plt.title("Weight Norm vs Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.plot_name))
    plt.close()

    if args.per_layer:
        llama_steps, llama_layers = compute_curve_per_layer(
            args.llama_dir, args.layer_prefix_depth, args.include_regex, args.exclude_regex
        )
        cola_steps, cola_layers = compute_curve_per_layer(
            args.cola_dir, args.layer_prefix_depth, args.include_regex, args.exclude_regex
        )

        per_layer_dir = os.path.join(args.output_dir, args.per_layer_dir)
        os.makedirs(per_layer_dir, exist_ok=True)

        write_per_layer_csv(
            os.path.join(per_layer_dir, "llama_per_layer.csv"),
            llama_steps,
            llama_layers,
            "llama",
        )
        write_per_layer_csv(
            os.path.join(per_layer_dir, "cola_per_layer.csv"),
            cola_steps,
            cola_layers,
            "cola",
        )

        all_layers = sorted(set(llama_layers.keys()) | set(cola_layers.keys()))
        for layer in all_layers:
            plt.figure(figsize=(8, 5))
            if layer in llama_layers:
                plt.plot(llama_steps, llama_layers[layer], label="llama")
            if layer in cola_layers:
                plt.plot(cola_steps, cola_layers[layer], label="cola")
            plt.xlabel("Steps")
            plt.ylabel("Weight Norm")
            plt.title(f"Weight Norm vs Steps: {layer}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = safe_filename(layer) + ".png"
            plt.savefig(os.path.join(per_layer_dir, fname))
            plt.close()


if __name__ == "__main__":
    main()