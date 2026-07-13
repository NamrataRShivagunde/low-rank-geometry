import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from models import Qwen2, Llama31, Mistral
from data import get_adv_bench_behaviors_50
from tester.LMEval import lm_eval_c4
from utils.plot import Landscape4Model, Landscape4ModelPCA
from transformers import AutoTokenizer
from exps.landscape.most.landscape_eval_utils import compute_nll_loss, get_c4_dataloader, load_model_from_args



parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["llama", "cola", "fira", "sltrain", "relora", "galore"], default="llama")
# huangyao's suggestions: ["advbench", "truthfulqa", "livecode", "gpqa", "math500"]
# But, livecode&math500 is hard to implement
parser.add_argument("--task", type=str, choices=["advbench", "truthfulqa", "humaneval", "gsm8k", "mmlu", "c4-val"], default="c4-val")
# Optional: path to a checkpoint file (torch .pt/.pth). If provided, the script will attempt a best-effort load.
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to load (optional)")
# Optional: output directory where landscape data and figures will be saved
parser.add_argument("--output_dir", type=str, default="ll-plots/data", help="Directory to save results and figures")
# Device override (auto-select cuda if available)
parser.add_argument("--device", type=str, default="cuda", help="torch device string, e.g. cuda or cpu")
# Landscape plot mode
parser.add_argument("--mode", type=str, choices=["2D", "3D", "2D_PCA"], default="2D", help="2D/3D random directions, or 2D_PCA for principal-component direction")
parser.add_argument("--component", type=int, default=1, help="1-based PCA component index for 2D_PCA mode (1=top component)")

# simage_save_path is the subdirectory under output_dir where images will be saved
parser.add_argument("--image_save_path", type=str, default="images", help="Subdirectory under output_dir to save landscape images")

# C4 evaluation options
parser.add_argument("--c4_dataset_path", type=str, default=None, help="Local path to saved C4 dataset (save_to_disk) or a jsonl file to evaluate")
parser.add_argument("--tokenizer", type=str, default="t5-base", help="Tokenizer HF id to use for C4 tokenization (defaults to checkpoint or model.tokenizer)")
parser.add_argument("--c4_max_examples", type=int, default=1000, help="Number of C4 examples to sample for evaluation (default: 2000)")
parser.add_argument("--c4_max_length", type=int, default=256, help="Max tokenization length for C4 samples")
parser.add_argument("--c4_batch_size", type=int, default=128, help="Batch size for C4 evaluation")

# plot x and y limits
parser.add_argument("--x_min", type=float, default=-0.005, help="Minimum x-coordinate for landscape plot")
parser.add_argument("--x_max", type=float, default=0.005, help="Maximum x-coordinate for landscape plot")
parser.add_argument("--x_interval", type=float, default=0.25e-3, help="Interval between x-coordinates for landscape plot")
parser.add_argument("--y_min", type=float, default=-5e-3, help="Minimum y-coordinate for landscape plot")       
parser.add_argument("--y_max", type=float, default=5e-3, help="Maximum y-coordinate for landscape plot")
parser.add_argument("--y_interval", type=float, default=1e-3, help="Interval between y-coordinates for landscape plot")

# SLTrain loading options (used only when --model sltrain)
parser.add_argument("--rank", type=int, default=128, help="SLTrain rank parameter")
parser.add_argument("--lora_alpha", type=float, default=32, help="SLTrain LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=0.0, help="SLTrain LoRA dropout")
parser.add_argument("--sp_ratio", type=float, default=0.01, help="SLTrain sparse ratio")
parser.add_argument("--train_scaling", default=False, action="store_true", help="Enable SLTrain trainable scaling")
parser.add_argument(
    "--target_modules",
    type=str,
    default="attn,mlp,attention",
    help="Comma-separated module name fragments to wrap for SLTrain",
)

args = parser.parse_args()

model = load_model_from_args(args)

device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
model.to(device)


if args.task == "c4-val":
    # Prefer evaluating a saved local C4 dataset via lm_eval_c4 when provided.
    if args.c4_dataset_path:
        print(f"Using saved C4 dataset at {args.c4_dataset_path} for evaluation")
        benchmark = lambda m, verbose=False: lm_eval_c4(
            m,
            dataset_path=args.c4_dataset_path,
            tokenizer_id=(args.tokenizer or args.checkpoint),
            max_examples=args.c4_max_examples,
            max_length=args.c4_max_length,
            batch_size=args.c4_batch_size,
        )
    else:
        print("Preparing C4 dataloader (sampling from online C4)...")
        if args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        else:
            raise ValueError("No tokenizer available for C4 tokenization. Provide --tokenizer or --checkpoint that contains a tokenizer.")

        pad_idx = tokenizer.pad_token_id
        dataloader = get_c4_dataloader(tokenizer, max_examples=args.c4_max_examples, max_length=args.c4_max_length, batch_size=args.c4_batch_size)
        benchmark = lambda m, verbose=False: compute_nll_loss(m, dataloader, pad_idx)
else:
    benchmark = lambda m, verbose=False: lm_eval_gsm8k(m, verbose=verbose)
    if hasattr(model, "generate_by_input_ids"):
        model.generate = model.generate_by_input_ids
    elif not hasattr(model, "generate"):
        print("Warning: model has no 'generate' or 'generate_by_input_ids' method; generation-based evals may fail.")


saving_name = args.task + "-" + args.model + "-" + args.output_dir.split("/")[-1]

# Save arrays directly under the user-specified output directory.
npy_dir = os.path.join(args.output_dir, "npy")
os.makedirs(npy_dir, exist_ok=True)

# image_save_path can be absolute, or a subdirectory name under output_dir.
if os.path.isabs(args.image_save_path):
    img_dir = args.image_save_path
else:
    img_dir = os.path.join(args.output_dir, args.image_save_path)
os.makedirs(img_dir, exist_ok=True)

print(f"Saving name: {saving_name}")
print(f"Image directory: {img_dir}")
print(f"NumPy directory: {npy_dir}")

drawer_mode = "2D" if args.mode == "2D_PCA" else args.mode
if args.mode == "2D_PCA":
    print("Using PCA-based principal directions for landscape slice")
    if args.component < 1:
        raise ValueError(f"--component must be >= 1, got {args.component}")
    drawer = Landscape4ModelPCA(
        model,
        benchmark,
        device=torch.device("cuda"),
        save_path=img_dir,
        mode=drawer_mode,
    )
    if hasattr(drawer, "set_pca_component_index"):
        drawer.set_pca_component_index(args.component - 1)
    print(f"Using PCA component #{args.component} (1-based)")
else:
    print("Using random Gaussian directions for landscape slice")
    drawer = Landscape4Model(
        model,
        benchmark,
        direction="Gaussian",
        device=torch.device("cuda"),
        save_path=img_dir,
        mode=drawer_mode,
    )

# 60M baseline and cola
if args.model in ["llama", "cola", "fira", "sltrain", "relora", "galore"]:
    # add args for x and y limits and intervals
    
    #coordinate_config = dict(x_min=-0.005, x_max=0.005, x_interval=0.25e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
    coordinate_config = dict(x_min=args.x_min, x_max=args.x_max, x_interval=args.x_interval, y_min=args.y_min, y_max=args.y_max, y_interval=args.y_interval)

# 60M llama with weighted norm gaussian
#coordinate_config = dict(x_min=-0.04, x_max=0.04, x_interval=0.001, y_min=-0.04, y_max=0.04, y_interval=0.001)

drawer.synthesize_coordinates(**coordinate_config)

"""
Save high-resolution landscape arrays (.npy) under:
    <output_dir>/npy/
and image outputs under:
    <output_dir>/<image_save_path>/
"""

drawer.find_direction()

result = drawer.compute_for_draw()
drawer.draw_figure(drawer.mesh_x, drawer.mesh_y, result, saving_name=f"{saving_name}.png")
drawer.draw_figure(drawer.mesh_x, drawer.mesh_y, result, saving_name=f"{saving_name}.pdf")
npy_path = os.path.join(npy_dir, f"{saving_name}-high.npy")
img_path = os.path.join(img_dir, f"{saving_name}.png")
pdf_path = os.path.join(img_dir, f"{saving_name}.pdf")
np.save(npy_path, result)
print(result)
print(f"Saved image: {img_path}")
print(f"Saved pdf: {pdf_path}")
print(f"Saved array: {npy_path}")