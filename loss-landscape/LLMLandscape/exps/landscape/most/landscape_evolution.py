"""
Generate loss landscape plots across training checkpoints to visualize landscape evolution.

Usage:
    python loss-landscape/LLMLandscape/exps/landscape/most/landscape_evolution_v2.py --checkpoints_base checkpoints/llama_60m-2025-12-08-21-00-08 \
    --model llama \
    --task c4-val \
    --c4_max_examples 5 \
    --c4_batch_size 64 \
    --c4_max_length 256 \
    --curve_grid \
    --grid_cols 5 \
    --x_min -0.005 --x_max 0.005 --x_interval 0.00025 \
    --output_dir output/landscape_evolution \
    --curve_grid_png llama_u_curves_grid.png \
    --tokenizer "t5-base" --y_min -0.005 --y_max 0.005 --y_interval 0.001

This script:
1. Finds checkpoint subdirectories (e.g., model_500, model_1000, ...) under --checkpoints_base
2. Selects up to --max_checkpoints evenly spaced checkpoints
3. For each checkpoint:
   - Loads the model
   - Computes loss landscape using the same method as landscape_lowrank.py
   - Saves individual PDF plots
4. Optionally creates a combined multi-page PDF showing landscape evolution

The landscape computation uses Gaussian perturbations and the specified task/benchmark.
"""

import argparse
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import landscape utilities (assuming we're running from repo root with PYTHONPATH set)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from utils.plot import Landscape4Model
from tester import lm_eval_truthfulqa, lm_eval_gsm8k, lm_eval_humaneval, lm_eval_mmlu
from tester.LMEval import lm_eval_c4
from data import get_adv_bench_behaviors_50
from tester import test_vanilla_harmful_output_rate
#from landscape_lowrank import compute_nll_loss

def compute_nll_loss(model, dataloader, pad_idx):
    """
    Token-normalized NLL computed over dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch[0].to(model_device)
            attention_mask = batch[1].to(model_device)

            labels = input_ids.clone()
            labels[labels == pad_idx] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            batch_loss = outputs.loss.item()
            batch_tokens = (labels != -100).sum().item()

            total_loss += batch_loss
            total_tokens += batch_tokens
            num_batches += 1
        
        total_loss = total_loss / num_batches

        return total_loss

def get_c4_dataloader(tokenizer: AutoTokenizer, max_examples: int = 2000, max_length: int = 128, batch_size: int = 4):
    """Load a sampled subset of the C4 validation split in streaming mode and return a PyTorch DataLoader."""
    print(f"Loading C4 validation split (up to {max_examples} examples) in streaming mode...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    dataset = dataset.shuffle(seed=42)

    streamed_samples = []
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        streamed_samples.append(example)

    tokenized = tokenizer([ex["text"] for ex in streamed_samples], truncation=True, padding="max_length", 
                         max_length=max_length, return_tensors="pt")

    dataset_torch = torch.utils.data.TensorDataset(tokenized["input_ids"], tokenized["attention_mask"]) 
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)
    print(f"C4 DataLoader ready with {len(dataset_torch)} examples.")
    return dataloader


def find_checkpoint_dirs(base_path, pattern=r"model_\d+", max_count=None):
    """Find checkpoint directories matching pattern, sorted by step number."""
    if not os.path.isdir(base_path):
        return []
    
    dirs = [d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d)) and re.match(pattern, d)]
    
    # Sort by numeric suffix
    def get_step_number(dirname):
        match = re.search(r'(\d+)$', dirname)
        return int(match.group(1)) if match else 0
    
    dirs.sort(key=get_step_number)
    
    if max_count and len(dirs) > max_count:
        # Select evenly spaced checkpoints
        indices = np.linspace(0, len(dirs) - 1, max_count, dtype=int)
        dirs = [dirs[i] for i in indices]
    
    return [os.path.join(base_path, d) for d in dirs]


def load_model(checkpoint_path, model_type, device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    if model_type == "llama":
        config = AutoConfig.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
    elif model_type == "cola":
        from training.CoLA.cola import ColaConfig, ColaForCausalLM
        config = ColaConfig.from_pretrained(checkpoint_path)
        model = ColaForCausalLM.from_pretrained(
            checkpoint_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def setup_benchmark(task, model, checkpoint_path, args):
    """Setup benchmark function based on task."""
    if task == "advbench":
        loader = get_adv_bench_behaviors_50()
        benchmark = lambda m, verbose=False: test_vanilla_harmful_output_rate(m, loader, verbose=verbose)
        if hasattr(model, 'generation_max_length'):
            model.generation_max_length = 100
            
    elif task == "truthfulqa":
        benchmark = lambda m, verbose=False: lm_eval_truthfulqa(m, verbose=verbose, limit=100)
        if hasattr(model, "generate_by_input_ids"):
            model.generate = model.generate_by_input_ids
            
    elif task == "humaneval":
        benchmark = lambda m, verbose=False: lm_eval_humaneval(m, verbose=verbose)
        if hasattr(model, "generate_by_input_ids"):
            model.generate = model.generate_by_input_ids
            
    elif task == "mmlu":
        benchmark = lambda m, verbose=False: lm_eval_mmlu(m, verbose=verbose)
        if hasattr(model, "generate_by_input_ids"):
            model.generate = model.generate_by_input_ids
            
    elif task == "c4-val":
        if args.c4_dataset_path:
            print(f"Using saved C4 dataset at {args.c4_dataset_path}")
            benchmark = lambda m, verbose=False: lm_eval_c4(
                m,
                dataset_path=args.c4_dataset_path,
                tokenizer_id=(args.tokenizer or checkpoint_path),
                max_examples=args.c4_max_examples,
                max_length=args.c4_max_length,
                batch_size=args.c4_batch_size,
            )
        else:
          
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            
            if tokenizer is None:
                raise ValueError("No tokenizer available for C4 tokenization")
            
            pad_idx = tokenizer.pad_token_id
            dataloader = get_c4_dataloader(tokenizer, max_examples=args.c4_max_examples, 
                                          max_length=args.c4_max_length, batch_size=args.c4_batch_size)
            benchmark = lambda m, verbose=False: compute_nll_loss(m, dataloader, pad_idx)
    else:
        # Default to gsm8k
        benchmark = lambda m, verbose=False: lm_eval_gsm8k(m, verbose=verbose)
        if hasattr(model, "generate_by_input_ids"):
            model.generate = model.generate_by_input_ids
    
    return benchmark


def compute_landscape_for_checkpoint(checkpoint_path, model_type, task, args, device):
    """Compute loss landscape for a single checkpoint."""
    # Load model
    model = load_model(checkpoint_path, model_type, device)
    
    # Setup benchmark
    benchmark = setup_benchmark(task, model, checkpoint_path, args)
    
    # Wrap benchmark with debug output and NaN detection
    eval_counter = [0]  # Use list to allow mutation in closure
    nan_count = [0]
    def benchmark_with_logging(m, verbose=False):
        eval_counter[0] += 1
        if eval_counter[0] == 1 or eval_counter[0] % 20 == 0:
            print(f"  Evaluation {eval_counter[0]}...", flush=True)
        result = benchmark(m, verbose=verbose)
        if np.isnan(result):
            nan_count[0] += 1
            if nan_count[0] <= 3:
                print(f"  WARNING: NaN loss at evaluation {eval_counter[0]} (NaN count: {nan_count[0]})", flush=True)
        if eval_counter[0] == 1:
            print(f"  First evaluation complete: loss={result:.4f}", flush=True)
        return result
    
    # Create landscape drawer
    drawer = Landscape4Model(
        model,
        benchmark_with_logging,
        mode="2D",  # Use 2D mode for surface plots
        direction="WeightNormGaussian",
        device=device,
    )
    
    # Configure coordinates
    coordinate_config = dict(
        x_min=args.x_min, 
        x_max=args.x_max, 
        x_interval=args.x_interval,
        y_min=args.y_min, 
        y_max=args.y_max, 
        y_interval=args.y_interval
    )
    
    drawer.synthesize_coordinates(**coordinate_config)
    
    # Compute landscape
    checkpoint_name = os.path.basename(checkpoint_path)
    grid_shape = drawer.mesh_x.shape
    total_evals = grid_shape[0] * grid_shape[1]
    print(f"\nComputing landscape for {checkpoint_name}...")
    print(f"  Grid shape: {grid_shape}, Total evaluations: {total_evals}")
    print(f"  Finding random direction...")
    
    if task == "truthfulqa":
        # Average over multiple random directions
        result = 0
        for _ in range(args.truthfulqa_repeats):
            drawer.find_direction()
            cur_result = drawer.compute_for_draw()
            result += cur_result
        result /= args.truthfulqa_repeats
    else:
        drawer.find_direction(seed=42)
        result = drawer.compute_for_draw()
    
    # Ensure result is 2D for plotting
    # compute_for_draw returns a 2D array with shape matching mesh_x and mesh_y
    # If it's somehow 1D, reshape it
    if result.ndim == 1:
        # Try to reshape based on mesh dimensions
        mesh_shape = drawer.mesh_x.shape
        result = result.reshape(mesh_shape)
    
    # Report NaN statistics
    num_nans = np.sum(np.isnan(result))
    if num_nans > 0:
        print(f"  WARNING: {num_nans}/{result.size} evaluations resulted in NaN ({100*num_nans/result.size:.1f}%)")
        print(f"  Consider reducing perturbation range (--x_min, --x_max, --y_min, --y_max)")
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    return result, drawer.mesh_x, drawer.mesh_y, checkpoint_name


def save_individual_pdf(mesh_x, mesh_y, result, checkpoint_name, output_path, task):
    """Save individual landscape plot as PDF."""
    # Debug: check shapes
    print(f"Debug - mesh_x shape: {mesh_x.shape}, mesh_y shape: {mesh_y.shape}, result shape: {result.shape}")
    
    # Ensure all arrays are 2D
    if result.ndim != 2:
        raise ValueError(f"Result must be 2D for surface plot, got shape {result.shape}")
    if mesh_x.shape != result.shape or mesh_y.shape != result.shape:
        raise ValueError(f"Shape mismatch: mesh_x {mesh_x.shape}, mesh_y {mesh_y.shape}, result {result.shape}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(mesh_x, mesh_y, result, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X Direction')
    ax.set_ylabel('Y Direction')
    ax.set_zlabel(f'{task} Loss')
    ax.set_title(f'Loss Landscape - {checkpoint_name}')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved individual PDF: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate loss landscape evolution across training checkpoints")
    
    # Required arguments
    parser.add_argument("--checkpoints_base", type=str, required=True,
                       help="Base directory containing checkpoint subdirectories (e.g., model_500, model_1000, ...)")
    parser.add_argument("--model", type=str, choices=["llama", "cola"], required=True,
                       help="Model type")
    parser.add_argument("--task", type=str, 
                       choices=["advbench", "truthfulqa", "humaneval", "gsm8k", "mmlu", "c4-val"],
                       default="c4-val",
                       help="Evaluation task")
    
    # Optional arguments
    parser.add_argument("--max_checkpoints", type=int, default=20,
                       help="Maximum number of checkpoints to process (default: 20)")
    parser.add_argument("--checkpoint_pattern", type=str, default=r"model_\d+",
                       help="Regex pattern for checkpoint directories (default: model_\\d+)")
    parser.add_argument("--output_dir", type=str, default="./output/landscape_evolution",
                       help="Directory to save output PDFs and data")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device string (e.g., cuda, cpu); auto-detected if not provided")
    
    # Landscape configuration
    # Using settings that worked previously: 80x80 grid
    parser.add_argument("--x_min", type=float, default=-0.04)
    parser.add_argument("--x_max", type=float, default=0.04)
    parser.add_argument("--x_interval", type=float, default=0.001)  # 80 points
    parser.add_argument("--y_min", type=float, default=-0.04)
    parser.add_argument("--y_max", type=float, default=0.04)
    parser.add_argument("--y_interval", type=float, default=0.001)  # 80 points
    # This gives an 80x80 = 6400 evaluation grid by default
    
    # C4 evaluation options
    parser.add_argument("--c4_dataset_path", type=str, default=None,
                       help="Local path to saved C4 dataset")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer HF id (defaults to checkpoint)")
    parser.add_argument("--c4_max_examples", type=int, default=2000)
    parser.add_argument("--c4_max_length", type=int, default=256)
    parser.add_argument("--c4_batch_size", type=int, default=128)
    
    # Task-specific options
    parser.add_argument("--truthfulqa_repeats", type=int, default=5,
                       help="Number of random directions to average for TruthfulQA (default: 5)")
    
    # Output options
    parser.add_argument("--combined_pdf", type=str, default="landscape_evolution_combined.pdf",
                       help="Filename for combined multi-page PDF")
    parser.add_argument("--save_numpy", action="store_true",
                       help="Save landscape data as .npy files")

    # Curve-only mode: compute a single scalar metric per checkpoint and plot a 1D curve
    parser.add_argument("--curve_only", action="store_true",
                       help="If set, skip landscape surfaces and plot a loss/metric curve across checkpoints")
    parser.add_argument("--curve_png", type=str, default="loss_curve.png",
                       help="Output filename for the curve PNG (saved under --output_dir)")
    parser.add_argument("--curve_csv", type=str, default="loss_curve.csv",
                       help="Optional CSV filename for (step, metric) values (saved under --output_dir)")

    # U-curve grid: plot 1D landscape curves (Gaussian direction, x-axis sweep) for each checkpoint
    parser.add_argument("--curve_grid", action="store_true",
                       help="If set, compute a 1D loss landscape curve per checkpoint and plot a side-by-side grid")
    parser.add_argument("--curve_grid_png", type=str, default="u_curves_grid.png",
                       help="Output filename for the grid of U-curves (saved under --output_dir)")
    parser.add_argument("--grid_cols", type=int, default=5,
                       help="Number of columns in the U-curve grid (rows auto-computed)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device) if args.device else \
             (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find checkpoints
    checkpoint_paths = find_checkpoint_dirs(args.checkpoints_base, 
                                           pattern=args.checkpoint_pattern,
                                           max_count=args.max_checkpoints)
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {args.checkpoints_base} matching pattern {args.checkpoint_pattern}")
    
    print(f"\nFound {len(checkpoint_paths)} checkpoints to process")
    print("Checkpoints:", [os.path.basename(p) for p in checkpoint_paths])
    
    if args.curve_only:
        # Compute a single scalar metric per checkpoint and plot a curve
        print("\nCurve-only mode: computing scalar metric per checkpoint...")
        points = []  # list of (step, metric)
        for checkpoint_path in tqdm(checkpoint_paths, desc="Checkpoints"):
            try:
                # Extract step number from directory name
                base = os.path.basename(checkpoint_path)
                step = 0
                m = re.search(r"(\d+)$", base)
                if m:
                    step = int(m.group(1))

                # Load model and benchmark
                model = load_model(checkpoint_path, args.model, device)
                benchmark = setup_benchmark(args.task, model, checkpoint_path, args)
                metric = float(benchmark(model, verbose=False))

                points.append((step, metric, base))
                # cleanup
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Skipping {checkpoint_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not points:
            raise SystemExit("No metrics computed; check checkpoints and task setup")

        # Sort by step
        points.sort(key=lambda x: x[0])
        steps = [p[0] for p in points]
        metrics = [p[1] for p in points]

        # Save CSV
        csv_path = os.path.join(args.output_dir, args.curve_csv)
        with open(csv_path, "w") as f:
            f.write("step,metric,checkpoint\n")
            for s, mval, name in points:
                f.write(f"{s},{mval:.6f},{name}\n")
        print(f"✓ Saved curve CSV: {csv_path}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, metrics, marker="o", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{args.task} metric")
        ax.set_title(f"{args.model} - {args.task} across checkpoints")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(args.output_dir, args.curve_png)
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"✓ Saved curve PNG: {png_path}")
        print("Done!")
        return

    if args.curve_grid:
        # Compute 1D (2D mode in class) Gaussian-direction loss curve per checkpoint, then plot a grid
        print("\nU-curve grid mode: computing per-checkpoint 1D loss curves...")
        curves = []  # list of dicts: {step, name, xs, ys}
        global_min, global_max = float('inf'), float('-inf')
        # Prepare directory to save per-checkpoint curves
        curves_npy_dir = os.path.join(args.output_dir, "curves_npy")
        os.makedirs(curves_npy_dir, exist_ok=True)
        # Prepare common x-axis
        xs = np.arange(args.x_min, args.x_max, args.x_interval)
        if xs.size == 0:
            raise SystemExit("x-axis range is empty; check --x_min/--x_max/--x_interval")
        for checkpoint_path in tqdm(checkpoint_paths, desc="Checkpoints"):

            base = os.path.basename(checkpoint_path)
            m = re.search(r"(\d+)$", base)
            step = int(m.group(1)) if m else 0

            # Load model and benchmark
            model = load_model(checkpoint_path, args.model, device)
            benchmark = setup_benchmark(args.task, model, checkpoint_path, args)

            # Landscape drawer in 2D mode for a 1D curve
            drawer = Landscape4Model(
                model,
                benchmark,
                mode="2D",
                direction="WeightNormGaussian",
                device=device,
            )
            # Use a single y-value by making the y-range contain exactly one point
            # Avoid empty mesh (y_min == y_max gives len(y) = 0 with np.arange)
            drawer.synthesize_coordinates(
                x_min=args.x_min, x_max=args.x_max, x_interval=args.x_interval,
                y_min=args.y_min, y_max=args.y_max, y_interval=args.y_interval
            )
            drawer.find_direction(seed=42)
            ys = drawer.compute_for_draw()  # 1D array

            # Track global y-limits for consistent axes across subplots
            if isinstance(ys, np.ndarray):
                cur_min, cur_max = float(np.nanmin(ys)), float(np.nanmax(ys))
                global_min = min(global_min, cur_min)
                global_max = max(global_max, cur_max)

            curves.append({"step": step, "name": base, "xs": xs.copy(), "ys": ys})
            
            # Save per-checkpoint curve arrays for reuse
            np.save(os.path.join(curves_npy_dir, f"{base}_ys.npy"), ys)
            np.save(os.path.join(curves_npy_dir, f"{base}_xs.npy"), xs)

            # cleanup
            del model
            torch.cuda.empty_cache()
  

    #     if not curves:
    #         raise SystemExit("No curves computed; check checkpoints and task setup")

    #     # Sort by step
    #     curves.sort(key=lambda d: d["step"])
    #     cols = max(1, int(args.grid_cols))
    #     rows = int(np.ceil(len(curves) / cols))
    #     fig_w, fig_h = cols * 4.0, rows * 4.0  # Increased height for better y-axis visibility
    #     fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    #     for i, c in enumerate(curves):
    #         r, col = i // cols, i % cols
    #         ax = axes[r][col]
    #         ys_relative = c["ys"] - np.min(c["ys"])
    #         min_idx = np.argmin(c["ys"])
    #         shifted_xs = c["xs"] - c["xs"][min_idx]
    #         ax.plot(shifted_xs, ys_relative, linewidth=1.5)
    #         ax.set_title(f"{c['name']}", fontsize=9)
    #         ax.set_xlabel("Perturbation (centered at min)")
    #         ax.set_ylabel(f"Relative {args.task} loss")
    #         ax.grid(True, alpha=0.25)
    #         # Set y-limits from 0 to the max relative loss for U-shape visibility
    #         if np.isfinite(ys_relative).all():
    #             ax.set_ylim(0, np.max(ys_relative))

    #     # Hide any empty axes
    #     total = rows * cols
    #     for j in range(len(curves), total):
    #         r, col = j // cols, j % cols
    #         axes[r][col].axis('off')

    #     plt.tight_layout()
    #     out_path = os.path.join(args.output_dir, args.curve_grid_png)
    #     fig.savefig(out_path, dpi=200)
    #     plt.close(fig)
    #     print(f"✓ Saved U-curve grid: {out_path}")
    #     print("Done!")
    #     return

    # # Process each checkpoint with landscape surfaces
    # results = []
    # combined_pdf_path = os.path.join(args.output_dir, args.combined_pdf)
    
    # with PdfPages(combined_pdf_path) as pdf:
    #     for checkpoint_path in tqdm(checkpoint_paths, desc="Processing checkpoints"):

    #         # Compute landscape
    #         result, mesh_x, mesh_y, checkpoint_name = compute_landscape_for_checkpoint(
    #             checkpoint_path, args.model, args.task, args, device
    #         )
            
    #         # Save individual PDF
    #         individual_pdf_path = os.path.join(args.output_dir, f"{checkpoint_name}_landscape.pdf")
    #         save_individual_pdf(mesh_x, mesh_y, result, checkpoint_name, individual_pdf_path, args.task)
            
    #         # Add to combined PDF
    #         fig = plt.figure(figsize=(10, 8))
    #         ax = fig.add_subplot(111, projection='3d')
    #         surf = ax.plot_surface(mesh_x, mesh_y, result, cmap='viridis', alpha=0.8)
    #         ax.set_xlabel('X Direction')
    #         ax.set_ylabel('Y Direction')
    #         ax.set_zlabel(f'{args.task} Loss')
    #         ax.set_title(f'Loss Landscape - {checkpoint_name}')
    #         fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    #         plt.tight_layout()
    #         pdf.savefig(fig, dpi=150, bbox_inches='tight')
    #         plt.close(fig)
            
    #         # Optionally save numpy data
    #         if args.save_numpy:
    #             npy_path = os.path.join(args.output_dir, f"{checkpoint_name}_landscape.npy")
    #             np.save(npy_path, result)
            
    #         results.append({
    #             'checkpoint': checkpoint_name,
    #             'path': checkpoint_path,
    #             'shape': result.shape,
    #             'min': float(result.min()),
    #             'max': float(result.max()),
    #             'mean': float(result.mean())
    #         })

    
    # print(f"\n✓ Combined PDF saved: {combined_pdf_path}")
    # print(f"✓ Individual PDFs saved in: {args.output_dir}")
    
    # Save summary
    # summary_path = os.path.join(args.output_dir, "summary.txt")
    # with open(summary_path, 'w') as f:
    #     f.write("Loss Landscape Evolution Summary\n")
    #     f.write("="*50 + "\n\n")
    #     f.write(f"Task: {args.task}\n")
    #     f.write(f"Model: {args.model}\n")
    #     f.write(f"Checkpoints processed: {len(results)}\n\n")
    #     for r in results:
    #         f.write(f"\n{r['checkpoint']}:\n")
    #         f.write(f"  Min loss: {r['min']:.6f}\n")
    #         f.write(f"  Max loss: {r['max']:.6f}\n")
    #         f.write(f"  Mean loss: {r['mean']:.6f}\n")
    
    # print(f"✓ Summary saved: {summary_path}")
    # print("\nDone!")


if __name__ == "__main__":
    main()
