# Loss Landscape & Geometric/Spectral Metrics

This directory is the **analysis** stage of the repo (companion to `training/` and
`evaluation/`). It takes the checkpoints produced by training and characterizes how each
pre-training method — `fullrank`, `galore`, `fira`, `cola`, `sltrain`, `relora` — shapes the
model's geometry over the course of training.

It is the code behind the metrics in the paper
**"Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training"**
(https://arxiv.org/abs/2605.13652). The core finding: low-rank methods are not fully equivalent
to full-rank training — nor to each other — even when validation perplexity is close, so we
describe the differences with 16 geometric/spectral diagnostics across four dimensions:

1. **Loss-landscape geometry** — along random directions and top-K PCA directions of the trajectory
2. **Checkpoint interpolation** — loss barriers between consecutive checkpoints and across methods
3. **Weight / update spectra** — rank, effective/stable rank, singular spectra of `W` and `ΔW`
4. **Activation similarity** — how closely a method's activations track the full-rank baseline

---

## Directory layout

```
loss-landscape/
├── LLMLandscape/                  # vendored loss-landscape engine (do NOT edit; code taken from https://arxiv.org/abs/2505.17646)
│   ├── utils/plot/                # Landscape4Model, Landscape4ModelPCA (random & PCA grids)
│   └── exps/landscape/most/       # landscape_eval_utils.py: load_model_from_args, get_c4_dataloader, compute_nll_loss
├── configs/
│   └── metrics.yaml               # default grid/params for landscape metrics
└── scripts/
    ├── single_checkpoint_scripts/ # one metric, one checkpoint 
    ├── multi_checkpoint_scripts/  # same metric swept across training steps / per model size
    ├── bash_scripts/              # ready-to-run launchers (canonical invocations + flags)
    └── supporting_scripts/        # helpers, e.g. list_checkpoints.py
```

- **`single_checkpoint_scripts/`** — the core metric implementations. Each is argparse-driven
  and computes one metric for one checkpoint. 
- **`multi_checkpoint_scripts/`** — thin wrappers that run a single-checkpoint metric across the
  full step trajectory, split per model size (`*_60m.py`, `*_130m.py`, `*_350m.py`), plus
  `*_overlay.py` plotters that combine methods onto one figure and `*_trend` CSV emitters.
- **`bash_scripts/`** — the command lines (paths, flags, CUDA device, checkpoint
  step lists). **Start here to see how a metric is invoked**, then adapt paths.

---

## The metrics

All live in `scripts/single_checkpoint_scripts/`

| # | Script | Metric | Dimension |
|---|--------|--------|-----------|
| 1 | `1_landscape_2d_lowrank_number_of_dir.py` | Loss landscape along N random directions | Landscape |
| 3 | `3_landscape_2d_pca_topk_components.py` | Loss landscape along top-K PCA directions of the trajectory | Landscape |
| 4 | `4_expected_sharpness.py` | Expected sharpness (and variance) from a 1D landscape `.npy` | Landscape |
| 5 | `5_ranks.py` | Rank / spectral metrics (stable rank, effective rank) on weight matrices | Spectra |
| 5b | `5b_ranks_init_to_final.py` | Singular spectrum of `ΔW = W_final − W_warmstart`, grouped by projection (ReLoRA-style) | Spectra |
| 5 | `5_ranks_updates.py` | Update magnitude + update rank between consecutive checkpoints | Spectra |
| 6 | `6_activations_comparison.py` | Activation L2 / cosine vs. the full-rank baseline | Activations |
| 7 | `7_1d_interpolation.py` | Loss interpolation / barrier between consecutive checkpoints | Interpolation |
| 9 | `9_1d_interpolation_cross_method.py` | Cross-method interpolation barrier (IMIB) at the same step | Interpolation |

Overlay/trend plotters for each metric live under
`multi_checkpoint_scripts/` with the same numeric prefix (e.g. `5_ranks_overlay.py`).

---

## Quick start

Install analysis dependencies and activate the environment:

```bash
pip install -r loss-landscape/requirements.txt
conda activate <VIRTUAL-ENV-NAME>
```

List the checkpoints available for a model size:


### Single checkpoint — 2D random-direction landscape

```bash
CUDA_VISIBLE_DEVICES=0 python \
  loss-landscape/scripts/single_checkpoint_scripts/1_landscape_2d_lowrank_number_of_dir.py \
  --model llama \
  --checkpoint CHECKPOINTS/models-60m/llama-60m/model_10000 \
  --task c4-val --tokenizer t5-base \
  --num_directions 1 \
  --c4_max_examples 1000 --c4_batch_size 128 --c4_max_length 256 \
  --x_min -0.003 --x_max 0.0031 --x_interval 0.00025 \
  --y_min -0.003 --y_max 0.003 --y_interval 0.00025 \
  --output_dir results/loss-landscape-2d/models-60m/llama-60m/model_10000
```

### Single checkpoint — rank / spectral metrics

```bash
CUDA_VISIBLE_DEVICES=0 python \
  loss-landscape/scripts/single_checkpoint_scripts/5_ranks.py \
  --checkpoint CHECKPOINTS/models-60m/llama-60m/model_10000 \
  --svd_device cuda --seed 42 \
  --output_dir results/rank-metrics/models-60m/llama-60m/model_10000
```

### Whole trajectory — sweep one metric across all steps for a size

```bash
CUDA_VISIBLE_DEVICES=0 python \
  loss-landscape/scripts/multi_checkpoint_scripts/5_ranks_multi_chkpt_60m.py \
  --checkpoint_root CHECKPOINTS/models-60m \
  --output_root results/rank-metrics/models-60m \
  --svd_device cuda --seed 42 \
  --only_models llama cola fira sltrain relora galore \
  --only_checkpoints model_1000 model_2000 model_3000 model_4000 model_5000 \
                     model_6000 model_7000 model_8000 model_9000 model_10000
```

The `bash_scripts/*.sh` launchers contain the canonical, copy-pasteable commands (including the
per-size checkpoint step lists) for every metric.

### Checkpoint step conventions ("all checkpoints")

| Size | Steps |
|------|-------|
| 60M  | `model_1000, model_2000, …, model_10000` |
| 130M | `model_2000, model_4000, …, model_20000` |
| 350M | `model_6000, model_12000, …, model_60000` |

Methods: `fullrank` (dir name `llama`), `galore`, `fira`, `cola`, `sltrain`, `relora`.

---

## How the scripts find the model

Each script auto-locates `LLMLandscape/` and the repo root by walking up from its own path, then
loads checkpoints through `exps/landscape/most/landscape_eval_utils.py`:

- `load_model_from_args(...)` — method-aware loading (handles CoLA custom registration and
  ReLoRA/SLTrain materialization so all methods load into a comparable dense form)
- `get_c4_dataloader(...)` — fixed C4 validation subset for loss/activation evaluation
- `compute_nll_loss(...)` — the shared NLL loss used by landscape/interpolation scripts

Because loading is centralized there, checkpoint-format quirks are handled in one place.

---

## Output conventions

Follow these when adding a metric

- **Per-checkpoint summaries** → `.json` and/or `.csv`
- **Trends across steps** → `.csv`
- **Plots** → `.png` and `.pdf` # pdf are good for papers
- **Save under** `results/<metric_name>/<model_size>/<method>/…`
- **`.npy` is reserved for loss-landscape grid data** (1D/2D loss grids, direction vectors).
  Scalar/tabular metrics (rank, stable/effective rank, sharpness, norms, …) must use CSV/JSON, and not `.npy`.

Example: `4_expected_sharpness.py` consumes a landscape `.npy` produced by script 1/3 and produces
sharpness as CSV/JSON — landscape geometry is cached as `.npy`, derived scalars are tabular.

