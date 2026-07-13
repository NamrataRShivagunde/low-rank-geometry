
# low-rank-geometry

This repository accompanies our paper **"Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training"** (https://arxiv.org/abs/2605.13652).

We study the optimization landscape of low-rank pre-training methods for LLMs — including GaLore, LoRA, CoLA, SLTrain, FiRA, and ReLoRA — and compare them against full-rank training through geometric and spectral diagnostics.

The repository is organized into three stages, each with its own README:

- **Training** (`training/`) — unified launcher for all low-rank pre-training methods (documented below).
- **Analysis** (`loss-landscape/`) — geometric & spectral metrics on the resulting checkpoints. See [`README_METRIC.md`](README_METRIC.md).
- **Evaluation** (`evaluation/`) — config-driven downstream evaluation via lm-evaluation-harness. See [`README_EVAL.md`](README_EVAL.md).


# Unified Training Integration
This folder now contains a single launcher that wraps all low-rank pre-training methods while keeping method code in their own repos.


## What is implemented

- Common entrypoint: `training/torchrun_main_common.py`
- Repo bootstrap helper: `training/setup_methods.py`
- Supported methods:
  - Full rank baseline: `fullrank`
  - Low-rank: `cola`, `galore`, `relora`, `fira`, `sltrain`
- Standard checkpoint naming and location:
  - Root directory: `checkpoints/`

## 1) Clone method repositories

Clone all method repos into `training/`:

```bash
python training/setup_methods.py
```

Or clone specific methods:

```bash
python training/setup_methods.py --methods relora switchlora fira sltrain
```

## 1.5) Install training dependencies

Use the unified requirements file for all training scripts:

```bash
pip install -r training/requirements.txt

cd ./training/SLTrain/sparse-lora
pip install --no-build-isolation .
```

## 2) Run training from one common entrypoint

All the bash scripts are given in `training/training_bash_scripts`.

For every run, the launcher writes checkpoints into a unique directory under the `save_dir` path.

Note: the code supports training with an offline-mode dataset. For this, download and save the dataset at a path and point the training to it.

## Citation
```bibtex
@article{shivagunde2026beyond,
  title={Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training},
  author={Shivagunde, Namrata and Deshpande, Vijeta and Muckatira, Sherin and Rumshisky, Anna},
  journal={arXiv preprint arXiv:2605.13652},
  year={2026}
}
```
