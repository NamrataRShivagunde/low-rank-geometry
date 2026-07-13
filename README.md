
# low-rank-geometry

This repository accompanies our paper **"Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training"** (https://arxiv.org/abs/2605.13652).

We study the optimization landscape of low-rank pre-training methods for LLMs — including GaLore, LoRA, CoLA, SLTrain, FiRA, and ReLoRA — and compare them against full-rank training through geometric and spectral diagnostics.


# Unified Training Integration
This folder now contains a single launcher that wraps all low-rank pre-training methods while keeping method code in their own repos.


## What is implemented

- Common entrypoint: `training/torchrun_main_common.py`
- Repo bootstrap helper: `training/setup_methods.py`
- Supported methods:
  - Full rank baseline: `fullrank`
  - Low-rank: `cola`, `galore`, `relora`, `fira`, `sltrain`
- Standard checkpoint naming and location:
  - Root directory: `CHECKPOINTS/`
  - Run directory format: `method_modelsize-timestamp`
  - Example: `cola_60m-2026-02-27-12-00-00`

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

### 1-step smoke test (60M)

Full-rank baseline:

```bash
python training/main.py --method fullrank --model-size 60m --smoke-test --monitor-memory
```

CoLA:

```bash
python training/main.py --method cola --model-size 60m --smoke-test --monitor-memory
```

GaLore:

```bash
python training/main.py --method galore --model-size 60m --smoke-test --monitor-memory
```

ReLoRA:

```bash
python training/main.py --method relora --model-size 60m --smoke-test --monitor-memory
```

SwitchLoRA (requires preprocessed dataset path):

```bash
python training/main.py --method switchlora --model-size 60m --dataset-path /path/to/preprocessed_c4 --smoke-test --monitor-memory
```

Fira:

```bash
python training/main.py --method fira --model-size 60m --smoke-test --monitor-memory
```

SLTrain:

```bash
python training/main.py --method sltrain --model-size 60m --smoke-test --monitor-memory
```

### Regular run (example)

```bash
python training/main.py \
  --method galore \
  --model-size 60m \
  --steps 10000 \
  --warmup-steps 1000 \
  --batch-size 128 \
  --total-batch-size 512 \
  --nproc-per-node 1 \
  --cuda-visible-devices 0 \
  --monitor-memory
```

## Outputs

For every run, launcher writes into a unique directory under `CHECKPOINTS/`:

- Model checkpoints from underlying method script
- `training_summary.json` containing:
  - method/model size
  - exact command
  - elapsed time
  - approximate throughput (`approx_steps_per_second`)
  - sampled peak GPU memory (`peak_memory_mb`, when enabled)
  - exit code

## Notes

- Additional method-specific flags can be appended with repeated `--extra-arg`.
- Use `--dry-run` to print the final command without launching training.
- configs - in sltrain original configs, the models were trained on 10% extra steps


## Citation
```bibtex
@article{shivagunde2026beyond,
  title={Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training},
  author={Shivagunde, Namrata and Deshpande, Vijeta and Muckatira, Sherin and Rumshisky, Anna},
  journal={arXiv preprint arXiv:2605.13652},
  year={2026}
}
```
