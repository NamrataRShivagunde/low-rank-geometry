# Evaluation (lm-evaluation-harness)

This directory contains config-driven downstream evaluation for checkpoints using EleutherAI lm-evaluation-harness.


## What this supports

- Evaluate one checkpoint, many checkpoints, or whole checkpoint directories (`model_*`).
- Tasks are defined in YAML config.
- Multi-seed evaluation (seed list set in config; shipped config uses 5 seeds).
- Optional bootstrap uncertainty via `bootstrap_iters` (useful when you have one model per method).
- Outputs include:
  - raw lm-eval JSON per checkpoint per seed
  - per-example analysis JSONL with prompt, prediction, and target
  - long-format CSV (`checkpoint, seed, task, metric, value`)
  - aggregate CSV (`mean/std/min/max` across seeds)

## Install

```bash
pip install -r evaluation/requirements.txt
```

## Config

Default config: `evaluation/configs/full_eval_all_tasks.yaml`

Tasks currently configured to evaluate on the tasks in the paper.

Default seeds:
- 42, 43, 44, 45, 46

Bootstrap uncertainty:
- Set `evaluation.bootstrap_iters` in config (e.g. `1000`) to request bootstrap-based stderr/CI from lm-eval.
- You can combine this with `seeds: [42]` if you want one deterministic run plus bootstrap uncertainty.

## Dry run (no evaluation)

```bash
python evaluation/run_lm_eval.py --config evaluation/configs/full_eval_all_tasks.yaml --dry-run

# Optional: override tokenizer from CLI
python evaluation/run_lm_eval.py --config evaluation/configs/full_eval_all_tasks.yaml --tokenizer t5-base --dry-run
```

## Run with config-defined checkpoints

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_lm_eval.py --config evaluation/configs/full_eval_all_tasks.yaml
```

For multi-GPU runs over the final checkpoints of each method, use the launchers
`evaluation/eval_60m.sh`, `evaluation/eval_130m.sh`, `evaluation/eval_350m.sh`.

## Run with explicit checkpoints/directories

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_lm_eval.py \
  --config evaluation/configs/full_eval_all_tasks.yaml \
  --checkpoints-dir /path/to/CHECKPOINTS/loss-landscape-checkpoints/llama_60m-2025-12-08-21-00-08 \
  --checkpoints-dir /path/to/CHECKPOINTS/loss-landscape-checkpoints/cola_60m-2025-12-08-20-59-46
```

A single checkpoint can be passed with `--checkpoint` (repeatable) instead of `--checkpoints-dir`.

## Output structure

Each run writes to:

- `evaluation/output/eval_YYYYMMDD_HHMMSS/resolved_plan.json`
- `evaluation/output/eval_YYYYMMDD_HHMMSS/raw/<checkpoint_name>/seed_<seed>.json`
- `evaluation/output/eval_YYYYMMDD_HHMMSS/raw/<checkpoint_name>/seed_<seed>_samples.jsonl`
- `evaluation/output/eval_YYYYMMDD_HHMMSS/results_long.csv`
- `evaluation/output/eval_YYYYMMDD_HHMMSS/results_aggregate.csv`
- `evaluation/output/eval_YYYYMMDD_HHMMSS/failures.json`
- `evaluation/output/eval_YYYYMMDD_HHMMSS/summary.json`
