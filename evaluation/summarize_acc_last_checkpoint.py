from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TASKS = [
    "piqa",
    "copa",
    "mrpc",
    "rte",
    "mnli",
    "arc_easy",
    "blimp",
    "sst2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize ACC mean/std over seeds for a target checkpoint step "
            "from evaluation/output run files, aggregating subtasks when present."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Evaluation run directory containing results_long.csv (e.g. evaluation/output/eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=10000,
        help="Checkpoint step to summarize (default: 10000)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task families to summarize (default: piqa copa mrpc rte mnli arc_easy blimp sst2)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: <run-dir>/acc_seed_summary_model_<step>.csv)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path (default: <run-dir>/acc_seed_summary_model_<step>.json)",
    )
    return parser.parse_args()


def _normalize_task_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _is_acc_metric(metric: str) -> bool:
    metric_l = metric.strip().lower()
    return metric_l.startswith("acc,")


def _is_target_checkpoint(checkpoint_path: str, checkpoint_name: str, step: int) -> bool:
    marker = f"model_{step}"
    return checkpoint_path.endswith("/" + marker) or checkpoint_name.endswith("__" + marker)


def _is_subtask(task_name: str, task_family: str) -> bool:
    # subtask examples: blimp_adjunct_island, mnli_matched, etc.
    return task_name.startswith(task_family + "_")


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_long = run_dir / "results_long.csv"
    if not results_long.exists():
        raise FileNotFoundError(f"results_long.csv not found: {results_long}")

    target_tasks = [_normalize_task_name(task) for task in args.tasks]
    step = int(args.checkpoint_step)

    # Collect per (checkpoint_name, seed, task_family) score.
    # For each task family and seed:
    #   - if subtasks exist, use mean(subtask acc)
    #   - else use mean(exact task acc)
    per_checkpoint_seed_task: dict[tuple[str, int, str], float] = {}

    # Temporary storage keyed by (checkpoint_name, seed, task_family)
    exact_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    subtask_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)

    with results_long.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            checkpoint = str(row.get("checkpoint", ""))
            checkpoint_name = str(row.get("checkpoint_name", ""))
            if not _is_target_checkpoint(checkpoint, checkpoint_name, step):
                continue

            metric = str(row.get("metric", ""))
            if not _is_acc_metric(metric):
                continue

            task_name = _normalize_task_name(str(row.get("task", "")))
            seed_raw = row.get("seed")
            value_raw = row.get("value")

            if seed_raw is None or value_raw is None:
                continue

            try:
                seed = int(seed_raw)
                value = float(value_raw)
            except (ValueError, TypeError):
                continue

            for task_family in target_tasks:
                key = (checkpoint_name, seed, task_family)
                if task_name == task_family:
                    exact_values[key].append(value)
                elif _is_subtask(task_name, task_family):
                    subtask_values[key].append(value)

    all_keys = set(exact_values.keys()) | set(subtask_values.keys())
    for key in all_keys:
        sub_values = subtask_values.get(key, [])
        if sub_values:
            per_checkpoint_seed_task[key] = _safe_mean(sub_values)
            continue

        base_values = exact_values.get(key, [])
        if base_values:
            per_checkpoint_seed_task[key] = _safe_mean(base_values)

    # Aggregate over seeds
    by_checkpoint_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (checkpoint_name, seed, task_family), score in per_checkpoint_seed_task.items():
        _ = seed
        by_checkpoint_task[(checkpoint_name, task_family)].append(score)

    rows: list[dict[str, Any]] = []
    for checkpoint_name, task_family in sorted(by_checkpoint_task.keys()):
        seed_scores = by_checkpoint_task[(checkpoint_name, task_family)]
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_step": step,
                "task": task_family,
                "num_seeds": len(seed_scores),
                "acc_mean": _safe_mean(seed_scores),
                "acc_std": _safe_std(seed_scores),
            }
        )

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else run_dir / f"acc_seed_summary_model_{step}.csv"
    )
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else run_dir / f"acc_seed_summary_model_{step}.json"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["checkpoint_name", "checkpoint_step", "task", "num_seeds", "acc_mean", "acc_std"],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2))

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint_step": step,
                "tasks": target_tasks,
                "num_rows": len(rows),
                "output_csv": str(output_csv),
                "output_json": str(output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
