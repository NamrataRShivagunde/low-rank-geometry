from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Ensure local lm-evaluation-harness is used instead of pip version
_EVAL_DIR = Path(__file__).resolve().parent
_LOCAL_LM_EVAL = _EVAL_DIR / "lm-evaluation-harness"

# Add to sys.path at the front to take priority
if _LOCAL_LM_EVAL.exists():
    _local_path = str(_LOCAL_LM_EVAL)
    if _local_path not in sys.path:
        sys.path.insert(0, _local_path)
    # Also update PYTHONPATH env var for subprocesses
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if _local_path not in current_pythonpath:
        os.environ["PYTHONPATH"] = f"{_local_path}:{current_pythonpath}" if current_pythonpath else _local_path


def patch_transformers_dtype_alias() -> None:
    try:
        import transformers
        from transformers import PreTrainedModel
    except Exception:
        return

    if getattr(PreTrainedModel, "_copilot_dtype_alias_patch", False):
        return

    original = PreTrainedModel.from_pretrained

    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")

        # Some lm-eval/transformers combinations pass gguf_file=None even when
        # the model __init__ does not accept it; strip it in that case.
        if "gguf_file" in kwargs:
            try:
                init_sig = inspect.signature(cls.__init__)
                init_params = init_sig.parameters
                accepts_var_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in init_params.values()
                )
                if (not accepts_var_kwargs) and ("gguf_file" not in init_params):
                    kwargs.pop("gguf_file", None)
            except Exception:
                pass

        return original.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    PreTrainedModel.from_pretrained = classmethod(_patched_from_pretrained)
    setattr(PreTrainedModel, "_copilot_dtype_alias_patch", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness for one or many checkpoints")
    parser.add_argument("--config", default="evaluation/configs/default_eval.yaml", help="YAML config path")
    parser.add_argument("--checkpoint", action="append", default=[], help="Single checkpoint path; repeatable")
    parser.add_argument("--checkpoints-dir", action="append", default=[], help="Directory containing model_* checkpoints; repeatable")
    parser.add_argument("--output-root", default=None, help="Override output root directory")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer override (HF repo ID or local path)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print run plan without executing")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")
    return loaded


_TASK_ALIASES = {
    "copa": "copa",
    "mrpc": "mrpc",
    "rte": "rte",
    "mnli": "mnli",
    "blimp": "blimp",
    "piqa": "piqa",
    "sst2": "sst2",
    "sst-2": "sst2",
    "qqp": "qqp",
    "qnli": "qnli",
    "boolq": "boolq",
    "multirc": "multirc",
    "multi-rc": "multirc",
    "wsc": "wsc",
    "cola": "cola",
    "arc-easy": "arc_easy",
    "arc_easy": "arc_easy",
    "arc easy": "arc_easy",
    "arc challenge": "arc_challenge",
    "arc-challenge": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "ifeval": "ifeval",
    "if-eval": "ifeval",
    "leaderboard_instruction_following": "leaderboard_instruction_following",
    "leaderboard-instruction-following": "leaderboard_instruction_following",
    "truthfulqa_mc1": "truthfulqa_mc1",
    "truthfulqa-mc1": "truthfulqa_mc1",
    "truthfulqa_mc2": "truthfulqa_mc2",
    "truthfulqa-mc2": "truthfulqa_mc2",
    "toxigen": "toxigen",
    "gsm8k": "gsm8k",
}


def normalize_task_name(task: str) -> str:
    normalized = task.strip().lower().replace("_", "-")
    return _TASK_ALIASES.get(normalized, normalized.replace("-", "_"))


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return int(digits), name
    return 10**18, name


def resolve_checkpoints(raw_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if not candidate.exists():
            continue

        if candidate.is_dir():
            model_dirs = sorted([p for p in candidate.glob("model_*") if p.is_dir()], key=checkpoint_sort_key)
            if model_dirs:
                resolved.extend(model_dirs)
                continue

            if (candidate / "config.json").exists():
                resolved.append(candidate)
                continue

        if candidate.is_file() and candidate.name == "config.json":
            resolved.append(candidate.parent)

    unique = sorted(set(resolved), key=lambda p: str(p))
    return unique


def build_model_args(pretrained_path: Path, extra: dict[str, Any]) -> str:
    merged = {"pretrained": str(pretrained_path)}
    merged.update(extra)

    entries: list[str] = []
    for key, value in merged.items():
        if isinstance(value, bool):
            value_text = "True" if value else "False"
        else:
            value_text = str(value)
        entries.append(f"{key}={value_text}")
    return ",".join(entries)


def has_local_tokenizer_files(checkpoint: Path) -> bool:
    candidates = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
    return any((checkpoint / name).exists() for name in candidates)


def checkpoint_label(path: Path) -> str:
    parent = path.parent.name
    return f"{parent}__{path.name}"


def infer_method_from_checkpoint(checkpoint: Path) -> str:
    base = checkpoint.parent.name.strip().lower()
    if "-" in base:
        return base.split("-", 1)[0]
    if "_" in base:
        return base.split("_", 1)[0]
    return base


def infer_size_from_checkpoint(checkpoint: Path) -> str:
    candidates = [checkpoint.parent.name, checkpoint.parent.parent.name]
    for candidate in candidates:
        text = candidate.strip().lower()
        match = re.search(r"(\d+[a-z])", text)
        if match:
            return match.group(1)
    return "unknown"


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(text).strip())
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "run"


def build_run_dir_name(
    tasks: list[str],
    checkpoints: list[Path],
    run_name: str | None,
) -> str:
    methods = sorted({infer_method_from_checkpoint(ckpt) for ckpt in checkpoints})
    method_tag = slugify("-".join(methods))
    task_tag = f"{len(tasks)}tasks"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = ["eval"]
    if run_name:
        parts.append(slugify(run_name))
    parts.extend([method_tag, task_tag, timestamp])
    return "__".join(parts)


def register_custom_model_types() -> None:
    # Register CoLA custom model so lm-eval hf backend can instantiate it
    # directly from checkpoints with model_type=cola.
    #
    # Note: ReLoRA/SLTrain are adapter-style checkpoints in this repo and are
    # materialized into dense weights before eval. They are not exposed as
    # Transformers PretrainedConfig + PreTrainedModel pairs with a custom
    # model_type like CoLA, so AutoConfig/AutoModel registration is not used
    # for them here.
    try:
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from transformers import AutoConfig, AutoModelForCausalLM
        from training.CoLA.cola import ColaConfig, ColaForCausalLM
    except Exception:
        return

    try:
        AutoConfig.register("cola", ColaConfig)
    except Exception:
        pass
    try:
        AutoModelForCausalLM.register(ColaConfig, ColaForCausalLM)
    except Exception:
        pass


def resolve_model_backend(method: str, default_model: str, overrides: dict[str, Any]) -> str:
    method_key = str(method).strip().lower()
    override = overrides.get(method_key)
    if override is not None:
        return str(override)

    # Map methods to their lm-eval model backends.
    # CoLA uses a custom CoLA model class that handles config rewriting.
    # Others use the default HF backend.
    method_default_map = {
        "llama": default_model,
        "galore": default_model,
        "fira": default_model,
        "cola": "cola",  # Use CoLA model class from local lm-eval
        "relora": default_model,
        "sltrain": default_model,
        "switchlora": default_model,
    }
    return str(method_default_map.get(method_key, default_model))


def materialize_relora_checkpoint(checkpoint: Path) -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    from safetensors.torch import load_file, save_file

    model_path = checkpoint / "model.safetensors"
    relora_cfg_path = checkpoint / "relora_config.json"

    if not model_path.exists() or not relora_cfg_path.exists():
        temp_dir = tempfile.TemporaryDirectory(prefix="lm_eval_ckpt_relora_missing_")
        temp_path = Path(temp_dir.name)
        for item in checkpoint.iterdir():
            destination = temp_path / item.name
            if item.is_file():
                shutil.copy2(item, destination)
            elif item.is_dir():
                shutil.copytree(item, destination)
        return temp_path, temp_dir, {"rewritten": False, "materialized": False, "reason": "missing relora config"}

    relora_cfg = json.loads(relora_cfg_path.read_text())
    rank = float(relora_cfg.get("r", 128))
    lora_alpha = float(relora_cfg.get("lora_alpha", 32))
    default_scale = (lora_alpha / rank) if rank > 0 else 1.0

    state = load_file(str(model_path), device="cpu")
    prefixes: set[str] = set()

    # keep a note on where lora is applied
    for key in state:
        if key.endswith(".lora_A.weight"):
            prefixes.add(key[: -len(".lora_A.weight")])

    new_state: dict[str, Any] = {}
    consumed: set[str] = set()

    for prefix in sorted(prefixes):
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"
        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.scaling"
        if a_key not in state or b_key not in state:
            continue

        a = state[a_key]
        b = state[b_key]
        scale = default_scale
        # if s_key in state:
        #     scale = float(state[s_key].reshape(-1)[0].tanh().item())

        delta = (b @ a) * scale
        if w_key in state:
            merged_weight = state[w_key] + delta.to(dtype=state[w_key].dtype)
        else:
            merged_weight = delta
        new_state[w_key] = merged_weight
        consumed.update({a_key, b_key, s_key, w_key})

    for key, value in state.items():
        if key in consumed:
            continue
        if ".lora_A.weight" in key or ".lora_B.weight" in key or key.endswith(".scaling"):
            continue
        new_state[key] = value

    temp_dir = tempfile.TemporaryDirectory(prefix="lm_eval_ckpt_relora_")
    temp_path = Path(temp_dir.name)
    for item in checkpoint.iterdir():
        if item.name == "model.safetensors":
            continue
        destination = temp_path / item.name
        if item.is_file():
            shutil.copy2(item, destination)
        elif item.is_dir():
            shutil.copytree(item, destination)
    save_file(new_state, str(temp_path / "model.safetensors"), metadata={"format": "pt"})

    return temp_path, temp_dir, {"rewritten": True, "materialized": True, "method": "relora"}


def materialize_sltrain_checkpoint(checkpoint: Path) -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    from safetensors.torch import load_file, save_file
    import torch

    model_path = checkpoint / "model.safetensors"
    splora_cfg_path = checkpoint / "splora_config.json"

    if not model_path.exists() or not splora_cfg_path.exists():
        temp_dir = tempfile.TemporaryDirectory(prefix="lm_eval_ckpt_sltrain_missing_")
        temp_path = Path(temp_dir.name)
        for item in checkpoint.iterdir():
            destination = temp_path / item.name
            if item.is_file():
                shutil.copy2(item, destination)
            elif item.is_dir():
                shutil.copytree(item, destination)
        return temp_path, temp_dir, {"rewritten": False, "materialized": False, "reason": "missing splora config"}

    splora_cfg = json.loads(splora_cfg_path.read_text())
    rank = float(splora_cfg.get("r", 128))
    lora_alpha = float(splora_cfg.get("lora_alpha", 32))
    default_scale = (lora_alpha / rank) if rank > 0 else 1.0

    state = load_file(str(model_path), device="cpu")
    prefixes: set[str] = set()
    for key in state:
        if key.endswith(".lora_A"):
            prefixes.add(key[: -len(".lora_A")])

    new_state: dict[str, Any] = {}
    consumed: set[str] = set()

    for prefix in sorted(prefixes):
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        i_key = f"{prefix}.sparse_index"
        v_key = f"{prefix}.sparse_value"
        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.scaling"
        if a_key not in state or b_key not in state:
            continue

        a = state[a_key]
        b = state[b_key]
        scale = default_scale
        if s_key in state:
            scale = float(state[s_key].reshape(-1)[0].tanh().item())

        merged_weight = (b @ a) * scale
        if i_key in state and v_key in state:
            sparse_index = state[i_key].long()
            sparse_value = state[v_key].to(dtype=merged_weight.dtype)
            sparse_weight = torch.zeros_like(merged_weight).reshape(-1)
            sparse_weight[sparse_index] = sparse_value
            merged_weight = merged_weight + sparse_weight.view_as(merged_weight)

        if w_key in state:
            merged_weight = state[w_key] + merged_weight.to(dtype=state[w_key].dtype)

        new_state[w_key] = merged_weight
        consumed.update({a_key, b_key, i_key, v_key, s_key, w_key})

    for key, value in state.items():
        if key in consumed:
            continue
        if ".lora_A" in key or ".lora_B" in key or ".sparse_index" in key or ".sparse_value" in key:
            continue
        if key.endswith(".scaling"):
            continue
        new_state[key] = value

    temp_dir = tempfile.TemporaryDirectory(prefix="lm_eval_ckpt_sltrain_")
    temp_path = Path(temp_dir.name)
    for item in checkpoint.iterdir():
        if item.name == "model.safetensors":
            continue
        destination = temp_path / item.name
        if item.is_file():
            shutil.copy2(item, destination)
        elif item.is_dir():
            shutil.copytree(item, destination)
    save_file(new_state, str(temp_path / "model.safetensors"), metadata={"format": "pt"})

    return temp_path, temp_dir, {"rewritten": True, "materialized": True, "method": "sltrain"}


def prepare_checkpoint_for_lm_eval(
    checkpoint: Path,
    method: str,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None, dict[str, Any]]:
    """Prepare a checkpoint for lm-eval based on training method.
    
    - ReLoRA/SLTrain: Materialize adapter weights → dense weights in temp dir
    - CoLA: Return as-is (custom model type, registered via register_custom_model_types)
    - Others (LLaMA, GaLore, Fira, etc.): Return as-is
    
    Returns:
        (prepared_checkpoint_path, temp_dir_or_none, metadata_dict)
    """
    if method == "relora":
        prepared, tmpdir, meta = materialize_relora_checkpoint(checkpoint)
        meta.setdefault("method", method)
        return prepared, tmpdir, meta

    if method == "sltrain":
        prepared, tmpdir, meta = materialize_sltrain_checkpoint(checkpoint)
        meta.setdefault("method", method)
        return prepared, tmpdir, meta

    if method == "cola":
        # CoLA: Return as-is. The CoLALM model class (from local lm-eval)
        # handles config rewriting internally when loading.
        return checkpoint, None, {
            "rewritten": False,
            "method": method,
            "materialized": False,
            "note": "CoLA checkpoint; CoLALM model class handles config rewriting",
        }

    # Default: full-rank models (llama, galore, fira, switchlora, etc.)
    # These load directly via HuggingFace transformers without modification.
    return checkpoint, None, {
        "rewritten": False,
        "method": method,
        "materialized": False,
        "note": f"Standard {method} checkpoint; no preparation needed",
    }


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    return None


def flatten_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_to_metrics = results.get("results")
    if not isinstance(task_to_metrics, dict):
        return rows

    for task_name, metrics in task_to_metrics.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            numeric = safe_float(metric_value)
            if numeric is None:
                continue
            rows.append(
                {
                    "task": str(task_name),
                    "metric": str(metric_name),
                    "value": numeric,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["checkpoint_name"]), str(row["task"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))

    output: list[dict[str, Any]] = []
    for (checkpoint_name, task, metric), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
        else:
            std = 0.0
        output.append(
            {
                "checkpoint_name": checkpoint_name,
                "task": task,
                "metric": metric,
                "num_seeds": len(values),
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def _sample_prompts(arguments: Any) -> list[str]:
    prompts: list[str] = []
    if isinstance(arguments, dict):
        for key in sorted(arguments):
            request = arguments[key]
            if isinstance(request, dict) and "arg_0" in request:
                prompts.append(str(request["arg_0"]))
    return prompts


def _sample_prediction(sample: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    filtered_resps = sample.get("filtered_resps")
    arguments = sample.get("arguments")

    if isinstance(filtered_resps, list):
        if filtered_resps and all(isinstance(item, str) for item in filtered_resps):
            return filtered_resps[0], {"source": "filtered_resps"}

        if filtered_resps and all(isinstance(item, list) for item in filtered_resps):
            choice_scores: list[float | None] = []
            choice_texts: list[str | None] = []

            if isinstance(arguments, dict):
                for key in sorted(arguments):
                    request = arguments[key]
                    if isinstance(request, dict):
                        choice_texts.append(
                            None if "arg_1" not in request else str(request["arg_1"])
                        )

            for item in filtered_resps:
                if not item:
                    choice_scores.append(None)
                    continue
                try:
                    choice_scores.append(float(item[0]))
                except Exception:
                    choice_scores.append(None)

            valid_scores = [score for score in choice_scores if score is not None]
            if valid_scores:
                best_index = max(
                    range(len(choice_scores)),
                    key=lambda index: choice_scores[index]
                    if choice_scores[index] is not None
                    else float("-inf"),
                )
                prediction = choice_texts[best_index] if best_index < len(choice_texts) else None
                if prediction is None:
                    prediction = best_index
                return prediction, {
                    "source": "filtered_resps",
                    "choice_index": best_index,
                    "choice_scores": choice_scores,
                }

    resps = sample.get("resps")
    if isinstance(resps, list) and resps:
        first_resp = resps[0]
        if isinstance(first_resp, list) and first_resp:
            return first_resp[0], {"source": "resps"}
        return first_resp, {"source": "resps"}

    return None, {"source": "unknown"}


def flatten_sample_records(
    samples: Any,
    *,
    checkpoint: str,
    checkpoint_name: str,
    method: str,
    size: str,
    seed: int,
    task: str,
) -> list[dict[str, Any]]:
    if not isinstance(samples, dict):
        return []

    task_samples = samples.get(task, [])
    if not isinstance(task_samples, list):
        return []

    rows: list[dict[str, Any]] = []
    for sample in task_samples:
        if not isinstance(sample, dict):
            continue

        prompts = _sample_prompts(sample.get("arguments"))
        prompt = prompts[0] if prompts else None
        prediction, prediction_meta = _sample_prediction(sample)

        metrics = {
            metric_name: sample.get(metric_name)
            for metric_name in sample.get("metrics", [])
            if metric_name in sample
        }

        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_name": checkpoint_name,
                "method": method,
                "size": size,
                "seed": seed,
                "task": task,
                "doc_id": sample.get("doc_id"),
                "filter": sample.get("filter"),
                "prompt": prompt,
                "prompts": prompts,
                "prediction": prediction,
                "prediction_meta": prediction_meta,
                "target": sample.get("target"),
                "metrics": metrics,
                "arguments": sample.get("arguments"),
                "resps": sample.get("resps"),
                "filtered_resps": sample.get("filtered_resps"),
                "doc": sample.get("doc"),
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
                "target_hash": sample.get("target_hash"),
            }
        )

    return rows


def main() -> None:

    # read args and config
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    eval_cfg = cfg.get("evaluation", {})
    if not isinstance(eval_cfg, dict):
        raise ValueError("Config key 'evaluation' must be a mapping")

    # read checkpoints
    config_paths = eval_cfg.get("checkpoint_paths", [])
    if not isinstance(config_paths, list):
        raise ValueError("evaluation.checkpoint_paths must be a list")

    cli_paths: list[str] = []
    cli_paths.extend(str(item) for item in args.checkpoint)
    cli_paths.extend(str(item) for item in args.checkpoints_dir)

    raw_checkpoint_paths = [str(item) for item in config_paths] + cli_paths
    checkpoints = resolve_checkpoints(raw_checkpoint_paths)
    if not checkpoints:
        raise ValueError("No valid checkpoints found from config/CLI paths")

    # read task list
    raw_tasks = eval_cfg.get("tasks", [])
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("evaluation.tasks must be a non-empty list")
    tasks = [normalize_task_name(str(task)) for task in raw_tasks]

    # set seed
    seeds = eval_cfg.get("seeds", [42])

    # default model backend (can be overridden per method below)
    default_model = str(eval_cfg.get("model", "hf"))
    method_model_overrides = eval_cfg.get("method_model_overrides", {})
    if not isinstance(method_model_overrides, dict):
        raise ValueError("evaluation.method_model_overrides must be a mapping")

    # other args
    device = str(eval_cfg.get("device", "cuda:0"))
    batch_size = str(eval_cfg.get("batch_size", "auto"))
    num_fewshot = int(eval_cfg.get("num_fewshot", 0))
    
    #bootstrap details
    bootstrap_iters = eval_cfg.get("bootstrap_iters", None)
    if bootstrap_iters is not None:
        bootstrap_iters = int(bootstrap_iters)
    
    limit = eval_cfg.get("limit", None)
    model_args_extra = eval_cfg.get("model_args_extra", {})
    if not isinstance(model_args_extra, dict):
        raise ValueError("evaluation.model_args_extra must be a mapping")
    method_model_args_extra = eval_cfg.get("method_model_args_extra", {})
    if not isinstance(method_model_args_extra, dict):
        raise ValueError("evaluation.method_model_args_extra must be a mapping")
    
    tokenizer_override = args.tokenizer if args.tokenizer is not None else eval_cfg.get("tokenizer", None)
    if tokenizer_override is not None:
        tokenizer_override = str(tokenizer_override)

    output_root = Path(args.output_root or str(eval_cfg.get("output_root", "evaluation/output")))
    if args.dry_run:
        targets: list[dict[str, str]] = []
        for checkpoint in checkpoints:
            method = infer_method_from_checkpoint(checkpoint)
            size = infer_size_from_checkpoint(checkpoint)
            for task in tasks:
                task_dir = output_root / size / method / checkpoint.name / f"{task}_result"
                targets.append(
                    {
                        "checkpoint": str(checkpoint),
                        "task": task,
                        "task_output_dir": str(task_dir),
                    }
                )
        plan = {
            "model": default_model,
            "method_model_overrides": method_model_overrides,
            "tasks": tasks,
            "seeds": seeds,
            "device": device,
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
            "bootstrap_iters": bootstrap_iters,
            "tokenizer_override": tokenizer_override,
            "limit": limit,
            "num_checkpoints": len(checkpoints),
            "checkpoints": [str(path) for path in checkpoints],
            "targets": targets,
        }
        print(json.dumps(plan, indent=2))
        return

    patch_transformers_dtype_alias()
    register_custom_model_types()

    from lm_eval import evaluator

    supported_kwargs = set(inspect.signature(evaluator.simple_evaluate).parameters.keys())

    for checkpoint in checkpoints:
        checkpoint_name = checkpoint_label(checkpoint)
        method = infer_method_from_checkpoint(checkpoint)
        size = infer_size_from_checkpoint(checkpoint)
        prepared_checkpoint, prepared_tmpdir, prep_meta = prepare_checkpoint_for_lm_eval(checkpoint, method)
        print(prepared_checkpoint, prep_meta)
        per_run_extra = dict(model_args_extra)
        per_method_extra = method_model_args_extra.get(method, {})
        if not isinstance(per_method_extra, dict):
            per_method_extra = {}
        per_run_extra.update(per_method_extra)
        if tokenizer_override:
            per_run_extra["tokenizer"] = tokenizer_override
        elif not has_local_tokenizer_files(prepared_checkpoint):
            for task in tasks:
                task_output_dir = output_root / size / method / checkpoint.name / f"{task}_result"
                task_output_dir.mkdir(parents=True, exist_ok=True)
                failure_payload = [
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_name": checkpoint_name,
                        "method": method,
                        "seed": None,
                        "task": task,
                        "prepared_checkpoint": str(prepared_checkpoint),
                        "checkpoint_rewritten": bool(prep_meta.get("rewritten", False)),
                        "error": (
                            "Tokenizer files not found in checkpoint and no evaluation.tokenizer override was provided. "
                            "Set evaluation.tokenizer in config (for example a LLaMA tokenizer repo/path)."
                        ),
                    }
                ]
                (task_output_dir / "failures.json").write_text(json.dumps(failure_payload, indent=2))
            if prepared_tmpdir is not None:
                prepared_tmpdir.cleanup()
            continue

        model_backend = resolve_model_backend(method, default_model, method_model_overrides)
        model_args = build_model_args(prepared_checkpoint, per_run_extra)

        print("model_backend",model_backend)
        print("model_args", model_args)
        print("tokenizer_override", tokenizer_override)

        for task in tasks:
            task_output_dir = output_root / size / method / checkpoint.name / f"{task}_result"
            task_raw_dir = task_output_dir / "raw"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            task_raw_dir.mkdir(parents=True, exist_ok=True)

            task_plan = {
                "model": default_model,
                "method_model_overrides": method_model_overrides,
                "task": task,
                "seeds": seeds,
                "device": device,
                "batch_size": batch_size,
                "num_fewshot": num_fewshot,
                "bootstrap_iters": bootstrap_iters,
                "tokenizer_override": tokenizer_override,
                "limit": limit,
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint_name,
                "method": method,
                "size": size,
                "task_output_dir": str(task_output_dir),
            }
            (task_output_dir / "resolved_plan.json").write_text(json.dumps(task_plan, indent=2))

            task_long_rows: list[dict[str, Any]] = []
            task_failures: list[dict[str, Any]] = []
            task_sample_files: list[str] = []

            for seed in seeds:
                task_out_file = task_raw_dir / f"seed_{seed}.json"

                kwargs: dict[str, Any] = {
                    "model": model_backend,
                    "model_args": model_args,
                    "tasks": [task],
                    "num_fewshot": num_fewshot,
                    "batch_size": batch_size,
                    "device": device,
                    "log_samples": True,
                    "random_seed": seed,
                    "numpy_random_seed": seed,
                    "torch_random_seed": seed,
                    "fewshot_random_seed": seed,
                }
                if bootstrap_iters is not None:
                    kwargs["bootstrap_iters"] = bootstrap_iters
                if limit is not None:
                    kwargs["limit"] = limit

                kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

                try:
                    print("kwargs:", kwargs)
                    result = evaluator.simple_evaluate(**kwargs)
                    print("result:", result)
                    task_out_file.write_text(json.dumps(result, indent=2, default=str))

                    sample_rows = flatten_sample_records(
                        result.get("samples", {}),
                        checkpoint=str(checkpoint),
                        checkpoint_name=checkpoint_name,
                        method=method,
                        size=size,
                        seed=seed,
                        task=task,
                    )
                    sample_file = task_raw_dir / f"seed_{seed}_samples.jsonl"
                    write_jsonl(sample_file, sample_rows)
                    task_sample_files.append(str(sample_file))

                    for metric_row in flatten_results(result):
                        task_long_rows.append(
                            {
                                "checkpoint": str(checkpoint),
                                "checkpoint_name": checkpoint_name,
                                "seed": seed,
                                "task": metric_row["task"],
                                "metric": metric_row["metric"],
                                "value": metric_row["value"],
                            }
                        )
                except Exception as exc:
                    task_failures.append(
                        {
                            "checkpoint": str(checkpoint),
                            "checkpoint_name": checkpoint_name,
                            "method": method,
                            "seed": seed,
                            "task": task,
                            "prepared_checkpoint": str(prepared_checkpoint),
                            "checkpoint_rewritten": bool(prep_meta.get("rewritten", False)),
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

            task_agg_rows = aggregate_rows(task_long_rows)

            write_csv(
                task_output_dir / "results_long.csv",
                task_long_rows,
                ["checkpoint", "checkpoint_name", "seed", "task", "metric", "value"],
            )
            write_csv(
                task_output_dir / "results_aggregate.csv",
                task_agg_rows,
                ["checkpoint_name", "task", "metric", "num_seeds", "mean", "std", "min", "max"],
            )

            (task_output_dir / "failures.json").write_text(json.dumps(task_failures, indent=2))

            task_summary = {
                "task_output_dir": str(task_output_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint_name,
                "method": method,
                "size": size,
                "num_seeds": len(seeds),
                "task": task,
                "bootstrap_iters": bootstrap_iters,
                "long_rows": len(task_long_rows),
                "aggregate_rows": len(task_agg_rows),
                "num_failures": len(task_failures),
                "files": {
                    "plan": str(task_output_dir / "resolved_plan.json"),
                    "results_long_csv": str(task_output_dir / "results_long.csv"),
                    "results_aggregate_csv": str(task_output_dir / "results_aggregate.csv"),
                    "samples_jsonl": task_sample_files,
                    "failures_json": str(task_output_dir / "failures.json"),
                },
            }
            (task_output_dir / "summary.json").write_text(json.dumps(task_summary, indent=2))

            print(json.dumps(task_summary, indent=2))

        if prepared_tmpdir is not None:
            prepared_tmpdir.cleanup()


if __name__ == "__main__":
    main()
