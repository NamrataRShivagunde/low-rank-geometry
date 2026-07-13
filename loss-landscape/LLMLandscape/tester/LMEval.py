import torch
import os
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from typing import Tuple, Dict, Iterable
from models import BaseModel
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import torch.utils.data


__all__ = ["lm_eval_gsm8k", "lm_eval_mmlu", "lm_eval_truthfulqa", "lm_eval_humaneval", "lm_eval_c4"]


def lm_eval_c4(
    model: BaseModel,
    dataset_path: str,
    tokenizer_id: str = None,
    split: str = None,
    max_examples: int = 1000,
    max_length: int = 256,
    batch_size: int = 128,
    target_tokens: int = 10_000_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
) -> float:
    """
    Compute token-normalized NLL on a saved C4 validation dataset.

    Parameters
    - model: repository model wrapper exposing HF-like forward (supports labels) and `tokenizer` attribute if available.
    - dataset_path: local path to a dataset saved with `datasets.Dataset.save_to_disk()` or a jsonl/text file (will attempt sensible fallbacks).
    - tokenizer_id: optional HF tokenizer id to use; if None the function will try `model.tokenizer` then `dataset_path` as an id.
    - split: optional dataset split name when loading via `load_dataset(dataset_path, split=split)`.
    - max_examples, max_length, batch_size: controls sampling/tokenization to limit memory for quick runs.
    - target_tokens: stop after this many non-pad tokens (mirrors other NLL helpers).

    Returns the token-normalized NLL (float; lower is better).
    """
    # Resolve tokenizer
    if tokenizer_id:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
    pad_idx = tokenizer.pad_token_id

    # Load dataset: prefer load_from_disk for saved datasets, fallback to load_dataset
    ds = None
    if os.path.isdir(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        # try to load as a single-file dataset (e.g., jsonl)
        ds = load_dataset("json", data_files={"validation": dataset_path}, split="validation")
  
    # we choose 1000 samples randomly with a fixed seed
    ds = ds.shuffle(seed=42)

    # Sample up to max_examples (materialize small slice)
    examples = []
    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        examples.append(ex)

    texts = [ex.get("text", ex.get("article", "")) for ex in examples]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    dataset_torch = torch.utils.data.TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)

    # Compute token-normalized NLL using model forward with labels (masking pad -> -100)
    model_device = device
    # If model parameters already on a device, prefer that
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(device)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            total_batches += 1
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)

            labels = input_ids.clone()
            labels[labels == pad_idx] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = float(outputs.loss.item()) if hasattr(outputs, "loss") else float(outputs["loss"]) if isinstance(outputs, dict) and "loss" in outputs else None
            total_loss += loss.detach()
        
        total_loss = total_loss / total_batches

        return total_loss


def lm_eval_gsm8k(
    model: BaseModel,
    task: str = "gsm8k_cot_llama",
    limit: int = 100,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    # Resolve tokenizer: accept tokenizer instance or HF id string on model.tokenizer
    tk = getattr(model, "tokenizer", None)
    if isinstance(tk, str):
        tk = AutoTokenizer.from_pretrained(tk)
    if tk is None:
        tk = AutoTokenizer.from_pretrained("t5-base")
    hf_model = HFLM(pretrained=model, tokenizer=tk, device=device)
    use_chat = getattr(tk, "chat_template", None) is not None
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        apply_chat_template=use_chat,
        fewshot_as_multiturn=True,
        log_samples=True,
        gen_kwargs=kwargs,
    )
    score = results["results"][task]["exact_match,flexible-extract"].item()
    print("GSM8k Score: ", score)
    print("-" * 10)
    return score


def lm_eval_mmlu(
    model: BaseModel,
    task: str = "mmlu_generative",
    limit: int = 10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    system_instruction: str = None,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    assert task in ["mmlu", "mmlu_continuation", "mmlu_generation", "mmlu_generative"]
    tk = getattr(model, "tokenizer", None)
    if isinstance(tk, str):
        tk = AutoTokenizer.from_pretrained(tk)
    if tk is None:
        tk = AutoTokenizer.from_pretrained("t5-base")

    hf_model = HFLM(pretrained=model, tokenizer=tk, device=device)
    use_chat = getattr(tk, "chat_template", None) is not None
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=10,
        apply_chat_template=use_chat,
        gen_kwargs=dict(do_sample=False),
        system_instruction=system_instruction,
    )
    results = results["results"]
    score = sum(
        [
            result["exact_match,get_response"] if task == "mmlu_generative" else result["acc,none"]
            for task_name, result in results.items()
        ]
    )
    score /= len(results)
    print("MMLU Score: ", score)
    print("-" * 10)
    return score


def lm_eval_truthfulqa(
    model: BaseModel,
    task: str = "truthfulqa_gen",
    limit: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    assert task in ["truthfulqa_mc1", "truthfulqa_mc2", "truthfulqa_gen"]
    tk = getattr(model, "tokenizer", None)
    if isinstance(tk, str):
        tk = AutoTokenizer.from_pretrained(tk)
    if tk is None:
        tk = AutoTokenizer.from_pretrained("t5-base")
    hf_model = HFLM(pretrained=model, tokenizer=tk, device=device)
    use_chat = getattr(tk, "chat_template", None) is not None
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        max_batch_size=128,
        apply_chat_template=use_chat,
        gen_kwargs=kwargs,
    )
    score = results["results"][task]["bleu_acc,none"] if "gen" in task else results["results"][task]["acc,none"]
    print("TruthfulQA Score: ", score)
    print("-" * 10)
    return score


def lm_eval_humaneval(
    model: BaseModel,
    task: str = "humaneval_instruct",
    limit: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    system_instruction: str = None,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/README.md
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # warning: Please read the warning!!!
    assert task in ["humaneval", "humaneval_64", "humaneval_instruct", "humaneval_instruct_64"]
    tk = getattr(model, "tokenizer", None)
    if isinstance(tk, str):
        tk = AutoTokenizer.from_pretrained(tk)
    if tk is None:
        tk = AutoTokenizer.from_pretrained("t5-base")
    hf_model = HFLM(pretrained=model, tokenizer=tk, device=device)
    use_chat = getattr(tk, "chat_template", None) is not None
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        apply_chat_template=use_chat,
        confirm_run_unsafe_code=True,  # warning: Please read the warning!!!
        system_instruction=system_instruction,
        gen_kwargs=kwargs,
    )
    score = results["results"][task]["pass@1,create_test"].item()
    print("Humaneval Score: ", score)
    print("-" * 10)
    return score
