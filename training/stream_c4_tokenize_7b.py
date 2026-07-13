#!/usr/bin/env python3
"""
Stream C4, tokenize fixed-length examples, shard progress to disk, and save a
final DatasetDict with `train` and `validation` splits.

Usage:
    python training/stream_c4_tokenize_7b.py --tokenizer t5-base --output-dir /data/c4_7b
"""
import argparse
import json
import os
import shutil
import time

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk, load_dataset
from transformers import AutoTokenizer


# For quick testing, keep these small. Change back to 7_000_000_000 / 10_000_000 for full runs.
TRAIN_TOKENS = 7_000_000_000
VAL_TOKENS = 10_000_000
MAX_LENGTH = 256
EXAMPLES_PER_SHARD = 10_000
BATCH_SIZE = 1000
SHUFFLE_SEED = 42


parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", default="t5-base")
parser.add_argument("--output-dir", default="datasets-7b/c4/tokenized")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
if tokenizer.pad_token_id is None:
    raise ValueError("Tokenizer must define pad_token_id when using padding='max_length'.")
start_time = time.time()


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )


def process_split(split_name, target_tokens):
    split_dir = os.path.join(args.output_dir, split_name)
    shard_dir = os.path.join(split_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    existing_shards = sorted(name for name in os.listdir(shard_dir) if name.startswith("shard_"))
    shard_idx = len(existing_shards)
    docs_done = 0
    tokens_done = 0

    if existing_shards:
        last_state = os.path.join(shard_dir, existing_shards[-1], "shard_state.json")
        if os.path.exists(last_state):
            with open(last_state) as f:
                state = json.load(f)
            docs_done = state["docs_consumed"]
            tokens_done = state["total_tokens"]

    print(
        f"[{split_name}] Resuming from shard {shard_idx}, "
        f"{tokens_done / 1e9:.4f}B tokens, {docs_done:,} docs already done",
        flush=True,
    )

    stream = load_dataset("allenai/c4", "en", split=split_name, streaming=True)
    stream = stream.shuffle(seed=SHUFFLE_SEED)
    stream = stream.skip(docs_done)
    tokenized_stream = stream.map(
        tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["text", "timestamp", "url"],
    )

    docs_seen = docs_done
    total_tokens = tokens_done
    shard_buf = []

    def make_shard(examples, current_shard_idx, current_docs_seen, current_total_tokens):
        path = os.path.join(shard_dir, f"shard_{current_shard_idx:04d}")
        Dataset.from_dict({"input_ids": examples}).save_to_disk(path)
        with open(os.path.join(path, "shard_state.json"), "w") as f:
            json.dump(
                {
                    "docs_consumed": current_docs_seen,
                    "total_tokens": current_total_tokens,
                },
                f,
                indent=2,
            )
        print(
            f"[{split_name}] shard {current_shard_idx:04d} saved "
            f"({len(examples):,} examples) {current_total_tokens / 1e9:.4f}B total",
            flush=True,
        )

    for doc in tokenized_stream:
        docs_seen += 1
        shard_buf.append(doc["input_ids"])
        if "attention_mask" in doc:
            total_tokens += int(sum(doc["attention_mask"]))
        else:
            total_tokens += int(sum(token != tokenizer.pad_token_id for token in doc["input_ids"]))

        if len(shard_buf) >= EXAMPLES_PER_SHARD:
            make_shard(shard_buf, shard_idx, docs_seen, total_tokens)
            shard_idx += 1
            shard_buf = []

        if total_tokens >= target_tokens:
            break

    if shard_buf:
        make_shard(shard_buf, shard_idx, docs_seen, total_tokens)

    all_shards = sorted(
        os.path.join(shard_dir, name)
        for name in os.listdir(shard_dir)
        if name.startswith("shard_")
    )
    split_dataset = concatenate_datasets([load_from_disk(path) for path in all_shards])

    split_info = {
        "split": split_name,
        "num_examples": len(split_dataset),
        "non_pad_tokens": total_tokens,
        "max_length": MAX_LENGTH,
        "examples_per_shard": EXAMPLES_PER_SHARD,
        "num_shards": len(all_shards),
    }

    print(
        f"[{split_name}] final size: {len(split_dataset):,} examples "
        f"({total_tokens / 1e6:.2f}M non-pad tokens processed)",
        flush=True,
    )
    return split_dataset, split_info, shard_dir


train, train_info, train_shard_dir = process_split("train", TRAIN_TOKENS)
val, val_info, val_shard_dir = process_split("validation", VAL_TOKENS)

dataset = DatasetDict({"train": train, "validation": val})
dataset.save_to_disk(args.output_dir)

with open(os.path.join(args.output_dir, "train_dataset_info.json"), "w") as f:
    json.dump(train_info, f, indent=2)

with open(os.path.join(args.output_dir, "validation_dataset_info.json"), "w") as f:
    json.dump(val_info, f, indent=2)

for shard_dir in [train_shard_dir, val_shard_dir]:
    if os.path.isdir(shard_dir):
        shutil.rmtree(shard_dir)

print("\nDone.", flush=True)
print(f"  train      : {len(train):,} examples", flush=True)
print(f"  validation : {len(val):,} examples", flush=True)
print(f"  saved to   : {args.output_dir}", flush=True)
print(f"  total time : {time.time() - start_time:.0f}s", flush=True)
