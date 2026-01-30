#!/usr/bin/env python3
"""Dump label_logits/log_probs from train output or rollout samples to JSON.

Notes:
- Train debug dumps store outputs under the key "rollout_data".
- Rollout debug dumps store samples under the key "samples".
- label_logits is produced during *train* forward-only log-prob computation and
  only exists if training was launched with --dump-label-logits.
"""

import argparse
import json
from pathlib import Path

import torch


def tensor_to_list(value):
    """Convert tensors or nested lists to JSON-serializable lists with full precision."""
    if isinstance(value, torch.Tensor):
        return value.float().tolist()
    if isinstance(value, list):
        return [tensor_to_list(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def extract_from_train_data(data, keys):
    """Extract requested keys from train debug dump (data['rollout_data'])."""
    rollout_data = data["rollout_data"]
    # Train outputs (e.g., label_logits/log_probs) are stored in rollout_data.
    print(f"Available train output keys: {list(rollout_data.keys())}")

    # These fields are metadata and help align outputs across ranks/rollouts.
    result = {
        "rollout_id": data.get("rollout_id"),
        "rank": data.get("rank"),
        "num_samples": len(rollout_data.get("label_logits", [])),
    }

    # Each key should map to a list of per-sample tensors, one tensor per sample.
    for key in keys:
        if key in rollout_data:
            values = rollout_data[key]
            result[key] = [tensor_to_list(v) for v in values]
            print(f"  {key}: {len(values)} samples")
            for i, v in enumerate(values):
                print(f"    Sample {i}: {len(v)} tokens")
        else:
            print(f"  {key}: NOT FOUND")

    return result


def extract_from_samples(data, keys):
    """Extract requested keys from rollout debug dump (data['samples'])."""
    samples = data["samples"]
    # Rollout samples are dictionaries created from Sample.to_dict().
    print(f"Available rollout sample keys: {list(samples[0].keys()) if samples else []}")

    # In rollout dumps, log_probs are stored as rollout_log_probs.
    key_aliases = {
        "log_probs": "rollout_log_probs",
    }

    # Metadata aligns outputs across ranks/rollouts.
    result = {
        "rollout_id": data.get("rollout_id"),
        "rank": data.get("rank"),
        "num_samples": len(samples),
    }

    # Each key should map to a list of per-sample lists/tensors.
    for key in keys:
        sample_key = key_aliases.get(key, key)
        if samples and sample_key in samples[0]:
            values = [s.get(sample_key) for s in samples]
            result[key] = [tensor_to_list(v) for v in values]
            print(f"  {key} (from {sample_key}): {len(values)} samples")
            for i, v in enumerate(values):
                print(f"    Sample {i}: {len(v)} tokens")
        else:
            print(f"  {key} (from {sample_key}): NOT FOUND")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to debug .pt file (train debug dump or rollout dump)",
    )
    parser.add_argument("-o", "--output", help="Output JSON path (default: input.json)")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["label_logits", "log_probs"],
        help="Keys to dump (default: label_logits log_probs)",
    )
    parser.add_argument(
        "--source",
        choices=["train", "rollout", "auto"],
        default="train",
        help=(
            "Where to read data from: "
            "'train' uses rollout_data (train outputs), "
            "'rollout' uses samples, "
            "'auto' tries rollout_data then samples. (default: train)"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    print(f"Loading {input_path}...")
    data = torch.load(input_path, map_location="cpu")

    if args.source == "train":
        if "rollout_data" not in data:
            raise KeyError("Expected 'rollout_data' in train debug data. Use --source rollout for rollout dumps.")
        result = extract_from_train_data(data, args.keys)
    elif args.source == "rollout":
        if "samples" not in data:
            raise KeyError("Expected 'samples' in rollout debug data. Use --source train for train dumps.")
        result = extract_from_samples(data, args.keys)
    else:
        if "rollout_data" in data:
            result = extract_from_train_data(data, args.keys)
        elif "samples" in data:
            result = extract_from_samples(data, args.keys)
        else:
            raise KeyError("Expected 'rollout_data' or 'samples' in input data")

    print(f"Writing {output_path}...")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
