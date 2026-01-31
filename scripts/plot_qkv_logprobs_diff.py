#!/usr/bin/env python3
"""Compare qkv formats (bshd vs thd) and plot mean abs diff of log-probs.

This script:
- Loads all train_data/*.pt under the provided dirs
- Merges samples via sample_indices (handles TP/DP sharding)
- Checks prompt/response lengths for consistency
- Computes mean abs diff across samples
- Bins by token index (default 50) and plots to PNG
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


TRUE_SET = {"1", "true", "yes", "y", "t"}
FALSE_SET = {"0", "false", "no", "n", "f"}


def parse_sp(value: str) -> str:
    v = value.strip().lower()
    if v in TRUE_SET:
        return "true"
    if v in FALSE_SET:
        return "false"
    return v


def load_merged_samples(train_data_dir: Path, key: str):
    files = sorted(train_data_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {train_data_dir}")

    by_index = {}
    prompt_lens = set()
    response_lens = set()
    total_lens = set()

    for f in files:
        data = torch.load(f, map_location="cpu")
        if "rollout_data" not in data:
            raise KeyError(f"Missing rollout_data in {f}")
        rd = data["rollout_data"]
        if key not in rd:
            raise KeyError(f"Missing {key} in {f}")

        sample_indices = rd.get("sample_indices")
        values = rd[key]
        if sample_indices is None:
            sample_indices = list(range(len(values)))
        if len(values) != len(sample_indices):
            raise ValueError(f"Length mismatch for {key} vs sample_indices in {f}")

        if "response_lengths" in rd:
            response_lens.update(rd["response_lengths"])
        if "total_lengths" in rd:
            total_lens.update(rd["total_lengths"])
        if "total_lengths" in rd and "response_lengths" in rd:
            for total, resp in zip(rd["total_lengths"], rd["response_lengths"], strict=True):
                prompt_lens.add(total - resp)

        for idx, val in zip(sample_indices, values, strict=True):
            existing = by_index.get(idx)
            if existing is None:
                by_index[idx] = val
            else:
                if not torch.equal(existing, val):
                    max_diff = (existing - val).abs().max().item()
                    raise ValueError(f"Mismatch for sample {idx} in {f}: max abs diff {max_diff}")

    indices = sorted(by_index.keys())
    samples = [by_index[i] for i in indices]
    lengths = [len(s) for s in samples]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent token lengths across samples: {sorted(set(lengths))}")

    arr = np.stack([s.detach().cpu().float().numpy() for s in samples], axis=0)
    return arr, indices, prompt_lens, response_lens, total_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thd-dir", required=True, help="Path to *_thd dump dir")
    parser.add_argument("--bshd-dir", required=True, help="Path to *_bshd dump dir")
    parser.add_argument("--key", default="log_probs", help="Key to compare (default: log_probs)")
    parser.add_argument("--bin-size", type=int, default=50, help="Bin size for x-axis")
    parser.add_argument("--out-dir", default="plots", help="Output directory for plot")
    parser.add_argument("--model-size", required=True, help="Model size number, e.g. 4")
    parser.add_argument("--tp", required=True, help="TP size")
    parser.add_argument("--dp", required=True, help="DP size")
    parser.add_argument("--sp", required=True, help="SP flag (true/false)")
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Also dump merged JSONs under json/ for both thd/bshd",
    )
    args = parser.parse_args()

    thd_train = Path(args.thd_dir) / "train_data"
    bshd_train = Path(args.bshd_dir) / "train_data"

    thd_arr, thd_indices, thd_prompt, thd_resp, thd_total = load_merged_samples(thd_train, args.key)
    bshd_arr, bshd_indices, bshd_prompt, bshd_resp, bshd_total = load_merged_samples(bshd_train, args.key)

    if thd_indices != bshd_indices:
        raise ValueError(f"Sample indices mismatch: thd={thd_indices} bshd={bshd_indices}")

    if thd_arr.shape != bshd_arr.shape:
        raise ValueError(f"Shape mismatch: thd={thd_arr.shape} bshd={bshd_arr.shape}")

    # Log length consistency warnings.
    def lens_msg(name, prompt, resp, total):
        return f"{name}: prompt_lens={sorted(prompt)} response_lens={sorted(resp)} total_lens={sorted(total)}"

    print(lens_msg("thd", thd_prompt, thd_resp, thd_total))
    print(lens_msg("bshd", bshd_prompt, bshd_resp, bshd_total))

    # Compute mean abs diff across samples.
    abs_diff = np.abs(bshd_arr - thd_arr)
    mean_diff = abs_diff.mean(axis=0)

    # Bin by token index.
    bin_size = max(1, args.bin_size)
    n = mean_diff.shape[0]
    xs = []
    ys = []
    for start in range(0, n, bin_size):
        end = min(start + bin_size, n)
        xs.append(start)
        ys.append(float(mean_diff[start:end].mean()))

    sp_str = parse_sp(args.sp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{args.model_size}b_tp{args.tp}_dp{args.dp}_sp{sp_str}_qkv_logprobs_diff.png"
    out_path = out_dir / out_name

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, linewidth=1.5)
    plt.xlabel(f"data index (binned by {bin_size})")
    plt.ylabel("logprobs abs diff (avg over samples)")
    plt.title(f"{args.model_size}b tp{args.tp} dp{args.dp} sp{sp_str} qkv {args.key} abs diff (bshd vs thd)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    if args.dump_json:
        json_dir = Path("json")
        json_dir.mkdir(parents=True, exist_ok=True)
        thd_json = json_dir / f"{args.model_size}b_tp{args.tp}_dp{args.dp}_sp{sp_str}_thd_{args.key}.json"
        bshd_json = json_dir / f"{args.model_size}b_tp{args.tp}_dp{args.dp}_sp{sp_str}_bshd_{args.key}.json"
        thd_payload = {"num_samples": thd_arr.shape[0], args.key: thd_arr.tolist()}
        bshd_payload = {"num_samples": bshd_arr.shape[0], args.key: bshd_arr.tolist()}
        thd_json.write_text(json.dumps(thd_payload, indent=2))
        bshd_json.write_text(json.dumps(bshd_payload, indent=2))
        print(f"Wrote JSON: {thd_json}")
        print(f"Wrote JSON: {bshd_json}")


if __name__ == "__main__":
    main()
