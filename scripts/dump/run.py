#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, *, check=True, env=None, allow_fail=False):
    if isinstance(cmd, (list, tuple)):
        cmd_display = " ".join(shlex.quote(str(c)) for c in cmd)
    else:
        cmd_display = cmd
    print(f"+ {cmd_display}")
    try:
        return subprocess.run(cmd, check=check, env=env)
    except subprocess.CalledProcessError as exc:
        if allow_fail:
            return exc
        raise


def load_model_args(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"MODEL_ARGS file not found: {path}")
    tokens = []
    in_block = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not in_block:
            if line.startswith("MODEL_ARGS=") and "(" in line:
                in_block = True
                # capture any tokens after '('
                line = line.split("(", 1)[1]
            else:
                continue
        if ")" in line:
            line = line.split(")", 1)[0]
            if line:
                tokens.extend(shlex.split(line, comments=True, posix=True))
            break
        if line:
            tokens.extend(shlex.split(line, comments=True, posix=True))
    if not tokens:
        raise ValueError(f"No MODEL_ARGS parsed from {path}")
    return tokens


def nvlink_count():
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return 0
        return len(re.findall(r"NV\d+", result.stdout))
    except FileNotFoundError:
        return 0


def get_model_config(model_size):
    """Get model configuration based on model size."""
    configs = {
        "4b": {
            "script": "qwen3-4B.sh",
            "hf_path": "/root/shared/Qwen3-4B",
            "ref_load": "/root/shared/Qwen3-4B_torch_dist/",
        },
        "30b": {
            "script": "qwen3-30B-A3B.sh",
            "hf_path": "/root/shared/Qwen3-30B-A3B",
            "ref_load": "/root/shared/Qwen3-30B-A3B_torch_dist/",
        },
        "32b": {
            "script": "qwen3-32B.sh",
            "hf_path": "/root/shared/Qwen3-32B",
            "ref_load": "/root/shared/Qwen3-32B_torch_dist/",
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return configs[model_size]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MILES training with configurable parallelism settings")
    parser.add_argument(
        "--tp", "--tp-size", type=int, default=4, dest="tp_size", help="Tensor parallel size (default: 4)"
    )
    parser.add_argument(
        "--dp", "--dp-size", type=int, default=1, dest="dp_size", help="Data parallel size (default: 1)"
    )
    parser.add_argument("--sp", "--sequence-parallel", action="store_true", dest="sp", help="Enable sequence parallel")
    parser.add_argument(
        "--no-sp", "--no-sequence-parallel", action="store_false", dest="sp", help="Disable sequence parallel"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["4b", "30b", "32b"],
        default="4b",
        help="Model size: 4b (Qwen3-4B), 30b (Qwen3-30B-A3B), 32b (Qwen3-32B) (default: 4b)",
    )
    parser.add_argument(
        "--qkv-format", type=str, choices=["thd", "bshd"], default="thd", help="QKV format (default: thd)"
    )
    parser.add_argument(
        "--sample-folder",
        type=str,
        default="16479",
        help="Sample folder name in /root/shared/debug_rollout/ (default: 16479)",
    )
    parser.add_argument(
        "--data-pad-size-multiplier",
        type=int,
        default=128,
        help="Multiplier for data padding size in data processing (default: 128)",
    )
    parser.set_defaults(sp=True)

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Print configuration
    print("=" * 80)
    print("MILES Training Configuration")
    print("=" * 80)
    print(f"Model:              {args.model}")
    print(f"Tensor Parallel:    {args.tp_size}")
    print(f"Data Parallel:      {args.dp_size}")
    print(f"Sequence Parallel:  {args.sp}")
    print(f"QKV Format:         {args.qkv_format}")
    print(f"Sample Folder:      {args.sample_folder}")
    print(f"Pad Size Multiplier: {args.data_pad_size_multiplier}")
    print("=" * 80)

    # Calculate total GPUs needed
    total_gpus = args.tp_size * args.dp_size
    print(f"Total GPUs needed:  {total_gpus}")
    print("=" * 80)

    # Get model configuration
    model_config = get_model_config(args.model)

    # for rerun the task
    run(["pkill", "-9", "sglang"], allow_fail=True, check=False)
    time.sleep(3)
    run(["ray", "stop", "--force"], allow_fail=True, check=False)
    run(["pkill", "-9", "ray"], allow_fail=True, check=False)
    # NOTE: Do NOT use "pkill -9 python" as it will kill the script itself!
    # Instead, kill specific process names like "miles" or "train.py"
    run(["pkill", "-9", "miles"], allow_fail=True, check=False)
    time.sleep(3)
    run(["pkill", "-9", "ray"], allow_fail=True, check=False)
    run(["pkill", "-9", "miles"], allow_fail=True, check=False)
    run(["pkill", "-9", "redis"], allow_fail=True, check=False)

    # will prevent ray from buffering stdout/stderr
    os.environ["PYTHONBUFFERED"] = "16"

    nv_count = nvlink_count()
    has_nvlink = 1 if nv_count > 0 else 0
    print(f"HAS_NVLINK: {has_nvlink} (detected {nv_count} NVLink references)")

    script_dir = Path(__file__).resolve().parent
    model_args = load_model_args(script_dir / f"../models/{model_config['script']}")

    ckpt_args = [
        "--hf-checkpoint",
        model_config["hf_path"],
        "--ref-load",
        model_config["ref_load"],
        # "--load", f"{model_config['hf_path']}_miles/",
        # "--save", f"{model_config['hf_path']}_miles/",
        # "--save-interval", "20",
    ]

    rollout_args = [
        "--prompt-data",
        "/root/shared/dapo-math-17k/dapo-math-17k.jsonl",
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        # "--rollout-shuffle",
        "--rm-type",
        "deepscaler",
        "--num-rollout",
        "1",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "8",
        "--rollout-max-response-len",
        "16384",
        "--rollout-temperature",
        "1",
        "--global-batch-size",
        "8",
        "--balance-data",
    ]

    eval_args = [
        # "--eval-interval", "20",
        # "--eval-prompt-data", "aime", "/root/aime-2024/aime-2024.jsonl",
        # "--n-samples-per-eval-prompt", "16",
        # "--eval-max-response-len", "16384",
        # "--eval-top-p", "1",
    ]

    perf_args = [
        "--tensor-model-parallel-size",
        str(args.tp_size),
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--qkv-format",
        args.qkv_format,
        "--data-pad-size-multiplier",
        str(args.data_pad_size_multiplier),
        "--max-tokens-per-gpu",
        "20480",
    ]

    # Configure batch size based on qkv format
    if args.qkv_format == "bshd":
        perf_args.extend(["--micro-batch-size", "1"])
    else:  # thd
        perf_args.append("--use-dynamic-batch-size")

    # Add sequence parallel if enabled
    if args.sp:
        perf_args.append("--sequence-parallel")

    grpo_args = [
        "--advantage-estimator",
        "grpo",
        "--use-kl-loss",
        "--kl-loss-coef",
        "0.00",
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        "0.00",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
    ]

    optimizer_args = [
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--optimizer-cpu-offload",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--use-precision-aware-optimizer",
    ]

    wandb_args = [
        # "--use-wandb",
        # "--wandb-project", "miles-dev",
        # "--wandb-group", "qwen3-30B-A3B-test",
        # "--wandb-key", os.environ.get("WANDB_KEY", ""),
    ]

    sglang_args = [
        "--rollout-num-gpus-per-engine",
        str(args.tp_size),
        "--sglang-mem-fraction-static",
        "0.7",
        "--sglang-cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
    ] + [str(x) for x in range(16, 257, 8)]

    misc_args = [
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
    ]

    # Build dump details path based on configuration
    sp_str = "sptrue" if args.sp else "spfalse"
    config_name = f"{args.model}_tp{args.tp_size}_dp{args.dp_size}_{sp_str}_{args.qkv_format}_{args.sample_folder}_m{args.data_pad_size_multiplier}"

    dump_path = f"/root/shared/dump_label_logits/{config_name}"
    dumper_dir = f"/root/shared/dump_layers/{config_name}"

    debug_args = [
        # "--debug-rollout-only",
        # "--save-debug-rollout-data", f"/root/shared/debug_rollout/{args.sample_folder}/{{rollout_id}}.pt",
        "--debug-train-only",
        "--load-debug-rollout-data",
        f"/root/shared/debug_rollout/{args.sample_folder}/{{rollout_id}}.pt",
        "--dump-label-logits",
        "--dump-details",
        dump_path,
    ]

    print(f"Dump details path:  {dump_path}")
    print(f"Dumper layer path:  {dumper_dir}")
    print(f"Sample data path:   /root/shared/debug_rollout/{args.sample_folder}/{{rollout_id}}.pt")

    # launch the master node of ray in container
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Set CUDA_VISIBLE_DEVICES based on total GPUs needed
    cuda_devices = ",".join(str(i) for i in range(total_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")

    run(
        [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            os.environ["MASTER_ADDR"],
            "--num-gpus",
            str(total_gpus),
            "--disable-usage-stats",
            "--dashboard-host=0.0.0.0",
            "--dashboard-port=8265",
        ]
    )

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": str(has_nvlink),
                "MEGATRON_DUMPER_ENABLE": "1",
                "MEGATRON_DUMPER_DIR": dumper_dir,
                "MEGATRON_DUMPER_WRITE_FILE": "1",
                "MEGATRON_DUMPER_AGGREGATE_TP": "0",
                "MEGATRON_DUMPER_DP_RANK_0_ONLY": "0",
                "MEGATRON_DUMPER_NAMES": "post_mlp",
            }
        }
    )

    cmd = [
        "ray",
        "job",
        "submit",
        "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={runtime_env_json}",
        "--",
        "python3",
        "train.py",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(total_gpus),
        "--colocate",
    ]

    cmd += model_args
    cmd += ckpt_args
    cmd += rollout_args
    cmd += optimizer_args
    cmd += grpo_args
    cmd += wandb_args
    cmd += perf_args
    cmd += eval_args
    cmd += sglang_args
    cmd += misc_args
    cmd += debug_args

    run(cmd)

    # Print summary for easy copying
    print("\n" + "=" * 80)
    print("Job Submitted Successfully!")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print("\nOutput Paths:")
    print(f"  Dump Details: {dump_path}")
    print(f"  Dumper Layer: {dumper_dir}")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
