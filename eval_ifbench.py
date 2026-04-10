#!/usr/bin/env python3
"""
eval_ifbench.py — Evaluate IFBench on a running sglang server using evalscope.

Usage:
    python -m agent.eval_bench.eval_ifbench \
        --api-url http://localhost:30000/v1 \
        --model-name Qwen3.5-35B-A3B

    # Multiple sglang servers (comma-separated)
    python -m agent.eval_bench.eval_ifbench \
        --api-url http://10.0.0.1:30000/v1,http://10.0.0.2:30000/v1 \
        --model-name Qwen3.5-35B-A3B

    # Custom sampling parameters
    python -m agent.eval_bench.eval_ifbench \
        --api-url http://localhost:30000/v1 \
        --model-name Qwen3.5-35B-A3B \
        --temperature 1.0 --top-p 0.95 --top-k 20 \
        --presence-penalty 1.5 --enable-thinking

Metrics reported:
    prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEBUG_LOG_PATH = "/root/miles/.cursor/debug-b8af16.log"
DEBUG_SESSION_ID = "b8af16"


def _append_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _build_generation_config(args) -> dict:
    """Build evalscope generation_config from CLI args."""
    g = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.top_k is not None:
        g["top_k"] = args.top_k
    if args.presence_penalty is not None:
        g["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        g["repetition_penalty"] = args.repetition_penalty

    extra_body = {}
    if args.enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if extra_body:
        g["extra_body"] = extra_body

    return g


def run_on_instance(
    instance_id: int,
    api_url: str,
    model_name: str,
    api_key: str,
    generation_config: dict,
    output_dir: str,
    batch_size: int,
    timeout: int,
    limit: int | None,
    benchmark: str,
) -> dict:
    """Run IFBench/IFEval evaluation on a single sglang instance."""
    # region agent log
    _append_debug_log(
        run_id="baseline",
        hypothesis_id="H5",
        location="eval_ifbench.py:95",
        message="eval instance started",
        data={
            "instance_id": int(instance_id),
            "api_url": api_url,
            "model_name": model_name,
            "benchmark": benchmark,
            "limit": limit,
        },
    )
    # endregion
    try:
        from evalscope.config import TaskConfig
        from evalscope.run import run_task
    except ImportError as e:
        return {
            "benchmark": benchmark,
            "status": "error",
            "error": (
                f"evalscope not installed: {e}\n"
                "  pip install evalscope\n"
                "  or: pip install -e third_party/evalscope"
            ),
        }

    instance_work_dir = os.path.join(output_dir, f"instance_{instance_id}")

    # Strip <think>...</think> before evaluation
    dataset_args = {benchmark: {"filters": {"remove_until": "</think>"}}}

    task_cfg = TaskConfig(
        model=model_name,
        api_url=api_url,
        api_key=api_key,
        eval_type="openai_api",
        datasets=[benchmark],
        dataset_args=dataset_args,
        generation_config=generation_config,
        eval_batch_size=batch_size,
        work_dir=instance_work_dir,
        judge_strategy="rule",
        limit=limit,
        use_cache=instance_work_dir,
        ignore_errors=True,
        model_args={"timeout": timeout},
    )

    print(f"[Instance {instance_id}] {benchmark} -> {api_url}")
    print(f"[Instance {instance_id}]   generation_config: {json.dumps(generation_config, ensure_ascii=False)}")

    try:
        run_task(task_cfg=task_cfg)
        # region agent log
        _append_debug_log(
            run_id="baseline",
            hypothesis_id="H6",
            location="eval_ifbench.py:148",
            message="eval run_task finished",
            data={
                "instance_id": int(instance_id),
                "status": "success",
                "work_dir": instance_work_dir,
                "api_url": api_url,
            },
        )
        # endregion
        _print_report(instance_work_dir, model_name, benchmark)
        print(f"[Instance {instance_id}] DONE: {benchmark}")
        return {"benchmark": benchmark, "status": "success"}
    except Exception as e:
        import traceback

        # region agent log
        _append_debug_log(
            run_id="baseline",
            hypothesis_id="H6",
            location="eval_ifbench.py:168",
            message="eval run_task failed",
            data={
                "instance_id": int(instance_id),
                "status": "error",
                "api_url": api_url,
                "error": str(e),
            },
        )
        # endregion
        print(f"[Instance {instance_id}] FAIL: {benchmark} - {e}")
        traceback.print_exc()
        return {"benchmark": benchmark, "status": "error", "error": str(e)}


def _print_report(work_dir: str, model_name: str, benchmark_name: str):
    """Read and print the evalscope report JSON."""
    for report_path in sorted(Path(work_dir).rglob(f"reports/{model_name}/{benchmark_name}*.json")):
        try:
            with open(report_path) as f:
                report = json.load(f)
        except Exception:
            continue

        score = report.get("score", 0)
        metrics = report.get("metrics", [])
        evaluated = metrics[0].get("num", 0) if metrics else 0

        print(f"\n{'=' * 60}")
        print(f"  {benchmark_name} Report")
        print(f"{'=' * 60}")
        print(f"  Score (primary): {score:.4f}")
        for m in metrics:
            print(f"  {m.get('name', '?')}: {m.get('value', 0):.4f} (n={m.get('num', 0)})")
        print(f"  Evaluated: {evaluated} samples")
        print(f"  Report: {report_path}")
        print(f"{'=' * 60}\n")
        return


def parse_api_urls(raw: str) -> list[str]:
    """Parse comma-separated URLs or ports into a list of base URLs."""
    urls = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("http://") or part.startswith("https://"):
            base = part.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            urls.append(base)
        else:
            urls.append(f"http://localhost:{int(part)}/v1")
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IFBench/IFEval on a running sglang server using evalscope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="sglang OpenAI-compatible base URL(s), comma-separated. " "E.g. http://host:30000/v1 or just 30000",
    )
    parser.add_argument("--model-name", required=True, help="Served model name")
    parser.add_argument("--api-key", default="EMPTY", help="API key (default: EMPTY)")
    parser.add_argument(
        "--benchmark",
        default="ifbench",
        choices=["ifbench", "ifeval"],
        help="Benchmark to run (default: ifbench)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/<model>/raw)")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode")

    # Eval control
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None, help="Max samples (for debugging)")

    args = parser.parse_args()

    api_urls = parse_api_urls(args.api_url)
    if not api_urls:
        print("ERROR: --api-url is empty")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join("results", args.model_name, "raw")
    os.makedirs(output_dir, exist_ok=True)

    gen_config = _build_generation_config(args)

    print(f"{'=' * 60}")
    print(f"  Benchmark:  {args.benchmark}")
    print(f"  Model:      {args.model_name}")
    print(f"  Instances:  {len(api_urls)} ({api_urls})")
    print(f"  Output:     {output_dir}")
    print(f"  Sampling:   temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"  Thinking:   {args.enable_thinking}")
    print(f"{'=' * 60}")

    if len(api_urls) == 1:
        result = run_on_instance(
            instance_id=0,
            api_url=api_urls[0],
            model_name=args.model_name,
            api_key=args.api_key,
            generation_config=gen_config,
            output_dir=output_dir,
            batch_size=args.batch_size,
            timeout=args.timeout,
            limit=args.limit,
            benchmark=args.benchmark,
        )
        if result["status"] != "success":
            sys.exit(1)
    else:
        # Parallel evaluation across multiple instances
        results = []
        with ProcessPoolExecutor(max_workers=len(api_urls)) as executor:
            futures = {}
            for i, url in enumerate(api_urls):
                future = executor.submit(
                    run_on_instance,
                    instance_id=i,
                    api_url=url,
                    model_name=args.model_name,
                    api_key=args.api_key,
                    generation_config=gen_config,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    limit=args.limit,
                    benchmark=args.benchmark,
                )
                futures[future] = i

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"benchmark": args.benchmark, "status": "error", "error": str(e)})

        num_fail = sum(1 for r in results if r["status"] != "success")
        if num_fail > 0:
            print(f"\nWARNING: {num_fail}/{len(results)} instance(s) failed")
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
