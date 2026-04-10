#!/bin/bash
###############################################################################
# run_ifbench.sh — IFBench evaluation against a running sglang server
#
# Requires: pip install evalscope
#
# Usage:
#   bash agent/eval_bench/run_ifbench.sh \
#       --api-url http://10.0.0.1:30000/v1 \
#       --model-name Qwen3.5-35B-A3B
#
#   # Multiple servers
#   bash agent/eval_bench/run_ifbench.sh \
#       --api-url http://10.0.0.1:30000/v1,http://10.0.0.2:30000/v1 \
#       --model-name Qwen3.5-35B-A3B
#
#   # With thinking mode and custom params
#   bash agent/eval_bench/run_ifbench.sh \
#       --api-url http://localhost:30000/v1 \
#       --model-name Qwen3.5-35B-A3B \
#       --enable-thinking
#
# All arguments are forwarded to eval_ifbench.py. Run with --help for details.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/eval_ifbench.py" ]; then
    REPO_DIR="$SCRIPT_DIR"
else
    REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

cd "$REPO_DIR"
# Default to the last 4 GPUs unless user already set it.
: "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"
export CUDA_VISIBLE_DEVICES
# region agent log
python3 - "$REPO_DIR" "$*" <<'PY'
import json
import os
import sys
import time

log_path = "/root/miles/.cursor/debug-b8af16.log"
payload = {
    "sessionId": "b8af16",
    "runId": "baseline",
    "hypothesisId": "H7",
    "location": "run_ifbench.sh:39",
    "message": "run_ifbench entrypoint reached",
    "data": {
        "repo_dir": sys.argv[1],
        "cwd": os.getcwd(),
        "args": sys.argv[2],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    "timestamp": int(time.time() * 1000),
}
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
# endregion
exec python3 "$REPO_DIR/eval_ifbench.py" "$@"
