#!/bin/bash
# Validation script for fully-async rollout pipeline
# Uses Qwen3-0.6B on 2 GPUs (last 2 GPUs via CUDA_VISIBLE_DEVICES)

set -ex
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=6,7

# Clean up any existing processes on these GPUs
ray stop --force 2>/dev/null || true
sleep 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-0.6B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/shared/Qwen3-0.6B
   --ref-load /root/shared/Qwen3-0.6B_torch_dist
)

ROLLOUT_ARGS=(
   --prompt-data /root/shared/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler

   --num-rollout 10
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --over-sampling-batch-size 8
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 8

   # Fully-async pipeline flags
   --fully-async-rollout
   --partial-rollout
   --max-buffer-staleness 3
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --actor-num-nodes 1
   --actor-num-gpus-per-node 1
   --update-weights-interval 1
)

# Start Ray with 2 GPUs
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
