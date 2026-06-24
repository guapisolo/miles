"""Qwen3.6-27B VLM RL on MathLLMs/MathVision, on the refactored rollout.

Refactored rollout: ``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`` switches the
default rollout to ``InferenceRolloutFn``, and the custom generate hook
(``examples/mathvision/rollout.py``) uses the new ``GenerateFnInput ->
GenerateFnOutput`` signature.

Qwen3.6-27B is a dense VLM (``Qwen3_5ForConditionalGeneration``: GatedDeltaNet
text decoder + vision tower). The model is built by megatron.bridge's OWN
implementation via ``--megatron-to-hf-mode bridge`` (``AutoBridge`` ->
``Qwen3VLModelProvider`` -> ``provider.provide()``); the per-model ``--spec``
from ``scripts/models/qwen3.6-27B.sh`` is parsed but UNUSED on the bridge path.

The vision tower is frozen via ``--freeze-vision-model`` (the bridge provider's
native ``freeze_vision_model`` flag, which calls ``model.freeze()``); the dense
provider does not freeze it by default. The non-smoke config reproduces the
customer's train<->rollout logprob-diff setup (32k context, lr 2e-6, ViT frozen)
scaled to 8xH200 colocate.

Usage::

    # smoke test (tiny batch, short responses, a few steps)
    MILES_SCRIPT_SMOKE=1 python examples/mathvision/run_mathvision.py

    # formal repro run (32k, ViT frozen, reduced batch, ~100 steps)
    python examples/mathvision/run_mathvision.py
"""

import os

from miles.utils.external_utils.command_utils import execute_train

MODEL_NAME = "Qwen3.6-27B"
HF_CKPT = f"/personal/models/{MODEL_NAME}"
MEGATRON_MODEL_TYPE = "qwen3.6-27B"  # scripts/models/qwen3.6-27B.sh (--spec ignored under bridge)

DATA_ROOT = "/root/datasets/mathvision_miles"
TRAIN_DATA = os.path.join(DATA_ROOT, "train.parquet")
EVAL_DATA = os.path.join(DATA_ROOT, "eval.parquet")
# Checkpoints land on /personal (big disk); 27B full ckpt (weights+optimizer) is large.
CKPT_DIR = "/personal/solo-logs/miles/repro-logprob-blowup/ckpts_27b"

NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "8"))
SMOKE = os.environ.get("MILES_SCRIPT_SMOKE", "0") == "1"


def prepare():
    if not os.path.isdir(HF_CKPT):
        raise FileNotFoundError(f"Missing {HF_CKPT}. Run: hf download Qwen/{MODEL_NAME} --local-dir {HF_CKPT}")
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Missing {TRAIN_DATA}. Run: python examples/mathvision/prepare_data.py")


def execute():
    if SMOKE:
        # train slice keeps the smoke run cheap; data path supports `@[start:end]`.
        train_path = f"{TRAIN_DATA}@[0:64]"
        num_rollout, rollout_bs, n_samples, gbs = 2, 8, 4, 32
        max_resp = 1024
        save_args = ""  # smoke: don't write a (huge) 27B checkpoint
        eval_args = (
            "--eval-interval 2 "
            f"--eval-prompt-data mathvision {EVAL_DATA}@[0:8] "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )
    else:
        # Customer-aligned repro of the train<->rollout logprob-diff blowup, scaled to
        # fit 8xH200 colocate: reduced batch (gbs 64) at the customer's 32k context.
        train_path = TRAIN_DATA
        num_rollout, rollout_bs, n_samples, gbs = 100, 8, 8, 64
        max_resp = 32768
        save_args = f"--save {CKPT_DIR} --save-interval 20 "
        eval_args = (
            "--eval-interval 25 "
            f"--eval-prompt-data mathvision {EVAL_DATA}@[0:64] "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 4096 "
            "--eval-top-k 1 "
        )

    ckpt_args = f"--hf-checkpoint {HF_CKPT} "

    rollout_args = (
        f"--prompt-data {train_path} "
        "--input-key problem "
        "--label-key answer "
        "--metadata-key metadata "
        '--multimodal-keys \'{"image": "images"}\' '
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": true}' "
        "--rollout-shuffle "
        "--custom-generate-function-path examples.mathvision.rollout.generate "
        "--custom-rm-path examples.mathvision.reward.reward "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_bs} "
        f"--n-samples-per-prompt {n_samples} "
        f"--rollout-max-response-len {max_resp} "
        "--rollout-temperature 1 "
        "--rollout-top-p 0.95 "
        "--rollout-top-k 20 "
        f"--global-batch-size {gbs} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 2e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # CPU Adam (Miles 27B recipe): offload the ~40GB/GPU optimizer state to host RAM
        # so 27B@32k fits on 8xH200. --use-precision-aware-optimizer is a required companion;
        # --overlap-cpu-optimizer-d2h-h2d overlaps the H2D/D2H transfers to hide latency.
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        # SGLang TP>1 produces garbage for Qwen3.5 (sgl-project/sglang#21039),
        # so keep one GPU per rollout engine.
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.8 "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load {HF_CKPT} "
        "--megatron-to-hf-mode bridge "
        # Freeze the ViT (customer setup): bridge provider's native freeze_vision_model,
        # which the dense Qwen3.5 provider leaves off by default.
        "--freeze-vision-model "
        # 27B @ 32k on 8xH200 memory budget. The OOM bottleneck is the 32k forward/backward
        # activations + [32k x 248k-vocab] logits (NOT weights -> TP8/PP2 don't help; TP8 also
        # breaks GDN at num_query_groups=4). Miles's documented 27B recipe (TP4 + CPU Adam +
        # mem 0.5) is validated at ~8k context; 32k is ~4x the activations, so we add CP2 to
        # split the 32k sequence across 2 GPUs (GDN supports CP via mamba_context_parallel;
        # CP splits the sequence, not heads, so it avoids the TP8 GQA-reshape break).
        # TP4 x CP2 = 8 GPUs (PP1, DP1). CPU Adam (optimizer_args) frees the optimizer too.
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        # VL models assert calculate_per_token_loss under CP>1 (see model_provider.py bridge branch).
        "--calculate-per-token-loss "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # Qwen3.5's GatedDeltaNet (linear-attn) in bridge mode uses Megatron's GDN,
        # which rejects packed (thd) sequences; use padded bshd + micro-batch-size.
        "--qkv-format bshd "
        "--micro-batch-size 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    misc_args = (
        "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " f"--rollout-num-gpus {NUM_GPUS} " "--colocate "
    )

    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project miles-mathvision "
            "--wandb-group qwen3.6-27b "
            f"--wandb-key '{key}' "
            "--disable-wandb-random-suffix "
        )
        if (key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    train_args = (
        f"{ckpt_args}{rollout_args}{eval_args}{grpo_args}{optimizer_args}"
        f"{sglang_args}{megatron_args}{save_args}{misc_args}{wandb_args}"
    )

    execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here — it is
            # incompatible with SGLang's TorchMemorySaver under --colocate and kills the
            # rollout engine. TP8 alone shards the 27B@32k memory enough to fit.
            **({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
