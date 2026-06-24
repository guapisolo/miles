"""Qwen3.5-4B VLM RL on MathLLMs/MathVision, on the refactored rollout.

Refactored rollout: ``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`` switches the
default rollout to ``InferenceRolloutFn``, and the custom generate hook
(``examples/mathvision/rollout.py``) uses the new ``GenerateFnInput ->
GenerateFnOutput`` signature.

Qwen3.5-4B is a VLM (``Qwen3_5ForConditionalGeneration``: GatedDeltaNet text
decoder + vision tower). Its vision tower is materialized on the training side
through ``--megatron-to-hf-mode bridge`` (``AutoBridge`` builds a
``Qwen35VLModelProvider``); the per-model ``--spec`` from
``scripts/models/qwen3.5-4B.sh`` is unused on the bridge path.

Usage::

    # smoke test (tiny batch, short responses, a few steps)
    MILES_SCRIPT_SMOKE=1 python examples/mathvision/run_mathvision.py

    # fuller run
    python examples/mathvision/run_mathvision.py
"""

import os

from miles.utils.external_utils.command_utils import execute_train

MODEL_NAME = "Qwen3.5-4B"
HF_CKPT = f"/root/models/{MODEL_NAME}"
MEGATRON_MODEL_TYPE = "qwen3.5-4B"  # scripts/models/qwen3.5-4B.sh (--spec ignored under bridge)

DATA_ROOT = "/root/datasets/mathvision_miles"
TRAIN_DATA = os.path.join(DATA_ROOT, "train.parquet")
EVAL_DATA = os.path.join(DATA_ROOT, "eval.parquet")

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
        eval_args = (
            "--eval-interval 2 "
            f"--eval-prompt-data mathvision {EVAL_DATA}@[0:8] "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )
    else:
        train_path = TRAIN_DATA
        num_rollout, rollout_bs, n_samples, gbs = 3000, 64, 8, 512
        max_resp = 4096
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data mathvision {EVAL_DATA} "
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
        "--rollout-shuffle "
        "--custom-generate-function-path examples.mathvision.rollout.generate "
        "--custom-rm-path examples.mathvision.reward.reward "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_bs} "
        f"--n-samples-per-prompt {n_samples} "
        f"--rollout-max-response-len {max_resp} "
        "--rollout-temperature 1 "
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
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        # SGLang TP>1 produces garbage for Qwen3.5 (sgl-project/sglang#21039),
        # so keep one GPU per rollout engine.
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load {HF_CKPT} "
        "--megatron-to-hf-mode bridge "
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
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
            "--wandb-group qwen3.5-4b "
            f"--wandb-key '{key}' "
            "--disable-wandb-random-suffix "
        )
        if (key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    train_args = (
        f"{ckpt_args}{rollout_args}{eval_args}{grpo_args}{optimizer_args}"
        f"{sglang_args}{megatron_args}{misc_args}{wandb_args}"
    )

    execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            **({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
