# MathVision VLM RL (Qwen3.5-4B, refactored rollout)

Single-turn GRPO on the [MathLLMs/MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)
visual-math benchmark with **Qwen3.5-4B**, built on the **refactored rollout**.

- **Refactored rollout** — `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` switches the
  default rollout to `InferenceRolloutFn`, and the custom generate hook
  ([`rollout.py`](./rollout.py)) uses the new `GenerateFnInput -> GenerateFnOutput`
  signature instead of the legacy `(args, sample, sampling_params)` form.
- **VLM via bridge mode** — `Qwen3.5-4B` (`Qwen3_5ForConditionalGeneration`) is a
  VLM: a GatedDeltaNet text decoder plus a vision tower. The vision tower is
  materialized on the training side with `--megatron-to-hf-mode bridge`
  (`AutoBridge` builds a `Qwen35VLModelProvider`). The per-model `--spec` in
  `scripts/models/qwen3.5-4B.sh` is only used on the non-bridge path and is
  ignored here.

## Data preparation

MathVision ships only `test` (3040) and `testmini` (304) splits and has **no
training split**, so we use `test` as the RL prompt source and `testmini` as the
held-out eval set.

```bash
hf download --repo-type dataset MathLLMs/MathVision --local-dir /root/datasets/MathVision
python examples/mathvision/prepare_data.py        # -> /root/datasets/mathvision_miles/{train,eval}.parquet
```

[`prepare_data.py`](./prepare_data.py) rewrites each row into the columns the
loader expects:

| Column | Used by | Content |
|---|---|---|
| `problem` | `--input-key` | question text with a single `<image>` placeholder + answer-format instruction |
| `answer` | `--label-key` | gold answer — a letter for multiple-choice, a value for free-form |
| `images` | `--multimodal-keys '{"image": "images"}'` | `[abs_path]` to the decoded figure on disk |
| `metadata` | `--metadata-key` + reward | `{id, options, is_choice, subject, level}` |

MathVision references images with `<imageN>` tags but provides exactly one
decoded figure per row, so rows that reference more than one image are dropped
(~11%); the rest are normalized to a single `<image>` placeholder.

## Reproduce

```bash
# smoke test: tiny batch, short responses, 2 steps + 1 eval (~minutes on 8xH200)
MILES_SCRIPT_SMOKE=1 python examples/mathvision/run_mathvision.py

# fuller run
python examples/mathvision/run_mathvision.py

# with wandb
WANDB_API_KEY=... python examples/mathvision/run_mathvision.py
```

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MILES_SCRIPT_SMOKE` | `0` | `1` for the small smoke config |
| `MILES_SCRIPT_NUM_GPUS` | `8` | Number of GPUs |
| `MILES_SCRIPT_EXTERNAL_RAY` | `0` | Use an external Ray cluster (`1` to enable) |
| `WANDB_API_KEY` | unset | Enables wandb logging when set |

## Reward

[`reward.py`](./reward.py) (`--custom-rm-path`) scores by row type recorded in
`metadata`:

- **multiple-choice** (`is_choice`): `1.0` when the selected option letter matches
  the gold letter. The model may emit the letter or the option content; content is
  mapped back to its letter before comparison.
- **free-form**: `1.0` when the boxed answer matches the gold value via the shared
  math grader (`grade_answer_verl`, numeric / sympy equivalence).

## Notes

- **Qwen3.5 needs `bshd`, not packed sequences** — under `--megatron-to-hf-mode bridge`
  the text decoder uses Megatron's GatedDeltaNet, which rejects packed (`thd`)
  sequences (`GDN does not support packed sequence for now`). The launcher therefore
  uses `--qkv-format bshd` + `--micro-batch-size 1` instead of `--use-dynamic-batch-size`.
- **SGLang TP** — Qwen3.5 produces garbage output under SGLang TP>1
  ([sgl-project/sglang#21039](https://github.com/sgl-project/sglang/issues/21039)),
  so the rollout uses one GPU per engine (`--rollout-num-gpus-per-engine 1`).
  Megatron training TP (`--tensor-model-parallel-size`) is independent.
- **Training on `test`** — MathVision has no train split; training on `test` and
  evaluating on `testmini` is fine for this example but is not a clean
  train/test separation for reporting benchmark numbers.
