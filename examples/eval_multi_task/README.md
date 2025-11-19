# Multi-Task Evaluation Example

## Configuring `multi_task.yaml`
- `eval.defaults` defines inference parameters shared by every dataset entry. Override them inside an individual dataset block if needed.
- `eval.datasets` enumerates the datasets to evaluate. Each entry should specify:
  - `name`: a short identifier that appears in logs and dashboards.
  - `path`: the path to the dataset JSONL file.
  - `rm_type`: which reward function to use for scoring.
  - `n_samples_per_eval_prompt`: how many candidate completions to generate per prompt.

## IFBench Notes
- When `ifbench` is used, `miles/rollout/rm_hub/ifbench.py` will automatically prepares the scoring environment, so no additional manual setup is required beyond providing the dataset path.

## HLE Notes
- Set your `OPENAI_API_KEY` (and optionally `HLE_JUDGE_MODEL`) before running evaluation. The default judge model is `o3-mini-2025-01-31`.
- When `hle` is configured, `miles/rollout/rm_hub/hle.py` clones the official [centerforaisafety/hle](https://github.com/centerforaisafety/hle) repo, installs `examples/eval_multi_task/requirements_hle.txt`, and imports the OpenAI-based judge.
- Place the evaluation set at `/root/hle/hle.jsonl` (or update `multi_task.yaml`). Each record should contain a `prompt` field (string or chat format) and a reference `label`.
