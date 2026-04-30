# Chat Template Verification

## Background

In agentic workflows (multi-turn tool-calling), miles uses sglang's **pretokenized prefix** mechanism to avoid re-tokenizing the entire conversation history on every turn. This requires the chat template to satisfy an **append-only invariant**: rendering the first *N* messages must produce a string that is an exact prefix of rendering all messages.

Many production chat templates default to behavior that is sensible for online serving but breaks this invariant — truncating historical reasoning at the last user boundary, branching on `loop.last`, or rendering the same turn differently depending on what comes after it. miles ships a **TITO** layer that auto-resolves a registered fixed chat template (and any required Jinja kwargs) per model family, so the same model can be used for both production serving (default behavior) and RL training (append-only behavior) without forking the template manually.

## Concepts

A miles agentic training run with a non-default chat template uses three CLI flags that work together:

| Flag | Description |
| :--- | :--- |
| `--tito-model` | TITO tokenizer family, e.g. `qwen3`, `qwen35`, `qwennext`, `glm47`. Use `default` to keep the model's HF-native chat template untouched. |
| `--tito-allowed-append-roles` | Roles the session may append after assistant turns (any subset of `{tool, user, system}`). `tool` is implicit in every surface. |
| `--apply-chat-template-kwargs` | Extra kwargs threaded into the Jinja render. Auto-merged from registered rows; explicit values win on conflict. |

Each `--tito-model` family registers a tuple of rows of the form `(allowed_roles, template, extra_kwargs)`, where `template` is either a path to a bundled `.jinja` file (a fixed template miles ships) or `None` (use the model's HF-native template).

At training startup miles resolves a row via **smallest-superset lookup**: pick the registered row whose `allowed_roles` is a superset of `--tito-allowed-append-roles` and has the smallest cardinality among all matching rows. If no row matches, or two rows are equally minimal supersets, miles raises `ValueError` at startup.

The matched row's `template` is assigned to `--chat-template-path`, and its `extra_kwargs` are merged per-key into `--apply-chat-template-kwargs` — only keys the user did not set are auto-filled. The chosen path and any auto-filled kwargs are logged so you can confirm what got applied.

## Verifying your model

Before training, run the verifier with the same `(--tito-model, --tito-allowed-append-roles)` pair you plan to train with:

```shell
python scripts/tools/verify_chat_template.py \
    --model <model-id-or-local-path> \
    --tito-model <family> \
    --tito-allowed-append-roles tool [user|system ...] \
    --thinking both
```

| Argument | Description |
| :--- | :--- |
| `--model` | HuggingFace model ID or local checkpoint path. Only the tokenizer files are loaded; weights are not required. |
| `--tito-model` | A registered TITO family. Required. |
| `--tito-allowed-append-roles` | The role surface you plan to train with. Default `tool`. |
| `--thinking {off,on,both}` | Thinking-mode coverage. Default to `both` for any model that supports thinking — bracketing bugs only surface when both modes are exercised against the same trajectory. |

The script exits with code `0` if every selected case passes, non-zero otherwise.

## Currently supported families

| TITO family | Append roles | Auto kwargs | Example models |
| :--- | :--- | :--- | :--- |
| `qwen3` | `{tool}` | — | `Qwen3-4B`, `Qwen3-0.6B` |
| `qwen3` | `{tool, user}` | `clear_thinking=False` | `Qwen3-4B`, `Qwen3-0.6B` |
| `qwen35` | `{tool}` | — | `Qwen3.5-0.8B` |
| `qwen35` | `{tool, user}` | `clear_thinking=False` | `Qwen3.5-0.8B` |
| `qwennext` | `{tool}` | — | `Qwen3-4B-Thinking-2507`, `Qwen3-Next-80B-A3B-Thinking` |
| `qwennext` | `{tool, user}` | `clear_thinking=False` | `Qwen3-4B-Thinking-2507`, `Qwen3-Next-80B-A3B-Thinking` |
| `glm47` | `{tool}` | — | `GLM-4.7-Flash`, `GLM-5` |
| `glm47` | `{tool, user}` | `clear_thinking=False` | `GLM-4.7-Flash`, `GLM-5` |
| `glm47` | `{tool, user, system}` | `clear_thinking=False` | `GLM-4.7-Flash`, `GLM-5` |

Models whose HF-native chat template is already append-only (e.g. `Qwen3-Instruct-2507`, `Qwen3-Next-Instruct`) can use `--tito-model default` and skip the resolver path entirely.

## When verification fails

If `verify_chat_template.py` fails, the chat template does not satisfy the append-only invariant under the requested `--tito-allowed-append-roles` surface. Adjust the chat template (or restrict the role surface) until the verifier passes. The bundled fixed templates under `miles/utils/chat_template_utils/templates/` are reference examples of how supported families resolve this.
