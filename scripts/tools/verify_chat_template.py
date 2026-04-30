#!/usr/bin/env python3
"""One-click verification: is a chat template append-only after last user message?

Usage examples::

    # Verify a local .jinja template file
    python scripts/tools/verify_chat_template.py --template path/to/template.jinja

    # Verify a HuggingFace model's chat template
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B

    # Resolve a bundled fixed template for a TITO tokenizer family
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \\
        --tito-model qwen3 --tito-allowed-append-roles tool

    # Restrict which append roles the session is allowed to use (tool is implicit)
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \\
        --tito-allowed-append-roles user

    # Run thinking cases: off (default) / on / both
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --thinking both
"""

from __future__ import annotations

import argparse
import json
import sys

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType


def _load_template_from_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load_template_from_model(
    model_id: str,
    *,
    tito_model: str | None,
    allowed_roles: list[str],
) -> tuple[str, str, dict]:
    """Load chat template for a HF model.

    Returns ``(template_str, source_description, resolved_kwargs)``. When the
    matched fixed-template row registers extra kwargs, the caller is expected
    to merge ``resolved_kwargs`` into its own template kwargs (with explicit
    user values winning on conflict).
    """
    if tito_model:
        from miles.utils.chat_template_utils import resolve_fixed_chat_template

        fixed_path, resolved_kwargs = resolve_fixed_chat_template(tito_model, allowed_roles)
        if fixed_path:
            return _load_template_from_file(fixed_path), f"fixed template: {fixed_path}", resolved_kwargs
        # No template path matched, but the row may still carry kwargs (e.g. an
        # HF-native + clear_thinking=False fix).  Fall through to HF-native
        # loading and bubble the kwargs up.
    else:
        resolved_kwargs = {}

    from miles.utils.chat_template_utils.template import load_hf_chat_template

    return load_hf_chat_template(model_id), f"HuggingFace: {model_id}", resolved_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a chat template is append-only after last user message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--template",
        metavar="PATH",
        help="Path to a local .jinja chat template file.",
    )
    source.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID (e.g. Qwen/Qwen3-0.6B).",
    )

    parser.add_argument(
        "--tito-model",
        choices=[t.value for t in TITOTokenizerType],
        default=None,
        help=(
            "When using --model, look up a bundled fixed template registered for this "
            "TITO tokenizer family under the given --tito-allowed-append-roles surface. "
            "Falls back to the HF default chat template when no entry is registered."
        ),
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        default=["tool"],
        choices=["tool", "user", "system"],
        metavar="ROLE",
        help=(
            "Roles the session may append after an assistant turn.  'tool' is "
            "implicitly always allowed (listing it is fine).  Trajectories that "
            "require roles outside this set are skipped.  Default: tool."
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "on", "both"],
        default="on",
        help=(
            "Thinking-mode filter.  off: non-thinking trajectories only.  "
            "on: thinking trajectories with enable_thinking=True.  "
            "both: non-thinking + thinking with enable_thinking={True,False}.  "
            "Default: on."
        ),
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default={},
        metavar="JSON",
        help=(
            "Extra kwargs threaded into the chat template on every render, as a "
            "JSON object (e.g. '{\"clear_thinking\": false}' for GLM).  Same "
            "convention as --apply-chat-template-kwargs in the training entrypoint."
        ),
    )

    args = parser.parse_args()

    extra_template_kwargs = args.chat_template_kwargs

    # ``--tito-allowed-append-roles`` lists the *optional* extra roles; ``tool``
    # is implicit for tool-capable agentic workflows and unioned in here so both
    # the fixed-template lookup and the trajectory filter see the same surface.
    allowed_roles = set(args.tito_allowed_append_roles) | {"tool"}

    # ── Load template ──────────────────────────────────────────────────
    if args.template:
        chat_template = _load_template_from_file(args.template)
        source_desc = f"file: {args.template}"
    else:
        chat_template, source_desc, resolved_kwargs = _load_template_from_model(
            args.model,
            tito_model=args.tito_model,
            allowed_roles=sorted(allowed_roles),
        )
        # Merge auto-resolved kwargs into the user's --chat-template-kwargs.
        # Explicit user values win on conflict; only keys the user did not set
        # are auto-filled.
        for key, value in resolved_kwargs.items():
            if key in extra_template_kwargs:
                continue
            extra_template_kwargs[key] = value
            print(f"Auto-set --chat-template-kwargs {key}={value!r} (from --tito-model={args.tito_model})")

    from miles.utils.test_utils.chat_template_verify import ALL_CASES, check_coverage, run_all_checks, select_cases

    is_thinking_filter = {"off": False, "on": True, "both": None}[args.thinking]
    selected = select_cases(allowed_append_roles=allowed_roles, is_thinking=is_thinking_filter)

    print(f"Template source:       {source_desc}")
    print(f"Allowed append roles:  {sorted(allowed_roles)}")
    print(f"Thinking mode:         {args.thinking}")
    if extra_template_kwargs:
        print(f"Template kwargs:       {extra_template_kwargs}")
    print(f"Selected trajectories: {len(selected)} of {len(ALL_CASES)} (after filtering)")
    print()

    # Global coverage sanity check — reports gaps in the mock-trajectory pool,
    # not gaps caused by the current CLI flags.  A gap here means some CLI
    # setting exercises no trajectory; fixing it requires adding a trajectory
    # in mock_trajectories.py.
    coverage = check_coverage()
    if coverage.missing:
        print("Trajectory coverage gaps ((thinking, append_roles \\ {tool}) with no trajectory):")
        for is_thinking, roles in coverage.missing:
            label = "thinking    " if is_thinking else "non-thinking"
            roles_str = "{" + ", ".join(roles) + "}" if roles else "{}"
            print(f"  - {label}  x  {roles_str}")
        print()

    # ── Run verification ───────────────────────────────────────────────
    results = run_all_checks(
        chat_template,
        allowed_append_roles=allowed_roles,
        thinking=args.thinking,
        extra_template_kwargs=extra_template_kwargs,
    )

    # ── Print results ──────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    max_name_len = max((len(r.case_name) for r in results), default=0)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = f"  [{status}] {r.case_name:<{max_name_len}}"
        if r.error:
            first_line = r.error.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            line += f"  -- {first_line}"
        print(line)

    print()
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")

    if failed:
        print("\nVerdict: FAIL - template is NOT append-only after last user message")
        return 1
    else:
        print("\nVerdict: PASS - template IS append-only after last user message")
        return 0


if __name__ == "__main__":
    sys.exit(main())
