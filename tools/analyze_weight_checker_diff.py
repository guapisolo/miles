#!/usr/bin/env python
import argparse
import collections
import pathlib
import re
import sys


DIFF_LINE_PATTERN = re.compile(r"name=([^\s]+)\s+max_abs_err=([^\s]+)")


def _parse_run_arg(raw: str) -> tuple[str, pathlib.Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --run '{raw}'. Expected LABEL=PATH.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --run '{raw}'. Expected LABEL=PATH.")
    return label, pathlib.Path(path)


def _iter_diff_names(log_path: pathlib.Path):
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Filter to actual tensor compare failure lines.
            if "get_tensor_info(actual)" not in line:
                continue
            match = DIFF_LINE_PATTERN.search(line)
            if match is None:
                continue
            yield match.group(1)


def _prefix_of(name: str) -> str:
    return name.split(".", 1)[0] if "." in name else name


def analyze_log(log_path: pathlib.Path) -> dict:
    names = list(_iter_diff_names(log_path))
    unique_names = set(names)
    prefix_counter = collections.Counter(_prefix_of(name) for name in unique_names)
    non_visual = sorted(name for name in unique_names if not name.startswith("visual."))

    return {
        "diff_lines": len(names),
        "unique_names": unique_names,
        "visual_unique": sum(1 for n in unique_names if n.startswith("visual.")),
        "non_visual_unique": len(non_visual),
        "non_visual_names": non_visual,
        "prefix_counter": prefix_counter,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze --check-weight-update-equal logs and summarize diff tensor names. "
            "Each run is provided as LABEL=PATH."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in LABEL=PATH format. Can be passed multiple times.",
    )
    parser.add_argument(
        "--non-visual-limit",
        type=int,
        default=200,
        help="Max non-visual diff names to print per run (default: 200).",
    )
    parser.add_argument(
        "--show-prefix-breakdown",
        action="store_true",
        help="Print unique-name prefix breakdown for each run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    parsed_runs: list[tuple[str, pathlib.Path]] = []
    for raw in args.run:
        try:
            label, path = _parse_run_arg(raw)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2
        parsed_runs.append((label, path))

    results = {}
    for label, path in parsed_runs:
        results[label] = analyze_log(path)

    print("| Run | diff_lines | unique_diff_names | visual_unique | non_visual_unique |")
    print("|---|---:|---:|---:|---:|")
    for label, _ in parsed_runs:
        res = results[label]
        print(
            f"| {label} | {res['diff_lines']} | {len(res['unique_names'])} | "
            f"{res['visual_unique']} | {res['non_visual_unique']} |"
        )

    print()
    for label, _ in parsed_runs:
        res = results[label]
        print(f"[{label}] non-visual unique diff names ({res['non_visual_unique']}):")
        if res["non_visual_unique"] == 0:
            print("  (none)")
        else:
            for name in res["non_visual_names"][: args.non_visual_limit]:
                print(f"  {name}")
            if res["non_visual_unique"] > args.non_visual_limit:
                print(f"  ... ({res['non_visual_unique'] - args.non_visual_limit} more)")
        print()

    if args.show_prefix_breakdown:
        for label, _ in parsed_runs:
            res = results[label]
            print(f"[{label}] prefix breakdown (unique names):")
            for prefix, count in sorted(res["prefix_counter"].items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {prefix}: {count}")
            print()

    if len(parsed_runs) >= 2:
        ordered_labels = [label for label, _ in parsed_runs]
        common = set.intersection(*(results[label]["unique_names"] for label in ordered_labels))
        print(f"Common unique diff names across all runs: {len(common)}")
        for label in ordered_labels:
            exclusive = results[label]["unique_names"] - common
            print(f"  {label} exclusive unique diff names: {len(exclusive)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
