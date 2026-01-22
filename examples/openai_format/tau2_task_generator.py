"""
Generate tau2-bench task ids into a jsonl file for miles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.run import get_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tau2-bench task ids for miles.")
    parser.add_argument("--task-set-name", default="retail", help="Task set name registered in tau2.")
    parser.add_argument("--task-split-name", default="base", help="Task split name.")
    parser.add_argument("--output-path", default="/root/tau2-bench/tasks.jsonl", help="Output jsonl path.")
    args = parser.parse_args()

    tasks = get_tasks(task_set_name=args.task_set_name, task_split_name=args.task_split_name)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps({"task_id": task.id}) + "\n")


if __name__ == "__main__":
    main()
