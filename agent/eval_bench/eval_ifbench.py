#!/usr/bin/env python3
import runpy
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(repo_root / "eval_ifbench.py"), run_name="__main__")
