#!/usr/bin/env python3
"""Standalone CPU-only stress driver for the Miles session server.

Thin CLI wrapper; the actual logic lives in
:mod:`miles.utils.test_utils.stress_cli` so pytest tests can import and
exercise it directly. See that module for the full description.
"""

from __future__ import annotations

import os

# Belt: CPU enforcement at the CLI layer.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from miles.utils.test_utils.stress_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
