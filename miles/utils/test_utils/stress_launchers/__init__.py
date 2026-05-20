"""Subprocess launchers for the CPU-only session-server stress harness.

Exposes :class:`StressProcessTrio` (the three-process orchestrator used by
``scripts/tools/stress_session_server_cpu.py``) plus the subprocess entry
points it spawns.

The harness deliberately puts mock SGLang and the real ``SessionServer``
into separate Python interpreters so the driver's RSS / CPU / event-loop
attribution stays clean; running them in the same process pollutes the
signal with GIL contention and shared FastAPI internals.
"""

from miles.utils.test_utils.stress_launchers.manager import StressProcessTrio, StressR3Spec

__all__ = ["StressProcessTrio", "StressR3Spec"]
