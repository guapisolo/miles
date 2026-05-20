# CPU Stress Sweep Report

- Matrix: N ∈ [4, 16] × trajectory_length ∈ [256, 1024]
- Tokens per turn: 512
- R3 inject: True (num_layers=28, topk=8)
- HF checkpoint: `Qwen/Qwen3-0.6B`
- Tracemalloc: False

## Cell Summary

| Cell | Status | Load (s) | Wall (s) | P50 (ms) | P99 (ms) | Driver RSS | Session RSS | Mock RSS | Replay |
|------|--------|----------|----------|----------|----------|-----------|-------------|----------|--------|
| N4_L256 | pass | 0.08 | 19.29 | 70.1 | 70.7 | 56.9 MiB | 1023.0 MiB | 1.0 GiB | — |
| N4_L1024 | pass | 0.11 | 19.95 | 43.0 | 66.5 | 66.0 MiB | 1023.0 MiB | 1020.5 MiB | — |
| N16_L256 | pass | 0.17 | 19.85 | 116.1 | 156.2 | 70.2 MiB | 1022.6 MiB | 1.0 GiB | — |
| N16_L1024 | pass | 0.31 | 20.48 | 132.7 | 197.3 | 85.9 MiB | 1.0 GiB | 1022.9 MiB | — |

## Failure Analysis

All cells passed; no failure analysis required.

## Methodology Notes

- Each cell starts a fresh `StressProcessTrio`; `SessionRegistry` is empty per cell.
- Verifier is always on and runs in streaming mode (per-session GET → verify → drop) so driver RSS is O(num_sessions) rather than O(num_sessions × trajectory_length).
- Latency P50/P99 is driver-side `time.perf_counter()` around `await append_turn`. No py-spy / flamegraph / async profiler is installed; AC-11.1 mandates rough numbers only.
- The plan's headline 5×3 matrix ({64,128,256,512,1024} × {5k,20k,50k}) requires ≥ 150 GB host RAM (R3 alone ≈ 50 GB at 1024×50k×28×8 int32 plus FastAPI buffering); run on a dedicated host via `--matrix-config`.
