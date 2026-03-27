# Generate Hub Rollout Design

This document explains how `generate_hub` fits into the rollout stack, what
responsibilities belong to each layer, and how the fully-async rollout mode
works end-to-end.

## 1. Scope and Terminology

There are two distinct abstractions:

1. **`GenerateFn`** — per-sample generation semantics.
   - Input: one `Sample`, generation state, sampling params.
   - Output: one `Sample` (or a list when multi-sample).
   - Lives in `generate_hub/`.

2. **`RolloutFn`** — rollout-step scheduling.
   - Input: rollout id, data source, rollout config.
   - Output: grouped samples for one training step.
   - Lives in `inference_rollout/`.

The key rule: `generate_hub` owns "how to turn one prompt into model calls".
Rollout code owns "how to schedule many prompts, filter, abort, and deliver
enough data for training".

## 2. Layering

```text
RolloutManager
  → InferenceRolloutFn._call_train
    → generate_rollout_async(continuous=True/False)
      → submit_generate_tasks
        → generate_and_rm_group
          → generate_and_rm
            → generate_hub.<variant>.generate
```

| Layer | Responsibility |
|---|---|
| `RolloutManager` | Lifecycle, servers, data source, rollout/eval entry |
| `InferenceRolloutFn` | Build `GenerateState`, route train vs eval |
| `generate_rollout_async` | Submit groups, collect, filter, abort |
| `generate_and_rm_group` | Fan out per-sample generation within a prompt group |
| `generate_hub.*.generate` | Backend-specific generation logic |

## 3. Standard vs Fully-Async Rollout

Both modes use the same function: `generate_rollout_async`. The only
difference is the **submission policy**, controlled by `continuous: bool`.

### 3.1 Standard mode (`continuous=False`)

```python
while len(data) + len(pendings) < target_data_size:
    samples = data_source(batch_size)
    pendings.update(submit_generate_tasks(state, samples))
```

Submits just enough tasks so that `collected + in_flight >= target`.
Conservative — once quota is reached, stops submitting and just waits.

### 3.2 Fully-async mode (`continuous=True`)

```python
while len(pendings) < args.over_sampling_batch_size:
    samples = data_source(batch_size)
    pendings.update(submit_generate_tasks(state, samples))
```

Keeps exactly `over_sampling_batch_size` tasks in-flight at all times,
regardless of how many samples are already collected. As one completes, a new
one is submitted immediately.

### 3.3 Everything else is shared

Both modes share the same:

- **Collection loop**: `asyncio.wait(FIRST_COMPLETED)` → process done tasks → dynamic filter
- **Abort**: once `target_data_size` valid samples are collected, remaining pendings go through `abort()`
- **Post-processing**: sort, `rollout_sample_filter`, `rollout_all_samples_process`, `state.reset()`

This is intentional. The fully-async feature is a one-line policy change, not
a separate code path.

## 4. End-to-End Code Path (fully-async)

### Step 1: CLI flag

```
--fully-async-rollout    (arguments.py)
```

### Step 2: InferenceRolloutFn routes to generate_rollout_async

```python
# inference_rollout_common.py
async def _call_train(self, input):
    output, aborted = await generate_rollout_async(
        self.state,
        input.rollout_id,
        self.data_source.get_samples,
        continuous=getattr(self.state.args, "fully_async_rollout", False),
    )
    self.data_source.add_samples(aborted)
    return output
```

No separate method, no persistent worker object. Just one keyword argument.

### Step 3: The main loop

```
┌─────────────────────────────────────────────────────┐
│ while len(data) < target_data_size:                 │
│                                                     │
│   1. SUBMIT: keep pendings at over_sampling_batch_  │
│      size by pulling from data_source               │
│                                                     │
│   2. WAIT: asyncio.wait(FIRST_COMPLETED)            │
│                                                     │
│   3. PROCESS: for each done task:                   │
│      - run dynamic_filter                           │
│      - if passes → append to data[]                 │
│      - else → record metric, discard                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Step 4: Abort and cleanup

Once `len(data) == rollout_batch_size`:

1. `abort(state, pendings, rollout_id)` — sends abort to all SGLang workers,
   awaits pending tasks, optionally collects partial samples
2. Sort data by sample index
3. Run `rollout_sample_filter` and `rollout_all_samples_process` hooks
4. `state.reset()` to clear abort flag for next rollout

### Step 5: Caller gets result

`_call_train` receives `(RolloutFnTrainOutput, aborted_samples)`, feeds
aborted samples back to `data_source.add_samples()` for reuse.

## 5. Why Fully-Async Helps

Standard mode's submission policy is `collected + pending >= target`. When
generation latency has high variance (e.g. agentic tool calls where some
prompts take 10x longer), this means:

- Early tasks finish, but no new tasks are submitted because the quota was met
- The pipeline stalls waiting for the slow tail
- GPU utilization drops

Fully-async mode decouples submission from collection. The in-flight count
stays constant, so fast-finishing slots immediately get refilled. The
tradeoff is that more work gets aborted at the end, but for long-tail
workloads the throughput improvement far outweighs the wasted compute.

## 6. Staleness and On-Policy Correctness

Staleness control is **not** a rollout-layer concern. The rollout function
generates samples within a single training step using the current policy
weights, then aborts everything before weight update. Samples never cross a
weight-update boundary.

If finer-grained staleness control is needed (e.g. rejecting samples that
were generated too many steps ago in a partial-rollout setting), that belongs
in the **data source** layer, which tracks sample provenance and decides what
to serve.

## 7. The Old Prototype (`examples/fully_async/`)

`examples/fully_async/fully_async_rollout.py` is an earlier proof-of-concept
with a different architecture:

- Global singleton worker in a separate thread with its own event loop
- No pause-before-weight-update boundary
- Queue contents can survive across rollout calls (policy leak)
- No dynamic filter integration
- No eval support

It remains as an example only. The production path is the `continuous=True`
parameter on the standard `generate_rollout_async`.

## 8. `generate_hub` Variants

All three existing generate variants work identically under both standard and
fully-async mode, because the scheduling change is above them in the stack.

| Variant | Description | Supports partial rollout |
|---|---|---|
| `single_turn` | One `/generate` call per sample | Yes |
| `multi_turn` | Multiple model turns with tool observations | No |
| `agentic_tool_call` | Wraps user-supplied async agent, traces session records | Yes (via session records) |

## 9. Design Rule

If a feature answers "how should one sample talk to the model?" →
`generate_hub`.

If it answers "how should many samples be scheduled and aligned with
training-step boundaries?" → `inference_rollout`.
