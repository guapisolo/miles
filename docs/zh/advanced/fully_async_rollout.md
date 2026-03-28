# Fully-Async Rollout 实现

## 概述

Fully-async rollout 是一种解耦生成和训练的推理策略。核心思想是**持续维持
最大数量的 in-flight 请求**，在收够 training batch 后 abort 剩余请求，
将 partial samples 缓存复用，并在 weight update 时通过 session server
透明暂停/恢复生成。

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│  Training Loop (train_async.py)                         │
│                                                         │
│  for rollout_id in range(start, num_rollout):           │
│    ┌──────────────────────────────────────────┐         │
│    │ rollout_manager.generate(rollout_id)      │         │
│    │  └─ generate_rollout_async(continuous=T)  │         │
│    │     ├─ 维持 in-flight = over_sampling_bs  │         │
│    │     ├─ 收够 rollout_batch_size → abort    │         │
│    │     └─ partial samples → buffer           │         │
│    └──────────────────────────────────────────┘         │
│                    ↓ samples                            │
│    actor.async_train(samples)                           │
│                    ↓                                    │
│    actor.update_weights()                               │
│      ├─ pause_sessions()  ← gate 关门                   │
│      ├─ sync weights to SGLang                          │
│      └─ resume_sessions() ← health check → gate 开门    │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Continuous 提交模式

**文件**: `miles/rollout/inference_rollout/inference_rollout_train.py`

标准模式（`continuous=False`）在 `collected + pending < target` 时才提交新请求，
有 long-tail stall 问题。Continuous 模式始终维持 `pending == over_sampling_batch_size`：

```python
if continuous:
    while len(pendings) < args.over_sampling_batch_size:
        samples = data_source(args.over_sampling_batch_size, rollout_id=rollout_id)
        pendings.update(submit_generate_tasks(state, samples))
```

当 `len(data) >= target_data_size` 时触发 abort，收集 partial samples。

### 2. Abort 与 Partial Sample 收集

**文件**: `miles/rollout/inference_rollout/inference_rollout_train.py`

收够 samples 后，`abort()` 做三件事：
1. 设 `state.aborted = True`，阻止新请求进入生成
2. 向所有 SGLang worker 发 `POST /abort_request {"abort_all": True}`
3. 等待所有 pending tasks 完成，收集带 response 的 partial samples

```python
async def abort(state, pendings, rollout_id):
    state.aborted = True
    # abort SGLang in-flight requests
    urls = await get_worker_urls(args)
    await asyncio.gather(*[post(url + "/abort_request", ...) for url in urls])

    aborted_samples = []
    async for group in as_completed_async(pendings):
        if not args.partial_rollout:
            continue
        for sample in group:
            if sample.response and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)
    return aborted_samples
```

Partial samples 被打上 `start_rollout_id` 元数据，标记它们是哪个 rollout 生成的。

### 3. Buffer 缓存与 Staleness 控制

**文件**: `miles/rollout/data_source.py`

`RolloutDataSourceWithBuffer` 维护一个 buffer，abort 收集的 partial samples
通过 `add_samples()` 回注 buffer，下一轮 `get_samples()` 优先从 buffer 取用。

**Staleness 过滤**：当设置 `--max-buffer-staleness N` 时，`pop_first` 会丢弃
`rollout_id - start_rollout_id > N` 的 stale groups：

```python
def pop_first(args, rollout_id, buffer, num_samples):
    max_staleness = getattr(args, "max_buffer_staleness", None)

    if max_staleness is not None and rollout_id is not None:
        fresh = []
        for group in buffer:
            start_id = group[0].metadata.get("start_rollout_id")
            if start_id is not None and (rollout_id - start_id) > max_staleness:
                continue  # 丢弃
            fresh.append(group)
        buffer[:] = fresh

    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

**固定 sample 数量保证**：stale samples 被丢弃后，`get_samples()` 自动从
parent data source（数据集）补足，确保每个 training step 的 sample 数量恒定。

**Metrics 上报**：丢弃数量通过 `rollout/buffer/stale_samples_discarded` 和
`rollout/buffer/remaining_size` 上报到 wandb。如果丢弃率过高，说明
`max_buffer_staleness` 设置不合理或 weight update 太频繁。

### 4. Session Server Abort/Resume

**文件**: `miles/rollout/session/session_server.py`, `sessions.py`

在 agentic 场景（多轮 tool call）中，session server 代理 agent 框架与 SGLang
之间的通信。Weight update 时通过 gate 机制透明暂停：

- **Pause**: 关闭 `resume_event`（gate 关门）+ 向 SGLang 发 abort
- **Resume**: poll `/health` 确认 SGLang ready → 打开 `resume_event`（gate 开门）

`chat_completions` handler 中的 gate 循环：

```python
while True:
    await backend.resume_event.wait()       # paused 时阻塞
    result = await backend.do_proxy(...)
    if not backend.is_paused():
        break                               # 正常，继续
    # abort 到达，丢弃 partial response，等 resume 后重发
```

只丢弃当前 turn 的 partial generation，之前 turn 的 token history 保留。
Resume 后 SGLang 对 prefix 做 re-prefill（利用 KV cache），用新权重
重新生成当前 turn。Agent 框架对此完全无感知。

### 5. Weight Update 集成

**文件**: `miles/backends/megatron_utils/actor.py`

`update_weights()` 方法内部包裹 pause/resume：

```python
def update_weights(self):
    if use_session_server:
        ray.get(self.rollout_manager.pause_sessions.remote())

    self.weight_updater.update_weights()    # 同步权重到 SGLang

    if use_session_server:
        ray.get(self.rollout_manager.resume_sessions.remote())
```

不需要修改 `train_async.py`，pause/resume 的调用完全封装在 actor 内部。

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--fully-async-rollout` | flag | False | 启用 continuous 提交模式 |
| `--partial-rollout` | flag | False | 启用 partial sample 收集和 buffer 复用 |
| `--max-buffer-staleness` | int | None | buffer 中 sample 的最大年龄（rollout_id 差值），超过则丢弃。None = 不过滤 |
| `--mask-offpolicy-in-partial-rollout` | flag | False | 对 partial sample 的 response tokens 做 loss mask |
| `--over-sampling-batch-size` | int | — | continuous 模式下维持的 in-flight 数量 |
| `--use-session-server` | flag | False | 启用 session server，weight update 时触发 pause/resume |

## 数据流

```
Dataset ──get_samples()──→ Buffer (partial samples 优先)
                              │
                              ↓ staleness filter
                              │
    ┌─────────────────────────┘
    │
    ↓
generate_rollout_async(continuous=True)
    │
    ├─ submit → SGLang → collect completed samples
    │                         │
    │                         ↓ len(data) >= target
    │                     abort()
    │                         │
    │                         ├─ completed samples → train
    │                         └─ partial samples ──add_samples()──→ Buffer
    │
    ↓
RolloutFnTrainOutput
    ├─ samples (固定 rollout_batch_size 个)
    └─ metrics (含 stale_samples_discarded)
```

## 跨 Policy 行为

一个 sample 可能跨越多次 weight update：
- Turn 1-3 由 policy v0 生成
- Weight update 发生，session server abort/resume
- Turn 4 由 policy v1 重新生成

这是设计上允许的——不要求严格 on-policy。`mask_offpolicy_in_partial_rollout`
可对旧 policy 生成的 response tokens 做 loss mask。

## 测试

| 测试文件 | 覆盖点 |
|---|---|
| `test_buffer_staleness.py` | staleness 过滤、fresh 保留、disabled 默认、混合、固定 sample count、metrics |
| `test_fully_async.py` | basic continuous 模式、inflight count、partial rollout、staleness + fully-async 组合 |
