# Session Server Abort/Resume 流水线

## 问题

在 fully-async rollout + agentic 场景下，单个 sample 的生成时间可能很长（多轮
tool call、长推理链）。当需要更新权重时，当前流水线会 abort 所有 in-flight 的
SGLang 请求并丢弃已完成的部分工作，下一轮从头开始，浪费大量算力。

## 方案

由 session server 透明处理 weight update 期间的 abort/resume。从 agent 框架
的视角看，它的 HTTP 请求只是变慢了——完全不知道中间发生了权重更新。

## 流水线总览

```
Training Loop (train_async.py)
│
├─ rollout_manager.generate()        ← 持续生成
│    └─ generate_rollout_async(continuous=True)
│         └─ agent 框架 → SessionServer → SGLang
│
├─ actor_model.async_train()         ← 用已收集的 samples 训练
│
└─ actor_model.update_weights()      ← 触发 abort/resume
     │
     ├─ 1. rollout_manager.pause_sessions()
     │      └─ POST /abort_sessions → SessionServer
     │           ├─ 关闭 resume_event（gate 关门）
     │           └─ POST /abort_request → SGLang（停止生成）
     │
     ├─ 2. weight_updater.update_weights()
     │      └─ 将新权重同步到 SGLang 引擎
     │
     └─ 3. rollout_manager.resume_sessions()
            └─ POST /resume_sessions → SessionServer
                 └─ 打开 resume_event（gate 开门）
                      └─ 所有阻塞的 handler 重新向 SGLang 发请求
```

## Session Server Gate 机制

核心是在 `chat_completions` handler 中用 `while` 循环包住 `do_proxy`：

```python
while True:
    await backend.resume_event.wait()   # paused 时阻塞
    result = await backend.do_proxy(...)
    if not backend.is_paused():
        break                           # 正常响应，继续处理
    # 被 abort 了，丢弃 partial response，回到 gate 等 resume
```

三种场景：

1. **请求到达时正常运行中**：直接通过 gate，`do_proxy` 正常与 SGLang 通信。

2. **请求到达时已 paused**：阻塞在 `resume_event.wait()`，直到 resume 后
   再发给 SGLang（此时已是新权重）。

3. **请求正在 SGLang 中生成时 abort 到达**：SGLang 返回 partial 200 响应。
   Handler 检查 `is_paused()` 为 True，丢弃 partial 结果，回到 gate 阻塞，
   等 resume 后用同样的请求重新发送。

所有场景下 agent 都只收到一个完整的响应。当前 turn 的部分生成结果被丢弃，
用新权重重新生成。

## 关键设计决策

- **丢弃当前 turn 的 partial generation**：只丢弃正在生成的 assistant
  message，之前 turn 的 token 仍保留在 session 的 token history 中。
  Resume 后重发同样的请求，SGLang 会对已有 prefix 做 re-prefill（利用
  KV cache / radix tree），然后用新权重重新生成当前 turn。

- **Gate 放在 session server 而非 rollout 层**：session server 已经持有
  到 agent 的 HTTP 连接，是吸收暂停的天然位置，agent 无需感知。

- **pause/resume 放在 `actor.update_weights()` 内部**：不需要改
  `train_async.py`。`update_weights` 方法已有 `self.rollout_manager`
  的引用，在权重同步前后加 pause/resume 即可。

- **允许跨 policy 的 sample**：一个 sample 的早期 turn 可能由旧 policy
  生成，被重试的 turn 由新 policy 生成。这是可接受的——不要求严格 on-policy。

## 修改的组件

| 文件 | 改动 |
|---|---|
| `session_server.py` | `resume_event`、`is_paused()`、`pause_sessions()`、`resume_sessions()` |
| `sessions.py` | `POST /abort_sessions`、`POST /resume_sessions`、`chat_completions` 中的 gate 循环 |
| `rollout.py` | `RolloutManager` 上的 `pause_sessions()`、`resume_sessions()` 方法 |
| `actor.py` | `update_weights()` 在权重同步前后调用 pause/resume |
