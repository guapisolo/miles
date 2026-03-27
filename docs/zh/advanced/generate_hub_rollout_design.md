# Generate Hub 与 Rollout Function 设计说明

这篇文档说明 `generate_hub` 在整个 rollout 栈里的位置、每一层应该承担的职责，以及你现在加的 fully async rollout 为什么方向是对的、但应该放在什么抽象层上。

先说结论：

- `generate_hub` 应该只负责 sample 局部的生成语义。
- fully async 是 rollout 调度问题，不是 `generate_hub` 问题。
- 现在新的 `PersistentRolloutWorker` 方向是合理的。
- 更早的“全局后台线程 worker 一直跑”的方案可以作为原型验证，但不适合作为最终抽象边界。

## 1. 范围与术语

当前 refactor 里，其实有两套不同层级的抽象，必须先拆开：

1. `GenerateFn`
   - 输入：单个 `Sample`、生成状态、采样参数。
   - 输出：一个 `Sample`，或者一组 `Sample`。
   - 职责：把“这一条样本”转换成一次或多次模型调用，再把结果写回训练样本。

2. `RolloutFn`
   - 输入：rollout id、data source、全局 rollout 配置。
   - 输出：这一轮训练或评测需要的样本分组。
   - 职责：调度很多个 sample group，做过滤、abort、回收，并保证这一轮拿到足够可用的数据。

这里最容易混淆的点是：`generate_hub` 这个名字看起来像“rollout 的中心”，但实际上它只应该负责“单条轨迹怎么生成”，而不应该负责“很多条轨迹怎么调度”。

## 2. 当前分层

现在的调用栈大致是：

```text
RolloutManager
  -> InferenceRolloutFn
    -> generate_rollout_async / generate_rollout_fully_async
      -> generate_and_rm_group
        -> generate_and_rm
          -> generate_hub.<variant>.generate
```

每层职责可以理解成：

- `RolloutManager`
  - 管生命周期、server、data source、rollout/eval 入口
- `InferenceRolloutFn`
  - 构建持久化 `GenerateState`
  - 选择 train 或 eval
  - 选择标准 rollout 调度或 fully async 调度
- `generate_rollout_*`
  - 提交 group、收集结果、做 dynamic filter、处理剩余请求 abort
- `generate_and_rm_group`
  - 在一个 prompt group 内并发跑多个 sample
  - 必要时跑 group reward model
- `generate_hub.*.generate`
  - 负责具体的生成逻辑、工具调用逻辑、agent 逻辑封装

这个分层本身是对的，因为它把“生成语义”和“调度语义”拆开了。

## 3. `GenerateFn` 应该负责什么

`GenerateFnInput` / `GenerateFnOutput` 定义了 `generate_hub` 的核心契约。一个 `generate_hub` 实现通常应该做的事是：

- 读取 `input.sample`、`input.state`、`input.sampling_params`
- 发起一次或多次后端请求
- 正确更新样本字段：
  - `tokens`
  - `response`
  - `response_length`
  - `status`
  - `rollout_log_probs`
  - `loss_mask`
- 返回：
  - 一个 `Sample`
  - 或多个 `Sample`

一个 `generate_hub` 实现通常不应该负责：

- 从 data source 拉数据
- oversampling 策略
- dynamic sampling filter
- rollout 级别的 abort 策略
- 持久 worker 生命周期
- 和 weight update 的同步边界

这些都不是“单条样本怎么生成”的问题，而是“这一轮 rollout 怎么组织”的问题。

## 4. 现有 `generate_hub` 变体

### 4.1 `single_turn`

`miles/rollout/generate_hub/single_turn.py` 是最基础的 `/generate` 单轮生成实现。

特点：

- 每个 sample 一次后端请求
- 支持 partial rollout resume
- 适合普通的一问一答式生成

如果你的环境本质上还是“一个 observation -> 一个 answer”，这是最自然的 generate function。

### 4.2 `multi_turn`

`miles/rollout/generate_hub/multi_turn.py` 在 `/generate` 之上实现了多轮工具调用。

特点：

- 一个 sample 可能触发多轮模型调用
- tool response 会作为 observation 追加回 token stream，并且 `loss_mask=0`
- 支持两种输出方式：
  - 合并成一个最终 sample
  - 用 `--generate-multi-samples` 返回每轮一个 sample
- 当前不支持 partial rollout

这个抽象适合“单条轨迹内部有多轮 LM 交互和工具观测”的场景。

### 4.3 `agentic_tool_call`

`miles/rollout/generate_hub/agentic_tool_call.py` 更像一个薄封装。它自己不实现 agent 决策，而是：

- 调用户通过 `--custom-agent-function-path` 提供的 agent 函数
- 用 Miles Router session 记录 OpenAI-format 请求
- 再把 session records 转成训练样本

特点：

- agent 逻辑在框架外，用户自由实现
- `generate_hub` 只负责 tracing 和 sample conversion
- 支持单 sample 合并输出或 multi-sample 输出
- 其 sample 构造方式是“从 session record 回放生成”，而不是直接操作 `/generate` 响应

这个设计是好的，因为它把“agent 怎么跑”交给用户，把“怎么记账、怎么还原成训练样本”留在框架里。

## 5. Fully Async 应该放在哪

fully async 应该放在 rollout 层，不应该放在 `generate_hub` 层。

原因很直接：

1. fully async 处理的是很多个 group，不是一个 sample。
2. 它依赖 data source 拉取策略和过滤策略。
3. 它必须和 weight update 做严格同步，才能保持 on-policy。
4. 它必须决定 in-flight group 和已完成但尚未消费的 group 怎么处理。
5. 它应该对所有 generate variant 都生效，而不是让每个 `generate_hub` 自己再写一遍 scheduler。

所以从抽象边界上说，把 fully async 放到
`inference_rollout_common.py` /
`inference_rollout_train.py`
这一层，比把它塞进 `generate_hub` 要干净得多。

## 6. 现在的 Fully Async 设计合理吗

结论是：合理，但要区分“当前主线实现”和“早期原型实现”。

- 当前主线方向是合理的。
- 更早的全局后台 worker 原型不适合作为最终结构。

### 6.1 为什么当前主线方向是合理的

新的 `PersistentRolloutWorker` 有几个很重要的优点：

1. 复用了标准生成路径。
   - worker 仍然走 `generate_and_rm_group`，所以 `single_turn`、`multi_turn`、`agentic_tool_call` 都不需要单独再写一个 async 版本。

2. async 逻辑集中在 rollout 层。
   - 提交、收集、abort、过滤、后处理，都留在 rollout 调度代码里，没有污染 `generate_hub`。

3. 它有明确的 weight update 边界。
   - `resume() -> collect() -> pause()`
   - `pause()` 会在下一次权重更新前 abort 掉在飞请求。
   - 这是最关键的正确性条件。如果没有这个边界，“fully async” 很容易变成“悄悄混入 off-policy 样本”。

4. 它保留了现有过滤和后处理挂钩。
   - dynamic filter、sample filter、all-samples post-process 仍然在 rollout 层工作。

5. 它确实适合长尾 agentic workload。
   - 尤其是多轮、工具调用、时延分布很长的时候，producer/consumer 形式的 worker 能显著减少 collector 端空等。

### 6.2 为什么旧的全局 worker 原型不够好

`examples/fully_async/fully_async_rollout.py` 里的更早原型能证明思路，但结构上有几个明显问题：

- 全局单例 worker 生命周期太粗
- 单独线程加单独 event loop，复杂度高
- 没有严格的“weight update 前 pause”边界
- queue 里的结果可能跨 rollout 调用残留
- 没有完整接入标准 dynamic filter 流程
- eval、异常处理、资源回收都比较弱

最大的风险其实不是代码风格，而是策略新鲜度。如果 worker 跨训练步一直跑，那么很容易在权重已经更新之后，还消费到旧策略生成的结果。对 on-policy RL 来说，这不是小问题，而是训练语义已经变了。

## 7. 应该怎么理解“fully async”

更安全的理解不是“后台永远持续生成”。

更合理的理解是：

- 在 rollout actor 内维护一个持久 producer
- rollout 活跃期间持续预取和回填
- 但在 weight update 之前必须强制打一个同步边界

这个版本虽然没有“永不停止的全局生成”那么激进，但它更符合真正的目标：尽量把 rollout 生成和训练重叠起来，同时不破坏 on-policy 边界。

所以从架构表述上，当前实现更准确的说法其实像：

- 持久化预取 rollout
- 有边界的 fully async rollout
- rollout 层 producer/consumer 调度

名字不是最关键的，关键是不能让样本跨过 policy update 边界泄漏。

## 8. 为什么这也和 OpenAI API 的抽象一致

OpenAI 自己的 API 设计里，其实也有相似的层次区分：

- tool calling、conversation state 这些属于请求语义
- background execution 属于调度语义

相关官方文档：

- Responses API create reference  
  <https://developers.openai.com/api/reference/resources/responses/methods/create/>
  - 包含 `background`
  - 包含 `conversation` / `previous_response_id`
  - 包含并行工具调用控制
- Responses migration guide  
  <https://developers.openai.com/api/docs/guides/migrate-to-responses/>
  - 说明了 Chat Completions 需要手动管理 conversation state，而 Responses 可以通过 `previous_response_id` 串接
- Chat Completions create reference  
  <https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/>
  - 包含并行 function/tool calling 控制

这对 Miles 的启发是：

- `generate_hub` 应该建模“请求或轨迹怎么和模型交互”
- fully async rollout 应该建模“这些交互如何被调度”

两者有关，但不应该混成一个抽象。

## 9. 当前还存在的风险和缺口

### 9.1 命名仍然容易让人误解

`generate_hub` 里装的是 generate function，不是完整 rollout function。后续文档和代码注释最好持续强调这一点。

### 9.2 fully async 还缺专门测试

当前对 `single_turn`、`multi_turn`、`agentic_tool_call` 的测试已经比较完整，但 fully async 路径本身还缺对等的专项覆盖。

建议补的测试包括：

- 标准 rollout 和 fully async rollout 在确定性 mock 下输出等价
- `pause()` 能正确 abort 在飞请求
- queue 中旧结果不会跨 weight update 泄漏
- partial rollout 行为与标准路径一致

### 9.3 有些算出来的数据会被故意丢弃

现在 `pause()` 会直接清掉 queue 中已经完成但还没消费的 group。这会浪费一部分算力，但对 on-policy 训练来说这是更安全的默认行为。如果以后想复用这些 group，必须显式引入 policy version 跟踪。

### 9.4 `multi_turn` 和 `agentic_tool_call` 的限制仍然存在

fully async 只解决调度问题，不会自动消除 generate 层已有的限制：

- `multi_turn` 仍然不支持 partial rollout
- `agentic_tool_call` 目前仍然走 OpenAI-format session wrapper，而不是原生 Responses API 路径

这些都还是 generate 层自己的问题，应该继续留在 generate 层处理。

## 10. 一个简单的设计判断规则

如果一个特性回答的是这个问题：

> “单条 sample / 单条 trajectory 应该怎样和模型交互？”

那它应该放在 `generate_hub`。

如果一个特性回答的是这个问题：

> “很多个 sample group 应该如何被调度、过滤、abort，并且和 rollout step 边界对齐？”

那它应该放在 rollout 层。

按这个规则：

- `single_turn`、`multi_turn`、`agentic_tool_call` 属于 `generate_hub`
- fully async 属于 `inference_rollout_*`

## 11. 最终判断

你现在的设计方向是对的。

更具体地说：

- 把 `generate_hub` 保持成纯 generate-function 库，这个方向是对的
- 把 fully async 逻辑放到 rollout scheduler 层，这个方向是对的
- 用 persistent worker，并在 weight update 前显式 `pause()`，这个方向是对的
- 早期全局后台线程版本更适合作为原型，不适合作为最终架构

如果接下来继续收敛实现，最值得做的不是把更多调度逻辑塞进 `generate_hub`，而是把 rollout 调度层本身继续显式化，同时保持 `generate_hub` 只负责 sample 局部的生成语义。
