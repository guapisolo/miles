# Session Server Fake Streaming 设计

状态:设计草案,待评审。参照实现为 NVIDIA-NeMo/ProRL-Agent-Server(commit `f0e8343a`,2026-06-25),其 streaming 写法已逐行核实,证据见下节。

读者是维护 Miles session server 与 TITO 数据路径的工程师;读完后应能判断:为什么后端必须保持非流式、fake streaming 在 `chat_completions` 三阶段流程中的插入点、以及错误语义和训练路径为什么完全不受影响。

## 参照实现确认:ProRL-Agent-Server 确实是 fake streaming

结论:确认。ProRL 的 gateway 对 `stream: true` 请求做的是 fake streaming、延迟一次性返回,全仓库不存在任何真流式后端路径,也没有人工 sleep 切片。证据链:

- `src/polar/gateway/server.py:648-668`:`proxy_request` 按 `openai_request.get("stream", False)` 分流到 `_handle_streaming` / `_handle_non_streaming`。
- `src/polar/gateway/server.py:704-768`:`_handle_streaming` 先把请求改为非流式(丢 `stream_options`、置 `stream=False`),`await state.inference.completion(...)` 拿到完整响应并写入 storage,然后用 `_response_to_stream_chunk` 把完整响应合成为单个 `chat.completion.chunk`,最后以 SSE `StreamingResponse` 一次性吐出该 chunk 与 `data: [DONE]`。
- `src/polar/gateway/server.py:771-810`:`_response_to_stream_chunk` 把完整 `message`(content / reasoning_content / tool_calls)整体塞进一个 delta,`finish_reason` 与 `usage` 附在同一 chunk 上。
- `src/polar/gateway/proxy.py:114-122`:后端客户端 `completion()` 的 docstring 即 "Non-streaming chat completion. Returns the full JSON response.",并强制 `request_copy["stream"] = False`。

所谓 "delay return" 指:SSE 连接建立后没有任何字节,直到后端完整生成结束,才一次性输出全部内容。ProRL 中响应在构造 `StreamingResponse` 之前就已拿到,因此上游错误仍能以真实 HTTP status code 的 JSON 返回。

## 动机

Session server 的 `POST /sessions/{id}/v1/chat/completions` 是 OpenAI 兼容端点,面向外部 agent framework。相当一部分 framework 或 SDK helper 会固定发送 `stream: true`,不提供关闭开关。OpenAI 方言的 agent CLI 即是实例:Codex 以 `wire_api = "chat"` 接入时按流式消费 chat completions,slime 为此在其 OpenAI adapter 实现了同样的 fake streaming(`slime/agent/adapters/openai.py::_render_stream`,其注释记录了真实 client 累积 tool_calls 分片出错的实战教训)。streaming 对 harness 不只是 UX:长生成的非流式请求会撞上各层 idle timeout,Anthropic 协议甚至在 SDK 层强制长请求走流式。本设计的服务对象是直连本端点的 OpenAI 方言 harness;fake streaming 的机制(后端非流式 + delay return)跨方言共享,但 wire 渲染是方言私有的,故各方言端点各自负责自己的渲染。

当前行为:session server 无条件注入 `return_meta_info=True`([core.py:169](miles/rollout/session/core.py#L169))后把 `stream: true` 原样转发,而 sglang-miles 在请求转换层直接拒绝该组合(`serving_chat.py:551-555`,"return_meta_info is not supported with streaming" → 400),经非 200 透传([core.py:207](miles/rollout/session/core.py#L207))原样回给 client、不记录——即现状是**确定性 400**,streaming client 一律无法工作。即使后端没有该 guard,SSE 字节流也会在 `json.loads` 处炸成 500,且流式 chunk 没有 `choices[0].message` 和 `meta_info`,TITO 无从谈起。

目标:client 用 `stream: true` 调用该端点时得到合法的 OpenAI streaming 响应,且 TITO 追踪、SessionRecord、训练路径与非流式请求完全一致。

## 约束

硬约束:

1. 后端请求必须保持非流式:TITO 校验依赖完整响应中的 `choices[0].message` 与 `meta_info.output_token_logprobs`([core.py:213-235](miles/rollout/session/core.py#L213-L235)),逐 chunk 响应无法满足;且 sglang-miles 本身拒绝 `return_meta_info` + `stream` 的组合(`serving_chat.py:551-555`),真流式透传在后端就不成立。
2. `SessionRecord` 与训练路径(`GET /sessions/{id}`)的数据形态不变。
3. 错误语义不变:后端非 200 仍不记录、原样透传;`UpstreamResponseError` 仍走 502 JSON。流开始前的失败返回 JSON error 也符合 OpenAI 官方行为,标准 SDK 都能处理。
4. 流式响应必须是合法 wire format:`data: {chat.completion.chunk}` 序列 + `data: [DONE]`,`text/event-stream`。

软约束:改动局限在 `miles/rollout/session/`,不引入新依赖,不改 `do_proxy` 传输层。

非目标(均不在 v1 范围,v1 只保证本端点的 streaming 正确性):

- 真 token 级流式:违反硬约束 1。
- token 切片的伪增量输出与 heartbeat:见方案 C 与风险 1。
- 泛化 `session_proxy` 路由的流式支持:其现状是整体缓冲后原样回放 SSE 字节,行为已可接受,保持不变。
- Anthropic 等非 OpenAI 方言:Claude Code 类 harness 归属未来的方言 adapter,该 adapter 以非流式调用本端点、自行渲染 Anthropic 事件语法,与本设计正交。先例:ProRL `transform/anthropic.py`;slime `slime/agent/adapters/anthropic.py::_render_stream`(同为 fake streaming,按 block 合成 `message_start` → 单 delta → `message_stop` 序列)。
- trajectory 层的多 segment 支持:独立设计轮。已收敛的方向备忘——miles 的 sample↔session 绑定是固定 URL 一比一,fork 必须落在 session 内部:session 内多条 `LinearTrajectory` + message 级贪心最长前缀分派,无全匹配则在最长重叠者的 record 边界 fork 拷贝 checkpoint,继承段不参与子线 loss(每响应只训一次);截断即 segment 终点,对截断 segment 的延伸请求 4xx 拒绝(原 lazy boundary repair 设计已废弃,见 TRUNCATION_HANDLING_DESIGN.md 状态注记)。

## 方案选择

方案 A,真流式透传:需要把 `do_proxy` 改成流式转发,且逐 chunk 拿不到 `message` / `meta_info`,[core.py:214](miles/rollout/session/core.py#L214) 的校验无法执行,直接违反硬约束 1、2。排除。

方案 B,ProRL 式 fake streaming(选定):请求准备阶段摘掉 `stream` 标志,后端路径与 TITO 三阶段流程零改动,仅在响应构造时按 client 原始意图切换 wire format。每条硬约束都由"后端视角完全等同非流式请求"这一性质直接满足。

方案 C,fake streaming 加 heartbeat 或 token 切片:没有任何约束要求它;代价是把 proxy await 移进 SSE generator,HTTP 200 必须在拿到结果前提交,上游错误退化为 in-stream error event,破坏硬约束 3。读者若问"为什么不顺手做得更像真流式"——答案是错误语义的代价,列为后续迭代方向而非本轮范围。

## 设计

改动集中在 `SessionCore.chat_completions` 与 `core.py` 的响应构造 helper:

1. Phase 1(锁内、序列化 `proxy_body` 之前):`client_stream = bool(request_body.pop("stream", False))`,同时 `request_body.pop("stream_options", None)`。`SessionRecord.request` 存的本来就是注入过 `logprobs` / `input_ids` 的改写请求,摘除 `stream` 与既有先例一致。
2. Phase 2、Phase 3 不变。
3. 响应构造:三个 `_chat_client_response(result, response)` 调用点(closing 跳过、`num_assistant` 竞态跳过、正常返回)统一按 `client_stream` 分支。流式分支:
  - 新 helper `_response_to_stream_chunk(response)`:仿照 ProRL 合成单个 chunk,`delta` 含 `role` / `content` / `reasoning_content`(如有)/ `tool_calls`(带 `index`),`finish_reason` 与 `usage` 附在同一 chunk;`id` / `created` / `model` 从完整响应透传。单个大 delta 协议合法:OpenAI streaming 的 delta 语义是增量拼接而非覆盖(SGLang 端按 offset 切片,`serving_chat.py` 的 `delta = content["text"][offset:]`),client 累加器对这个 delta 做一次拼接即得完整内容。
  - wire format:`data: {chunk}\n\ndata: [DONE]\n\n`,`media_type="text/event-stream"`,headers 带 `Cache-Control: no-cache` 与 `X-Accel-Buffering: no`(防反向代理缓冲),不复用上游 JSON 响应的 headers。
  - 用普通 `Response` 而非 `StreamingResponse`:整个 payload 在构造时已知,ProRL 的 generator 也只 yield 一次,普通 `Response` 语义相同且贴合 `core.py` 现有的 `Response` 风格。
  - chunk 不携带 `meta_info`:选择流式的 client 是标准 OpenAI 协议消费方,训练路径从 `GET /sessions/{id}` 拿完整数据;因此 `_strip_replay_payloads` 也无需作用于 chunk。
4. 错误路径零改动:所有失败(transport 502、非 200 透传、`UpstreamResponseError`)都发生在流式响应构造之前,仍返回带真实 status code 的 JSON。

与 ProRL 的有意偏差:普通 `Response` 代替 `StreamingResponse`(语义相同);无多方言 transformer 层(session server 只有 OpenAI chat 一种方言);`usage` 无条件附在唯一 chunk 上与 ProRL 相同,但与严格 OpenAI 协议的 `stream_options.include_usage` 语义有偏差,主流 SDK 对多出的 `usage` 字段兼容。

## 行为 case 清单

请求侧:

- R1 `stream` 缺省或 `false`:现有 JSON 路径,字节级不变(回归基线)。
- R2 `stream: true`:走 fake streaming 分支;`stream` 与 `stream_options` 均在 Phase 1 被 pop,后端与 `SessionRecord` 不见这两个字段。
- R3 `stream: true` + `stream_options.include_usage`:同 R2;usage 无条件附在唯一 chunk 上(见偏差说明),不单独发空 `choices` 的 usage chunk。
- R4 `stream` 非布尔值:按 `bool()` 真值处理(比 slime 的 `is True` 宽、与 ProRL 一致);合法 client 只会发布尔,不为畸形输入加分支。

成功响应侧(均为单 chunk + `data: [DONE]`,`id`/`created`/`model` 从上游响应透传):

- S1 纯文本 `finish_reason="stop"`:delta 含 `role`+`content`。
- S2 工具调用 `finish_reason="tool_calls"`:全部 `tool_calls` 带 `index` 放在**同一个** chunk(slime 实战教训:分片会被部分 client 拼成不可解析的 arguments),`content` 若非空一并携带。
- S3 `reasoning_content` 存在(SGLang 扩展):附在 delta 上,与 ProRL/slime 一致。
- S4 截断 `finish_reason="length"`:正常出流,本层只透传 finish_reason;截断后的 segment 终止与延伸拒绝属 trajectory 层方向(见非目标),不在本层。
- S5 `content` 为空串(如纯工具调用):delta 携带空串,合法;`content` 为 `None` 属错误侧 E3。

错误侧(全部发生在 SSE 构造之前,一律返回真实 status code 的 JSON,与 OpenAI"流开始前失败返回 JSON error"的行为一致):

- E1 后端非 200(如 400 context too long):原样透传、不记录(现状不变)。
- E2 transport 失败:502 JSON(现状不变)。
- E3 `UpstreamResponseError`(缺 meta_info / logprobs 数量不符 / content 为 None):502 JSON(现状不变)。
- E4 session 不存在或 closing:404 JSON(现状不变)。

竞态侧(Phase 3 的两个跳过路径,行为同正常返回、仅不更新状态):

- C1 proxy 期间 session 被 DELETE(closing):按 client 原始意图返回——stream 请求返回 SSE 形态。
- C2 `num_assistant` 竞态不符:同 C1。

假设声明:响应恒为单 choice(`choices[0]`),与 `chat_completions` 现有假设一致,`n>1` 不在支持范围。

## 风险与开放问题

已知风险 1:fake streaming 不提供任何防 timeout 保证。ProRL 的写法里响应在构造 `StreamingResponse` 之前才拿到,生成期间 client 连 HTTP response headers 都收不到;本设计同构,timeout 时钟与非流式请求完全等价。链路上有三个时钟:client 到 session server 的 read timeout(httpx 语义为相邻字节间隔,零字节期间持续累计;OpenAI SDK 默认 600s)、中间反向代理如有(nginx `proxy_read_timeout` 默认仅 60s)、session server 到 SGLang 的 `miles_router_timeout`(默认 600s,[server.py:35](miles/rollout/session/server.py#L35),系统硬上限,与 stream 无关)。保证方式是部署约束:client read timeout ≥ 最大生成时长,并与 `miles_router_timeout` 对齐;这与非流式请求的要求相同,本设计未引入恶化。若要主动保证,机制是方案 C 的 SSE comment heartbeat(`:` 开头的行,SSE 规范要求解析器忽略,OpenAI SDK 与 ProRL 自己的 `_parse_sse_lines` 均如此处理;ProRL 的 `/events` 端点即先例,`stream_events(heartbeat_seconds=15.0)`,超时即发 ping),但它要求先提交 200 再等结果,上游错误退化为 in-stream error、SDK 基于 status code 的自动重试失效,且只能修 idle timeout、修不了 client 的总 deadline。升级触发条件:第一个实际接入的 framework 的 read timeout 不可配置或小于生成上限。

已知风险 2:`usage` 附带方式偏离严格协议(见上),若遇到严格校验 chunk schema 的 client 再按 `include_usage` 语义拆分末尾空 `choices` 的 usage chunk。

开放问题:是否存在接入方依赖首 chunk 尽早到达做 UX(spinner、首 token 延迟统计)。需要等第一个真实 framework 接入后观察,决定是否升级到方案 C;影响面仅限响应构造分支,不影响本轮其余设计。

## 验证

在 [tests/fast/router/test_sessions.py](tests/fast/router/test_sessions.py) 现有 harness(`MockSGLangServer` 忽略 `stream` 字段、恒返回完整 JSON,无需改动)上新增:

- S1:`stream: true` 纯文本请求,断言 200 + `text/event-stream`,响应体恰为一条 chunk 加 `data: [DONE]`,chunk 的 `delta.content` / `finish_reason` / `usage` 与非流式响应一致,且不含 `routed_experts` / `indexer_topk`。
- R2:断言 mock 后端收到的 payload 不含 `stream` / `stream_options`。
- 硬约束 2:流式请求后 `GET /sessions/{id}`,record 与等价非流式请求完全一致。
- S2:mock 返回 tool_calls 响应,断言全部 tool_calls 带 `index` 在同一 chunk 内、`json.loads(arguments)` 可解析。
- S4:mock 返回 `finish_reason="length"`,断言 chunk 透传该 finish_reason。
- E1:后端非 200 + `stream: true`,JSON error 原样透传(硬约束 3)。
- R1:`stream: false` 与缺省行为不变(回归)。
- 端到端形态:用 openai SDK 的 `create(stream=True)` 原始迭代器消费一次假流,断言累计结果等于非流式响应内容(客户端兼容性的最低验证)。

