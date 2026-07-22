# TITO Truncation Handling 设计

状态：已废弃（deprecated，2026-07-21），暂存备查、不进评审。废弃原因：动机前提"agent rollout 允许截断 turn 后继续提交后续 turn"被推翻——新方向为截断即 lineage 终点，session server 对截断后的延伸请求直接返回 4xx（fail loud），lazy boundary repair 因此失去存在条件；`merge_samples` 现有"遇非 COMPLETED 即停"行为从 workaround 升格为正确终态。后续如需恢复"截断后继续"的场景，再以本文为起点重启。

目标基线：`design/tito-truncation-handling`，包含 PR #1605 的 session-side sample assembly 重构提交 `b9f420c9c` 和 `1faecafa3`。

读者是维护 Miles session rollout、TITO tokenizer、sample assembly 与训练数据路径的工程师；读完后应能判断 truncated assistant boundary 在何时被修复、哪些运行时数据必须保持 raw、synthetic token 如何获得 zero loss mask，以及 CPU/GPU 验证是否覆盖 rollback、replay 与 CP 风险。

## 核心决策

上一个 truncated turn 返回时，不补 token，不改 `output_token_logprobs`，不改 `SessionRecord`，也不改该 turn 的 trajectory checkpoint。runtime 保存的仍是 SGLang 实际收到和生成的 token。

只有当下一 turn 真正到来时，`LinearTrajectory.prepare_pretokenized` 才把上一条 record 的只读上下文交给 `TITOTokenizer.merge_tokens`。`merge_tokens` 在新 turn 的 junction 处允许一种严格受限的 drift：上一 assistant 的 raw completion 可以缺少 canonical chat template 的尾部，但必须是 canonical completion 的 exact token prefix。缺失尾部被临时补进这一次真实发送给 backend 的 `input_ids`，旧 checkpoint 和旧 record 均不回写。

训练侧不保存 tail sidecar，也不修改 per-turn Sample。只有最终 `merge_samples` 合并相邻 turn 时，tail 才因为已经存在于后一 turn 的真实 prompt 中而自然落入 observation bridge，并获得 `loss_mask=0`、`rollout_log_probs=0.0` 及现有 OPD bridge 的零值语义。

这里的“最后 merge sample 再改”专指训练表示中的 `response_length`、mask、logprob 和 OPD 字段。`merge_samples` 不负责凭空追加 token；否则那些 token 从未进入下一轮推理 prompt，无法修复真实 rollout 的上下文。

## 背景与动机

Agent rollout 允许一个 assistant turn 因 `max_tokens` 命中 `finish_reason == "length"` 后，继续提交 tool、user 或 system 消息并生成后续 turn。

当前 `miles/rollout/generate_utils/sample_utils.py::merge_samples` 在累计 Sample 的 `status` 不是 `COMPLETED` 时立即停止，因此 intermediate truncated turn 后面的合法记录会被丢弃。这是保护 prefix 和 replay 完整性的 workaround，不是完整的 truncation 语义。

TITO 的下一 turn prompt 以模型实际生成 token 为 prefix，再增量拼接非 assistant 消息。agent 保存并回传的却是 SGLang parser 生成的 OAI assistant message。截断可能发生在 `</think>` 或消息结束 token 之前；重新对 OAI message 执行 chat template 时，模板会补出模型没有生成的收尾序列，导致 raw checkpoint 与 canonical message history 在 assistant 尾部发生差异。

本设计的第一目标是保持 runtime `TokenInfo` 的真实性，第二目标才是修复 continuation。一个 truncated turn 在没有后续请求时不需要被“伪装成闭合”；只有后续模型输入依赖这个边界时，才按需计算 closure。

## 成功标准

- 上一 turn 的 `choice.finish_reason` 只有精确等于 `"length"` 时才允许 truncation repair；`stop`、`tool_calls`、`abort` 和其他值不添加 truncation tail。
- 上一 turn 返回后，其 client response、`SessionRecord.request/response`、`meta_info.output_token_logprobs`、`completion_tokens` 和 raw trajectory checkpoint 均保持不变。
- 下一 turn 的 `merge_tokens` 只允许在旧 prefix 尾部追加由 canonical diff 证明的 token，不允许 fuzzy diff、longest-common-prefix 猜测、decode/re-encode 或替换真实 token。
- 截断发生在 thinking 结束 token 之前时，tail 可以包含 `</think>`、消息结束 token 和必要换行；截断发生在 thinking 结束 token 之后时，只补尚缺的 canonical 尾部。
- 上一 turn 的 tools 与 effective chat-template kwargs 用于重建上一 assistant；下一 turn 的 tools 用于 tokenize 新增 non-assistant suffix，二者不得混用。
- intermediate truncated turn 后的成功记录必须进入单 Sample 合并结果；generic `merge_samples`、本地 `max_seq_len` 截断和 `ABORTED` 默认仍是 terminal。
- tail 在 merged response span 中必须 `loss_mask=0`、`rollout_log_probs=0.0`，且不能伪造上一 turn 的 logprob、`routed_experts` 或 `indexer_topk`。
- final truncated turn 没有后续请求时保持 raw，不在训练 Sample 尾部物化没有消费者的 synthetic closure。
- regular CP 与 Ulysses CP 在 interior zero-mask bridge 跨切分边界时保持 token、`response_length` 和 mask 对齐。

## 已验证的现状与约束

PR #1605 之后，`SessionCore.collect_samples` 在 session server 内按 `compute_samples_from_openai_records` → `truncate_samples_by_total_tokens` → `merge_samples` 的顺序组装 Sample；本方案必须基于这条新路径。

`SessionCore.chat_completions` 的 Phase 1 已把 `prepare_pretokenized` 返回值作为 backend 的真实 `input_ids`。Phase 3 则在请求成功后把 `prompt_token_ids + completion_token_ids` 保存为新 checkpoint，并把同一真实 prompt 写入下一条 record。因此 lazy repair 具有天然的事务边界：下一请求失败时旧状态仍是 raw，重试会从同一 record 确定性地重新推导 tail。

`compute_samples_from_openai_records` 从每条 record 的真实 `output_token_logprobs` 构造 per-turn Sample。它在处理下一条 record 时直接跳到该 record 的 `prompt_token_ids` 长度，因此 intermediate closure 已存在于下一 prompt 时，不需要 sidecar，也不需要放宽最终 accumulated-token cursor 检查。

`_merge_sample_pair` 已把两个生成 turn 之间的整个 token gap 视为 observation bridge：增加 `response_length`，填充 `loss_mask=0`、`rollout_log_probs=0.0`，并采用后一 turn 覆盖完整 prefix 的 replay payload。lazy closure 正好复用这套语义。

`truncate_samples_by_total_tokens` 也会把 Sample 标成 `TRUNCATED`，但它会丢弃后续 turn 并立即停止；这个状态不证明存在经过 lazy repair 的 successor。不能仅凭 `Sample.status == TRUNCATED` 全局放宽 merge。

当前 `compute_session_mismatch` 直接比较 closed canonical history 与 raw latest checkpoint。若 final turn 是 `finish_reason == "length"`，缺少 closure 是预期状态，却会被报告为 hard special-token mismatch；诊断层必须改成只读 virtual repair，不能为了让指标变绿而修改 checkpoint。

## 范围与非目标

本轮处理 `finish_reason == "length"` 且 parser 生成的 OAI assistant message 能以 append-only token tail 还原 raw completion 的情况。

本轮不恢复 `abort`、请求失败或缺失后续 replay payload 的轨迹；`_introduces_replay_gap` 的保护继续生效。

本轮不改变本地 `max_seq_len` 截断、reward、length penalty、merged Sample 最终 status 或 truncation metric 的既有语义。

本轮不把 synthetic closure 写回上一 turn，不为它制造 rollout logprob 或 routing/indexer 数据，也不在 final truncated Sample 尾部补无消费者的 token。

如果 reasoning/tool parser 丢弃或改写 completion 中间内容，使 raw completion 不是 canonical assistant serialization 的 exact token prefix，本轮显式阻止 continuation，不猜测被丢弃的内容。

## 运行时数据流

设第一轮 prompt 为 `P`，模型因 length 生成 raw token `A`，canonical 缺失尾部为 `S`，下一轮 non-assistant suffix 与 assistant opener 为 `E`，下一轮模型输出为 `B`：

```text
turn 0 返回后：
  checkpoint              = P + A
  record[0].request       = P
  record[0].response      = raw A, finish_reason=length

turn 1 发请求前：
  merge_tokens            = P + A + S + E
  backend input_ids       = P + A + S + E
  turn 0 checkpoint       = P + A                 # 不回写

turn 1 成功后：
  checkpoint              = P + A + S + E + B
  record[1].request       = P + A + S + E
  record[1].response      = raw B

sample assembly：
  sample[0].tokens        = P + A
  sample[1].tokens        = P + A + S + E + B

merge_samples：
  tokens                  = P + A + S + E + B
  loss_mask               = mask(A) + 0...(S + E) + mask(B)
  rollout_log_probs       = logp(A) + 0...(S + E) + logp(B)
```

如果 turn 1 请求失败或没有发生，`S` 不进入任何持久状态。若 turn 1 成功，`S` 已成为真实 prompt 的一部分，后续 checkpoint、record 和 replay payload自然覆盖它。

## `TITOTokenizer` 设计

### 只读 previous-turn context

在 tokenizer 模块定义 ephemeral、不可持久化的上下文：

```python
@dataclass(frozen=True)
class PreviousAssistantTurn:
    finish_reason: str
    request_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    tools: list[dict[str, Any]] | None
    chat_template_kwargs: dict[str, Any]
```

`LinearTrajectory.prepare_pretokenized` 必须先执行现有 rollback，再从 rollback 后的 `records[-1]` 构造该对象。它不增加 `LinearTrajectory` 字段，不修改 `SessionRecord`，也不缓存 tail。

生产路径在存在上一 checkpoint 时总是传入 previous-turn context，使 `merge_tokens` 能显式检查 finish reason 和 record/checkpoint 对齐；单元测试可以传 `None` 测试无 session context 的基础 tokenizer 行为。

`merge_tokens` 的接口扩展为：

```python
def merge_tokens(
    self,
    old_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    pretokenized_token_ids: list[int],
    tools: list[dict[str, Any]] | None = None,
    *,
    previous_turn: PreviousAssistantTurn | None = None,
) -> list[int]:
    ...
```

这里的 `tools` 属于下一请求；上一请求的 tools 只能从 `previous_turn.tools` 读取。

### Base template method

现有 Qwen、GLM 和 MiniMax 都 override 整个 `merge_tokens`，容易让某个 family 绕过 truncation repair。基础类应把流程固定为一个 template method：

```text
raw pretokenized prefix
  -> _repair_previous_truncated_assistant(...)  # generic S + optional family placeholder
  -> tokenize_additional_non_assistant(...)     # 使用下一请求 tools
  -> _prepare_prefix_for_incremental(...)       # normal family junction hook
  -> concatenate
```

`_repair_previous_truncated_assistant` 内部先推导 generic canonical tail，再调用一个默认 no-op 的 `_append_truncated_boundary_placeholder` hook；目前只有 GLM 需要后者。Qwen、GLM 和 MiniMax 的正常 junction 行为收缩到 `_prepare_prefix_for_incremental`，不再复制主流程。后续新增 family 也会默认经过 repair gate。

### Strict canonical tail diff

`_repair_previous_truncated_assistant` 先验证：

- `pretokenized_token_ids == previous_turn.prompt_token_ids + previous_turn.completion_token_ids`。
- `old_messages == previous_turn.request_messages + [previous_turn.assistant_message]`，比较沿用 session 已有的 message normalization 语义。
- `finish_reason != "length"` 时直接返回原 prefix，不运行 truncation diff。

对 `finish_reason == "length"`，使用上一请求的 tools 和 effective kwargs 计算：

```text
P = apply_chat_template(previous request messages, add_generation_prompt=True)
F = apply_chat_template(previous request messages + assistant message, add_generation_prompt=False)
A = previous raw completion_token_ids
```

随后执行严格 token 级验证：

1. `P` 必须是 `F` 的 exact token prefix。
2. `C = F[len(P):]` 是 canonical assistant completion。
3. `A` 必须是 `C` 的 exact token prefix。
4. `S = C[len(A):]` 是唯一允许追加的 generic tail。
5. 验证 `A + S == C` 后，返回 `pretokenized_token_ids + S`。

差分必须直接在完整 token 序列上做，不能先做字符串 suffix diff 后单独 tokenize `S`，因为 BPE 边界和 special token 会破坏等价性。

如果 `A == C`，`S` 可以为空；是否允许 continuation 仍由 `finish_reason == "length"` 和成功 successor record 证明，不能依赖 tail 非空。

### Template kwargs 的约束

tail 必须由上一请求实际生效的 chat-template kwargs 推导。当前本地 `render_messages` 只读取 tokenizer 实例的固定 `chat_template_kwargs`，但 `SessionCore` 允许请求逐轮覆盖并转发给 backend；若两者不同，严格 diff 的前提不成立。

本 patch 选择在同一 TITO session 内冻结 effective `chat_template_kwargs`：请求可以重复传相同值，但不能逐 turn 改变；不一致时在发送 backend 前抛出 `MessageValidationError`。上一 record 中的 effective kwargs仍进入 context用于一致性断言。

如果产品必须支持逐 turn kwargs 变化，需要另行把 previous/next effective kwargs贯穿 `render_messages`、所有 segment helper 和 `tokenize_additional_non_assistant`，并证明模板切换仍是 append-only；这不是 truncation patch中的隐式 fallback。

### Family boundary 行为

Qwen3、Qwen3.5、Qwen3-Next、Nemotron3 以及共享 `<|im_end|>\n` 语义的 family 不硬编码 truncation closure。thinking 内截断时 strict diff自然得到 `</think>`、消息结束 token 和换行；answer 内截断时只得到消息尾。repair 后若 prefix 已以换行结束，现有 newline hook不得重复追加；正常 stop 仍由原 hook补 stop token之后的换行。

MiniMax 保留与 Qwen 同构的 message-end newline hook；generic tail 已含换行时同样不得重复。

GLM-4.7 的 canonical assistant serialization通常补 thinking closure，但下一 role opener由增量 suffix产生。为复用现有“trim ambiguous boundary，再 append真实 boundary”的 junction contract，`_append_truncated_boundary_placeholder` 在 generic tail 后临时追加一个确定性的 `<|user|>`；随后的 `_prepare_prefix_for_incremental` 按现有逻辑trim它，再由 incremental suffix追加真实 `<|user|>`、`<|observation|>` 或 `<|system|>`。placeholder不会进入 backend `input_ids`、record、checkpoint或 Sample；选择 `<|user|>` 而不是 `<|observation|>` 没有语义差异。

若 canonical tail本身已经以 GLM ambiguous boundary结束，hook不得重复追加 placeholder。现有 `max_trim_tokens == 1` 仍只处理模型真实输出的一个 trailing ambiguous token。

## `LinearTrajectory` 与事务语义

`update_pretokenized_state` 保持现有 contract：只保存这次 backend 真实使用的 `prompt_token_ids` 加上真实 `completion_token_ids`。上一 turn 的 checkpoint永远不被追加 synthetic tail。

第一次 continuation 的 `prepare_pretokenized` 返回 virtual repaired prompt，但不修改 trajectory。只有 backend成功且 Phase 3并发检查通过后，这个 prompt才作为新 record和新 checkpoint的一部分提交。

这带来以下行为：

- backend非 200、parser失败或请求取消：旧 raw checkpoint不变。
- 同一请求重试：从相同 raw checkpoint和 record重新推导相同 tail。
- concurrent请求：只有通过现有 `expected_num_assistant` 检查的结果能提交。
- rollback：先同步裁剪 `messages`、`trajectory_token_ids` 和 `records`，再读取新的 `records[-1]`；若回到 truncated checkpoint，下一次 continuation重新 repair。
- strict diff失败：本次 continuation在发 backend前失败，不产生 successor record。

lazy方案的固有时机是：truncated response本身可以成功返回，而不可修复的 parser drift要到第一次 continuation才暴露。若产品要求提前发现，可以在 turn提交时执行只读“可修复性验证”，但仍不得改 checkpoint、record或 TokenInfo。

## Sample assembly 与 continuation gate

`compute_samples_from_openai_records` 不增加 sidecar逻辑，也不改变 per-turn Sample：

- 前一 Sample只含 raw `A` 及其真实 logprob/replay。
- 后一 Sample的 prompt是 backend真实收到的 `P + A + S + E`。
- final accumulated-token cursor仍严格等于 latest raw checkpoint长度。
- `multi_samples=True` 时 `S` 位于后一 Sample的 prompt，不在任何 assistant response mask中。

`merge_samples` 增加显式 per-transition参数，例如：

```python
def merge_samples(
    samples: list[Sample],
    tokenizer,
    *,
    continuation_safe: Sequence[bool] | None = None,
) -> Sample:
    ...
```

`continuation_safe[i]` 表示原始 `samples[i]` 是否可以被 `samples[i + 1]` 延续。generic调用默认全部为 `False`。

session assembly在 `truncate_samples_by_total_tokens` 之后取 `surviving_records = records[:len(samples)]`，再临时推导标记，不写入 Sample或 wire schema：

```text
continuation_safe[i] =
    records[i].response.choices[0].finish_reason == "length"
    and records[i + 1] exists
```

successor record只有在 lazy-repaired prompt真实发给 backend且请求成功后才会存在，因此它是 runtime repair成功的持久证据。`_merge_sample_pair` 现有的 exact token-prefix断言提供第二层验证；replay-gap guard继续独立执行。

`_merge_sample_pair` 仅在当前 transition标记为 safe时允许 predecessor status为 `TRUNCATED`，`ABORTED` 永不放行。合并后 status仍取 successor status，因此“intermediate truncated，final completed”的最终 Sample为 `COMPLETED`，原始 truncation仍可从 records观察。

本地 `max_seq_len` 截断发生在 merge之前并丢弃后续 samples：

- successor仍保留时，`S` 已位于其 prompt并计入真实物理长度。
- successor output被裁剪时，其 prompt内 `S` 仍可进入 bridge。
- successor因预算被整体丢弃时，结果退回前一 raw truncated Sample，不物化 `S`。
- 预算产生的 terminal `TRUNCATED` 没有相邻 surviving successor，不能获得 continuation permission。

## Final truncation 与 mismatch 诊断

final turn为 `finish_reason == "length"` 时没有下一请求，因此 checkpoint、record和 Sample都只保存 raw completion，status保持 `TRUNCATED`。不应在 `merge_samples` 尾部追加 closure：没有后续训练 token依赖它，也没有真实 replay或 logprob覆盖它。

`compute_session_mismatch` 对 latest length record执行只读 virtual comparison：

1. 用同一 strict diff推导 canonical `S`。
2. 令 `actual_for_compare = session.token_ids + S`。
3. 用现有 comparator比较 canonical full history与 `actual_for_compare`。
4. `metadata["accumulated_token_ids"]` 继续返回 raw `session.token_ids`，供 sample cursor使用。

GLM placeholder属于 junction实现细节，不加入 `actual_for_compare`；这里只追加 chat template真正产生的 canonical tail。

intermediate truncation之后若已有成功 successor，latest checkpoint已经包含真实 repaired prompt，正常 comparison无需特殊处理。

现有任何直接拿 raw `accumulated_token_ids` 重新执行 canonical comparison 的 e2e或诊断工具，都必须改为复用同一只读 virtual-repair helper；不能为了维持“raw metadata等于comparison input”的旧假设而暴露tail sidecar或改写metadata。

## Loss、replay、长度与 CP

merged Sample的 `obs_len` 自然等于 `len(S + E)` 加上既有 turn间环境内容。tail与 tool/user/system文本共享 observation bridge语义：`loss_mask=0`、`rollout_log_probs=0.0`、`teacher_log_probs/opd_reverse_kl=0.0`，top-k OPD metadata填空列表。

上一 Sample不含 `S`，因此不需要伪造上一 turn的 `routed_experts` 或 `indexer_topk`。后一请求真实 input已经包含 `S`，其 full-prefix replay payload自然覆盖该位置；merged Sample继续采用后一 turn payload。PR #1672 的 `_introduces_replay_gap` guard不得移除。

merged `response_length` 按现有定义包含 observation bridge，因此会比旧 workaround多出 `len(S)`；`effective_response_length` 只统计有效 mask，不增加。依赖 raw `response_length` 的 length penalty或性能统计会有少量 drift，这是接受的既有 bridge语义。

PPO GAE会遍历 physical response span中的 interior zero-mask位置。默认 `gamma == 1.0` 且 `lambd == 1.0` 时不增加折扣；非默认值下 closure会像现有 observation token一样增加折扣步数，必须作为已知训练语义记录。

MTP当前直接使用完整 `batch["tokens"]` 作为 labels，不服从 rollout `loss_mask`。因此 intermediate closure和现有 observation token一样可能参与 MTP auxiliary loss。若验收要求“所有 objective都不能训练 synthetic token”，必须另加 MTP provenance mask或暂不支持该组合；仅要求 policy/OPD zero loss时，本设计已经满足。

CP不要求 response mask中的 `0` 只出现在两端，关键 invariant是 `len(loss_mask) == response_length` 及各 per-token字段等长。仍需用 regular zigzag CP和Ulysses CP测试 closure完全位于 shard内及跨 shard boundary两种情况。

## 验证计划

### CPU：tokenizer strict diff

在 `tests/fast/utils/chat_template_utils/test_tito_tokenizer.py` 增加真实 tokenizer truncation matrix：

- Qwen3在 `</think>` 前截断，断言 `S` 含 thinking closure、message end和必要换行，且 `A + S == C`。
- Qwen3在 `</think>` 后的 partial answer截断，断言 `S` 只含剩余 message closure。
- Qwen正常 `stop`、`tool_calls` 与已闭合输出不走 truncation diff，也不重复 newline。
- GLM-4.7覆盖 thinking前后截断，并用下一消息分别为 user、tool observation和system验证 placeholder被trim、真实 role opener只出现一次。
- MiniMax及每个注册TITO family至少覆盖一个answer内截断roundtrip；支持reasoning的family再覆盖thinking内截断。
- raw completion中间token不一致、checkpoint与record不一致、messages与record不一致时fail-loud。
- `finish_reason` 参数化覆盖 `length`、`stop`、`tool_calls`、`abort` 与未知值，证明只有 `length` 追加 `S`。
- 上一/下一请求使用不同 tools，断言tail render只读取上一 tools，incremental suffix只读取下一 tools。
- effective `chat_template_kwargs` 相同可继续，不同则在发backend前拒绝。

### CPU：trajectory、事务与 rollback

在 `tests/fast/router/test_linear_trajectory.py` 和 scripted session backend tests中覆盖：

- length response提交后，checkpoint精确等于 `P + A`，record和client response保持raw，没有sidecar。
- 下一次 `prepare_pretokenized` 返回 `P + A + S + E`，但调用后旧checkpoint仍为 `P + A`。
- 下一backend失败或不提交时，重试产生相同prompt；成功后新record.request和checkpoint才包含 `S`。
- rollback到raw truncated checkpoint后，从rollback后的 `records[-1]` 重新repair；不得使用被丢弃turn的finish reason、tools或kwargs。
- strict diff失败时不调用backend，不改变 `messages`、`trajectory_token_ids`、`records` 或 `num_assistant`。
- final length的 `compute_session_mismatch` 通过virtual repair不产生预期缺尾的hard mismatch，同时 `accumulated_token_ids` 仍为raw。

CPU backend必须真实返回被cut的 output IDs和 `finish_reason == "length"`；现有普通mock path忽略 `max_tokens`，不能用“传很小的值”伪装coverage。

### CPU：sample merge 与训练安全

在 `tests/fast/rollout/session/test_samples.py` 和 `tests/fast/router/test_session_samples_op.py` 增加：

- intermediate length → completed：per-turn Sample保持raw，merged Sample保留后续turn，精确断言 `S + E` 位于zero-mask/zero-logprob bridge。
- final length：Sample只含raw output并保持 `TRUNCATED`，没有terminal synthetic tail。
- `multi_samples=True`：前一Sample只含raw response，后一Sample的prompt含 `S`。
- generic `merge_samples` 未传permission时仍在 `TRUNCATED` 停止；显式safe transition才放行。
- `ABORTED`、未知status、本地 `max_seq_len` truncation和缺少successor record一概不放行。
- GLM真实trailing ambiguous token仍最多trim一个，不误删raw output；placeholder不出现在任何持久token序列。
- routed-expert/indexer replay开启时，后一full-prefix ndarray覆盖bridge并通过 `Sample.validate()`；缺失payload仍触发replay-gap safe stop。
- teacher logprob、OPD reverse KL和top-k metadata在bridge上的填充值与长度正确。
- train-data conversion后保留interior zero span，`effective_response_length` 不包含 `S`。

在 regular CP 与 `tests/fast/backends/training_utils/test_ulysses_cp_utils.py` 中构造含interior zero closure的mask，覆盖closure位于单shard及跨shard两类切法。

### GPU：真实 length finish

复用现有 Qwen3 与 GLM-4.7 session-server e2e lane，不新增模型启动。

在确定的中间步骤临时把单次请求 `max_tokens` 调得很小，并配合 `min_tokens/ignore_eos` 稳定得到 `finish_reason == "length"`；driver必须先断言确实发生length，再恢复sampling参数，追加带唯一marker的user follow-up并完成至少一个正常assistant turn。

最终断言：

- latest merged Sample为 `COMPLETED`，包含post-truncation marker和后续assistant输出。
- mask中存在“有效生成 → zero bridge → 后续有效生成”的区间。
- Qwen lane验证thinking/message closure，GLM lane验证ambiguous placeholder trim/replace。
- `tito_session_mismatch` 无hard mismatch。
- 旧workaround会停在length turn，因此后续有效mask与marker断言能够直接区分新旧行为。

## 方案对比

| 方案 | runtime真实性 | 下一轮prompt正确性 | 训练安全性 | 结论 |
| --- | --- | --- | --- | --- |
| turn提交时把tail写入checkpoint并保存record sidecar | checkpoint不再等于raw TokenInfo | 正确 | 需要额外alignment状态 | 不选 |
| 把tail写入上一response或per-turn Sample | 破坏logprob与公开response语义 | 正确 | replay缺行且可能成为loss target | 不选 |
| 只在最终 `merge_samples` 创建tail | runtime保持raw | 错误，下一轮从未看到closure | 无法证明真实rollout上下文 | 不选 |
| 下一 `merge_tokens` lazy repair，最终merge只赋训练语义 | 上一turn完全raw | 正确，closure进入真实 `input_ids` | 复用现有zero bridge与后一replay | 选择 |

固定模型closure表也不选。它无法同时覆盖thinking前后截断，并会随chat template漂移；通用strict diff负责assistant serialization，family hook只处理无法由该serialization表达的junction quirk。

## 实施顺序

1. 将 `TITOTokenizer.merge_tokens` 重构为base template method，实现 previous-turn context和strict canonical tail diff，再迁移Qwen、GLM、MiniMax junction hook。
2. 在 `LinearTrajectory.prepare_pretokenized` 的rollback之后从latest record构造context，保持 `SessionCore` Phase 3 raw commit不变，并冻结session内effective kwargs。
3. 给 `compute_session_mismatch` 增加final-length只读virtual repair，保持 `accumulated_token_ids` raw。
4. 给 `merge_samples` 增加per-transition `continuation_safe`，由server-side surviving records临时推导，不改变Sample或wire schema。
5. 补齐tokenizer、trajectory、sample/replay/OPD、CP CPU tests。
6. 修改现有Qwen3和GLM-4.7 GPU lane，真实制造一次length finish并证明后续turn进入merged Sample。

每一步都必须独立保持raw response字段和Sample shape invariant；不得先全局放宽 `TRUNCATED` merge再依赖后续patch补安全条件。

## 风险与待确认项

### Parser append-only能力

reasoning/tool parser可能在截断的复合或并行tool call中保留已完成call却丢弃未完成片段，使OAI message无法重建raw completion。strict diff会在第一次continuation前失败。合入前应对所有声明支持TITO的family跑reasoning、plain text和tool-call truncation matrix；高频失败应修parser保留raw partial content，而不是让TITO猜测。

### Lazy failure时机

lazy repair优先保证上一response和TokenInfo真实性，代价是不可修复性在下一请求才暴露。是否增加turn提交时的只读preflight validation是产品选择；即使增加，也不能把tail写回runtime state。

### MTP与非默认PPO折扣

如果“synthetic token必须loss mask为0”只约束rollout policy/OPD，本设计满足；如果约束所有objective，MTP需要额外工作。非默认 `gamma/lambd` 下physical bridge长度也会影响GAE折扣，这与已有observation span同类，但需训练owner确认。

### Raw length drift

intermediate closure进入真实下一prompt后，会增加总token数和merged raw `response_length`，可能轻微影响长度惩罚、max-seq预算与性能统计；`effective_response_length` 不变。任何按真实模型生成长度计费的调用方都应读取生成元数据，而不是从merged physical span反推。
