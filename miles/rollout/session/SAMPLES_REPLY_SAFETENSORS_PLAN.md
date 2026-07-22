# 用 Safetensors 替换 Samples Reply 信封格式

状态：拟议的修改方案；生产代码尚未改动。决策记录：本方案曾一度被 JSON+base64 替代方案取代（理由是最简单、engine 一跳已有同款惯用法），随后在同 fixture 实测中 base64 被否决——combined 达现行 envelope 的 5.1 倍，瓶颈是 137 MB base64 字符串穿过 CPython json 编解码的逐字符文本扫描（`json.dumps` 单项 773 ms）；safetensors 原型实测反而比现行 envelope 快 1.8 倍、内存峰值减半。本方案据该实测数据恢复为定案，实测明细见"方案选型"。

目标：PR `radixark/miles#1605`，分支 `refactor/session-sample-assembly`（base `main`），当前 head `cb11a4cb69`（rebase 到新 main 后的同两条 commit；本文引用的全部锚定文件在新旧 head 间逐字节一致）。实施时发现该 rebase 带入了一个既有 breakage：#1703 把 `Sample.session_id` 改名为 `routing_key`，而 `samples.py` 的 `TEMPLATE_FIELDS` 与两个测试文件未同步，字段划分断言在 import 时即失败——本改动顺带修复（划分表与测试同步为 `routing_key`），这正是该断言要抓的字段漂移。改动形态是对该分支的 refactor：删除 `miles/rollout/session/samples.py` 中自 `samples.py:165` 的注释 `# Wire envelope of the samples reply: u64-length-prefixed JSON meta, then the raw binary segments (no base64).` 起至文件末尾（`samples.py:348`）的全部实现，重建为新文件 `miles/rollout/session/samples_codec.py`——一个操作原语模块：纯函数对 `encode_samples_reply`/`decode_samples_reply`（`bytes` ↔ `SamplesReply`）及其 wire 契约，不含 HTTP、session 状态或组装逻辑。`samples.py` 只保留组装与截断函数（`samples.py:1-163`），不再含任何 wire 概念。该拆分可行的前提已核实：组装半区对编解码符号零代码引用（仅 docstring 提及），迁移无耦合。

读者：session-server 样本组装路径的 reviewer 和实现者。读完本文后，应能判断 wire 格式选型、精确的改动面、兼容性边界，以及验证是否闭合了相关的正确性与传输成本风险。

## 动机与成功标准

PR 1605 正确地把 records 到 `Sample` 的组装移到了拥有 session 的进程中，但它最终从 session server 回复 driver 时使用了一个仓库本地的二进制容器：8 字节的 metadata 长度、包含手工计算的 dtype/shape/offset/nbytes 值的 JSON，以及拼接起来的原始 ndarray 字节。在受支持的路径上尚未观察到确定性的数据损坏，但 framing、张量布局校验和重建全部由本地实现和维护，而这本是一个标准的张量序列化问题。

目标是仅用 `safetensors.numpy` 替换那个自定义容器，使仓库只拥有 `Sample` 字段与命名张量之间的小型映射，而 framing、dtype/shape 存储、越界校验和字节解析由一个现成的 tensor-only 包负责。这个改动的首要目的是降低正确性风险和代码所有权；编解码提速与更低的内存峰值是实测确认的附带收益（数据见"方案选型"），driver 是单进程，解码耗时与瞬时内存直接决定它的 event loop 阻塞时长。

Miles 侧的映射代码不 overfit 到当前的字段集合：每字段的不可约知识（归一化 dtype、严格 dtype、解码还原形态、`null` 还原值）以一张声明式 spec 表表达，编码器和解码器是这张表上的两个循环，而不是逐字段复制的分支代码；新增张量字段只需加一行表。spec 表与标量字段列表的并集在 import 时断言等于 `COMPUTED_FIELDS`，使新增字段而未提供 wire 表示大声失败——这同时堵上现行代码的一个真实缺口：今天往 `COMPUTED_FIELDS` 加字段但忘改编码器，字段会静默掉出 wire，解码后保持 template 值。

当以下条件全部满足时，改动即为完成：

- `encode_envelope`、`decode_envelope`、`_LEN`、`_read_segment` 以及所有手工的 offset/dtype/shape/nbytes 记账代码全部删除。
- `encode_samples_reply` 仍返回 `bytes`，`decode_samples_reply` 仍返回相同的 `SamplesReply`，且所有解码出的 `Sample` 值和 Python/NumPy 类型保持不变（数组 flags 不在契约内：解码数组从 `np.frombuffer` 的只读视图变为 `safetensors.numpy.load` 的可写副本，顺带消除 `torch.from_numpy` 的只读警告）；二者及 wire 契约整体迁至 `samples_codec.py`，函数签名不变，import 路径变化（该端点未发布，无兼容负担），全部 import 方同 PR 更新。
- `samples_codec.py` 是自包含的操作原语：除 `Sample`/NumPy/safetensors 外不依赖 session 包的其他模块，`samples.py` 不再出现 wire、envelope、tensor 名等概念。
- `POST /sessions/{session_id}/samples`、其请求 JSON、状态码、`application/octet-stream`、超时、不重试行为、session 删除、组装顺序、截断、merge 以及 driver 侧 overlay 保持不变。
- 普通的 `safetensors.numpy.load(payload)` 无需任何 Miles 特有的外层 framing 即可打开该回复。
- 非连续的 ndarray 输入在编解码边界处显式连续化之后可以精确往返。
- 畸形容器和缺失的必需张量引用会大声失败，而不是被解释为合法的 `None` 值。
- 在 `COMPUTED_FIELDS` 中新增字段而未在 spec 表或标量列表中提供 wire 表示，会在 import 时被断言拦截，而不是静默掉出 wire。
- 在约 100 MiB 的生产规模回复上，payload 相对当前信封的增长不超过 `max(4 KiB, 0.01%)`，且编码加解码时间的中位数不超过当前信封基线的 110%。

## 已验证的当前数据流

有两个不同的操作在讨论中都被称为 "merge"，本方案对二者都不做改动：

1. `SessionCore.collect_samples` 把若干 turn records 转换为逐 turn 的 samples，并在 `multi_samples=False` 时于序列化前调用 `merge_samples`。
2. 已组装完成的结果被序列化一次，从拥有 session 的 session server 传输到 Miles rollout driver。

本方案只改动第 2 步中的序列化容器。

```text
Session records in the owning process
    -> SessionCore.collect_samples
    -> compute_samples_from_openai_records
    -> truncate_samples_by_total_tokens, when requested
    -> merge_samples, when multi_samples=False
    -> encode_samples_reply                         [change starts]
    -> HTTP application/octet-stream
    -> OpenAIEndpointTracer.collect_samples
    -> decode_samples_reply                         [change ends]
    -> Sample objects with driver template overlay
    -> RolloutManager.convert_samples_to_train_data
    -> split_train_data_by_dp / Ray
    -> training actor get_rollout_data
    -> NumPy R3 converted with torch.from_numpy
    -> fill_replay_data aligns R3 with training microbatches
    -> Megatron replay queues feed the model forward
```

相关代码锚点：

- HTTP 路由注册在 `miles/rollout/session/sessions.py:75`，委托给 `SessionCore.collect_samples`。
- 服务端的计算、截断、可选 merge 和编码运行在 `miles/rollout/session/core.py:146-183`。
- 原始字节的客户端 POST 与解码运行在 `miles/rollout/generate_utils/openai_endpoint_utils.py:54-83`。
- 整个 wire 编解码段自 `miles/rollout/session/samples.py:165` 的 wire-envelope 注释起延伸到文件末尾（`samples.py:348`），即本次 refactor 的删除重建区段：`_LEN`/`encode_envelope`/`decode_envelope`（165-185）、`COMPUTED_FIELDS`/`TEMPLATE_FIELDS` 及划分完整性 assert（188-232）、`_SEGMENT_DTYPES`/`_SEGMENT_FIELDS`（234-235）、`_OPD_STUDENT_TOP_LOGPROBS_KEY`（237）、`SamplesReply`（240-246）、`encode_samples_reply`/`decode_samples_reply`/`_read_segment`（249-316）、`_assert_overlay_template_defaults`（319-348）。
- 解码后的 R3 在 `miles/ray/rollout/train_data_conversion.py:70-74` 进入训练数据 dict，在 `miles/ray/rollout/train_data_conversion.py:146-167` 经过 DP 切分后仍保留，并在 `miles/backends/training_utils/data.py:110-113` 从 NumPy 转换为 torch。
- Megatron 并不把 R3 作为普通的 `GPTModel` batch 参数接收。`miles/backends/megatron_utils/actor.py:394-409` 调用 `fill_replay_data`，后者按相同的 microbatch 顺序读取 R3 字段、校验 token 维度、做 padding/切片，并在模型 forward 之前把它们记录进 replay 队列。

因此下游的答案是精确的：R3 通过 `rollout_data` 和 `DataIterator` 参与训练 batch 准备，随后 Megatron 通过 replay 队列而不是常规的 `GPTModel` 输入 key 列表消费它。序列化器的替换必须保持 ndarray 的 shape、dtype、值、`None` 和零尺寸张量不变，以使这条下游路径不受影响。

## 为什么当前的信封是不必要的风险

当前编解码器拥有四个相互独立的正确性机制：

- `_LEN = struct.Struct(">Q")` 定义并解析一个私有外层帧。
- 编码器为每个张量计算字节 offset 和字节长度。
- JSON 在张量字节之外重复存储每个张量的 dtype 和 shape。
- 解码器信任这些值，切分 body，调用 `np.frombuffer`，再 reshape 结果。

解码器自身并不证明各 segment 互不重叠、所有被引用的范围都在界内、body 被完全覆盖，或者 `nbytes == prod(shape) * itemsize`；这些失败被留给切片或 reshape 的偶然行为。补齐所有这些检查会增加自定义协议代码量，这与可靠性动机相冲突。

`safetensors` 已经把命名张量的 dtype、shape、offsets 和数据存储在一个经过校验的格式中。用 `safetensors 0.8.0` 做的一次本地探测确认：截断的 payload、尾随字节和空 payload 都会以 `SafetensorError` 被拒绝。剩下的由 Miles 拥有的契约只剩"哪个 `Sample` 字段映射到哪个张量名"和"哪些标量字段仍留在 JSON 中"。

性能上信封也不占优：它的纯 Python 组装（`tobytes()` → `b"".join()` → 与 meta 前缀拼接）对 100 MiB 数据做约三次全量拷贝，实测 combined 249 ms、编码侧 Python 层内存峰值 313 MB；safetensors 原型在同一 fixture 上 combined 142 ms、峰值 106 MB（数据见"方案选型"）。

## 约束与显式非目标

### 硬约束

- wire 字节可以原子地改变，因为该端点是在当前未合入的分支上引入的，没有必须保持兼容的已发布 reader。
- Python 函数签名、HTTP 契约、解码出的 `SamplesReply` 以及 `Sample` overlay 语义不得改变。
- `tokens` 在 wire 上必须仍归一化为 `int64`，并恢复为 Python 的 `list[int]`。
- `rollout_log_probs` 在 wire 上必须仍归一化为 `float64` 并恢复为 Python list，保持当前的往返精度。
- `rollout_routed_experts` 和 `rollout_indexer_topk` 必须仍是 NumPy 数组，保持 `int32` 与原始 shape 的当前字段契约；预期之外的 R3 dtype 必须失败，而不是被静默转换。
- 解码侧必须把每个张量字段的 dtype 校验在 spec 表声明的契约上，容器里 dtype 不符必须 raise。
- 合法的 `None` 与合法的零尺寸张量必须保持可区分。
- metadata 引用了却缺失的张量必须 raise；缺失绝不能静默变成 `None`。
- 编码器在把每个张量交给 `safetensors` 之前必须将其物化为连续内存。
- 该包必须是直接依赖，因为生产代码直接 import 它。

### 偏好约束

- 删除的自定义编解码代码要多于新增的。
- 编解码器作为操作原语独立成 `miles/rollout/session/samples_codec.py`：一对纯函数加它们的 wire 契约常量，不 import HTTP 层、session 状态或组装函数；`samples.py` 反向也不 import codec。
- 每字段知识用声明式 spec 表表达而不是逐字段复制的分支；泛化止步于"表驱动 + import 时覆盖断言"，不做通用对象序列化、按类型自动分发或 codec 注册表。
- 逻辑标量 metadata 的形状尽量贴近当前表示，使 overlay 代码保持肉眼可审。
- 不承诺 zero-copy 行为；实测 payload 大小和延迟，而不是从包内部实现去推断。

### 非目标

- 不改变 session server 内部几十个 turn 级结果的计算、截断或 merge 方式。
- 不改变哪些 `Sample` 字段被归类为 `COMPUTED_FIELDS` 或 `TEMPLATE_FIELDS`。
- 不改动 Mooncake，也不引入新的传输层；Mooncake 并不能消除该边界上对定义良好的内存 payload 格式的需求。
- 不添加压缩、base64、pickle、分块、流式、共享内存或通用的 codec 注册表。
- 不添加双读、版本协商、格式标志位或回退到旧信封的兜底。
- 不改变现有的客户端生命周期，其中 DELETE 在 payload 解码之前尝试。
- 不声称解决 PR 1605 中独立的 session-server fan-in 和峰值 RSS 容量问题。

## 方案选型

| 候选方案 | 可靠性面 | 适配复杂度 | Payload 特征 | 结论 |
| --- | --- | --- | --- | --- |
| 保留自定义信封并补充校验 | Miles 仍将永久拥有 framing、边界、dtype、shape 和兼容性 | 最高 | 原始大小 | 拒绝：维护问题原样保留 |
| JSON 加 base64（`pybase64`） | JSON 自定界，字节校验归 b64decode/frombuffer/reshape | 低：无新依赖，engine 一跳已有同款惯用法 | 张量字节 +33% | 拒绝：实测 combined 为 envelope 的 5.1 倍（见下） |
| `pickle` | 能表示一切，但允许通用对象反序列化 | 最低 | 原始大小 | 拒绝：契约比所需更宽且更不安全 |
| `numpy.savez` | 现成的 NumPy 格式，配合 `allow_pickle=False` 对数值数组安全 | 需要 `BytesIO`、ZIP 归档处理和显式的 pickle 纪律 | 原始大小加 ZIP metadata | 可行，但不是最窄的仓库本地契合 |
| `safetensors.numpy` | tensor-only 的已校验格式，Miles 其他地方已在使用 | 直接的内存 `bytes` save/load API | 原始大小加一个小 header | 选定 |

三个主要候选在同一 fixture（100,000 token、64 层 top-k 4 的 `int32` R3 张量，约 102 MB 张量数据）上用忠实复刻的独立原型实测（envelope 复刻结果 249 ms 与真实代码 smoke 的 269 ms 吻合，证明复刻可信）：

| 方案 | payload | encode 中位数 | decode 中位数 | combined | encode/decode Python 层内存峰值 |
| --- | --- | --- | --- | --- | --- |
| envelope（现行） | 104.2 MB | 149 ms | 100 ms | 249 ms | 313 / 216 MB |
| safetensors | 104.2 MB | 95 ms | 48 ms | 142 ms | 106 / 112 MB |
| JSON + base64 | 138.9 MB | 994 ms | 262 ms | 1265 ms | 449 / 278 MB |

差距的机制：safetensors 与 envelope 都把二进制放在 JSON 之外、只走 memcpy 量级的操作，其中 safetensors 由 Rust 侧单缓冲写入/读出，比纯 Python 的三次全量拷贝更少动数据；base64 则强迫 102 MB 字节膨胀成 137 MB ASCII 后穿过 CPython json 的逐字符转义扫描（`json.dumps` 单项 773 ms，约 180 MB/s），这笔钱正是信封当年把二进制放到 JSON 外面所回避的。不带 R3 的常见回复（payload 约 2 MB）三个方案都在 30 ms 以内，选型由带 R3 的大回复决定。decode 侧运行在单进程 driver 的 event loop 里，safetensors 把每 100 MiB 回复的阻塞时长从 100 ms 降到 48 ms。

仓库已在 `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/delta.py` 中 import `safetensors.numpy`，当前环境通过 `transformers` 带有 `safetensors 0.8.0`。实现仍会把 `safetensors` 加入 `requirements.txt`，因为生产代码的直接 import 不得依赖一个无关包的传递依赖。

内存态的 `safetensors.numpy.load(data: bytes)` API 返回张量，但不暴露 safetensors 的 header metadata。手工解析 header 或写临时文件会重新制造正在被移除的复杂度，因此回复中的标量 JSON 会存成一个名为 `_samples_meta` 的普通 `uint8` 张量。不使用 `__metadata__` 这个名字，因为它被 safetensors 格式保留。

## 提议的 wire schema

payload 是恰好一个 safetensors 字节缓冲区。它没有任何 Miles 特有的前缀或后缀。

张量名是确定性的：

- `_samples_meta` 以 rank 为一的 `uint8` 数组存放紧凑的 UTF-8 JSON。
- `sample.{index}.tokens` 存放单个 sample 的 `int64` token 数组。
- `sample.{index}.rollout_log_probs` 在非 `None` 时存放单个 sample 的 `float64` log-probability 数组。
- `sample.{index}.rollout_routed_experts` 在非 `None` 时存放 routed-expert ndarray。
- `sample.{index}.rollout_indexer_topk` 在非 `None` 时存放 indexer ndarray。

JSON 张量保留标量/list 字段，并把每个张量字段映射为其精确张量名或 `null`：

```json
{
  "samples": [
    {
      "response": "...",
      "response_length": 2,
      "loss_mask": [1, 1],
      "status": "completed",
      "weight_versions": ["w1"],
      "prefix_cache_info": {
        "cached_tokens": 2,
        "total_prompt_tokens": 3
      },
      "tensors": {
        "tokens": "sample.0.tokens",
        "rollout_log_probs": "sample.0.rollout_log_probs",
        "rollout_routed_experts": "sample.0.rollout_routed_experts",
        "rollout_indexer_topk": null
      }
    }
  ],
  "session_metadata": {},
  "empty_reason": null
}
```

schema 有意不含版本字段。编码器和解码器在同一个 revision 中一起改动，不支持新旧混布，而版本分支会在没有现存 reader 需要服务的情况下平添代码。格式本身对字段集合零假设：`tensors` 映射的 key 就是字段名，新增字段不需要改变 schema 结构。

### 字段规则

每字段的 wire 行为由 spec 表声明（见逐文件修改），当前四行的语义为：

| 字段 | 编码表示 | 解码表示 | `null` 还原值 | dtype 契约 |
| --- | --- | --- | --- | --- |
| `tokens` | 归一化 `np.int64` 后 `np.ascontiguousarray` | `.tolist()` | `[]` | `int64` |
| `rollout_log_probs` | 归一化 `np.float64` 后 `np.ascontiguousarray` | `.tolist()` | `None` | `float64`，保持无损 f64 往返 |
| `rollout_routed_experts` | 严格校验 `int32` 后 `np.ascontiguousarray` | `np.ndarray` | `None` | `int32`，编码解码两侧都校验 |
| `rollout_indexer_topk` | 严格校验 `int32` 后 `np.ascontiguousarray` | `np.ndarray` | `None` | `int32`，编码解码两侧都校验 |

`_samples_meta` 自身必须是 rank 为一的 `uint8`，否则解码器 raise `ValueError`。

`None` 只由 JSON `null` 表示；被引用却缺失的张量属于畸形数据。零尺寸的 ndarray 由一个带零维度的真实 safetensors 条目表示，且必须解码回 ndarray 而非 `None`。

### 为什么连续化转换是强制的

已安装的 NumPy 适配器要求连续的稠密张量，但它并不能可靠地在序列化前拒绝所有非连续视图。一次独立的本地探测把 transpose、strided-slice 和 reverse 视图序列化时未抛错，随后解码出与输入不同的值。因此在唯一的编解码边界处做 `np.ascontiguousarray` 是正确性要求，即使常见截断路径产出的数组通常是连续的。

## 逐文件详细修改

### 1. `requirements.txt`

在现有按字母序排列的依赖列表中加入 `safetensors>=0.8.0`（带版本下界：畸形容器的精确异常契约在 0.8.0 上实测验证，7 节的畸形测试会拦截未来版本的行为漂移）。不要加第二个包、optional extra 或条件 marker；这与仓库现有的直接使用和依赖风格一致。

### 2. `miles/rollout/session/samples.py`

只删不建：删除自 `samples.py:165` 的 wire-envelope 注释起至文件末尾的整段，以及 `import struct`、`import dataclasses`/`deepcopy` 等仅被该段使用的 import（逐一核对后删）。组装与截断函数（`compute_samples_from_openai_records`、`_compute_sample_from_openai_record`、`truncate_samples_by_total_tokens`）逐字不动。模块 docstring 从 "Training-sample assembly and its wire codec" 收窄为仅描述组装，wire 相关的段落移入新模块的 docstring。

### 3. `miles/rollout/session/samples_codec.py`（新文件）

samples reply 的操作原语：`encode_samples_reply`/`decode_samples_reply` 纯函数对及其 wire 契约。依赖面只有 `Sample`、NumPy、`safetensors.numpy` 和标准库；禁止 import HTTP 层、session 状态或 `samples.py`。

从被删区段原样迁入（语义与代码均不变）：

- `COMPUTED_FIELDS`、`TEMPLATE_FIELDS` 及其划分完整性 assert。
- `_OPD_STUDENT_TOP_LOGPROBS_KEY`（仅被 `_assert_overlay_template_defaults` 引用，二者都在本区段内）。
- `SamplesReply` dataclass。
- `_assert_overlay_template_defaults` 及其 docstring。

不迁入（随信封一起消失）：`_LEN`、`encode_envelope`、`decode_envelope`、`_read_segment`、wire-envelope 注释本身，以及所有 offset/nbytes 记账。

新模块的常量：

- `_SAMPLES_META_KEY = "_samples_meta"`。
- 用一张 spec 表取代 `_SEGMENT_FIELDS`/`_SEGMENT_DTYPES`，并声明标量字段列表和覆盖断言：

```python
@dataclasses.dataclass(frozen=True)
class _TensorSpec:
    normalize_dtype: np.dtype | None  # np.asarray target on encode; None keeps the caller's dtype
    wire_dtype: np.dtype              # pinned on both sides; encode-side mismatch check subsumes the R3 strict check
    restore_list: bool                # decode: .tolist() vs keep ndarray
    null_factory: Callable[[], object]  # decode value for JSON null; factory so no instance is shared across samples


_TENSOR_SPECS = {
    "tokens": _TensorSpec(np.dtype(np.int64), np.dtype(np.int64), True, list),
    "rollout_log_probs": _TensorSpec(np.dtype(np.float64), np.dtype(np.float64), True, lambda: None),
    "rollout_routed_experts": _TensorSpec(None, np.dtype(np.int32), False, lambda: None),
    "rollout_indexer_topk": _TensorSpec(None, np.dtype(np.int32), False, lambda: None),
}
_SCALAR_FIELDS = ("response", "response_length", "loss_mask", "status", "weight_versions", "prefix_cache_info")

assert set(_TENSOR_SPECS) | set(_SCALAR_FIELDS) == set(COMPUTED_FIELDS) and not set(_TENSOR_SPECS) & set(
    _SCALAR_FIELDS
), "every COMPUTED field must have exactly one wire representation (tensor spec or scalar)"
```

`encode_samples_reply` 将按以下顺序重写：

1. 对 samples 做 enumerate，使每个张量名 `sample.{index}.{field}` 稳定且唯一。
2. 对 `_TENSOR_SPECS` 循环：源值为 `None` 时在 sample JSON 的 `tensors` 映射写 `null`；否则 `arr = np.asarray(value, dtype=spec.normalize_dtype)`，若 `spec.strict_dtype` 非空且 `arr.dtype` 不符则 raise `ValueError`，然后 `tensors[name] = np.ascontiguousarray(arr)` 并记录张量名。
3. 标量字段按 `_SCALAR_FIELDS` 顺序写入 sample JSON，其中 `status` 取 `.value`、`prefix_cache_info` 取 `.to_dict()`，其余直接赋值——与今天完全一致。
4. 顶层的 `session_metadata` 和 `empty_reason` 与今天完全一致地保留。
5. 用 `json.dumps(..., ensure_ascii=False, separators=(",", ":"))` 序列化，编码 UTF-8，把字节以 `np.frombuffer(meta_bytes, dtype=np.uint8)` 暴露为 rank 为一的 `uint8` ndarray，存到 `_SAMPLES_META_KEY` 下。
6. 直接返回 `safetensors.numpy.save(tensors)`。

`decode_samples_reply` 将按以下顺序重写：

1. 调用 `safetensors.numpy.load(payload)`，让无效容器的 `SafetensorError` 向上传播。
2. 用直接 pop 取出 `_SAMPLES_META_KEY`，使其缺失时抛 `KeyError`。
3. 校验 metadata 张量是 rank 为一且为 `uint8`；否则 raise `ValueError`。
4. 解码 UTF-8 并调用 `json.loads`。
5. 保持当前的空回复行为：当 `meta["samples"]` 为空时跳过 `_assert_overlay_template_defaults`，返回 `empty_reason` 和 `session_metadata`。
6. 对每个 sample：deepcopy `input_sample`；对 `_TENSOR_SPECS` 循环，读取 JSON `tensors` 映射的引用，`null` 时赋 `spec.null_factory()`（每 sample 新实例，绝不共享）；否则先校验引用名等于重算的确定性名称 `sample.{index}.{field}`（不符 raise `ValueError`，使同 dtype 字段的引用互换可检测），再用 `tensors.pop(name)` 消费（缺失抛 `KeyError`），校验 `arr.dtype == spec.wire_dtype` 否则 raise `ValueError`，最后按 `spec.restore_list` 决定 `.tolist()` 还是保留 ndarray。
7. 标量字段按 `_SCALAR_FIELDS` 循环赋值，`status` 经 `Sample.Status(...)`、`prefix_cache_info` 经 `Sample.PrefixCacheInfo.from_dict(...)` 还原——与今天完全一致。
8. 全部 sample 消费完毕后，若容器仍有未被引用的剩余张量则 raise `ValueError`（fail-closed 的廉价部分：容器内容与 metadata 引用必须精确互覆盖）。
9. 返回类型和字段含义均不变的 `SamplesReply`。

新增张量字段的流程从此是：归入 `COMPUTED_FIELDS`，在 `_TENSOR_SPECS` 加一行；漏掉任何一步都被 import 断言拦截。不要在此之上再加通用对象序列化、按类型自动分发或跨模块 codec 注册表。

模块 docstring 描述命名的 safetensors 张量与 overlay 契约（接收自 `samples.py` docstring 中 wire 相关的段落），不再出现长度前缀帧和原始二进制 segments 的措辞。

### 4. `miles/rollout/session/core.py`

`core.py:25` 的 import 拆为两条：`compute_samples_from_openai_records`/`truncate_samples_by_total_tokens` 仍来自 `.samples`，`encode_samples_reply` 改自 `.samples_codec`。保持 `_samples_response` 的字节级行为不变，仅把其过时的 docstring 从 "one codec envelope (JSON meta + raw binary segments)" 更新为 "one safetensors binary payload"。不改路由、response class、状态码或 media type。

### 5. `miles/utils/http_utils.py`

保持 `post_bytes_no_retry` 不变，把措辞 "the reply is a binary envelope" 更新为 "the reply is a binary payload"，以免之后的代码搜索暗示已移除的私有信封仍然存在。这是纯文档改动。

### 6. 仅 import 行更新的文件

以下三处把 `miles.rollout.session.samples` 改为 `miles.rollout.session.samples_codec`，除 import 行外不动任何断言或行为：

- `miles/rollout/generate_utils/openai_endpoint_utils.py:10`（`SamplesReply`、`decode_samples_reply`）。
- `tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py:19`（`encode_samples_reply`）。
- `tests/fast/router/test_session_samples_op.py:30`（`decode_samples_reply`）。

`tests/fast/rollout/session/test_samples.py:13` import 的是组装函数，留在 `samples.py`，不改。

### 7. `tests/fast/rollout/session/test_samples_codec.py`

import 改自 `samples_codec`。保留所有现有断言：`test_field_partition_is_total_and_disjoint`、`test_round_trip_overlays_computed_and_keeps_template`、`test_multi_sample_reply_keeps_per_sample_segments`（改名为 `..._tensors`）、`test_empty_reply_round_trips_reason_and_skips_defaults_guard`、`test_defaults_guard_rejects_evolved_template`。把过时的测试/注释措辞从 "segments" 改为 "tensors"。

新增以下聚焦覆盖：

1. `test_safetensors_container_round_trips_non_contiguous_replay_tensors`：把 `rollout_routed_experts` 和 `rollout_indexer_topk` 都设为非连续的 `int32` 视图，编码一次，用 `safetensors.numpy.load` 直接打开字节，断言精确的预期张量 key 和 `_samples_meta` 的 dtype/shape，再经 Miles 解码，断言精确的值、shape、dtype 和 Python list 还原。
2. `test_zero_size_tensor_is_distinct_from_none`：把一个空的 `(0, layers, topk)` R3 ndarray 与一个 `None` 张量字段一起编码，断言解码后前者返回空 ndarray、后者返回 `None`。
3. `test_null_tokens_restore_fresh_empty_lists`：构造两个 tokens 为 `null` 的 sample，断言解码出的两个 `[]` 不是同一个对象（`null_factory` 反共享保证）。
4. `test_malformed_safetensors_reply_fails_loudly`：参数化以下用例——截断/空的 safetensors 缓冲区要求 `SafetensorError`；缺少 `_samples_meta` 的合法容器要求 `KeyError`；`_samples_meta` 为非 `uint8` dtype 或非 rank 一 shape 要求 `ValueError`；合法 metadata 引用缺失张量要求 `KeyError`；容器内张量 dtype 与 spec 契约不符要求 `ValueError`；引用名与确定性名称不符要求 `ValueError`；容器携带未被引用的剩余张量要求 `ValueError`。每个用例断言其精确的异常类型，而不是接受一个宽泛的并集。
5. `test_encode_rejects_non_int32_replay_dtype`：R3 字段传入 `int64` ndarray，要求编码侧 `ValueError`，而不是静默转换。

不要复制现有的 router golden 断言，也不要仅为测试添加包装层。

### 有意不改的文件

- `miles/rollout/session/sessions.py`：路由和请求解析保持不变。
- `miles/rollout/generate_utils/openai_endpoint_utils.py`：除第 6 节的 import 行外，原始 POST、DELETE 生命周期和对 `decode_samples_reply` 的调用保持不变。
- `miles/rollout/generate_hub/agentic_tool_call.py`：empty-reason 映射和 metadata 优先级保持不变。
- `miles/ray/rollout/train_data_conversion.py`：解码后的 `Sample` 契约得到保持。
- `miles/backends/training_utils/data.py`、`miles/backends/training_utils/replay_data.py` 和 Megatron 代码：ndarray 消费方保持不变。
- `tests/fast/router/test_session_samples_op.py` 和 `tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py`：除第 6 节的 import 行外，它们是验证目标，不是编辑目标。
- `tests/manual/session/bench_session_server_overhead.py`：当前 benchmark 只压测 chat 和可选的 `GET /sessions/{id}`，不调用 `/samples`，因此不会被悄悄挪用为编解码证据。
- `miles/rollout/session/TRUNCATION_HANDLING_DESIGN.md`：这个已存在的未跟踪用户文件与本改动无关，禁止触碰。

## 失败行为

| 失败 | 预期归属与结果 |
| --- | --- |
| 无效/截断/带尾随字节的 safetensors 字节 | `safetensors.numpy.load` 抛出 `SafetensorError`；客户端采集大声失败 |
| 缺少 `_samples_meta` | 直接 pop 抛出 `KeyError` |
| `_samples_meta` 的 dtype 或 rank 错误 | Miles 解码器抛出 `ValueError` |
| 无效的 UTF-8 或 JSON | 标准 decode/JSON 异常向上传播 |
| JSON 指名的张量不存在 | `tensors.pop` 抛出 `KeyError` |
| JSON 引用名与确定性名称 `sample.{index}.{field}` 不符 | Miles 解码器抛出 `ValueError` |
| 容器携带未被任何引用消费的剩余张量 | Miles 解码器抛出 `ValueError` |
| 容器内张量 dtype 与 spec 契约不符 | Miles 解码器抛出 `ValueError` |
| JSON 显式存储 `null` | 解码器返回该字段 spec 声明的 `null` 还原值 |
| R3 dtype 不是契约约定的 `int32` | Miles 编码器抛出 `ValueError`，而不是静默转换索引 |
| `COMPUTED_FIELDS` 新增字段但 spec/标量列表未覆盖 | import 时断言失败 |
| 编码时其他不受支持或无效的 ndarray | `np.ascontiguousarray` 或 `safetensors.numpy.save` 抛错；服务端返回未处理的 500，与今天的编解码 bug 表现一致 |

不尝试回退到旧信封的兜底。`OpenAIEndpointTracer.collect_samples` 仍在其 `finally` 块中于解码之前尝试 session DELETE，因此解码失败在拥有的 session 被删除后依旧不可重试。这是既有的生命周期行为，显式处于本次仅编解码改动的范围之外。

## 实施顺序

1. 在编辑生产代码之前，在固定的 benchmark fixture 上采集当前信封的 payload 长度和编解码耗时。
2. 添加直接依赖，删除 `samples.py` 自 `samples.py:165` 注释起的整段，新建 `samples_codec.py` 承载 safetensors 编解码段（含原样迁入的字段划分、`SamplesReply` 和 overlay 守卫），并更新第 4、6、7 节列出的全部 import 方。
3. 更新 `core.py`、`http_utils.py` 和编解码测试名/注释中的相邻术语。
4. 添加正向、零尺寸、畸形 payload 和编码侧 dtype 拒绝的编解码测试。
5. 先运行聚焦的编解码测试；只修复由新格式引起的失败。
6. 运行 HTTP router 和客户端生命周期测试，证明编解码器之外的边界没有移动。
7. 运行下游转换和 replay 测试，确认被保持的 NumPy 契约仍能抵达训练 replay 路径。
8. 用 safetensors 重跑同一 benchmark fixture，套用 payload/延迟门限。
9. 对恰好被触碰的文件运行 pre-commit，并检查最终 diff 中是否存在任何兼容分支、死信封代码或无关编辑。

编码器和解码器必须一起落地。一侧旧一侧新的中间状态在设计上即为无效。

## 验证计划

### 功能测试

```bash
pytest -q tests/fast/rollout/session/test_samples_codec.py
pytest -q tests/fast/router/test_session_samples_op.py
pytest -q tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py
pytest -q tests/fast/ray/rollout/test_train_data_conversion.py
pytest -q tests/fast/backends/training_utils/test_replay_data.py
```

编解码测试证明精确的值/类型和新的包边界。router 测试证明真实的 `application/octet-stream` 响应和经由服务端组装的解码。客户端测试证明 POST/DELETE/错误行为。最后两个测试是训练数据和 replay 契约的下游守卫；它们不替代编解码断言。

### 静态检查

```bash
pre-commit run --files \
  requirements.txt \
  miles/rollout/session/samples.py \
  miles/rollout/session/samples_codec.py \
  miles/rollout/session/core.py \
  miles/rollout/generate_utils/openai_endpoint_utils.py \
  miles/utils/http_utils.py \
  tests/fast/rollout/session/test_samples_codec.py \
  tests/fast/router/test_session_samples_op.py \
  tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py \
  miles/rollout/session/SAMPLES_REPLY_SAFETENSORS_PLAN.md
```

编辑之后，`rg -n "encode_envelope|decode_envelope|_read_segment|struct.Struct|raw binary segments|binary envelope" miles/rollout/session miles/utils/http_utils.py tests/fast/rollout/session` 必须不再返回任何编解码实现引用。

### 编解码 benchmark

在编辑当前编解码器之前和实现 safetensors 之后各运行一次下面的精确命令，仅把输出文件名从 `/tmp/samples-codec-before.json` 改为 `/tmp/samples-codec-after.json`。脚本顶部的双路径 import 使同一条命令在改动前后都可运行：改动前 codec 在 `samples.py`，改动后在 `samples_codec.py`。脚本顶部的双路径 import 使同一条命令在改动前后都可运行：改动前 codec 在 `samples.py`，改动后在 `samples_codec.py`。两次命令在同一台其余时间空闲的主机和同一 Python 环境上运行。fixture 固定为 100,000 个 replay token 位置、64 层、top-k 为 4、一个 `int32` routed-expert 张量、`int64` tokens、`float64` rollout log probabilities 以及相同的标量 JSON 字段。两次预热之后是九次计量迭代。单次迭代的 combined 值是该迭代实测的编码时间加上实测的解码时间；`combined_median_s` 是这九个逐迭代总和的中位数，而不是两个各自独立选出的中位数之和。

```bash
python - <<'PY' | tee /tmp/samples-codec-before.json
import gc
import importlib.metadata
import json
import platform
import statistics
import sys
import time

import numpy as np

try:  # after: codec lives in its own primitive module
    from miles.rollout.session.samples_codec import decode_samples_reply, encode_samples_reply
except ImportError:  # before: codec still lives in samples.py
    from miles.rollout.session.samples import decode_samples_reply, encode_samples_reply
from miles.utils.types import Sample

num_tokens = 100_000
sample = Sample()
sample.tokens = list(range(num_tokens + 1))
sample.response = ""
sample.response_length = num_tokens
sample.loss_mask = [1] * num_tokens
sample.rollout_log_probs = [-0.5] * num_tokens
sample.rollout_routed_experts = np.arange(num_tokens * 64 * 4, dtype=np.int32).reshape(num_tokens, 64, 4)
sample.rollout_indexer_topk = None
sample.status = Sample.Status.COMPLETED

def measure_once():
    started = time.perf_counter()
    payload = encode_samples_reply([sample], {"max_trim_tokens": 0}, None)
    encoded = time.perf_counter()
    reply = decode_samples_reply(payload, Sample())
    decoded = time.perf_counter()
    return {
        "encode_s": encoded - started,
        "decode_s": decoded - encoded,
        "combined_s": decoded - started,
        "payload_bytes": len(payload),
        "reply": reply,
    }

for _ in range(2):
    measure_once()

measurements = []
last_reply = None
for _ in range(9):
    gc.collect()
    result = measure_once()
    last_reply = result.pop("reply")
    measurements.append(result)

(decoded_sample,) = last_reply.samples
assert decoded_sample.tokens == sample.tokens
assert decoded_sample.rollout_log_probs == sample.rollout_log_probs
np.testing.assert_array_equal(decoded_sample.rollout_routed_experts, sample.rollout_routed_experts)
assert decoded_sample.rollout_routed_experts.dtype == np.int32
payload_lengths = {item["payload_bytes"] for item in measurements}
assert len(payload_lengths) == 1
print(
    json.dumps(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "safetensors": importlib.metadata.version("safetensors"),
            "iterations": len(measurements),
            "payload_bytes": payload_lengths.pop(),
            "encode_median_s": statistics.median(item["encode_s"] for item in measurements),
            "decode_median_s": statistics.median(item["decode_s"] for item in measurements),
            "combined_median_s": statistics.median(item["combined_s"] for item in measurements),
        },
        sort_keys=True,
    )
)
PY
```

验收门限：

- `new_payload_bytes - old_payload_bytes <= max(4096, old_payload_bytes * 0.0001)`。
- `new_encode_decode_median <= old_encode_decode_median * 1.10`。
- 每个解码字段在与编解码测试相同的断言下与源精确一致。

参考基线：本命令此前对当前信封做过一次 smoke 运行，九次迭代得到 `104200553` payload 字节、`0.1654s` 编码中位数、`0.1027s` 解码中位数和 `0.2691s` 的逐迭代 combined 中位数。独立的 safetensors 原型在同构造 fixture 上实测 payload 104.2 MB、编码中位数 95 ms、解码中位数 48 ms、combined 142 ms，编码/解码的 Python 层内存峰值 106/112 MB（对比信封复刻的 313/216 MB）。

实施后验收实测（本节精确命令，同机前后各一次）：before `104200553` B / combined `0.2642s`（encode `0.1620` / decode `0.1008`），after `104200794` B / combined `0.1755s`（encode `0.1195` / decode `0.0523`）。payload 增长 241 B（门限 10420 B）、combined 为基线的 0.66 倍（门限 1.10 倍），两条门限均以显著余量通过。

现有的 `32 sessions x 50 turns` 手动 benchmark 不是这次替换的验收门限，因为其当前的负载生成器不调用 `/samples`。真实的网络/fan-in 压测可以在更大范围的 PR 1605 容量评审中单独进行，但本方案不声称这个编解码 microbenchmark 能解决事件循环排队或峰值 RSS 问题。

## 评审清单

- diff 是否删除了所有手工字节 offset 和 shape/dtype 解析器？
- `samples_codec.py` 是否是自包含的操作原语（不 import HTTP/session 状态/`samples.py`），`samples.py` 是否已无任何 wire 概念，第 4、6、7 节的 import 方是否全部更新？
- `_samples_meta` 是否是普通的 `uint8` 张量，而不是被保留的 `__metadata__` key？
- 每个输入数组在 `safetensors.numpy.save` 之前是否都经过 `np.ascontiguousarray`？
- 编码器和解码器是否都是 `_TENSOR_SPECS`/`_SCALAR_FIELDS` 上的循环，而不是逐字段复制的分支？import 时的覆盖断言是否存在？
- 两个 R3 字段是否在编码侧拒绝 `int32` 之外的 dtype、解码侧校验容器 dtype 与契约一致，而不是静默转换？
- JSON `null` 是否仍是合法 `None` 的唯一表示，且与零尺寸张量可区分？
- 非 null 的 metadata 引用是否使用直接张量查找，并在缺失时失败？
- 畸形用例测试是否断言精确的 `SafetensorError`、`KeyError` 或 `ValueError` 路径，包括 metadata 的 dtype 和 rank 校验？
- tokens/log probabilities 是否还原为 Python list，而两个 R3 字段保持为 NumPy 数组？
- `COMPUTED_FIELDS`、`TEMPLATE_FIELDS`、overlay 默认值、路由、HTTP media type、超时、DELETE、截断和 merge 是否都未被触碰？
- `safetensors` 是否以 `>=0.8.0` 下界直接声明在 `requirements.txt` 中？
- 解码侧是否校验引用名等于确定性名称、消费后容器无剩余张量？
- PR 描述是否把 R3 编码侧严格 `int32` 拒绝标注为 deliberate behavior change（现行为任意 dtype 均可往返），并注明该 `ValueError` 走 500 而非 422 的既有归属？
- 聚焦测试、下游守卫、pre-commit 和编解码 benchmark 是否全部通过？8/32 路并发 decode 冒烟是否已作为 PR 1605 容量评审的记录项（不设门限）安排？

## 风险、上线与回滚

### 混合版本部署

旧解码器读不了新 payload，新解码器也读不了旧 payload。这被接受，因为该端点尚未从 `main` 发布，且两侧在同一个 PR 中一起改动。同 revision 部署不是运维承诺，而是结构事实（面板评审中三方独立核查）：session server 由 driver 进程经 `multiprocessing.get_context("spawn")` 从同一解释器、同一 miles 安装启动（`miles/ray/rollout/router_manager.py:139-141`），两端不可能分离升级。如果该启动结构在合入前发生变化，必须重新打开本方案，而不是悄悄加兜底。

### 包行为与内存

safetensors 原型在 100 MiB fixture 上实测比现行信封更快（combined 142 ms 对 249 ms）且 Python 层内存峰值更低（106/112 MB 对 313/216 MB），对单进程 driver 的 event loop 阻塞时长约为信封的一半。但原型不等于实现，验收以实现后的 benchmark 为准；PR 1605 更大范围的峰值 RSS 和并发 fan-in 问题仍是单独的验证工作。

### DELETE 之后的解码失败

客户端在解码回复之前删除 session。因此畸形 payload 会大声失败，但无法再从原 session 重新获取。本方案保持该行为；改进可观测性或调整生命周期顺序需要单独的改动，因为它影响序列化之外的失败语义。

### 回滚

回滚是对这次原子编解码改动的完整 revert，包括编码器和解码器两侧、直接依赖和测试。不要只回滚一侧，也不要保留一个休眠的旧解码器。

## 开放问题

这次编解码器替换没有阻塞性的设计问题。唯一必须保持成立的假设是原子的同 revision 部署和当前 tensor-only 的 `Sample` 字段集合；如果新出现必需的非张量二进制字段或混合版本上线，需要的是新的设计决策，而不是藏在这个编解码器里的扩展。
