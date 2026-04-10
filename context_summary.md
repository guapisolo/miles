# IFBench 问题排查上下文总结

## 1) 背景与目标

- 客户反馈：同样使用 IFBench 评测集与相近采样参数时，`miles` 分数约 `0.50~0.52`，`sglang-server` 约 `0.60`。
- 重点怀疑点：
  - `linear_attention.A_log` 在 miles 侧出现 `float32 -> bf16`，参与 recurrence 后误差累积放大。
  - torch-dist 离线转 HF 时未 merge MoE kernel，离线 merge 后效果对齐；希望定位 online-update 侧可否修复。

## 2) 评测参数（本次统一口径）

- `top_p: 0.95`
- `temperature: 1.0`
- `presence_penalty: 1.5`
- `top_k: 20`
- `extra_body.chat_template_kwargs.enable_thinking: true`

## 3) 已完成代码改动

### 3.1 脚本与入口修正

- 修改 `run_ifbench.sh`：
  - 默认 GPU 改为后 4 卡：`CUDA_VISIBLE_DEVICES=4,5,6,7`（若外部已设置则不覆盖）。
  - 评测执行路径改为直接调用本地文件：`python3 "$REPO_DIR/eval_ifbench.py"`，避免依赖不存在的模块路径。
- 新增兼容入口：
  - `agent/eval_bench/run_ifbench.sh`
  - `agent/eval_bench/eval_ifbench.py`
  - `agent/__init__.py`
  - `agent/eval_bench/__init__.py`

### 3.2 调试埋点（用于 runtime evidence）

- `miles_plugins/models/qwen3_5.py`：
  - 增加 H1~H4 埋点，记录 `A_log`/`g`/kernel 输入输出 dtype 与关键统计。
- `eval_ifbench.py`：
  - 增加 H5~H6 埋点，记录评测实例启动、`run_task` 成功/失败与目标 `api_url`。
- `run_ifbench.sh`：
  - 增加 H7 埋点，确认脚本入口是否命中本仓库版本。

## 4) 已验证结果（关键）

- `bash agent/eval_bench/run_ifbench.sh --help`：可正常执行。
- `bash run_ifbench.sh --help`：可正常执行。
- 运行时证据确认：
  - H7 已命中（本机执行可写入调试日志），证明脚本入口和日志链路本身可用。

## 5) 当前结论

- **IFBench 脚本“可以跑起来”**（入口与参数解析均已验证）。
- 目前尚未拿到一次“真实业务评测请求”对应的完整模型侧日志，因此：
  - H1~H4（`A_log` dtype 及 recurrence 路径）尚未最终确认或排除；
  - 需要一次稳定复现并保留日志文件用于最终定因与修复闭环。

## 6) 建议的标准执行命令（绝对路径，避免歧义）

```bash
bash /root/miles/agent/eval_bench/run_ifbench.sh \
  --api-url <MILES_API_URL>/v1 \
  --model-name default \
  --enable-thinking \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --presence-penalty 1.5 \
  --output-dir /tmp/ifbench/evalscope
```

## 7) 后续建议

- 若要继续“修分数差异”而非只保证脚本可跑，下一步应：
  1. 用上面绝对路径命令复跑一次；
  2. 保留评测产物与调试日志；
  3. 基于日志对 H1~H4 做确认/排除，再做最小修复（例如 `A_log` 显式保持 `fp32` 状态路径）并做回归验证。
