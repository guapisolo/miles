# 设计文档:CI 历史指标 Gate Helper

状态:Draft / 待团队 review。本文是实现前的方案,不是已落地实现的说明。

## 读者与目标

读者是 miles CI 与训练后端的工程师:读完后你应当能判断「这套机制要不要采纳」「两层 gate 的边界是否合理」「受控存储选型和清理语义能不能接受」,并据此 review 后续实现。

本文要回答的设计问题:在现有 CI 之上,如何按 test(文件路径)收集**跨 commit** 的训练指标历史,用「hardcoded 安全值 + 数据驱动统计」两层 gate 拦截回归,并支持 PR 归因、手动查询历史、清理坏数据。**wandb 在本方案里只是指标的搜集入口(只写、不回读)**;本文**不**涉及把任意 code/数据通过 hook 上传(那是 future),也不替换现有单 run 内的对齐检查。

下面先列已被代码验证的现状判断,后续每条约束和设计决策都回指到它们。

## 已验证的现状(证据)

- **test 身份就是文件路径**:CI 注册表 `CIRegistry` 以 `filename` 为 key,通过 AST 解析每个 test 文件顶层的 `register_cuda_ci(est_time, suite, ...)`(`tests/ci/ci_register.py:36`、`:69`);`run_suite`/`ci_utils` 也是按文件路径调度执行的(CUDA test 以 `python3 <file>` 子进程跑,CPU test 走 pytest)。所以「用文件路径区分 test」与现状完全一致,且**调度侧已经持有这个完整路径**。
- **指标搜集已统一走 tracking 抽象,wandb 是其中一个 backend**:训练调用 `log(args, metrics, step_key)`(`miles/utils/tracking_utils/__init__.py:13`),`TrackingManager.log` 把同一次调用 fan-out 给所有启用的 backend(`base.py:147`);backend 经 `BACKEND_REGISTRY` 注册,新增 backend 的步骤在 `base.py:7-12` 有明确文档(subclass `TrackingBackend` → 注册 → 加 `--use-<name>` flag)。`WandbBackend.log` 只是 `wandb.log(metrics)`(`base.py:48`)。**这就是接入新「搜集出口」的天然位置。**
- **commit/PR 上下文在运行期已就绪**:`GITHUB_COMMIT_NAME = {github.sha}_{pr_number||non-pr}` 注入每个 CI job(`.github/workflows/_run-ci.yml:57`);`get_default_wandb_args(test_file)` 已经拿到 test 文件路径并据此配置 wandb(`miles/utils/external_utils/command_utils.py:208`、`:219`)。即「完整 test 路径 + commit + PR」无需新探针即可获得。
- **要 gate 的 4 个 metric key 已流经 tracking 抽象**:`train/grad_norm`(`miles/backends/training_utils/log_utils.py:449`)、`train/train_rollout_logprob_abs_diff`、`train/ppo_kl`、`train/train_rollout_kl`(`miles/backends/training_utils/loss_hub/losses.py`)。
- **跨 run 历史比较目前完全不存在**:现有 logp/grad 检查都是**单 run 内**对齐——`logprob_comparator` 比同一 run 的 tp1 vs tp2pp2cp2 dump;`--ci-save/load-grad-norm` 的基线是同一 run 内临时写的(`miles/backends/training_utils/ci_utils.py:27`);`ppo_kl < 1e-9`、`eval/gsm8k >= 0.55` 等是**硬编码 magic number**(`ci_utils.py:12`、各 e2e test)。没有任何跨 commit 历史被用来 gate 或归因。
- **`/data/miles_ci` 是 per-host、不跨机共享**:它从各宿主机 bind-mount(`tests/ci/README.md`),而 CI 分布在 scitix-72/73、novita-host2/4 多台机器。所以它**不能**当跨 run 历史存储;天然跨机共享的只有云端。
- **repo 无现成云/DB 依赖**:`requirements.txt` 只有 `wandb`、`polars`、`blobfile`(已用于 `gs://`/`s3://` checkpoint I/O),没有任何 SQL/对象存储 SDK。受控存储是新基建,最省的地基是 `blobfile` + `polars`(均为现有依赖)。

## 1. Motivation

现状有三个痛点,都指向「缺一份能跨 commit 比较的指标历史」:

1. **正确性阈值是手维护的 magic number,既粗又难统计**:每个 test 的每个关键 metric 都要人手挑一个阈值硬编码进代码(`ppo_kl < 1e-9`、`eval/gsm8k >= 0.55`)。这种值只能取「直觉上不该达到的安全值」,无法反映该 test 该 metric 的真实统计分布;为每个 test 每个 metric 都算并硬编码统计值太麻烦,于是没人做,subtle 回归就漏掉了。
2. **CI 挂了难定位是哪个 PR 弄挂的**:现有检查只给单点 pass/fail,没有跨 commit 的趋势。希望挂的时候能看到这个 metric 最近的变化曲线,辅助归因到具体 PR。
3. **指标其实每次都被搜集了,却没沉淀成可比较的历史**:metrics 已经流经 tracking 抽象(并上传到 wandb),但没有任何一份「按 test 按 commit 可查询、可清理」的历史被留下来用于 gate 或归因。

由此本轮想要的能力:按 test(文件路径)收集跨 commit 的指标历史 → 用历史自动算统计值来 gate(先做 4 个 logp/grad metric)→ 出问题时能查趋势、归因 PR → 发现某个历史点本身是坏的(被坏 run 污染)时能快速清掉。

## 2. Constraints(从 motivation 推出)

硬约束(必须满足):

- **C1 跨机共享**:历史存储必须跨所有 CI host 共享 → 上云。(来自「收集跨 commit 历史 + PR 归因」;`/data/miles_ci` per-host 做不到)
- **C2 test 身份 = 完整文件路径**:存储与 gate 的 canonical key 是 repo 相对的完整 test 文件路径,等同 `CIRegistry.filename`,由调度侧(已持有该路径)赋予,**不**派生自 wandb 的 project/run 命名。(来自「区分不同 test」)
- **C3 step1 搜集只接 tracking 抽象现有的指标**:不新增上传/采集探针,直接在 `log()` fan-out 处接一个新出口,拿已经在打的 4 个 wandb metric。(来自需求 1.2 step1)
- **C4 PR/commit 可归因**:每条历史点必须带 commit sha + PR number。(来自痛点 2;`GITHUB_COMMIT_NAME` 已提供)
- **C5 冷启动可用**:新 test 或历史不足(<5 点)时,gate 不能因「没历史」就误判或失效——必须有一层与历史无关、永远可用的兜底。(来自「test 多样」+ historical gate 需要 ≥5 点)
- **C6 可清理且清理后统计立即正确**:能按 (test, commit/run) 失效/删除坏数据点,删除后历史均值立刻反映。(来自需求 1.3)
- **C7 抗 run-to-run 抖动**:gate 默认 20% **相对**容差;仅对在所选 reducer 下会落在 ≈0 的 metric(如 step-0 的 `ppo_kl`)再加一个**绝对地板**,以避开近零奇点(见 4.3)。(来自痛点 1 + 近零 metric 的现实)
- **C8 不回查 wandb**:wandb 仅作搜集入口(只写);gate 判定与历史读取一律走受控存储,任何环节不得回查 wandb API。(来自用户明确约束)

软约束(强烈偏好,可权衡):

- 少引新依赖,尽量复用已有的 tracking 抽象 / polars / blobfile。
- gate 与 test 解耦:新增 gate、改阈值、清理、re-baseline 都不应改动 test 训练代码。
- 支持用户手动查询某 (test, metric) 的历史值序列。

非目标(本轮不做):

- 通过「灵活 hook 上传任意 code/数据」(需求 1.2 future)——step1 只摄入 tracking 抽象里现有的 4 个标量;接入点(新 backend)预留扩展但不实现额外 hook。
- 不替换现有单 run 内对齐检查(`comparator`/dumper/`--ci-load-grad-norm`),两套并存。
- 不做实时 dashboard——wandb 已经提供曲线可视化。

## 3. Proposal selection(用约束筛选)

### 3.1 历史存储的真实来源

- **repo 内 per-test golden 文件**(扩展 `--ci-save/load-grad-norm` 到提交进 repo 的跨 run golden):虽可随 git 跨机,但更新靠手动 PR、C6 清理=改文件 diff、C4 的 PR 归因弱、且只存单点不存历史序列(无法满足「手动查询历史值」与统计 gate)→ **淘汰**。
- **回查 wandb 云端历史**:技术上 wandb 已按 commit/PR 存了历史,但用户明确否决(C8):清理只能删/标 wandb run(粒度粗、易误删他人 run、无受控 schema),且会让 gate 强依赖 wandb 在线 + API 限流 → **淘汰**。
- **在搜集入口旁路一份到独立受控存储(选中)**:在 `log()` fan-out 处加一个 `CIHistoryBackend`,与 `WandbBackend` 并列、吃同一份 metrics(满足 C3,wandb 维持只写);该出口把 gate 相关标量落到上云的受控存储,行 key 为完整 `test_path`(C1/C2/C4);gate 判定只读这份受控 schema(C8 不碰 wandb、C6 清理=删/标行、C7 统计在受控数据上算干净、支持手动查询历史序列)。代价:多一套 backend + store + gate step。

### 3.2 gate 的比较语义

用同一把尺子(C5 冷启动、C7 抗抖、痛点 1 catch subtle 回归、痛点 2 归因)量三种语义:

- **纯 pinned 安全值 + 紧容差**:C5 满足(与历史无关),但只能取粗安全值、漏报 subtle 回归,且要手维护精确统计值(motivation 明确反对)→ 单独不够。
- **纯历史统计带**:catch subtle 回归与归因好,但 C5 冷启动直接失败(<5 点没法判)、历史被污染时无任何保护 → 单独不够。
- **两层 gate(选中)**:`hard gate`(hardcoded 粗安全值 + 默认 20% 容差,bound to test,与历史无关,永远可用)正面解决 C5 与兜底;`historical gate`(≥5 个 trusted 历史点,偏离均值超默认 20% 即失败)解决 subtle 回归、自校准、PR 归因。两者互补:hard gate 无冷启动问题但钝,historical gate 锐但需历史;并且 hard gate 顺带充当 historical 基线的「准入过滤」(见 4.3)。

结论:在以上约束下,「搜集入口旁路 + 独立受控存储 + 两层 gate + 独立 test-后 gate 步骤」是最直接的组合,而不是功能最多的组合。

## 4. Proposal design

### 4.1 数据流

一次 CI job 内,一个 test 文件的指标从产生到 gate 的完整时间线(泳道 = 进程/系统边界;时间自上而下):

```
 时刻   训练进程 python3 <test_file>            run_suite gate step (同 job, test 后)        Neon: gate_history          带外: 人工 / CLI
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 t0   run_suite 设 env MILES_CI_GATE_RECORD_DIR=<d>, 然后 launch python3 <test_file>
 t1   训练循环每步: log(args, metrics, step_key)
        │  fan-out  (TrackingManager.log, base.py:147)
        ├─► WandbBackend.log ───────────────────────────────────────────────────────────►  wandb cloud   [搜集入口, 只写, C8 不回读]
        └─► CIHistoryBackend.log: 进程内累积 {metric_key: [(step, value), …]} (仅 4 个 gate key)
 t2   finish_tracking() → CIHistoryBackend.finish():
        把累积的 series 写 <d>/<run_id>.json   (本地文件, 不含身份)
 t3   test 子进程返回 (pass / fail)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 t4   [仅当 test pass]                          AST 解析 <test_file> 的 register_ci_gate(...) specs
 t5                                             读 <d>/*.json, 按各 metric 的 reducer 压成 per-(metric, sub_label) 当前标量
 t6                                             赋身份: test_path = CIRegistry.filename;  (sha, pr) ← GITHUB_COMMIT_NAME
 t7                                             对每个 (metric, sub_label):
                                                  hard gate:  |cur - hard_ref| > tol ?
                                                  SELECT value WHERE test_path,metric,(sub),trusted ─────►  读 trusted 历史序列
                                                                                                   ◄─────  [≥5 点] 算 mean
                                                  historical gate(≥5 trusted 点才跑): |cur - mean| > 20% ?
                                                  INSERT(…, value=cur, trusted = 全部 active gate 通过) ──►  落一行 (始终写)
 t8                                             任一 active gate 失败 → gate step 失败 → CI job 失败
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 (带外, 任意时刻)                                                                          SELECT / UPDATE  ◄──── query / clean
                                                                                          (UPDATE trusted=false; mean 立即正确)
```

**阶段 A — 指标产生(训练进程,沿用现状,零改动)**:test 被 `run_suite` 以 `python3 <test_file>` 启动(CUDA;`ci_utils.py`),训练循环每步调 `log(args, metrics, step_key)`(`__init__.py:13`),`TrackingManager.log` 把同一份 metrics dict fan-out 给所有启用 backend(`base.py:147`)。这条链现状已存在,本方案不动它。

**阶段 B — 进程内捕获 → 本地 record(新增 `CIHistoryBackend`)**:新 backend 与 `WandbBackend` 并列吃同一份 metrics,只挑出 4 个 gate key,在内存里累积 `{metric_key: [(step, value), …]}`(量级:每 run 几千步 × 4 float,可忽略)。`finish()` 时把累积 series 序列化成一个**本地 JSON 文件**,落在 `run_suite` 通过 env `MILES_CI_GATE_RECORD_DIR` 指定的目录里,文件名用该训练 run 自己的 `run_id`(一个 test 文件若跑多个训练 run/role,就是多份文件,见 6)。这份 record **不含身份、不写云、不读 wandb**——它纯粹是「训练进程 → gate step」的进程间交接物(C8 要求当前值不能回查 wandb,只能来自进程内捕获)。reducer 不在这里做:backend 只负责"忠实 dump series",压成标量的口径(4.4)留给 gate step,使 reducer 配置集中在一处。

**阶段 C — gate step(`run_suite` 内,test 返回后,同一 job)**:这是逻辑与 I/O 的集中点,顺序为 t4–t8:
1. 仅当 test 子进程 pass 才进入(test 本身崩了已经 fail,gate 无意义);
2. 静态 AST 解析该 `<test_file>` 的 `register_ci_gate(...)`,拿到每个 metric 的 `hard_ref / rel / abs_floor / reducer / one_sided`(4.5);
3. 读 `MILES_CI_GATE_RECORD_DIR` 下的本地 record,对每个 (metric, sub_label) 应用 reducer → 当前标量 `cur`;
4. 赋身份:`test_path` 取 `run_suite` 手里的 `CIRegistry.filename`(完整路径,身份只在这一处产生,与 wandb 命名无关 → C2),`(commit_sha, pr_number)` 解析自 env `GITHUB_COMMIT_NAME`(C4);
5. 对每个 (metric, sub_label) 跑 hard gate(总是)+ historical gate(`SELECT` 到 trusted 历史 ≥5 点才跑);
6. 无论 gate 结果如何都 `INSERT` 一行(让趋势/PR 归因始终看得到这次),`trusted` = 本次是否通过**所有 active gate**(见 4.3);
7. 收集所有 (metric, sub) 的失败,有任一失败则 gate step 失败 → CI job 失败。

**阶段 D — query / clean(带外,人工)**:任意时刻,用 CLI 或 Neon 网页对 `gate_history` 做 `SELECT`(查趋势/归因 PR)或 `UPDATE trusted=false`(清坏点)。因为 historical gate 每次都现算 trusted 行的 mean,清理后下一次 gate 立即用上正确基线(C6)。这条线与 A–C 完全解耦,不经过训练进程也不经过 wandb。

各跳的数据形态(具体):

| 跳 | 位置 | 形态 |
|---|---|---|
| 训练内 | `log()` 入参 | `{"train/grad_norm": 1.02, "train/ppo_kl": 3e-9, …, "train/step": N}`(每步一份) |
| 本地 record | `<d>/<run_id>.json` | `{"run_id": "...", "sub_label": null, "series": {"train/grad_norm": [[1,1.1],[2,1.05],…], …}}` |
| reduce 后 | gate step 内存 | `{"train/grad_norm": 1.03, "train/ppo_kl": 4.2e-3, …}`(每 metric 一个标量) |
| Neon 行 | `gate_history` | `(test_path, metric_key, sub_label, commit_sha, pr_number, run_id, value, ts, trusted)` |

身份归属与并发:身份(`test_path`)**只在 gate step 这一处**由 `CIRegistry.filename` 赋予,所以 wandb 是否用 stem 命名、会不会撞名,都与本方案身份无关(C2 天然满足)。并发:多个 CI job 同时 `INSERT` 由 Postgres 原子处理、无锁争用(这正是相对 git 方案的优势);读历史是一致性 `SELECT`,不受并发写干扰。

### 4.2 受控存储:Neon(managed serverless Postgres)

存储**已定为 Neon**(serverless Postgres)。理由:我们的数据量(几千行 tiny scalar)稳在其 Free 档内(0.5 GB/project 存储、100 CU-hours/project 计算,$0、免信用卡),钱不构成因素;原生 SQL 查询 + 网页表格 UI 正好补上扁平存储「过滤/聚合要客户端算」的短板,清理是原子 `UPDATE`,满足 C1/C4/C6。代价(实现期要落):新增一个长期连接串 secret `NEON_DATABASE_URL`(注入 `.github/workflows/_run-ci.yml` 的 job env)、`requirements.txt` 加 `psycopg`、gate step 多一个对 DB 的网络依赖(见 6)。

单表 schema(行 = 一个 (test, metric, run) 的 reduce 后标量):

```sql
CREATE TABLE gate_history (
    id          bigserial        PRIMARY KEY,
    test_path   text             NOT NULL,   -- = CIRegistry.filename(完整路径, 满足 C2)
    metric_key  text             NOT NULL,   -- train/grad_norm 等
    sub_label   text,                        -- 一文件多 run/role 时区分(见 6)
    commit_sha  text             NOT NULL,
    pr_number   integer,                     -- 来自 GITHUB_COMMIT_NAME, 满足 C4
    run_id      text             NOT NULL,
    value       double precision NOT NULL,
    ts          timestamptz      NOT NULL DEFAULT now(),
    trusted     boolean          NOT NULL DEFAULT true
);
CREATE INDEX ON gate_history (test_path, metric_key, trusted);
```

- 读历史(historical gate):`SELECT value FROM gate_history WHERE test_path=$1 AND metric_key=$2 AND trusted ORDER BY ts DESC LIMIT :N`。
- 写当前 run:gate 判完一次 `INSERT`(无论 gate 结果都写,供趋势/归因可见),`trusted` = 本次是否通过**所有 active gate**(见 4.3)。
- 身份字段(`test_path`/`commit_sha`/`pr_number`)由 gate step 赋予,不依赖 wandb 命名(满足 C2/C8)。

### 4.3 两层 gate 的判定逻辑

per-metric 容差以**相对为默认**,沿用现有 `math.isclose(rel_tol, abs_tol)` 的 rel-OR-abs 语义(`miles/backends/training_utils/ci_utils.py:54`)。判 fail 的统一形式:`|current - ref| > max(rel * |ref|, abs_floor)`,`rel` 默认 0.20;`abs_floor` **仅对在所选 reducer 下会落在 ≈0 的 metric 才设非零值**(避开近零奇点:参考值≈0 时固定相对带宽的绝对宽度趋零,本就 ≈0 的量上的浮点噪声会撑出巨大相对偏差却无实质偏移)。对 `grad_norm`(量级 O(0.1–10))`abs_floor=0` 即可,纯相对 20% 足够;对 step-0 的 `ppo_kl`(≈1e-9)等才需要 `abs_floor`。哪些 metric 落在近零、需要地板,由 4.4 的 reducer 决定。

- **hard gate(总是跑,与历史无关 → 满足 C5)**:`ref` = 该 (test, metric) 的 hardcoded 安全值,bound to test、作为参数声明(见 4.5)。可配成双边(偏离 ref 超容差即 fail)或单边上限(`current > ref` 即 fail,适合「越大越坏」的 metric 如 `grad_norm`)。语义是「直觉上不该达到的安全线」,不是统计值。
- **historical gate(≥5 个 trusted 历史点才跑)**:`ref` = trusted 历史点的均值;`current` 偏离均值超默认 20%(同样 rel-OR-abs)即 fail。trusted 点 <5 时本层**不激活**,只剩 hard gate(冷启动 / 新 test 安全)。
- **trusted 准入**:一行总是 `INSERT`(让趋势/PR 归因看得到每次),但 `trusted` = 本次是否通过**所有 active gate**(hard 必跑;historical 在 ≥5 trusted 点时跑)。理由:漂移(historical 失败)但过 hard 的 run 不应静默进入基线——否则基线缓慢跟随漂移、掩盖慢回归;漂移要成为"新常态"需人工经 clean/rebaseline 接受。不形成循环:`historical_ok` 是对**既有** trusted 基线算的,再决定本行 trusted。手动清理可把某行 `trusted` 翻成 false(见 4.6)。

### 4.4 per-run 标量 reducer

每个 run 的 metric 是一条时间序列,gate 前必须 reduce 成标量,且 reducer 决定该 metric 是否落在近零区(进而决定要不要 `abs_floor`)。提议 **per-metric 可配 reducer**,默认取**最后一个训练 step 的值**(收敛后的代表值);`grad_norm` 这类抖动大的可配「最后 K 步均值」;`ppo_kl` 现有硬检查发生在 step 0(那里 ≈0),故若沿用 step-0 口径就必须配 `abs_floor`。具体每个 metric 取什么口径需在实现时定死并写进 gate spec(见 7. 开放问题)。

### 4.5 hard gate 参数的声明位置

为满足「bound to test、作为参数」且与现有 `register_cuda_ci` 风格一致,提议在 test 文件里加一个**声明式 gate spec**(例如紧挨 `register_cuda_ci` 的 `register_ci_gate(metric, hard_ref, rel=0.2, abs_floor=..., reducer=..., one_sided=...)`,同样用 AST 静态解析、运行期 no-op)。这样新增/调阈值是改声明而非改训练逻辑,gate step 在执行前静态读取该 test 的 spec。

### 4.6 清理与 re-baseline(需求 1.3)

清理是 Neon 上的一次原子 `UPDATE`,可走 CLI(`python -m tests.ci.history_gate`)或直接在 Neon 网页 SQL editor / 表格 UI 操作:

- `query --test <path> --metric <key>`:`SELECT ... ORDER BY ts`,打印 trusted 历史序列(value、sha、pr、ts),即痛点 2 的趋势/PR 归因入口;Neon 控制台也能直接点开看。
- `clean --test <path> [--metric <key>] [--commit <sha> | --run <id> | --pr <n>]`:`UPDATE gate_history SET trusted=false WHERE ...`(原子)。因为 historical 均值只在 `trusted` 行上算,清理后均值立即正确(满足 C6)。
- `rebaseline`:无需特殊操作——清掉坏点后,后续通过 hard gate 的好 run 会自然把均值拉回;要立即重置可用 `clean` 把旧点全部失效。

### 4.7 搜集出口与 gate step 的边界

搜集出口 = 新增的 `CIHistoryBackend`(按 `base.py:7-12` 标准扩展:subclass + 注册 + `--use-ci-history` flag,启用条件 mirror `get_default_wandb_args` 的 CI gating)。它与 `WandbBackend` 并列吃同一份 `log()`,在进程内累积 4 个 gate metric,`finish()` 时按 reducer 压成标量、写一份**本地 per-run record**;它**不写云存储、不含身份**,只做进程内交接。gate step(run_suite 内,test 子进程返回后)读本地 record(当前值)+ 受控存储(trusted 历史),赋身份、跑两层 gate、并把当前行写入受控存储(`trusted` = 是否通过所有 active gate,见 4.3)。这样:wandb 维持纯搜集入口(C8),gate 全程不碰 wandb,gate 逻辑/身份/store I/O 全集中在 gate step(满足解耦软约束)。

### 4.8 本轮做 vs 后续

本轮:`CIHistoryBackend` + 本地 record + 受控存储 + 两层 gate + gate spec 声明 + query/clean CLI。后续(非目标):需求 1.2 的「灵活 hook 上传任意 code/数据」——`CIHistoryBackend`/record schema 的 `value` 字段可推广为 blob 引用,hook 在搜集出口处接入,但本轮不实现。

## 5. 主线自检

motivation(magic number 粗、难归因、搜集到的指标没沉淀成历史)→ constraints(C1 上云、C2 完整路径为 key、C3 接搜集入口、C4 带 PR、C5 冷启动、C6 可清理、C7 相对默认+近零加地板、C8 不回查 wandb)→ selection(搜集入口旁路 + 受控存储 + 两层 gate)→ design(backend/本地 record/两层逻辑/reducer/声明/清理/落点)→ risks。每条约束都回指某个痛点或用户约束,每个设计决策都回指某条约束;无悬空章节。

## 6. 已知风险(接受)

- **近零 metric 的容差需逐个标地板**:`ppo_kl`(可 ≈1e-9)、两个 KL/diff 在 on-policy regime 可逼近 0,纯相对 20% 会误报。靠 4.3 的 rel-OR-abs + per-metric `abs_floor` 兜,但每个近零 metric 的 `abs_floor` 仍需人工标一次(且与 reducer 口径绑定)。`grad_norm` 不受此影响。
- **一个 test 文件可能产生多条序列**:有的 test 在一次 `python3 <file>` 里跑多个配置/role(如 parallel_check 的 tp1、tp2pp2cp2),会产出多组 metric。本地 record 与 store 行需带 sub-run/role 标签,gate key 退化为 (test_path, metric, sub_label);step1 可先只 gate 代表性的那一组。
- **本地 record 丢失**:训练在 `finish()` 前崩溃则无 record;但此时 test 本身已失败,gate 无意义,可接受。
- **gate step 依赖 Neon 可达**:DB 不可达 = gate step flaky(与「任何跨机云存储都要联网」同类,跑不掉);Free 档 5 分钟自动挂起带来的冷启动延迟(亚秒~几秒)对 gate 可忽略。供应商风险:关键路径依赖第三方 Free 服务,但数据是普通 SQL 行、迁移成本极低,锁定低。

## 7. 开放问题(尚未拍板)

- ~~受控存储最终选型~~ **已定:Neon(Free 档,$0)**。实现期需开一个 Neon project、建表(见 4.2)、把连接串配成 GitHub secret `NEON_DATABASE_URL` 并注入 `_run-ci.yml`。
- **每个 metric 的 reducer 口径**:4 个 metric 各自取 last-step / last-K-mean / step-0 / max 中的哪个,需逐个定死(直接决定要不要 `abs_floor`)。
- **每个近零 metric 的 hard `ref` 与 `abs_floor` 初值**:谁来标、标多少(尤其 `ppo_kl`)。
- **trusted 头 5 点的可信度**:新 test 最早 5 个点天然无人验证,「hard-gate 通过即 trusted」是否足够,还是要人工确认首个基线。
- **清理权限归属**:谁能 `clean`/失效历史点,是否要审计。
