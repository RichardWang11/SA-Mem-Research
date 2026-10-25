# TSM Memory 交接：结构、恢复、验收与全量查询

本文面向接手同学。命令均从本文件所在目录执行，即仓库内
`memory_repair/`。本包携带已经生成的 Memory，
恢复时无需重新抽取实体/事实，也无需重新生成文档 embedding。

## 1. 这个包对应什么实验

- 官方基线：`0da3633`，包含 v0.1.1 (`7b45bf0`)。
- 修复提交：`d7c124c1e1c53b9afab882e9d080d5eb9ba57a83`。
- 初始 Memory 交付提交：`823a59e`；恢复工具与本说明随之后续提交。
- `code/` 是本次实验实际使用的代码快照，包含已有的 API 兼容和 Trace。
- 图抽取/原答案模型：SiliconFlow `Qwen/Qwen3-30B-A3B-Instruct-2507`。
- Dense embedding：RelayRouter `text-embedding-3-small`，1536 维。
- 检索为原始 `IndexFlatIP`，未增加向量归一化，最终 Top-K=20。
- 本次修复建库包含 10 个 Conversation 的全部 2871 个 TSM round；
  2871 个源文本均已对齐 Neo4j Episode、全局索引、leaf 与缓存。
- 这是“修复后的 TSM 复现”，不是未经修改的官方结果。随后相同321题QA与Mem0口径LLM-J已完成：主评测169/244=69.26%，详见[对照报告](../results/repaired_qa321_20261025/REPORT.md)。全量1986题尚未运行。

旧缺陷是每个月都从 `leaf_0` 编号，合并时覆盖同名 key；同时窗口内的
局部 `i` 被用于读取全局 `original_text[i]`，导致文本错配。
修复稳定了 `chunk_N -> leaf_N`，校验内容一致性，并在最终 Dense Tree
保留全部源 round，包括没有带时间事实边的源文本。Dense leaf 总数从
317 恢复为 2871。修复没有改变 Query Parser、Temporal Rerank、Answer Prompt。
细节与测试见 `REPAIR_REPORT.md`、`code/docs/DENSE_MEMORY_REPAIR.md`。

随后在真实 query-only Smoke 中补充修复 `4aeb080`：Graphiti 的 `graph.json`
第三列是属性字典，旧 `build_graph` 只接受数字权重。当前读取器兼容两种格式，
保留 fact/relation 等属性及旧权重合并规则。恢复包中的 `code/` 与仓库 `TSM/`
均已包含此兼容修复，已通过两个真实查询 Smoke；源数据和 FAISS 向量没有改变。

“完整”指原 TSM Loader 产生的所有 round：相邻不同 speaker 配成一轮；
同 speaker 位置跳过、最后落单的话语省略。它并不保证每条原始 utterance
都保留。这一 Loader 行为未修改。

## 2. Memory 结构及对应关系

每个 `memory/conv-XX.tar.gz` 内是：

```text
conv-XX/
  cache/
    tree.json                   # 全局 leaf_N -> 原始完整 round 文本
    dense_index.faiss           # 可以直接加载的向量索引
    dense_index_mapping.json    # FAISS 行号 -> leaf_N，附 Tree hash
    source_mapping.json         # leaf / Episode UUID / dialogue ID 的来源映射
    snapshots.json              # 时间窗口、Persona/Topic 摘要、窗口内 leaf
    graph.json                  # TSM NetworkX 兼容图，非完整 Neo4j 备份
    index.json                  # 实体名称 -> Episode UUID
    appearance_count.json       # 实体出现次数
    text_edge_mapping.json      # Episode UUID -> 原文、图边等兼容信息
    cluster_index.json          # 聚类构建使用的实体索引
  neo4j_export/
    nodes.json                  # 节点标签和全部属性，包含 embedding
    relationships.json          # 两端 UUID、关系类型及全部关系属性
```

原始对话 `dia_id` 通过 `source_mapping.json` 对应到 round/`leaf_N`。
`Episode` 是原始 user-only round 的图节点，`name=chunk_N`；对应 leaf
保存 user+assistant 完整 round。图中 Entity 是实体节点；`MENTIONS`
连接 Episode 与 Entity；`RELATES_TO` 连接 Entity 与 Entity，保存
`fact`、`episodes`、`valid_at`、`invalid_at` 等。一个 round 可贡献多条事实，
一条事实也可能关联多个 Episode，因此事实边与 leaf 不是一一对应。

FAISS 文档是 leaf；本版本的 Persona/Topic 摘要留在 Snapshot，未作为
FAISS 文档。Query 选 Snapshot 不会提前裁剪 leaf 的 Dense 搜索范围。
Graphiti 时间事实通过 Episode 的 `chunk_N` 关联 leaf，作用于 Dense 候选重排。

所有档案的 SHA-256、代码版本、数据版本和 API 用量在 `manifest.json`。
图数据以逻辑 JSON 导出，时间值序列化为 ISO 字符串；下面的导入器会将它们
恢复为真正的 Neo4j 时间类型。不能直接把这些时间字符串原样当作属性导入。

## 3. 安装运行环境

原实验实际验证环境是 Python 3.12.3；官方发布要求 Python 3.11。
建议接手先沿用 Python 3.12 并记录本机版本；`environment.freeze.txt`
提供原环境版本参考。先安装运行依赖和原实验使用的 spaCy small 模型：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python -m spacy download en_core_web_sm
python handoff_memory.py inspect
```

`inspect` 不访问数据库或模型 API，会校验压缩包 SHA-256、2871 条来源映射、
FAISS 维数/文档数/Tree hash、向量非零/有限值，以及逐向量自检索。
模型安装影响 Query 时间解析，正式比较时应固定 spaCy 模型版本。

## 4. 启动本机 Neo4j 并导入

使用 Neo4j 5.26 Community。下面是本机已有 Docker 时的示例，密码由接手同学
自己设置，不能提交到 Git：

```bash
read -rsp 'New Neo4j password (at least 8 characters): ' NEO4J_PASSWORD
export NEO4J_PASSWORD
docker run -d --name tsm-neo4j \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH="neo4j/$NEO4J_PASSWORD" \
  -v tsm-neo4j-data:/data neo4j:5.26-community
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_DATABASE=neo4j
```

已有数据库的同学直接设置上述四个环境变量。等待 Neo4j 就绪后：

```bash
# 先验证输入，不写库、不调用模型
python handoff_memory.py restore

# 将 10 个 Conversation 的完整图导入；命令结束会自动逐属性验收
python handoff_memory.py restore --execute

# 后续随时重新验收，不写库
python handoff_memory.py test
```

导入器创建 UUID/group_id/时间索引，导入 Entity、Episodic、MENTIONS、RELATES_TO，
并恢复四类时间字段。现有目标 group 或 UUID 冲突时会拒绝导入，不会覆盖/删除。
若中途中断，请使用干净数据库重试；工具不自动清理半成品。

想先测试一个 Conversation 可用 `--conversation conv-26`。如果先导入了一个
Conversation，再导入其它 Conversation 时也要分别指定；不应对已有 group
再次执行默认全量导入。

`--namespace` 只用于隔离的数据库恢复验收：它会改写 UUID 与 group，不能直接
配合后面的原版 QA 缓存使用。正式 QA 请使用默认、未重命名的原始 group。

## 5. 验收标准与可视化

默认全量恢复的期望值：

| 项目 | 数量 |
|---|---:|
| Conversation | 10 |
| Episodic / source round / FAISS vector | 各 2871 |
| 全部图节点 | 5447 |
| 全部图关系 | 8490 |

`test` 除计数外，还比较每个节点/关系的全部属性、两端 UUID、类型和 embedding，
并用恢复后的 `valid_at` 执行真实时间范围查询。通过后才能使用该数据库评测。
所有校验均不产生模型 API 成本。

在 Neo4j Browser (`http://127.0.0.1:7474`) 登录后可以运行：

```cypher
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.group_id = 'locomo_tsm_siliconflow_qwen_full_tmux_20261021_conv-26'
RETURN a, r, b LIMIT 50;
```

```cypher
MATCH (a)-[r:RELATES_TO]->(b)
WHERE r.group_id = 'locomo_tsm_siliconflow_qwen_full_tmux_20261021_conv-26'
  AND r.valid_at >= datetime('2023-05-01T00:00:00Z')
  AND r.valid_at < datetime('2023-06-01T00:00:00Z')
RETURN a.name, r.fact, b.name, r.valid_at, r.invalid_at, r.episodes LIMIT 20;
```

时间/事实内容沿用原 Graphiti 抽取；恢复成功只证明数据完整恢复，不保证抽取的
事件时间语义正确，也不代表后续回答一定正确。

## 6. 准备全量 1986 题查询

不用重建 Memory。以下命令解出缓存、生成本机配置，不调用 API：

```bash
python handoff_memory.py prepare --subset all --work-dir "$PWD/qa_full"
```

默认使用原实验的 Qwen/SiliconFlow 答案模型与 RelayRouter embedding 服务。
需要切换服务时在 prepare 中添加 `--llm-base-url`、`--llm-model`、
`--embedding-base-url`。更换模型/服务应记录为实验配置差异；查询 embedding
必须与已建库的 `text-embedding-3-small` / 1536 维保持一致。

设置自己的 API 凭据后开始查询（这里会产生 Query embedding 和 Answer API 费用）：

```bash
read -rsp 'SiliconFlow API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
read -rsp 'RelayRouter embedding API key: ' OPENAI_EMBEDDING_API_KEY
export OPENAI_EMBEDDING_API_KEY
python run_queries.py --config "$PWD/qa_full/query_config.yaml"
```

`run_queries.py` 仅加载缓存；使用相同的 TSM 检索/Answer 实现。
它处理原入口按当天日期找缓存的路径问题，每次请求日志使用独立 attempt 目录，
最终检查是否存在 1986 个唯一 QID，并检查空输出/API 失败占位符。
不要在此流程中传入 `--force-rebuild` 或启用 `reset_group`。

结果与日志：

```text
qa_full/
  cache/locomo10/item_N_all_sessions/  # 已恢复的完整缓存
  answers/Locomo/*results.json        # 答案、证据、Retrieval Trace
  request_trace/attempt_NNN/          # Query embedding 与 Answer 请求/响应
  query_config.yaml                  # 本机有效配置，不包含密钥
```

## 7. 只测试原来的 321 题

另建工作目录：

```bash
python handoff_memory.py prepare --subset 321 --work-dir "$PWD/qa_321"
python run_queries.py --config "$PWD/qa_321/query_config.yaml"
```

两种模式的全部历史 Memory 完全相同：`all` 的 `query_filter_path=null`，
`321` 只在 Answer 前筛选指定 QID。不能只构建 Query 涉及的局部历史。

全量 1986 题由 1540 道非 Adversarial 和 446 道 Adversarial 构成；
321 题子集由 244 道主评测题和 77 道 Adversarial 构成。
Adversarial 不具备普通 `answer`，需要单独拒答评估；不能混算普通 Accuracy。
`code/scripts/evaluate_predictions.py` 只提供诊断 EM/F1 与结果导出，不是
论文的 LLM-J。之前针对 321 题的专用 Judge 脚本也不能直接用于全量 1986 题。
本交接入口覆盖 Memory 恢复、全量 QA 和 Trace，未宣称已完成全量 LLM-J。

## 8. 已经执行过的交接验证

2026-10-25 在原实验 Python 环境执行：

- 10 个压缩包的离线 FAISS/来源映射检查通过。
- 全量 Query 配置准备通过，确认 `query_filter_path=null`、10 个完整缓存。
- `conv-26` 用隔离 namespace 实际导入 Neo4j：426 节点、811 关系；
  导入后全部属性和真实时间范围查询通过。
- 本说明的交接恢复测试没有调用模型，也没有测量新的 QA/LLM-J 指标。

如报告实验结果，应同时记录代码 commit、模型/服务、数据 SHA-256、Top-K、
Parser 环境、Prompt/Judge、样本筛选方式和 API 用量。
