# TSM reproduction and LoCoMo temporal-retrieval audit

本分支只负责 TSM 复现与交接，不包含 SA-Mem 源码。研究问题是：Dense Top-20 是否遗漏时间相关证据，以及时间先筛是否能恢复。**H1 尚未完成验证；当前材料不证明 Temporal-first 的 QA 提升。**

## 先看什么

1. [环境配置](docs/environment.md)：独立 Python 环境、requirements 和 Neo4j。
2. [Smoke 交接](docs/smoke.md)：已归档的真实单 Query 运行，以及可选在线复跑。
3. [实验结果与限制](docs/results.md)：321 题、QA 指标、上游缺陷和候选案例。
4. [代码与数据说明](docs/provenance.md)：官方代码、本地适配、数据来源和日志范围。
5. [后续工作清单](docs/handoff.md)。

## 当前事实

| 项目 | 已完成工作 |
|---|---|
| 数据 | LoCoMo 10个 Conversation，完整历史输入；JEV显式时间Query共321，主评测244、Adversarial 77 |
| 历史运行 | Qwen/Qwen3-30B-A3B-Instruct-2507，batch 1，Top-20，Graphiti预算16k，52,352份请求Trace |
| 默认新运行配置 | **官方仓库 batch 8 / parallel_api_calls 8 / max_parallel_llm 8**；归档结果不会改写成batch 8 |
| QA | 主评测官方LoCoMo词级F1 2.96%；**论文GPT-4.1-mini LLM-J Accuracy未运行** |
| 上游缺陷 | 时间窗口leaf覆盖、来源文本错配、英文日期参考时间回退；已核对固定官方Commit |
| H1案例 | conv-26_11官方Gold排23/24但时间恢复未确认；conv-48_105补充支持证据31→6的离线排名投影 |

公开实现的缺陷保持原状，不为了研究假设修复 Memory 或实现新的检索架构。历史 smoke 虽跑通，但Dense Tree覆盖不完整，不能当作效果验收通过。

## 无API成本的快速检查

在本仓库根目录执行（只需要Python标准库）：

```bash
python3 scripts/verify_bundle.py
python3 scripts/check_recorded_smoke.py
python3 scripts/inspect_cases.py
```

第一条核对文件哈希并扫描常见密钥模式；第二条读取完整真实smoke归档；第三条检查保存的候选排名证据。它们不连接Neo4j，不调用LLM，也不重新执行检索。

## 文件布局

```text
TSM/                    固定的TSM研究源码快照，含上游代码和已发生的适配/Trace改动
configs/                默认官方参数、历史Qwen batch1参数
scripts/                复跑、离线验收、评分和案例查看入口
selection/              LoCoMo标注与321个Query ID；smoke只选conv-26_11
results/smoke/          完整真实smoke压缩包（所有原始请求、缓存、答案、日志）
results/full/           321题答案压缩包、逐题分数、验收汇总
results/cases/          只读Neo4j案例与候选的原始记录
manifests/              来源、环境、全量Trace索引和校验哈希
patches/                相对固定官方Commit的全部研究改动
reports/                历史研究报告，当前解释以docs/为准
```

本分支不包含API密钥、Neo4j密码、虚拟环境、完整LoCoMo下载文件或Neo4j数据库卷。Smoke归档和研究结果含公开LoCoMo派生文本，见[来源说明](docs/provenance.md)。
