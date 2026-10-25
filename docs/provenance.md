# 源码、数据与交接边界

## 官方和本地代码

- 上游：https://github.com/cosmicexotic/TSM
- 官方基础Commit：`0da3633ba8e27e9f8122872ff775587116948f5e`
- 研究分支已提交HEAD：`7c1534dcf4ef4afb41afed6cb01746ec0b140d1e`
- 本地未提交适配与Trace改动：全部包含在TSM/快照和patch中。

`TSM/`是运行后采集的研究源码快照，不包含其.git、不嵌套仓库、不包含SA-Mem。不能声称它是运行开始时归档的不可变快照。`manifests/source_snapshot.json`给出逐文件哈希和原始dirty状态。

更新：当前 `TSM/` 已合入源实验仓库修复提交 `d7c124c`（leaf 全局身份、来源对齐和全量源文本保留），另含恢复脚本、回归测试和修复说明。历史源码哈希见 `manifests/source_snapshot_historical.json`，历史树代码见 `patches/historical_tree_builder.py.txt`，本次修复补丁见 `patches/temporal_tree_repair.patch`。`memory_repair/code/` 是本次构建修复时的完整实际代码快照。

真实 query-only Smoke 后另补 `4aeb080`，兼容 Graphiti `graph.json` 的属性字典格式；旧数字权重格式仍沿用原合并行为。此修复只影响缓存反序列化，补丁见 `patches/query_cache_compatibility.patch`。`memory_repair/code/` 同步该后续修复。

`patches/research_vs_upstream.patch`包含Query筛选、Retrieval Trace、API请求Trace、Graphiti Episode初始化兼容、输出预算传递、禁用不支持的可选参数、Chat Completions JSON模式适配。没有修复上述树构建和时间错误。

Chat Completions适配器会向提示词追加JSON Schema，结构保证不等同于Responses严格Structured Outputs。部分Trace在追加前记录messages；阅读原始prompt时需结合适配器代码。

Python代码保留上游MIT许可，见TSM/LICENSE。Graphiti依赖源码和提示词未整包vendoring，仅依赖固定版本；历史请求包含运行时实际提示词。

## 论文与仓库差异

论文：https://aclanthology.org/2026.findings-acl.1496.pdf

- 论文附录Top-K=25，公开LoCoMo配置max_chunk_setting=20；历史运行用20。
- 论文LoCoMo参考时间描述为session start time；公开代码采用最后session时间+1天，历史运行沿用代码。
- 官方仓库batch8、外层并发8；历史运行获准改成batch1、外层并发1、Graphiti SEMAPHORE_LIMIT=1。
- TSM Snapshot内部仍有自己的线程池，不能把max_parallel_llm=1解释成每个阶段绝对单线程。
- 默认新运行profile恢复仓库batch和并发8；不改写历史记录。`reset_group=false`、新Group前缀和新目录是交接入口的数据隔离措施。

## 数据

官方LoCoMo：https://github.com/snap-research/locomo

固定数据Commit：`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。由TSM官方脚本下载并校验；本次在 `memory_repair/locomo10.json` 补充同一固定版本完整数据用于跨机器恢复。

selection/annotations_locomo1986.csv是用户提供分类文件中的LoCoMo部分；原文件实际还包含500条LongMemEval。本交接包不重新生成或推断JEV标签，来源标记tsm_active也不替代本地Parser实际结果。

结果、案例、Smoke原始请求包含LoCoMo派生文本和生成内容，按上游 CC BY-NC 4.0 条件提供，保留原始来源和修改说明；许可文本见 results/full/LOCOMO_LICENSE.txt。不因本仓库软件MIT许可而改变。

## 原始日志范围

Smoke：完整console、request_trace、cache、answers全部打包。它不是数据库dump。

全量：保留321题原始答案和检索Trace、逐题评分、案例数据库只读输出、52,352个原始请求文件的名称/大小/SHA256索引；没有把全部请求正文放进Git。索引在manifests/full_request_trace_inventory.jsonl。

原服务器位置仅作为定位信息，不是脚本依赖：

```text
/workspace/research/tsm_temporal_first/runs/siliconflow_qwen_full_tmux_20261021
/workspace/research/tsm_temporal_first/runs/siliconflow_qwen_rawtrace_smoke_20261021_v3
```

不包含密钥、密码、数据库卷、venv。扫描程序同时检查压缩包内部常见密钥模式，但模式扫描不是所有敏感信息的完备检测。

已确认问题的无API离线复现：`python3 scripts/reproduce_upstream_issues.py`。
