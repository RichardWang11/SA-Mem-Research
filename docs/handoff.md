# 接手顺序与本地上传计划

1. 运行verify_bundle.py和check_recorded_smoke.py，查看真实输入/输出和已知问题。
2. 阅读results.md、provenance.md；确认没有将F1写成论文Accuracy，也没有将离线排名投影写成QA救回。
3. 如需在线复跑，先dry-run确认profile；默认官方仓库batch8，历史结果使用batch1。
4. 2026-10-25 用户授权修复 Dense Tree 构建错误。当前 TSM 源码已合入修复，历史运行记录保留。接手复用完整 Memory 时优先阅读 [Memory 恢复与全量查询交接](../memory_repair/HANDOFF_ZH.md)；不要把修复后的结果与历史损坏建库结果混用。Query 时间解析和检索架构未改。
5. 修复后321题QA与Mem0提示词+gpt-4o-mini Judge已完成，详见results/repaired_qa321_20261025/REPORT.md。论文口径GPT-4.1-mini Judge仍待执行。Adversarial单独评估，不能把空参考答案直接套进主评测Judge。接手前先明确付费规模和Judge配置，保存所有请求与逐题标签。
6. 继续找H1必须区分：库内Dense Miss、建库时未进入索引、Parser/图谱时间错误、Gold标注遗漏补充证据。不能把mapping_unknown算Recall失败。

## 常用入口

复用已修复 Memory，无需重新抽取或重新生成文档 embedding：

```bash
cd memory_repair
python handoff_memory.py inspect
# 设置 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE 后：
python handoff_memory.py restore --execute
python handoff_memory.py test
python handoff_memory.py prepare --subset all --work-dir "$PWD/qa_full"
# 设置自己的 Answer 与 Query embedding API 凭据后才会产生 API 调用：
python run_queries.py --config "$PWD/qa_full/query_config.yaml"
```

`--subset all` 回答全部 1986 题；`--subset 321` 回答原子集，两者 Memory 相同。
这些命令的依赖、连接、验收标准和结果路径详见完整交接文件。
下面的 `run_tsm.py` 则是重新建库入口，其 `--scope full` 历史定义仍为 321 题。
该入口的 `official-default` 表示沿用官方参数；当前 TSM 源码已含构建修复，
因此新建库应标记为 repaired reproduction，不应称作未修改官方 Baseline。

```bash
# 标准库即可运行，完全离线
python3 scripts/verify_bundle.py
python3 scripts/check_recorded_smoke.py
python3 scripts/inspect_cases.py

# 在TSM虚拟环境中，默认dry-run
python scripts/run_tsm.py --scope smoke
python scripts/run_tsm.py --scope smoke --profile historical-qwen-batch1

# 评分已上传的历史答案，使用独立新输出目录
python scripts/score_existing_qa.py --out .local/rescored

# 用户自己有原始run目录和Neo4j后才使用；数据库只读
python scripts/search_temporal_candidates.py --run /path/to/original-run --out .local/new-graph-search
```

scripts/audit_run.py用于历史Qwen batch1数据的完整目录审计，需要原始数据、配置和cache日期，模型常量描述历史设置；不是任意新profile的通用验收器。使用--help查看参数。gold_mapping中旧字段graphiti_edge_uuids实际是Episode UUID，事实Edge UUID请看案例数据库原始查询结果。

## Git计划

目标：RichardWang11/SA-Mem-Research。

本地分支：research/tsm-locomo-temporal-audit，为不继承主分支文件的orphan分支，仅TSM材料，不含SA-Mem源文件。该分支以独立根提交发布，仅包含本次 TSM 交接包。

README、默认batch8与历史batch1区别、Smoke包含范围、依赖版本、文件哈希、密钥模式扫描和Git文件清单已在提交前检查。该分支用于独立浏览和checkout；不要直接合并进main，以免误改主分支的SA-Mem文件。
