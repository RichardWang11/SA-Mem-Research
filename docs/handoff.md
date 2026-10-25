# 接手顺序与本地上传计划

1. 运行verify_bundle.py和check_recorded_smoke.py，查看真实输入/输出和已知问题。
2. 阅读results.md、provenance.md；确认没有将F1写成论文Accuracy，也没有将离线排名投影写成QA救回。
3. 如需在线复跑，先dry-run确认profile；默认官方仓库batch8，历史结果使用batch1。
4. 当前用户决定保持官方实现，不修复原算法。新增分析与上游问题必须分开记录。
5. 论文口径GPT-4.1-mini Judge待执行。Adversarial单独评估，不能把空参考答案直接套进主评测Judge。接手前先明确付费规模和Judge配置，保存所有请求与逐题标签。
6. 继续找H1必须区分：库内Dense Miss、建库时未进入索引、Parser/图谱时间错误、Gold标注遗漏补充证据。不能把mapping_unknown算Recall失败。

## 常用入口

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
