# Smoke 案例交接

## A. 已完成真实运行：无需API即可查看

归档：`results/smoke/recorded_conv26_smoke.tar.gz`。

历史 Run ID：`siliconflow_qwen_rawtrace_smoke_20261021_v3`。使用完整conv-26历史，构建后只回答 `conv-26_11`：Where did Caroline move from 4 years ago? Gold为Sweden。

该运行使用Qwen/SiliconFlow、RelayRouter text-embedding-3-small、batch1、并发配置1、Graphiti输出预算16384；不等于默认batch8配置。归档包括完整请求链，而非只提供一两个成功请求。

```bash
python3 scripts/check_recorded_smoke.py
tar -tzf results/smoke/recorded_conv26_smoke.tar.gz | head
mkdir -p .local/recorded_smoke
tar -xzf results/smoke/recorded_conv26_smoke.tar.gz -C .local/recorded_smoke
```

解压内容：

- `console.log`：构建阶段、时间窗口、查询和结束记录。
- `request_trace/*.json`：所有请求/返回、Token和错误。
- `cache/locomo10/20261021/item_0_all_sessions/`：tree、FAISS、snapshots和文本映射。
- `answers/Locomo/*results.json`：1道问题的答案与完整检索Trace。

查看顺序：结果中的retrieval_trace → tree.json中的leaf → text_edge_mapping中的Episode → request_trace中的模型输入输出。该归档不是Neo4j数据库备份，解压不会恢复数据库。

**Smoke证明链路能完成，不证明抽取质量和Memory完整。** 原答案错误，后续发现Dense Tree只有少量leaf、时间参考回退等问题。`check_recorded_smoke.py` 会明确显示输入Round记录数和leaf数，不能把检查成功当作研究质量通过。

## B. 新机器在线复跑

先按[环境说明](environment.md)准备依赖和Neo4j，再下载原始固定版本数据：

```bash
python TSM/scripts/download_data.py locomo
python scripts/run_tsm.py --scope smoke
```

第二条默认仅dry-run，无模型成本。默认profile为 `official-default`：模型GPT-4o-mini、batch8、并发配置8、Top20、输出预算8192。仓库参数是8；论文并没有在附录明确给出这个batch值。

配置凭据后显式执行（会付费；完整conv-26构建可能持续数小时）：

```bash
set -a; source .env; set +a
python scripts/run_tsm.py --scope smoke --execute
```

要复跑归档使用的历史设置，必须显式选择：

```bash
python scripts/run_tsm.py --scope smoke --profile historical-qwen-batch1
python scripts/run_tsm.py --scope smoke --profile historical-qwen-batch1 --execute
```

历史profile使用SiliconFlow LLM和RelayRouter Embedding；在.env中配置相应两个Key。入口不读取原服务器的密钥文件。

入口为每次执行创建新目录和随机Group前缀，`reset_group=false`，不删除已有实验数据。记录resolved配置、命令、退出码及完整请求Trace。重新运行不是逐字节重现历史答案；模型服务和依赖差异可能影响结果。

全量入口：`python scripts/run_tsm.py --scope full`，仍默认dry-run。全量固定筛选321道Query，不缩减各Conversation的历史。
