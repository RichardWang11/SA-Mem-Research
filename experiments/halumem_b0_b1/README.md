# SA-Mem HaluMem B0 → B1 Baseline

这个公开仓库只用于展示 HaluMem-Medium 上 **B0 原始实现** 与 **B1 local temporal
resolution 修复** 的代码关系和实验结果。

本目录是从 `SA-Mem-Research/main` 隔离出来的实验包，不覆盖 Research 仓库根目录的
现有源码。B0/B1 的代码快照沿用之前正式实验使用的 `SA-Mem-core` 源码 lineage。

## 相关链接

- [SA-Mem-core B0 source (`main`, `46dfaa8`)](https://github.com/jianminYa/SA-Mem-core/tree/46dfaa8)
- [SA-Mem-core B1 source branch](https://github.com/jianminYa/SA-Mem-core/tree/halumem-b1-local-temporal)
- [SA-Mem-core B0 → B1 compare](https://github.com/jianminYa/SA-Mem-core/compare/main...halumem-b1-local-temporal)
- [独立 B0/B1 结果仓库](https://github.com/jianminYa/halumem-b0-b1-baseline)
- [本分支 GitHub 页面](https://github.com/RichardWang11/SA-Mem-Research/tree/research/halumem-b0-b1-baseline/experiments/halumem_b0_b1)

`SA-Mem-Research/main` 与 `SA-Mem-core/main` 不是逐文件相同的快照；因此这里保留了
可审计的 B0/B1 对应源码，而不是把 B1 patch 直接覆盖到 Research 根目录。

## 如何阅读 commit 历史

仓库按两次提交组织：

1. **第一个 commit：B0 baseline**
   - `code/b0/`：原始 SA-Mem construction 相关代码；
   - `results/b0/`：B0 的 memory、retrieval、QA 和汇总结果。
2. **第二个 commit：B1 modification**
   - 新增 `code/b1/`：与 B0 对应的 B1 代码；
   - 新增 `results/b1/`：B1 结果；
   - 新增 `diff/B0_TO_B1.patch`：只包含 B0 → B1 的核心代码差异；
   - 本 README 更新为 B0/B1 对照说明。

因此可以直接打开 GitHub 的 commit diff，看到“先有 B0，再加入 B1”的实际修改，而不是
把 B0、B1 混在一个初始快照里。

## B0 与 B1 的核心区别

```text
B0: Pass1 → temporal tool → follow-up LLM → Pass2
B1: Pass1(+temporal_expressions) → local resolver → Pass2
                         └─失败时 fallback 到 B0 tool + follow-up
```

B1 没有删除 raw dialogue、Pass2 或最终 MemBlock schema。它只改变 Pass1 的时间处理：

- B0 的 tool 参数只有 `expression + observation_time`，follow-up 需要自行把日期映射回 mention；
- B1 Pass1 额外输出 `mention_index + expression`；
- Python 本地调用 `resolve_temporal_expression()`，把日期补回对应 mention；
- 如果表达式覆盖不完整、日期无效、resolver warning 或 JSON 不可靠，则回到 B0 原始路径。

详细修改位置：

| 文件 | B1 中的变化 |
|---|---|
| `build_impl_graph.py` | B1 Pass1、本地 temporal resolve、coverage 校验和 fallback |
| `build_prompts.py` | B1 Pass1 的中间输出格式 |
| `memblock_extractor.py` | B1 分支、tool/follow-up instrumentation 与构建流程接入 |
| `temporal_resolution_tool.py` | 原有日期 resolver；B1 复用它，不重新发明日期算法 |

## 第一个 commit 中的 B0

B0 代码来源：`SA-Mem-core` 的 `main`，commit `46dfaa8`。

- [B0 code snapshot](code/b0/)
- [B0 result artifacts](results/b0/)
- [B0 summary](results/b0/summary.json)

### B0 实验配置

| 项目 | 配置 |
|---|---|
| Dataset | HaluMem-Medium，10 users，1,722 questions |
| LLM | `gpt-4o-mini` |
| Embedding | `text-embedding-3-small` |
| Retrieval | Top-20；post-hoc gold-session audit |
| QA context | Top-20 |
| QA repeats | 1 |

### B0 结果

| Metric | B0 |
|---|---:|
| Memory units | 2,434 |
| Construction tokens（10-user estimate） | 19,578,022 |
| Hit@1 | 35.25% |
| Hit@5 | 54.74% |
| Hit@10 | 61.41% |
| Hit@20 | 68.56% |
| MRR | 0.4479 |
| Mean gold rank | 32.85 |
| QA | 928/1,722 = 53.89% |

`memory_units.jsonl` 和 `retrieval_full.jsonl` 是逐条结果；不包含原始 dataset 文件、embedding
cache、API 配置或 credentials。

## 第二个 commit 加入的 B1

B1 代码来源：`SA-Mem-core` 的 `halumem-b1-local-temporal`，commit `510d0f3`。

- [B1 code snapshot](code/b1/)
- [B1 result artifacts](results/b1/)
- [B1 summary](results/b1/summary.json)
- [B0 → B1 core patch](diff/B0_TO_B1.patch)

对应的四个核心文件可以直接逐个比较：

```text
code/b0/build_impl_graph.py  ↔ code/b1/build_impl_graph.py
code/b0/build_prompts.py     ↔ code/b1/build_prompts.py
code/b0/memblock_extractor.py ↔ code/b1/memblock_extractor.py
code/b0/temporal_resolution_tool.py ↔ code/b1/temporal_resolution_tool.py
```

其中 `temporal_resolution_tool.py` 在 B0 中已经存在，B1 复用原有 resolver；实际 B1
逻辑主要集中在前三个文件。

## B0 vs B1 实验结果

两边使用同一批 HaluMem-Medium 10-user-v2 数据、同一模型配置和同一 retrieval/QA 评测。
construction 是已完成 source run 的 10-user estimate；本次 v2 只重新执行了 embedding、
retrieval 和一次 QA/judge，没有重新 construction。

| Metric | B0 | B1 |
|---|---:|---:|
| Memory units | 2,434 | 2,405 |
| Construction tokens（10-user estimate） | 19,578,022 | 15,304,975 |
| Construction saving | — | 21.83% |
| Hit@1 | 35.25% | 34.32% |
| Hit@5 | 54.74% | 53.49% |
| Hit@10 | 61.41% | 59.86% |
| Hit@20 | 68.56% | 67.08% |
| MRR | 0.4479 | 0.4374 |
| Mean gold rank | 32.85 | 33.97 |
| QA correct | 928/1,722 | 947/1,722 |
| QA accuracy | 53.89% | 54.99% |

### 结果解释

B1 construction estimate 比 B0 低约 21.83%，主要节省来自把可可靠解析的 temporal
follow-up 转为本地零 token 处理。B1 的 retrieval 指标略低，但一次 QA accuracy 高
1.10 个百分点；这只能作为待验证现象，不能据此证明 B1 一定提升了模型能力。QA 只有
一次，且 B0/B1 memory 内容并不完全相同，后续应重复 QA 并单独分析 relative-time 问题。

### 可审计结果文件

- B0：[`results/b0/`](results/b0/)
- B1：[`results/b1/`](results/b1/)
- 每个 variant 的 `memory_units.jsonl`：构建出的 memory units；
- `retrieval_full.jsonl`：逐 question 的完整 retrieval ranking；
- `retrieval_posthoc_gold_session.json`：Hit@K、MRR 和 gold-session rank；
- `qa_results_1repeat.jsonl`：逐 question QA/judge 结果；
- `summary.json`：只保留该 variant 的可读汇总。

完整原始数据集、embedding cache、API 配置和 credentials 不在仓库中。
