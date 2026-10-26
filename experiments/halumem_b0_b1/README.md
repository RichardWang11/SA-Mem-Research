# SA-Mem HaluMem B0 → B1 Baseline

这个公开仓库只用于展示 HaluMem-Medium 上 **B0 原始实现** 与 **B1 local temporal
resolution 修复** 的代码关系和实验结果。

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

## B1 结果将在下一次 commit 加入

B1 commit 加入后，本 README 的后半部分会补充：

- B0/B1 的代码文件级 diff；
- B0/B1 construction token、retrieval、QA 对比；
- B1 local resolve 和 fallback 的实际路径；
- B1 结果文件和可审计命令。
