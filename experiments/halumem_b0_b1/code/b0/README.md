# B0 代码快照

本目录来自 `SA-Mem-core` 的 `main` commit `46dfaa8`，只保留与 HaluMem memory
construction 直接相关的文件。

关键入口：

- `build_impl_graph.py`：B0 construction 流程；
- `build_prompts.py`：Pass1/Pass2 prompt；
- `memblock_extractor.py`：MemBlock 抽取和 tool/follow-up 调用；
- `temporal_resolution_tool.py`：B0 使用的日期 resolver 和 function schema。

B1 对应版本会在后续 commit 的 `code/b1/` 中加入；可通过根目录
`diff/B0_TO_B1.patch` 查看核心变化。
