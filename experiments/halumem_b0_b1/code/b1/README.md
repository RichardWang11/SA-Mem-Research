# B1 代码快照

本目录来自 `SA-Mem-core` 的 `halumem-b1-local-temporal` 分支，核心 commit 为
`510d0f3`。文件名与 `code/b0/` 一一对应，便于直接比较。

B1 的新增路径是：

```text
local-temporal Pass1 JSON
→ 校验 temporal_expressions 是否覆盖 mentions
→ mention_index 对应的本地 resolve_temporal_expression()
→ 将日期追加到对应 mention
→ 原有 Pass2
```

如果 JSON、mention 映射、session 日期或本地解析结果不可靠，则 fallback 到 B0 的
tool + follow-up 路径。

建议同时查看根目录的 `diff/B0_TO_B1.patch`，它是从 B0 commit 到 B1 commit 生成的
核心代码 diff。
