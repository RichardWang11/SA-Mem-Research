# 实验结果与结论边界

## 样本和执行

数据使用LoCoMo固定Commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` 的locomo10.json，SHA256为 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`。保留完整历史，筛选JEV显式时间321题；不是官方Temporal类321题。

| Conversation | 主评测 | Adversarial | 合计 |
|---|---:|---:|---:|
| conv-26 | 12 | 2 | 14 |
| conv-30 | 4 | 3 | 7 |
| conv-41 | 33 | 5 | 38 |
| conv-42 | 37 | 13 | 50 |
| conv-43 | 28 | 10 | 38 |
| conv-44 | 16 | 4 | 20 |
| conv-47 | 29 | 13 | 42 |
| conv-48 | 27 | 8 | 35 |
| conv-49 | 31 | 10 | 41 |
| conv-50 | 27 | 9 | 36 |
| 总计 | 244 | 77 | 321 |

全量运行约38小时44分，2026-10-23 06:45:50 UTC完成。共52,352个Trace，52,349 completed、3个结构化校验error，后续重试成功。结果321个唯一QID，无缺失或多余。

Smoke与全量是两次独立运行。Smoke有4,300个Trace、1个已恢复结构化错误；conv-26有32个Dense leaf。全量conv-26有34个leaf。不要混用两次运行的缓存和排名。

## QA指标

| 范围 | 指标 | 值 |
|---|---|---:|
| 主评测244题 | LoCoMo官方类别规则词级F1 | 2.9568% |
| Adversarial 77题 | LoCoMo原评分器拒答短语命中 | 11.6883%（9/77） |
| 主评测244题 | TSM诊断EM | 0% |
| 主评测244题 | TSM诊断Token F1 | 2.5418% |
| 主评测244题 | **TSM论文LLM-J Accuracy** | **未运行** |

论文4.1使用GPT-4.1-mini Judge和附录A.4的CORRECT/WRONG提示，不能用这里的F1替代。当前仓库的TSM脚本只有诊断EM/F1，没找到论文Judge runner。论文使用的Qwen骨干与本次模型相同，但默认仓库配置是GPT-4o-mini；本次batch1和协议适配等仍需独立说明。

按原始完整生成答案评分，没有事后提取简答。主评测平均127词，长解释会降低词面F1；同时Memory覆盖缺陷真实存在。不能把低分全部归因于格式，也不能从F1推断二元Accuracy。

重新离线评分（需evaluation依赖，不调用API）：

```bash
python scripts/score_existing_qa.py --out .local/rescored
```

## 已确认的上游实现问题

1. 月份窗口重复从leaf_0编号，合并字典覆盖。
2. 窗口局部下标i用于全局original_text[i]，实际文本错配到历史开头。
3. _reference_time仅识别数字年月日，LoCoMo英文日期回退到2000年。

构建文件与官方Commit逐字节一致，日期函数与官方AST一致；证据见 `results/full/issue_provenance.json`。我们不修复这些逻辑，记录其对公开实现复现的影响。不能据此推断论文内部实现或论文分数有误。

每个Conversation有181至334个输入Round记录，最终只有23至44个Dense leaf。所谓7.5%～16.6%的覆盖率是“保存的第一条消息文本能在Dense原文中找到”的比例，不是独立标注的事实抽取正确率。

## H1候选

- `conv-26_11`：两条官方Gold分别Dense第24、23名，均没进最终上下文；模型没有答Sweden。图谱的Sweden事实时间是2023，Query是2019，未证明时间先筛可恢复。
- `conv-48_105`：官方Gold D4:3的“工程里程碑”已召回，但直接解释净水器项目的D4:5–6在leaf_34、第31名；时间先筛的现存库语义排名第6。原回答有部分答案，无Judge前不能声称QA由错变对。Parser把4 February解析为整月，该离线投影也不证明事件精确时间无误。
- 扩展图谱只读筛查得到48题168个时间匹配的Top20外候选，其中155个可在时间先筛投影中进入Top20。它们大多不相关，不是168条Gold恢复。没有新增确认案例。

当前状态：严格官方Gold-ID Dense Miss 1例；额外直接支持证据时间恢复候选1例；已确认QA救回0例。H1未证实也未否定，H2双路召回未实现。

## Trace和Token

原Trace汇总总Token为23,878,802：Graphiti 22,003,152、Embedding 222,302、其他摘要等941,098、Generation 712,250。原程序Token统计文件为0，不能用它计算成本。提供者计费、缓存折扣、SDK内部重试并不一定都能从这些日志完整还原，人民币成本未核定。

Parser主评测命中202/244；候选中出现时间匹配Chunk 44/244。不是Parser正确率，也不是Gold命中率。Trace只保存匹配到候选的Edge，不能把空Trace直接解释为图谱没有任何时间Edge。
