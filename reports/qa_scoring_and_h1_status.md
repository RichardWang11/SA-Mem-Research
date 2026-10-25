# 已有运行的 QA 评分与 H1 案例状态

这是对已完成、保持原状的 Qwen batch-1 TSM 运行的事后评分。不是修复实验，不包含付费 Judge 请求。

## H1 状态

已扫描 321 条 Trace；40 道 Query 的全部 Gold 可映射到当前 Dense leaf。找到一个已确认 Dense Miss：conv-26_11，两条 Gold 位于 Dense 排名 23 和 24，最终上下文均不含，模型拒答。Query 时间范围为 2019 年；相关 Sweden 事实 Edge 时间为 2023 年，搬迁来源 Episode 未查到事实 Edge。因此没有已验证的“时间索引准确且 Temporal-first 可恢复”的完整案例。

原始 Memory 构建缺陷继续影响结论；未映射证据不能直接算 Dense Recall 失败。

## 评分来源和方法

固定官方 evaluator Commit：`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。
源码 SHA256：`8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd`。

从固定官方源码提取并执行未修改的 normalize_answer、exact_match_score、f1_score、f1、eval_question_answering 函数；只排除未调用的 BERTScore 依赖及其他无关函数。NLTK 3.9.1 安装在独立 evaluator_deps 目录，不修改实验虚拟环境。

原样使用完整 hypothesis；没有事后提取简答、没有重新生成答案。Adversarial 原始 QA 无 answer 字段时补空字符串以满足评分器接口，类别5判分不使用该字段。官方返回的 context recall 在未给 context ID 时默认1，故本报告完全不使用它。

## 结果

| 范围 | 样本数 | 指标 | 分数 |
|---|---:|---|---:|
| 主评测集 | 244 | 官方类别规则词级 F1 | 2.96% |
| Adversarial | 77 | 官方拒答短语规则命中 | 11.69%（9/77） |
| 主评测集 | 244 | TSM 自带诊断 EM | 0.00% |
| 主评测集 | 244 | TSM 自带诊断 Token F1 | 2.54% |

这些不是二元语义 Accuracy。不要将 F1 与 Adversarial 二元规则混合后宣称总体 Accuracy。

| 官方类型 | 数量 | 类别规则得分 |
|---|---:|---:|
| Multi-hop | 15 | 9.11% |
| Temporal | 44 | 1.65% |
| Open-domain | 9 | 1.01% |
| Single-hop | 176 | 2.86% |
| Adversarial（短语规则） | 77 | 11.69% |

## 解释限制

主评测答案平均 127.4 个英文空白分词，长解释会压低词面 F1。

Adversarial 官方规则只检查 no information available / not mentioned 两个短语；其他等义拒答可能不被计分，解释中出现该短语也可能误判。

低分同时受到已确认的 Memory 缺失和答案格式影响，尚不能分解两者贡献，也不代表论文报告的分数被推翻。

Parser 命中 202/244；最终候选映射到非空时间匹配 Edge 的题目 44/244。后者不是全部数据库时间查询的激活率，也不是 Gold 时间命中率。

完整逐题分数：../results/full/qa_per_question_scores.jsonl；汇总：../results/full/qa_scoring_summary.json；运行脚本：../scripts/score_existing_qa.py。

本次评分更新了之前“官方 evaluator 尚未运行”的状态；之前报告保留为历史记录。
