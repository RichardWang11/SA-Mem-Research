# 更深入的 H1 案例复查

本次保留原版算法及已有结果，仅进行离线检索结果分析和 Neo4j 只读查询，没有 LLM、Embedding 请求。

## 新发现：conv-48_105

Question: What milestone did Jolene achieve recently on 4 February, 2023?

Gold answer: Design and build a sustainable water purifier for a rural community。

官方 Gold 标注 D4:3 对应 leaf_33（Dense 第5名），内容仅为“取得重大工程里程碑”，无法单独说明具体项目。后续 D4:5 明确说明设计和建造可持续净水器，D4:6 说明帮助农村居民；两条组成 leaf_34，实际 Dense 第31名，未进入最终上下文。

图谱存在直接关联 chunk_34 的事实：I had to plan and research a lot to design and build a sustainable water purifier for a rural community in need. 时间 valid_at=2023-02-04T09:48:00+00:00。Query Parser 实际输出 2023-02 全月，不是精确2月4日。该 Edge 落在原 Parser 范围内。

只读查询时间范围内全部 Edge，通过 Episode 名和原文双重校验映射现有 leaf，候选为 leaf_32、leaf_26、leaf_25、leaf_35、leaf_30、leaf_34。保持现有语义排名后，leaf_34 从31名升到时间候选中的第6名。已确认原 Dense Top100 去重后覆盖全部38个现存 leaf，所以这个顺序投影不缺少现存库候选。不是实际修改或运行 A2。

这证明已有库中存在“直接答案支持证据不在Dense Top20，但现有时间条件先筛可纳入Top20”的候选。严格说它不是新增官方Gold-ID漏召回：D4:3已经召回。基线回答提到了净水项目但缺少可持续净水器细节，论文Judge较宽松，未判分前不能称为确定错题或QA救回。

时间精度也有限：D4:3说last week，Edge标为消息当天；这里只确认符合当前Parser月范围，未证明实际事件日期绝对准确。

## 原有案例：conv-26_11

两条官方Gold D3:13、D4:3分别排第24和23名，均不在最终上下文，模型未答Sweden。属于严格Gold-ID Dense Miss。Query解析2019年，但相关Sweden事实标为2023年，搬迁Episode没有查到引用它的事实Edge，因此现有时间索引恢复能力未确认。

## 对答案短语的扩展检查

扫描321道题的现存全库leaf，在Top20外检索长度至少4且非纯数字的完整Gold短语，得到6题；另用Porter词干和多词覆盖复查，未找到额外强候选。短语命中仅作候选生成，不作为正确证据判定。

| Query | Top20外出现的答案内容 | 复核结果 |
|---|---|---|
| conv-26_11 | Sweden，第23名 | 真正官方Gold漏召回，但时间索引不支持恢复 |
| conv-48_105 | 完整净水器答案，第31名 | 新增直接支持证据候选，时间先筛排名第6 |
| conv-49_52 | painting，第24名 | 5月绘画信息，问题问9月共同计划，不是匹配时间证据 |
| conv-49_85 | watercolor painting，第21名 | 原Top20和回答已有绘画信息，不能认定必要证据漏召回；时间需进一步核查 |
| conv-50_43 | Boston，第22、26名 | 3/4月旅行信息，问题问10月3日；答案词命中不构成对应时间证据 |
| conv-50_108 | Boston，第24、26、28名 | 3/4月信息，问题问9月1日订机票；无法据此认定恢复 |

## 更新后的结论

之前“只有一个”的结论仅限官方Gold ID直接映射，有漏掉补充支持证据的局限。目前有1个严格官方Gold-ID Dense Miss和1个新增时间先筛可恢复的直接支持证据候选；两个案例性质不同，不能合并成两个已确认QA救回。

文件：../results/cases/deeper_candidate_conv48_105.json.gz 保存原始问题、完整Trace、Neo4j时间候选；../results/full/deeper_h1_verified_projection.json 保存内容校验后的排名投影；../scripts/inspect_cases.py 可重现投影。
