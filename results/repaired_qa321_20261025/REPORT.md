# 修复完整 Dense Memory 后的 LoCoMo 显式时间 QA 对照

本报告比较建库修复前后的相同 321 个 QID；244 道主评测与 77 道 Adversarial 分开。
两者均使用 Qwen/Qwen3-30B-A3B-Instruct-2507、Top-20、相同 Answer Prompt，
Judge 均为 RelayRouter gpt-4o-mini + Mem0 论文附录 A 提示词。
这不是 TSM 论文 GPT-4.1-mini Judge 口径，也不是 LoCoMo 全量 1986 题。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 主评测 LLM-J | 37/244 (15.16%) | 169/244 (69.26%) |
| Adversarial 合成拒答参考 LLM-J | 69/77 | 43/77 |
| Adversarial 官方拒答短语规则 | 9/77 | 4/77 |

主评测净变化：+54.10 个百分点；错→对 139，对→错 7。

Adversarial 的普通 gold answer 为空，合成参考被明确标注，不能混算为主评测标准 Accuracy。

## 证据覆盖

Gold Dialogue ID 通过实际 source_mapping 对齐 leaf；仅在全部 Gold 均可可靠映射的题目上统计召回。
主评测可映射 242/244，mapping_unknown 1，无 Gold 1。

| 主评测证据指标 | 修复前 | 修复后 |
|---|---:|---:|
| Gold leaf Recall@20 | 35/277 (12.64%) | 196/277 (70.76%) |
| Gold leaf Recall@100 | 37/277 (13.36%) | 262/277 (94.58%) |
| 最终上下文 Gold leaf 覆盖率 | 35/277 (12.64%) | 196/277 (70.76%) |

## 修复后仍错误的可追踪样例

以下只列主评测，按 QID 排序。未召回与已召回仍答错分开记录；Gold 标注不一定穷尽全部可用证据。

### conv-26_11 — dense_top20_missing_some_or_all_gold

问题：Where did Caroline move from 4 years ago?
Gold：Sweden
生成答案：After carefully reviewing all the provided memories, there is no information about where Caroline moved from 4 years ago.

The memories contain details about Caroline's life in 2023, including her involvement with adoption, mentorship programs for LGBTQ youth, family events, and personal growth. However, none of the timestamps or content mention her previous residence or any move that occurred 4 years ago (around 2019).

Therefore, based on the available information, the answer cannot be determined.
Gold leaf：leaf_23, leaf_29

### conv-26_137 — dense_top20_missing_some_or_all_gold

问题：What painting did Melanie show to Caroline on October 13, 2023?
Gold：A painting inspired by sunsets with a pink sky.
生成答案：To determine what painting Melanie showed to Caroline on October 13, 2023, let's analyze the relevant memory:

- **Date**: 10:31 am on 13 October, 2023  
- **User (Caroline)**: "Wow Mel, that's stunning! Love the colors and the chilled-out sunset vibe. What made you paint it? I've been trying out abstract stuff recently. It's kinda freeing, just putting my feelings on the canvas without too much of a plan. It's like a cool form of self-expression."  
- **Assistant (Melanie)**: "Thanks, Caroline! I painted it because it was calming. I've done an abstract painting too, take a look! I love how ar
Gold leaf：leaf_178

### conv-26_40 — dense_top20_missing_some_or_all_gold

问题：How many times has Melanie gone to the beach in 2023?
Gold：2
生成答案：To determine how many times Melanie has gone to the beach in 2023, let's analyze the relevant memories:

1. **Date: 8:56 pm on 20 July, 2023**  
   User: "Wow, Melanie, what a beautiful moment! Lucky you to have such an awesome family!"  
   Assistant: "Seeing my kids' faces so happy at the beach was the best! We don't go often, usually only once or twice a year. But those times are always special to spend time together and chill."

   This message confirms that Melanie went to the beach **once in 2023**, specifically in July. She mentions that they don’t go often—only once or twice a year—and
Gold leaf：leaf_52, leaf_96

### conv-26_45 — gold_in_final_context_but_answer_wrong

问题：When did Caroline attend a pride parade in August?
Gold：The Friday before 14 August 2023
生成答案：To determine when Caroline attended a pride parade in August, let's analyze the relevant memories:

1. **Date: 2:24 pm on 14 August, 2023**  
   User: "Hey Caroline! Last night was amazing! We celebrated my daughter's birthday with a concert surrounded by music, joy and the warm summer breeze. Seeing my kids' smiles was so awesome, and I'm so thankful for our special moments together."  
   Assistant: "Wow, sounds wonderful! Your love for your kids is so awesome. What concert was it? The advocacy event was a cool experience - so much love and support, amazing!"

   Note: The assistant refers t
Gold leaf：leaf_106

### conv-41_0 — dense_top20_missing_some_or_all_gold

问题：Who did Maria have dinner with on May 3, 2023?
Gold：her mother
生成答案：Let's analyze the memories to determine who Maria had dinner with on May 3, 2023.

First, check if there is any memory mentioning a dinner on or around May 3, 2023.

Looking through the memories:

- The most recent memory before May 3, 2023, is from **3:18 pm on 4 May, 2023**:
  > User: Yeah, Maria! Family is the most important thing for me. They're my biggest support. We have each other's back through good times and bad.

  This is a conversation about family, but it doesn't mention dinner.

- The next relevant memory is from **6:29 pm on 7 July, 2023**:
  > User: Thanks, Maria. Your support 
Gold leaf：leaf_129

### conv-41_107 — dense_top20_missing_some_or_all_gold

问题：What new activity did Maria start recently, as mentioned on 3 June, 2023?
Gold：volunteering at a local dog shelter once a month
生成答案：Let's analyze the memories to determine what new activity Maria started recently, as mentioned on 3 June, 2023.

First, check if there is any memory dated 3 June, 2023. Looking through the provided memories, there is no entry with that exact date.

Now, check for any mention of Maria starting a new activity around that time. The closest relevant memory is:

- Date: 6:20 pm on 3 August, 2023  
  User: Maria, that's great! That picture shows a lot of joy. What got you started at that place?  
  Assistant: I started volunteering here about a year ago after witnessing a family struggling on the st
Gold leaf：leaf_174

### conv-41_110 — dense_top20_missing_some_or_all_gold

问题：What exciting news did Maria share on 16 June, 2023?
Gold：joined a gym
生成答案：After reviewing all the memories, there is no record of any conversation or event involving Maria on **16 June, 2023**. The closest relevant entries are:

- **13 August 2023**: Maria mentions her "little one" and how much joy she brings.
- **25 May 2023**: Maria shares that she dropped off baked goods at a homeless shelter.
- **6 May 2023**: Maria mentions joining a church.
- **3 August 2023**: Maria talks about volunteering at shelters and her plans for the next month.

There is no mention of any news shared by Maria on **16 June 2023** in the provided memories.

Therefore, based on the avail
Gold leaf：leaf_188

### conv-41_128 — gold_in_final_context_but_answer_wrong

问题：In what activity did Maria and her church friends participate in July 2023?
Gold：hiking
生成答案：To determine the activity Maria and her church friends participated in July 2023, let's analyze the relevant memories:

1. **Date: 3:34 pm on 17 July, 2023**  
   User: "Thanks, Maria! It's great to be part of this organization and work with such passionate people. We're like a family - always supporting each other. Do anything fun lately?"  
   Assistant: "Yeah, last weekend I had a picnic with some friends from church. We chilled under the trees, played games, and ate yummy food. It was great!"

This memory explicitly states that on **17 July 2023**, Maria had a **picnic with friends from ch
Gold leaf：leaf_249

### conv-41_149 — gold_in_final_context_but_answer_wrong

问题：What is John currently doing as a volunteer in August 2023?
Gold：mentoring students at a local school
生成答案：Based on the memories provided, John is currently volunteering as a member of the fire-fighting brigade in August 2023.

This is confirmed by the memory dated **11:08 am on 16 August, 2023**, where John states:  
> "Hey Maria! Guess what? I'm now part of the fire-fighting brigade. I'm super excited to be involved and help out my community!"

This is the most recent information about John's volunteer activities, and it directly answers the question about what he is doing in August 2023.
Gold leaf：leaf_304

### conv-41_151 — dense_top20_missing_some_or_all_gold

问题：What did the donations help John's community acquire on 16 August, 2023?
Gold：a brand new fire truck
生成答案：On 16 August, 2023, the donations helped John's community acquire items such as canned food, toiletries, and clothes to support the local fire station.
Gold leaf：leaf_320

## 补充词面指标

修复后主评测官方类别规则词级 F1 为 8.20%，历史值为 2.9568%。
主评测生成答案平均 175.5 词，未进行事后简答提取。
该 F1 与二元 LLM-J 不是同一指标，不能相互替代。

## 结论边界

这次比较修复了全量源文本进入 Dense Index 的前提。得分变化反映完整索引恢复后的结果，
不能直接证明 Temporal-first 优于 Dense-first。后续需要在同一完整 Memory 上筛查真实 Dense Miss，
再核验 Gold 对应事件边的时间和 Episode 映射。本轮没有实施 Temporal-first。
并发只发生在不同 Conversation 的只读 QA；每个 Conversation 内逐题执行。
外层初始 3 路，随后将 5 个排队 Conversation 提前独立执行，峰值 8 路；提前结果复用，未重复调用。
历史 Judge 汇总的子集字段缺失，本次使用 CSV 与原始逐题标签重新分组，未修改标签或重调模型。
API 响应 Trace 及 Judge 显式重试均保留；SDK 内部重试可能只有 console 日志，
因此 API Token 合计是已记录返回响应的用量，不保证覆盖提供者全部计费。

