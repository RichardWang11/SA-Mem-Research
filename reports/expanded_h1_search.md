# 扩展 H1 候选搜索

保持官方算法和已有运行原状；本次只读数据库和缓存，没有调用生成/Embedding模型。

检索范围：全部321题；269题有非空Parser时间范围。对10个Conversation的图谱来源事实做只读扫描，按现有Parser范围筛选，检查Episode名称和原文是否与现存leaf同时一致，再查该leaf是否位于原Dense Top20之外。

得到48道题的168个Query-leaf候选，其中155个候选在时间先筛后的现存库语义顺序中位于Top20内。数量仅代表符合时间条件并可从Dense20之外补入，不代表正确证据，更不代表答案救回。

扩展Gold前后上下文也产生13道候选，已将相邻却无关的对话排除出确认案例。答案词干覆盖检索用于找同义候选，不能替代语义判断。

本轮没有新增可以确认的H1支持案例。之前conv-48_105仍是最强的“直接答案补充证据可恢复”候选；conv-26_11仍是明确Gold-ID Dense Miss但时间图谱无法确认恢复的案例。

排除例：conv-26_31问露营日期，leaf_33从Dense31升至时间候选8，但内容是职业咨询，不能算正确证据。conv-48_102问机器人项目，候选leaf_34说净水器，项目不符。conv-48_26问书名Avalanche，候选为家人纪念、瑜伽等，不能提供答案。conv-50_73问Tokyo音乐节，候选讨论Boston等其他地点，无直接答案支持。

现有Dense库每个Conversation只有23至44个leaf，且集中于历史开头；因此可供发现的真实库内Dense Miss本来就受限。不能从本轮没有新增确认案例推出H1不存在。也不能把168个时间候选计成168条Gold恢复。

原始候选保存在 ../results/cases/all_temporal_outside20_candidates.json.gz，汇总在 ../results/full/expanded_search_summary.json，可重跑脚本为 ../scripts/search_temporal_candidates.py（需要当前Neo4j环境变量）。
