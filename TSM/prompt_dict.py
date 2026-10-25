Prompts = {

    "summarize_details_v0":\
"""You are a helpful assistant that summarizes the details of a chat history. You will be given a round of a conversation. You need to summarize given content. The summary should include the main characters, the main plot and some other details. You need to return the summary in a concise manner without any additional fictive information. The length of the summary should be about 500 tokens.
Here is the content:
Content: {content}
Now, please summarize the content.
Summary: """,

"summarize_details":\
"""You are a professional conversation summarizer. You will be given a round of conversation that you need to summarize comprehensively yet concisely.

Please organize your summary using the following structure:
1. Main participants and their roles (2-3 sentences)
2. Core topic and purpose of the conversation (1-2 sentences)
3. Key discussion points and conclusions reached (3-5 points)
4. Important details (if necessary)

Requirements for the summary:
- Remain objective and include only information present in the original content
- Highlight the most important information while omitting minor details
- Keep the summary length between 400-500 words
- Use clear and concise language

Here is the content:
Content: {content}

Now, please summarize the content.
Summary: """,

    "summarize_summary_v0":\
"""You are a helpful assistant that further summarizes the summaries of chat history. You will be given a series of summaries of rounds of conversations. You need to summarize the summaries in a concise manner. The length of the summary should be about 500 tokens.
Here is the summaries:
Summary: {summary}
Now, please summarize the summary based on the question.
Summary: """,

    "summarize_summary":\
"""You are an expert meta-summarizer tasked with condensing multiple conversation summaries into a cohesive overview. You will be given a series of summaries from different conversation rounds.

Please create a comprehensive meta-summary that:
1. Identifies recurring themes and patterns across all summaries
2. Highlights the progression of topics or ideas throughout the conversations
3. Captures the most significant points and conclusions
4. Preserves critical context from the original summaries

Requirements for your meta-summary:
- Maintain a logical flow that connects related information across summaries
- Eliminate redundancies while preserving important details
- Prioritize information based on relevance and importance
- Keep the meta-summary length between 400-500 words
- Use clear, concise language with appropriate transitions

Here are the summaries to consolidate:
Summary: {summary}

Now, please create a meta-summary that effectively synthesizes these summaries.
Meta-Summary: """,


    "QA_prompt_options":\
"""You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be an option among "A", "B", "C", and "D" that supported by the given evidences and matches the question. You should not assume any information beyond the evidence. You should only output the option. The format of the Evidence is keyEntity1_keyEntity2: Related chunks, which means the related chunks contain the information of the relationship between keyEntity1 and keyEntity2, where keyEntity1 and keyEntity2 are the entities in the question.

Question: {question}
Evidence: {evidence}

Answer: """,


    "QA_prompt_answer":\
"""You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be a short sentence that supported by the given edidences and matches the requirements of the question. You should not assume any information beyond the evidence. You should only output the answer. The format of the Evidence is keyEntity1_keyEntity2: Related chunks, which means the related chunks contain the information of the relationship between keyEntity1 and keyEntity2, where keyEntity1 and keyEntity2 are the entities in the question.

Question: {question}
Evidence: {evidence}

Answer: """,

    "QA_prompt_answer_system":\
"""
You are an expert at answering questions based on the provided memories. Your task is to provide accurate and concise answers to the questions by leveraging the information given in the memories.

Guidelines:
- Extract relevant information from the memories based on the question.
- If no relevant information is found, make sure you don't say no information is found. Instead, accept the question and provide a general response.
- Ensure that the answers are clear, concise, and directly address the question.

Here are the details of the task:
""",

    "QA_prompt_answer_improved":\
""" Answer the question based on query and memories.

User Memories:{evidence}
Query:{question}

Answer:""",

    "QA_prompt_answer_improved_v2":\
""" Answer the question based on query and memories.

User Memories:{evidence}
Query:{question}

FACTS provide relevant context for the current conversation.

# Key facts with their date ranges
# format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

Answer:""",


"zep_locomo_prompt":\
"""You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.

# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
8. Write a short answer in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Context:

{context}

Question: {question}
Answer:""",

"zep_locomo_example": \
'''Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct Answer: March 15, 2023
Explanation:
Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.
''',

    "QA_prompt_mirix_improved":\
'''You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),  calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories. Do not confuse character names mentioned in memories with the actual users who created those memories.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:

{evidence}

Question Date: {question_date}
Question: {question}
Answer:''',

    "QA_prompt_answer_zh":\
"""你是一位擅长回答问题的助手。你将收到一个问题和一些证据，请根据证据回答问题。答案应该是一个简短的句子，支持给定的证据，并符合问题的要求。不要假设任何证据之外的信息。你只需要输出答案。证据的格式是keyEntity1_keyEntity2: Related chunks，表示相关片段包含keyEntity1和keyEntity2之间的关系信息，其中keyEntity1和keyEntity2是问题中的实体。

问题：{question}
证据：{evidence}

答案：""",

    "QA_prompt_simple":\
"""Question time:{question_date} and question:{question}\nPlease answer the question based on the following memories: {evidence}""",

    "summarize_details_zh":\
"""你是一位擅长总结文本内容的助手。你将收到一部分文本内容，需要对其进行总结。总结内容应包括：主要实体信息、实体之间的关系以及规章制度的内容。请以简洁的方式进行总结，不要添加任何虚构信息。总结的长度应控制在大约1000个token左右。

以下是文本内容：
内容：{content}

现在，请对以上内容进行总结。
总结：""",

    "summarize_summary_zh":\
"""你是一位擅长总结文本内容的助手。你将收到一系列文本内容的总结，需要对这些总结进行进一步的总结。总结内容应包括：主要实体信息、实体之间的关系以及规章制度的内容。请以简洁的方式进行总结，不要添加任何虚构信息。总结的长度应控制在大约1000个token左右。

以下是总结内容：
总结：{summary}

现在，请根据问题对以上总结进行进一步的总结。
总结：""",

    "temporal_extraction_prompt_zh":\
"""你是一个专门提取图中边关系时间信息的AI助手。你只关注与边关系描述直接相关的时间信息。

<上下文>
{context}
</上下文>

<参考时间戳>
{reference_timestamp}
</参考时间戳>

<关系事实>
{entity_pair}
</关系事实>

重要提示：只有当时间信息是提供的关系事实的一部分时才提取时间信息。否则忽略提到的时间。
如果只提到相对时间（例如10年前，2分钟前），请根据提供的参考时间戳尽力确定具体日期。
如果关系不是持续性的，但你仍能确定日期，只设置 valid_at。

定义：
- valid_at: 边关系描述的事实变为真实或建立时的日期和时间。
- invalid_at: 边关系描述的事实停止为真实或结束时的日期和时间。

任务：
分析文本并确定是否有属于边关系事实的日期。只有当日期明确关系到关系本身的形成或改变时才设置日期。

指导原则：
1. 使用 ISO 8601 格式 (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) 表示日期时间。
2. 使用参考时间戳作为确定 valid_at 和 invalid_at 日期的当前时间。
3. 如果事实用现在时态书写，使用参考时间戳作为 valid_at 日期。
4. 如果没有找到建立或改变关系的时间信息，将字段留空。
5. 不要从相关事件推断日期。只使用直接陈述建立或改变关系的日期。
6. 对于直接关系到关系的相对时间提及，根据参考时间戳计算实际日期时间。
7. 如果只提到日期而没有具体时间，使用该日期的 00:00:00（午夜）。
8. 如果只提到年份，使用该年1月1日的 00:00:00。
9. 始终包含时区偏移（如果没有提到特定时区，使用 Z 表示 UTC）。
10. 讨论某事不再真实的事实应该有 valid_at，表示被否定的事实变为真实的时间。

请以JSON格式返回结果：
{
    "valid_at": "YYYY-MM-DDTHH:MM:SS.SSSSSSZ 或 null",
    "invalid_at": "YYYY-MM-DDTHH:MM:SS.SSSSSSZ 或 null"
}""",

    "temporal_extraction_prompt_en":\
"""You are an AI assistant that extracts datetime information for graph edges, focusing only on dates directly related to the establishment or change of the relationship described in the entity pair.

<CONTEXT>
{context}
</CONTEXT>

<REFERENCE TIMESTAMP>
{reference_timestamp}
</REFERENCE TIMESTAMP>

<ENTITY PAIR>
{entity_pair}
</ENTITY PAIR>

IMPORTANT: Only extract time information if it is part of the provided entity pair. Otherwise ignore the time mentioned. Make sure to do your best to determine the dates if only the relative time is mentioned. (eg 10 years ago, 2 mins ago) based on the provided reference timestamp
If the relationship is not of spanning nature, but you are still able to determine the dates, set the valid_at only.
Definitions:
- valid_at: The date and time when the relationship described by the entity pair became true or was established.
- invalid_at: The date and time when the relationship described by the entity pair stopped being true or ended.

Task:
Analyze the conversation and determine if there are dates that are part of the entity pair. Only set dates if they explicitly relate to the formation or alteration of the relationship itself.

Guidelines:
1. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) for datetimes.
2. Use the reference timestamp as the current time when determining the valid_at and invalid_at dates.
3. If the entity pair is written in the present tense, use the Reference Timestamp for the valid_at date
4. If no temporal information is found that establishes or changes the relationship, leave the fields as null.
5. Do not infer dates from related events. Only use dates that are directly stated to establish or change the relationship.
6. For relative time mentions directly related to the relationship, calculate the actual datetime based on the reference timestamp.
7. If only a date is mentioned without a specific time, use 00:00:00 (midnight) for that date.
8. If only year is mentioned, use January 1st of that year at 00:00:00.
9. Always include the time zone offset (use Z for UTC if no specific time zone is mentioned).
10. A entity pair discussing that something is no longer true should have a valid_at according to when the negated entity pair became true.

Please return the result in a list of JSON format:
[
    {{
        "entity1": "entity1",
        "entity2": "entity2",
        "relationship": "relationship",
        "valid_at": "YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null",
        "invalid_at": "YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null"
    }}
]""",

}