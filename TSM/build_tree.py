import multiprocessing as mp
from typing import List
import json
import os
from prompt_dict import Prompts
from utils import sequential_split, Timer
import time
from tqdm import tqdm

# def sequential_merge(chunks:List[str],
#                      tokenizer:AutoTokenizer,
#                      overlap:int)->str:
#     '''
#     Merge the chunks into a single text.
#     '''
#     res = chunks[0]
#     for i in range(1, len(chunks)):
#         ids = tokenizer(chunks[i], return_tensors="pt")["input_ids"][0][overlap:]
#         res += tokenizer.decode(ids)
#     return res

def load_cache_summary(cache_path:str)->List[str]:
    '''
    Load the summary from the cache file.
    The cache file is a json file, name as {book_id}_summary.json.
    keys are:
    chunk_id:{
        "text: str,
        "children": List[str],
        "parent": str,
    }
    '''
    with open(cache_path, "r") as f:
        return json.load(f)

def call_llm_unified(llm, prompt: str) -> str:
    """
    统一的LLM调用接口，支持公开 OpenAI-compatible API 和本地 pipeline。
    """
    if isinstance(llm, dict) and llm.get("type") == "openai_compatible":
        from provider.public_llm_provider import call_llm
        try:
            messages = [{"role": "user", "content": prompt}]
            response = call_llm(
                messages=messages,
                model=llm.get("backend_model", llm["model"]),
                temperature=llm.get("temperature", 0),
                max_tokens=llm.get("max_tokens", 1024),
                base_url=llm.get("base_url"),
                api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"),
                max_retries=llm.get("max_retries", 3),
                timeout=llm.get("timeout", 120),
            )
            if not response:
                print("⚠️ API调用返回None，使用默认摘要")
                return "摘要生成失败"
            return response
        except Exception as e:
            print(f"❌ API调用出错: {e}")
            return "摘要生成失败"

    # Local wrappers remain supported for backwards-compatible research use.
    if hasattr(llm, "get_llm_model_answer"):
        try:
            return llm.get_llm_model_answer(prompt)
        except Exception as e:
            print(f"❌ vLLM调用出错: {e}")
            return "摘要生成失败"

    # Local HuggingFace pipeline.
    try:
        response = llm(prompt)[0]["generated_text"]
        return response[len(prompt):]
    except Exception as e:
        print(f"❌ Pipeline调用出错: {e}")
        return "摘要生成失败"

def summarize_leaf(text:str, llm, language:str)->str:
    '''
    Summarize the text into chunks.
    '''
    if language == "en":
        prompt = Prompts["summarize_details"].format(content=text)
    else:
        prompt = Prompts["summarize_details_zh"].format(content=text)

    return call_llm_unified(llm, prompt)

def summarize_summary(text:str, llm, language:str)->str:
    '''
    Summarize the summary into chunks.
    '''
    if language == "en":
        prompt = Prompts["summarize_summary"].format(summary=text)
    else:
        prompt = Prompts["summarize_summary_zh"].format(summary=text)

    return call_llm_unified(llm, prompt)

def parallel_summarize_chunks(args):
    """
    并行摘要生成的工作函数
    args: (chunks, llm, language, summary_type, chunk_index)
    """
    chunks, llm, language, summary_type, chunk_index = args

    try:
        if summary_type == "leaf":
            summary = summarize_leaf(chunks, llm, language)
        else:  # summary
            summary = summarize_summary(chunks, llm, language)

        return {
            'success': True,
            'summary': summary,
            'chunk_index': chunk_index,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'summary': "摘要生成失败",
            'chunk_index': chunk_index,
            'error': str(e)
        }

def build_tree(text_chunks:List[str], llm, cache_folder:str, merge_num:int, language:str, parallel_api_calls:int = 4, original_text:List = None):
    '''
    Build the tree from the text with parallel API calls support.

    Args:
        text_chunks: 文本块列表（用于构建树和生成摘要）
        llm: LLM配置
        cache_folder: 缓存文件夹路径
        merge_num: 每个摘要合并的文本块数量
        language: 语言类型
        parallel_api_calls: 并行API调用数，默认为4
        original_text: 原始文本（包含full_text和user_only_text的字典列表），用于保存完整的turn到leaf节点
    '''
    build_start_time = time.time()
    if os.path.exists(os.path.join(cache_folder, "tree.json")):
        return load_cache_summary(os.path.join(cache_folder, "tree.json")), -1

    cache = {}

    # leaf ids in the format of "leaf_{i}"
    # due to the leaf has no children, it is set as None.
    for i in range(len(text_chunks)):
        # 如果提供了original_text且是字典格式，使用full_text保存完整的turn
        leaf_text = text_chunks[i]
        if original_text and i < len(original_text):
            original_chunk = original_text[i]
            if isinstance(original_chunk, dict) and "full_text" in original_chunk:
                leaf_text = original_chunk["full_text"]

        cache["leaf_{}".format(i)] = {
            "text": leaf_text,
            "children": None,
            "parent": None,
        }

    # do summarize for the first level with parallel API calls
    print(f"开始第一层摘要生成（并行化，进程数：{parallel_api_calls}）...")
    summary_tasks = []
    summary_id_count = 0

    # 准备并行任务
    for i in range(0, len(text_chunks), merge_num):
        merged_chunks = text_chunks[i:i+merge_num]
        summary_tasks.append((merged_chunks, llm, language, "leaf", summary_id_count))
        summary_id_count += 1

    # 检查是否需要并行化（仅当使用API模式时启用并行化）
    should_parallelize = isinstance(llm, dict) and llm.get("type") == "openai_compatible" and len(summary_tasks) > 1

    if should_parallelize and parallel_api_calls > 1:
        # 使用进程池并行执行摘要生成
        print(f"启用并行API调用，进程数：{parallel_api_calls}")
        with mp.Pool(processes=min(parallel_api_calls, len(summary_tasks))) as pool:
            results = list(tqdm(
                pool.imap(parallel_summarize_chunks, summary_tasks),
                total=len(summary_tasks),
                desc="并行摘要生成",
                ncols=80
            ))

        # 处理并行结果
        for result in results:
            if result['success']:
                chunk_index = result['chunk_index']
                i = chunk_index * merge_num
                cache["summary_0_{}".format(chunk_index)] = {
                    "text": result['summary'],
                    "children": [f"leaf_{j}" for j in range(i, min(i+merge_num, len(text_chunks)))],
                    "parent": [],
                }
                for j in range(i, min(i+merge_num, len(text_chunks))):
                    cache["leaf_{}".format(j)]["parent"] = "summary_0_{}".format(chunk_index)
            else:
                print(f"❌ 块 {result['chunk_index']} 摘要生成失败: {result['error']}")
    else:
        # 串行执行摘要生成
        print("使用串行API调用")
        summary_id_count = 0
        for i in tqdm(range(0, len(text_chunks), merge_num), desc="摘要生成"):
            merged_chunks = text_chunks[i:i+merge_num]
            summary = summarize_leaf(merged_chunks, llm, language)
            cache["summary_0_{}".format(summary_id_count)] = {
                "text": summary,
                "children": [f"leaf_{j}" for j in range(i, min(i+merge_num, len(text_chunks)))],
                "parent": [],
            }
            summary_id_count += 1
            for j in range(i, min(i+merge_num, len(text_chunks))):
                cache["leaf_{}".format(j)]["parent"]="summary_0_{}".format(summary_id_count-1)

    to_summarize = [f"summary_0_{i}" for i in range(summary_id_count)]
    to_summarize = [cache[i]["text"] for i in to_summarize]
    level = 1
    max_levels = 10  # 添加最大层级限制，防止无限循环

    # do summarize for the rest levels with parallel API calls for higher levels
    if level < 3:  # 仅在前几层使用并行化，因为高层级节点较少
        print(f"第{level}层及以上使用并行API调用")

    while len(to_summarize) > 1.2 * merge_num and level < max_levels:
        new_summary_id_count = 0

        if level < 3 and isinstance(llm, dict) and llm.get("type") == "openai_compatible" and parallel_api_calls > 1:
            # 高层级使用并行化
            summary_tasks = []
            for i in range(0, len(to_summarize), merge_num):
                merged_chunks = "\n\n".join(to_summarize[i:i+merge_num])
                summary_tasks.append((merged_chunks, llm, language, "summary", new_summary_id_count))
                new_summary_id_count += 1

            if len(summary_tasks) > 1:
                with mp.Pool(processes=min(parallel_api_calls, len(summary_tasks))) as pool:
                    results = list(tqdm(
                        pool.imap(parallel_summarize_chunks, summary_tasks),
                        total=len(summary_tasks),
                        desc=f"第{level}层并行摘要生成",
                        ncols=80
                    ))

                # 处理并行结果
                for result in results:
                    if result['success']:
                        chunk_index = result['chunk_index']
                        i = chunk_index * merge_num
                        cache["summary_{}_{}".format(level, chunk_index)] = {
                            "text": result['summary'],
                            "children": [f"summary_{level-1}_{j}" for j in range(i, min(i+merge_num, len(to_summarize)))],
                            "parent": [],
                        }
                        for j in range(i, min(i+merge_num, len(to_summarize))):
                            cache["summary_{}_{}".format(level-1, j)]["parent"] = f"summary_{level}_{chunk_index}"
                    else:
                        print(f"❌ 层级 {level} 块 {result['chunk_index']} 摘要生成失败: {result['error']}")
            else:
                # 单个任务，直接执行
                merged_chunks = "\n\n".join(to_summarize[0:merge_num])
                summary = summarize_summary(merged_chunks, llm, language)
                cache["summary_{}_{}".format(level, 0)] = {
                    "text": summary,
                    "children": [f"summary_{level-1}_{j}" for j in range(0, min(merge_num, len(to_summarize)))],
                    "parent": [],
                }
        else:
            # 低层级或串行模式
            for i in tqdm(range(0, len(to_summarize), merge_num), desc=f"第{level}层摘要生成"):
                # for summary, there is no overlap.
                merged_chunks = "\n\n".join(to_summarize[i:i+merge_num])
                # for summary, using different prompt.
                summary = summarize_summary(merged_chunks, llm, language)
                # key format: summary_{level}_{i}
                cache["summary_{}_{}".format(level, new_summary_id_count)] = {
                    "text": summary,
                    "children": [f"summary_{level-1}_{j}" for j in range(i, min(i+merge_num, len(to_summarize)))],
                    "parent": [],
                }
                new_summary_id_count += 1
                for j in range(i, min(i+merge_num, len(to_summarize))):
                    cache["summary_{}_{}".format(level-1, j)]["parent"] = f"summary_{level}_{new_summary_id_count-1}"

        # update the to_summarize list.
        to_summarize = [f"summary_{level}_{i}" for i in range(new_summary_id_count)]
        level += 1

    if level >= max_levels:
        print(f"警告: 达到最大层级限制 {max_levels}，停止构建")
    else:
        print(f"树构建完成，共构建 {level-1} 层")
    # save the cache.
    with open(os.path.join(cache_folder, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    build_end_time = time.time()
    return_time = build_end_time - build_start_time
    return cache, return_time

if __name__ == "__main__":
    pass
