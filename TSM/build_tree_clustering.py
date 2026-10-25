# -*- coding: utf-8 -*-
"""
基于实体聚类的树构建模块
使用GMM聚类对实体进行语义分组，生成用户persona作为树的第一层
"""

import json
import os
import time
import logging
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入聚类工具
from sklearn.mixture import GaussianMixture
try:
    import umap
except ImportError:  # Checked when clustering is actually requested.
    umap = None

# 导入现有模块
from utils import Timer
from prompt_dict import Prompts

logger = logging.getLogger(__name__)

# ===================== 统一的LLM调用函数（支持vLLM和MT API） =====================

def _call_llm_unified(messages: List[dict], llm_config: dict, op_type: str | None = None) -> str:
    """Call the configured public OpenAI-compatible backend."""
    try:
        from provider.public_llm_provider import call_llm
        return call_llm(
            messages=messages,
            model=llm_config.get("backend_model", llm_config["model"]),
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 2000),
            base_url=llm_config.get("base_url"),
            api_key_env=llm_config.get("api_key_env", "OPENAI_API_KEY"),
            max_retries=llm_config.get("max_retries", 3),
            timeout=llm_config.get("timeout", 120),
            op_type=op_type,
        ) or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

# ===================== GMM聚类功能 =====================

def perform_clustering(embeddings: np.ndarray, dim: int = 2, threshold: float = 0.1, verbose: bool = True) -> List[np.ndarray]:
    """
    执行GMM聚类（简化版，从_cluster_utils.py改编）

    Args:
        embeddings: 实体嵌入矩阵
        dim: UMAP降维目标维度
        threshold: 软聚类阈值
        verbose: 是否打印详细信息

    Returns:
        每个实体的聚类标签列表
    """
    if len(embeddings) < 2:
        # 如果实体太少，全部归为一类
        return [np.array([0]) for _ in range(len(embeddings))]
    if umap is None:
        raise ImportError("umap-learn is required for entity clustering")

    # UMAP降维
    if verbose:
        logger.info(f"UMAP降维: {embeddings.shape} -> ({len(embeddings)}, {dim})")

    n_neighbors = min(15, len(embeddings) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=min(dim, len(embeddings) - 1), metric='cosine')
    reduced_embeddings = reducer.fit_transform(embeddings)

    # 确定最优聚类数（使用BIC）
    max_clusters = min(50, len(embeddings))
    # 设置最小聚类数：如果实体数较多，至少分成多个聚类
    # 对于大量实体（>100），至少分成5-10个聚类；对于中等数量（>50），至少3-5个；对于少量（>20），至少2个
    if len(embeddings) > 100:
        min_clusters = max(5, min(10, len(embeddings) // 20))
    elif len(embeddings) > 50:
        min_clusters = max(3, min(5, len(embeddings) // 15))
    elif len(embeddings) > 20:
        min_clusters = 2
    else:
        min_clusters = 1

    n_clusters_range = range(min_clusters, min(max_clusters + 1, len(embeddings) + 1))
    bics = []

    if verbose:
        logger.info(f"聚类数搜索范围: {min_clusters} 到 {min(max_clusters, len(embeddings))} (实体数: {len(embeddings)})")

    for n in n_clusters_range:
        try:
            gm = GaussianMixture(n_components=n, random_state=42, n_init=3)
            gm.fit(reduced_embeddings)
            bic = gm.bic(reduced_embeddings)
            bics.append(bic)

            if verbose and len(bics) <= 5:  # 只打印前几个BIC值
                logger.debug(f"  n_clusters={n}: BIC={bic:.2f}")

            # 改进的早停条件：如果BIC连续增加2次，停止搜索
            if len(bics) >= 3 and bics[-1] > bics[-2] and bics[-2] > bics[-3]:
                if verbose:
                    logger.info(f"BIC连续增加，提前停止搜索（n={n}）")
                break
        except Exception as e:
            logger.warning(f"GMM聚类失败 (n={n}): {e}")
            # 如果失败，使用之前成功的最大n值
            if len(bics) > 0:
                break
            continue

    if len(bics) == 0:
        logger.warning("BIC计算失败，使用最小聚类数")
        optimal_clusters = min_clusters
    else:
        optimal_clusters = n_clusters_range[np.argmin(bics)]
        # 确保至少是最小聚类数
        optimal_clusters = max(optimal_clusters, min_clusters)

    if verbose:
        logger.info(f"最优聚类数: {optimal_clusters} (BIC范围: {min(bics):.2f} - {max(bics):.2f})")

    # 执行GMM聚类
    gm = GaussianMixture(n_components=optimal_clusters, random_state=42, n_init=5)
    gm.fit(reduced_embeddings)

    # 获取软聚类概率
    probs = gm.predict_proba(reduced_embeddings)

    # 根据阈值确定每个实体所属的聚类
    labels = []
    for prob in probs:
        cluster_ids = np.where(prob > threshold)[0]
        if len(cluster_ids) == 0:
            # 如果没有聚类超过阈值，选择概率最高的
            cluster_ids = np.array([np.argmax(prob)])
        labels.append(cluster_ids)

    return labels

# ===================== Persona生成功能 =====================

def generate_persona_summary(
    texts: List[str],
    entities: List[str],
    llm_config: dict,
    language: str = "zh"
) -> str:
    """
    生成用户persona摘要（支持超长文本的智能截断）

    Args:
        texts: 文本块列表
        entities: 实体名称列表
        llm_config: LLM配置
        language: 语言设置

    Returns:
        生成的persona摘要
    """
    # 超长预警阈值
    WARNING_THRESHOLD = 10000  # 字符数
    CRITICAL_THRESHOLD = 50000  # 字符数

    # vLLM上下文限制：65536 tokens，保守估计约200000字符（50000 tokens用于输入，留出空间给prompt和生成）
    # 1 token ≈ 4字符，65536 tokens ≈ 262144字符，但需要留出空间
    MAX_INPUT_CHARS = 200000  # 最大输入字符数（约50000 tokens）

    # 智能截断：如果文本太长，保留前面的文本块直到达到限制
    total_text = '\n\n'.join(texts)
    total_length = len(total_text)

    # 如果超过最大限制，进行截断
    if total_length > MAX_INPUT_CHARS:
        logger.warning(f"⚠️ 文本超长（{total_length}字符），超过vLLM限制，进行智能截断...")
        logger.warning(f"   原始文本块数: {len(texts)}")

        # 逐步添加文本块，直到接近限制
        truncated_texts = []
        current_length = 0
        for text in texts:
            text_with_separator = '\n\n' + text if truncated_texts else text
            if current_length + len(text_with_separator) > MAX_INPUT_CHARS:
                # 如果添加这个文本块会超过限制，尝试截断这个文本块本身
                remaining_chars = MAX_INPUT_CHARS - current_length - len('\n\n')
                if remaining_chars > 1000:  # 至少保留1000字符才有意义
                    truncated_text = text[:remaining_chars] + "\n\n[文本已截断...]"
                    truncated_texts.append(truncated_text)
                    current_length += len('\n\n' + truncated_text)
                break
            truncated_texts.append(text)
            current_length += len(text_with_separator)

        total_text = '\n\n'.join(truncated_texts)
        total_length = len(total_text)
        logger.warning(f"   截断后文本块数: {len(truncated_texts)}")
        logger.warning(f"   截断后总长度: {total_length} 字符")
    else:
        # 计算总文本长度（未截断的情况）
        total_length = len(total_text)

    # 打印预警信息
    if total_length > CRITICAL_THRESHOLD:
        logger.warning(f"⚠️ 严重警告: 聚类文本超长！总长度: {total_length} 字符 (>{CRITICAL_THRESHOLD})")
        logger.warning(f"   涉及实体数: {len(entities)}")
        logger.warning(f"   文本块数: {len(texts)}")
    elif total_length > WARNING_THRESHOLD:
        logger.info(f"⚠️ 警告: 聚类文本较长！总长度: {total_length} 字符")
        logger.info(f"   涉及实体数: {len(entities)}")
        logger.info(f"   文本块数: {len(texts)}")
    else:
        logger.info(f"ℹ️ 正常: 聚类文本长度: {total_length} 字符")

    # 构建prompt
    if language == "zh":
        prompt = f"""
基于以下文本内容和涉及的实体，生成一个详细的用户画像(persona)摘要：

涉及的主要实体（共{len(entities)}个）：
{', '.join(entities)}

相关文本片段（共{len(texts)}段，总长度{total_length}字符）：
{total_text}

请生成一个全面的用户画像，包括但不限于：
1. 用户的主要特征、性格和行为模式
2. 用户关注的核心话题和兴趣领域
3. 用户与这些实体的具体关系
4. 用户的动机、目标和价值观
5. 用户的知识背景和专业领域
6. 任何其他从文本中推断出的重要特征

注意：请确保摘要涵盖所有重要信息，不要遗漏关键细节。
"""
    else:
        prompt = f"""
Based on the following text content and entities, generate a comprehensive user persona summary:

Main entities involved ({len(entities)} total):
{', '.join(entities)}

Related text snippets ({len(texts)} segments, {total_length} characters total):
{total_text}

Please generate a comprehensive user persona including but not limited to:
1. User's main characteristics, personality, and behavior patterns
2. Core topics and areas of interest
3. User's specific relationships with these entities
4. User's motivations, goals, and values
5. User's knowledge background and professional domains
6. Any other important characteristics inferred from the text

Note: Ensure the summary covers all important information without missing key details.
"""

    # 调用LLM生成摘要（使用统一函数，支持vLLM和MT API）
    messages = [{"role": "user", "content": prompt}]
    summary = _call_llm_unified(messages, llm_config)

    if not summary:
        logger.error("Persona摘要生成失败")
        return "摘要生成失败"

    return summary

# ===================== 简化树构建备用方案 =====================

def _build_simple_tree(text_chunks: List[str], llm_config: dict, cache_folder: str,
                      merge_num: int, language: str) -> Tuple[dict, float, dict]:
    """
    简化的树构建方法，作为最后的备用方案

    Args:
        text_chunks: 文本块列表
        llm_config: LLM配置
        cache_folder: 缓存文件夹
        merge_num: 合并数量
        language: 语言

    Returns:
        (树结构, 耗时, 空索引)
    """
    start_time = time.time()
    logger.info("使用简化树构建方法")

    cache = {}

    # 创建叶子节点
    for i, chunk in enumerate(text_chunks):
        leaf_id = f"leaf_{i}"
        cache[leaf_id] = {
            "text": chunk,
            "children": None,
            "parent": None
        }

    # 创建一个根节点，包含所有叶子
    root_id = "summary_0_0"
    leaf_ids = [f"leaf_{i}" for i in range(len(text_chunks))]

    cache[root_id] = {
        "text": "简化树构建 - 所有文本块的集合",
        "children": leaf_ids,
        "parent": []
    }

    # 设置父节点关系
    for leaf_id in leaf_ids:
        cache[leaf_id]["parent"] = root_id

    build_time = time.time() - start_time
    logger.info(f"简化树构建完成，共 {len(cache)} 个节点，耗时 {build_time:.2f} 秒")

    return cache, build_time, {}

# ===================== 主要功能函数 =====================

def get_entities_with_embeddings(graphiti_driver, group_id: str) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    从Neo4j读取所有实体及其嵌入向量

    Args:
        graphiti_driver: Neo4j驱动
        group_id: 组ID

    Returns:
        (实体列表, 嵌入向量列表)
    """
    query = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id
    RETURN n.uuid as uuid,
           n.name as name,
           n.summary as summary,
           n.entity_type as type,
           n.name_embedding as embedding
    """

    # Neo4j 驱动是异步的，需要特殊处理
    import asyncio
    try:
        # 尝试同步调用
        result = graphiti_driver.execute_query(query, group_id=group_id)
        # 如果返回的是协程，需要等待
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)
    except Exception as e:
        logger.error(f"Neo4j查询失败: {e}")
        return [], []

    # 添加调试信息
    logger.info(f"Neo4j查询结果类型: {type(result)}")
    if hasattr(result, 'records'):
        logger.info(f"查询返回 {len(result.records)} 条记录")
    else:
        logger.warning(f"查询结果没有records属性: {result}")

    entities = []
    embeddings = []

    if hasattr(result, 'records'):
        for record in result.records:
            # 提取实体信息
            entity = {
                'uuid': record.get('uuid'),
                'name': record.get('name'),
                'summary': record.get('summary'),
                'type': record.get('type')
            }
            entities.append(entity)

            # 提取嵌入向量
            embedding = record.get('embedding')
            if embedding is not None:
                embeddings.append(np.array(embedding))
            else:
                # 如果没有嵌入，使用随机向量（应该不会发生）
                logger.warning(f"实体 {entity['name']} 缺少嵌入向量")
                embeddings.append(np.random.randn(1536))

    return entities, embeddings

def get_entity_chunk_mapping(graphiti_driver, group_id: str) -> Dict[str, List[Dict]]:
    """
    获取实体与文本块(episodes)的映射关系

    Args:
        graphiti_driver: Neo4j驱动
        group_id: 组ID

    Returns:
        entity_name -> chunk信息列表的映射
    """
    query = """
    MATCH (e:Episodic)-[:MENTIONS]->(n:Entity)
    WHERE n.group_id = $group_id
    RETURN n.name as entity_name,
           n.uuid as entity_uuid,
           e.name as episode_name,
           e.content as chunk_content,
           e.uuid as episode_uuid
    """

    # Neo4j 驱动是异步的，需要特殊处理
    import asyncio
    try:
        # 尝试同步调用
        result = graphiti_driver.execute_query(query, group_id=group_id)
        # 如果返回的是协程，需要等待
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)
    except Exception as e:
        logger.error(f"Neo4j查询失败: {e}")
        return [], []

    # 构建映射：entity -> chunks
    entity_to_chunks = defaultdict(list)

    if hasattr(result, 'records'):
        for record in result.records:
            entity_name = record.get('entity_name')
            chunk_info = {
                'episode_name': record.get('episode_name'),
                'content': record.get('chunk_content'),
                'uuid': record.get('episode_uuid')
            }
            entity_to_chunks[entity_name].append(chunk_info)

    return dict(entity_to_chunks)

def generate_entity_cluster_summary(
    entities: List[Dict],
    cluster_id: int,
    llm_config: dict,
    language: str = "zh"
) -> str:
    """
    基于聚类内实体的summary生成实体层面的聚类摘要

    Args:
        entities: 聚类中的实体列表（包含 name 与 summary 字段）
        cluster_id: 聚类ID
        llm_config: LLM配置
        language: 语言设置

    Returns:
        生成的实体聚类摘要
    """
    # 收集实体summary
    lines = []
    entity_names = []
    for ent in entities:
        name = ent.get("name", "")
        summary = (ent.get("summary") or "").strip()
        if summary:
            lines.append(f"实体: {name}\n摘要: {summary}")
            entity_names.append(name)

    if not lines:
        logger.warning(f"聚类 {cluster_id} 实体缺少有效summary，生成占位摘要")
        return f"聚类 {cluster_id} 的实体总结信息不足（共{len(entities)}个实体）"

    # 组织提示词
    lines_text = "\n\n".join(lines)
    if language == "zh":
        prompt = f"""
请基于以下实体的summary，为该聚类生成一个实体层面的综合摘要：

{lines_text}

请提炼：
1. 该聚类实体的共同主题/领域
2. 关键属性与差异点
3. 这些实体之间的潜在关联或层次
4. 值得关注的模式与异常

要求：简洁、结构化、避免重复原文。
"""
    else:
        prompt = f"""
Based on the following entity summaries, produce a consolidated entity-level cluster summary:

{lines_text}

Summarize:
1) Common themes/domains
2) Key attributes and differences
3) Potential relationships/hierarchy among entities
4) Notable patterns/anomalies

Be concise and structured, avoid verbatim repetition.
"""

    # 调用LLM生成摘要（使用统一函数，支持vLLM和MT API）
    messages = [{"role": "user", "content": prompt}]
    summary = _call_llm_unified(messages, llm_config)

    if not summary:
        logger.error(f"实体聚类摘要生成失败: cluster {cluster_id}")
        return f"聚类 {cluster_id} 实体摘要生成失败"

    return summary

def build_tree_with_clustering(
    text_chunks: List[str],
    llm_config: dict,
    cache_folder: str,
    graphiti_driver,
    group_id: str,
    language: str = "zh",
    cluster_threshold: float = 0.1,
    merge_num: int = 5,
    index: dict = None,
    text_edge_mapping: dict = None,
    original_text: List = None  # 原始文本（包含full_text和user_only_text的字典列表）
) -> Tuple[dict, float, dict]:
    """
    基于实体聚类构建层次化树结构

    Args:
        text_chunks: 原始文本块列表
        llm_config: LLM配置
        cache_folder: 缓存文件夹路径
        graphiti_driver: Neo4j驱动
        group_id: 图数据的group_id
        language: 语言设置
        cluster_threshold: 聚类阈值
        merge_num: 合并数量
        index: 实体索引（可选）
        text_edge_mapping: 文本边映射（可选）

    Returns:
        (树结构字典, 耗时, 实体索引映射)
    """
    start_time = time.time()

    # 检查缓存
    tree_path = os.path.join(cache_folder, "tree.json")
    index_path = os.path.join(cache_folder, "cluster_index.json")

    if os.path.exists(tree_path):
        logger.info("从缓存加载树结构")
        with open(tree_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)

        # 尝试加载聚类索引缓存
        cluster_index = {}
        if os.path.exists(index_path):
            logger.info("从缓存加载聚类索引")
            with open(index_path, 'r', encoding='utf-8') as f:
                cluster_index = json.load(f)
        else:
            logger.warning("未找到聚类索引缓存，可能影响检索性能")
            # 如果传入了原始索引，使用它作为备用
            if index:
                cluster_index = index

        return cache, -1, cluster_index

    # 1. 从Neo4j读取实体和嵌入
    logger.info("从Neo4j读取实体和嵌入...")
    entities, entity_embeddings = get_entities_with_embeddings(graphiti_driver, group_id)
    logger.info(f"读取到 {len(entities)} 个实体")

    if len(entities) == 0:
        logger.warning("未找到实体，使用传统树构建方法")
        # 回退到传统方法
        try:
            import importlib.util
            import sys

            # 动态导入原始build_tree模块，避免循环导入
            spec = importlib.util.find_spec("build_tree")
            if spec is not None and spec.origin != __file__:
                build_tree_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(build_tree_module)
                return build_tree_module.build_tree(text_chunks, llm_config, cache_folder, merge_num, language)
            else:
                logger.error("无法找到原始build_tree模块，使用简化树构建")
                return _build_simple_tree(text_chunks, llm_config, cache_folder, merge_num, language)
        except Exception as e:
            logger.error(f"回退到传统方法失败: {e}，使用简化树构建")
            return _build_simple_tree(text_chunks, llm_config, cache_folder, merge_num, language)

    # 获取实体到文本块的映射
    entity_to_chunks = get_entity_chunk_mapping(graphiti_driver, group_id)

    # 2. 执行GMM聚类
    logger.info("执行GMM聚类...")
    embeddings_matrix = np.array(entity_embeddings)
    cluster_labels = perform_clustering(
        embeddings_matrix,
        dim=2,
        threshold=cluster_threshold,
        verbose=True
    )

    # 3. 根据聚类组织文本块
    logger.info("根据聚类结果组织文本块...")
    clusters = defaultdict(list)

    for i, entity in enumerate(entities):
        entity_name = entity['name']
        # 软聚类：一个实体可能属于多个cluster
        for cluster_id in cluster_labels[i]:
            clusters[int(cluster_id)].append({
                'entity': entity,
                'chunks': entity_to_chunks.get(entity_name, [])
            })

    logger.info(f"形成 {len(clusters)} 个聚类")

    # 打印聚类统计信息
    print(f"\n===== 聚类统计信息 =====")
    print(f"总聚类数: {len(clusters)}")
    print(f"总实体数: {len(entities)}")
    print(f"总文本块数: {len(text_chunks)}")

    # 检查不同聚类是否包含相同的文本块
    cluster_chunk_mapping = {}  # cluster_id -> set of chunk names
    for cluster_id, cluster_data in clusters.items():
        cluster_chunks = set()
        for item in cluster_data:
            for chunk_info in item['chunks']:
                episode_name = chunk_info.get('episode_name', '')
                if episode_name:
                    cluster_chunks.add(episode_name)
        cluster_chunk_mapping[cluster_id] = cluster_chunks

    # 检查重复的文本块
    chunk_to_clusters = {}  # chunk_name -> list of cluster_ids
    for cluster_id, chunks in cluster_chunk_mapping.items():
        for chunk in chunks:
            if chunk not in chunk_to_clusters:
                chunk_to_clusters[chunk] = []
            chunk_to_clusters[chunk].append(cluster_id)

    # 找出被多个聚类共享的文本块
    shared_chunks = {chunk: clusters for chunk, clusters in chunk_to_clusters.items() if len(clusters) > 1}

    if shared_chunks:
        print(f"\n⚠️  警告: 发现 {len(shared_chunks)} 个文本块被多个聚类共享:")
        for chunk, cluster_list in shared_chunks.items():
            print(f"   - 文本块 '{chunk}' 被聚类 {cluster_list} 共享")

        # 统计聚类间的重复情况
        cluster_pairs = {}
        for chunk, cluster_list in shared_chunks.items():
            for i in range(len(cluster_list)):
                for j in range(i+1, len(cluster_list)):
                    pair = tuple(sorted([cluster_list[i], cluster_list[j]]))
                    cluster_pairs[pair] = cluster_pairs.get(pair, 0) + 1

        if cluster_pairs:
            print(f"\n聚类间重复情况:")
            for (cluster1, cluster2), count in sorted(cluster_pairs.items()):
                print(f"   - 聚类 {cluster1} 和聚类 {cluster2} 共享 {count} 个文本块")
    else:
        print(f"\n✅ 所有聚类都包含不同的文本块，没有重复")

    # 4. 初始化缓存结构
    cache = {}

    # 收集所有唯一的chunk内容，用于构建叶子节点
    all_chunks = {}  # episode_name -> content 的映射
    chunk_id_mapping = {}  # episode_name -> leaf_id 的映射

    # 创建text_chunks到leaf_id的直接映射
    text_chunks_to_leaf = {}
    for i, chunk_content in enumerate(text_chunks):
        text_chunks_to_leaf[chunk_content.strip()] = f"leaf_{i}"

    # 确保所有text_chunks都有对应的leaf节点
    # 为所有text_chunks创建leaf节点，不管是否被实体引用
    for i, chunk_content in enumerate(text_chunks):
        leaf_id = f"leaf_{i}"
        # 如果这个chunk没有被实体引用，给它一个默认的episode_name
        episode_name = f"default_episode_{i}"
        all_chunks[episode_name] = chunk_content
        chunk_id_mapping[episode_name] = leaf_id

    # 5. 为每个聚类生成两条并行的一层摘要：persona摘要与实体总结摘要
    first_layer_summaries = []
    first_layer_entity_summaries = []
    cluster_stats = {}

    # 定义处理单个聚类的函数
    def process_single_cluster(args):
        cluster_id, cluster_data = args

        # 收集聚类中的所有文本
        cluster_texts = []
        cluster_entities = []
        cluster_episode_names = set()  # 使用episode_name而不是索引

        # 为了能够在主线程恢复映射关系，我们需要保留文本和名称的对应列表
        cluster_episode_names_list = []
        cluster_texts_list = []

        for item in cluster_data:
            cluster_entities.append(item['entity']['name'])

            # 收集相关的文本内容
            for chunk_info in item['chunks']:
                # 直接使用chunk内容
                content = chunk_info.get('content', '')
                episode_name = chunk_info.get('episode_name', '')

                if content:
                    cluster_texts.append(content)
                    cluster_episode_names.add(episode_name)
                    cluster_texts_list.append(content)
                    cluster_episode_names_list.append(episode_name)

        # 去重实体名称
        cluster_entities = list(set(cluster_entities))
        cluster_texts = list(set(cluster_texts))

        # 统计信息
        total_chars = sum(len(text) for text in cluster_texts)
        stats = {
            'entity_count': len(cluster_entities),
            'chunk_count': len(cluster_texts),
            'total_chars': total_chars
        }

        # 如果没有文本，跳过这个聚类
        if len(cluster_texts) == 0:
            return None

        # 生成persona摘要
        persona_summary = generate_persona_summary(
            cluster_texts,
            cluster_entities,
            llm_config,
            language
        )

        # 同时基于实体summary生成实体总结摘要
        entity_level_summary = generate_entity_cluster_summary(
            [item['entity'] for item in cluster_data],
            cluster_id,
            llm_config,
            language
        )

        return {
            'cluster_id': cluster_id,
            'stats': stats,
            'cluster_episode_names': cluster_episode_names,
            'cluster_texts_list': cluster_texts_list,
            'cluster_episode_names_list': cluster_episode_names_list,
            'persona_summary': persona_summary,
            'entity_level_summary': entity_level_summary
        }

    # 使用线程池并发执行
    max_workers = min(10, len(clusters))  # 限制最大并发数
    logger.info(f"开始并发生成聚类摘要，并发数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_cluster = {
            executor.submit(process_single_cluster, (cid, cdata)): cid
            for cid, cdata in clusters.items()
        }

        # 处理结果
        for future in tqdm(as_completed(future_to_cluster), total=len(clusters), desc="生成Persona摘要"):
            cluster_id = future_to_cluster[future]
            try:
                result = future.result()
                if not result:
                    logger.warning(f"聚类 {cluster_id} 处理结果为空，跳过")
                    continue

                # 更新统计信息
                cluster_stats[cluster_id] = result['stats']

                # 打印日志
                print(f"\n聚类 {cluster_id}:")
                print(f"  - 实体数: {result['stats']['entity_count']}")
                print(f"  - 文本块数: {result['stats']['chunk_count']}")
                print(f"  - 总字符数: {result['stats']['total_chars']}")
                if result['stats']['total_chars'] > 100000:
                    print(f"  ⚠️ 超大聚类！可能需要特殊处理")

                # 补充映射逻辑：如果episode_name不在mapping中，尝试通过content匹配
                for content, episode_name in zip(result['cluster_texts_list'], result['cluster_episode_names_list']):
                    if episode_name and episode_name not in chunk_id_mapping:
                        content_stripped = content.strip()
                        if content_stripped in text_chunks_to_leaf:
                            chunk_id_mapping[episode_name] = text_chunks_to_leaf[content_stripped]
                            all_chunks[episode_name] = content

                # 存储节点
                node_id = f"summary_0_{cluster_id}"

                # 映射 episode_name 到 leaf_id
                cluster_leaf_ids = []
                for episode_name in sorted(result['cluster_episode_names']):
                    if episode_name in chunk_id_mapping:
                        cluster_leaf_ids.append(chunk_id_mapping[episode_name])
                    else:
                        logger.warning(f"Episode {episode_name} 没有找到对应的leaf_id")

                cache[node_id] = {
                    "text": result['persona_summary'],
                    "children": cluster_leaf_ids,
                    "parent": [],
                    "stats": result['stats']
                }

                first_layer_summaries.append(result['persona_summary'])

                # 存储实体总结节点
                entity_node_id = f"entity_summary_0_{cluster_id}"
                cache[entity_node_id] = {
                    "text": result['entity_level_summary'],
                    "children": [node_id],
                    "parent": []
                }

                first_layer_entity_summaries.append(result['entity_level_summary'])

            except Exception as e:
                logger.error(f"聚类 {cluster_id} 处理失败: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # 创建所有叶子节点，确保所有text_chunks都有对应的leaf节点
    logger.info(f"创建 {len(text_chunks)} 个叶子节点")
    for i, chunk_content in enumerate(text_chunks):
        leaf_id = f"leaf_{i}"

        # 如果提供了original_text且是字典格式，使用full_text保存完整的turn
        leaf_text = chunk_content
        if original_text and i < len(original_text):
            original_chunk = original_text[i]
            if isinstance(original_chunk, dict) and "full_text" in original_chunk:
                leaf_text = original_chunk["full_text"]
                logger.debug(f"Leaf节点 {i} 使用full_text保存完整的turn")

        cache[leaf_id] = {
            "text": leaf_text,
            "children": None,
            "parent": None
        }

    logger.info(f"叶子节点创建完成，共 {len([k for k in cache.keys() if k.startswith('leaf_')])} 个leaf节点")

    # 建立叶子节点的父节点关系
    for node_id, node_data in cache.items():
        if node_id.startswith("summary_0_"):
            children = node_data.get("children", [])
            # 确保children是列表类型，防止None值导致TypeError
            if children is None:
                children = []
            for child_name in children:
                if child_name in cache:
                    cache[child_name]["parent"] = node_id

    # 打印汇总统计
    if cluster_stats:
        print(f"\n===== 聚类汇总 =====")
        avg_entities = np.mean([s['entity_count'] for s in cluster_stats.values()])
        avg_chunks = np.mean([s['chunk_count'] for s in cluster_stats.values()])
        avg_chars = np.mean([s['total_chars'] for s in cluster_stats.values()])
        max_chars = max([s['total_chars'] for s in cluster_stats.values()])

        print(f"平均每聚类实体数: {avg_entities:.1f}")
        print(f"平均每聚类文本块数: {avg_chunks:.1f}")
        print(f"平均每聚类字符数: {avg_chars:.0f}")
        print(f"最大聚类字符数: {max_chars}")

    # 6. Bug2修复: 只构建第一层，不构建更高层
    # 保留persona链和entity_summary链的第一层，但不向上合并
    logger.info("树结构已构建完成（一层树：persona链 summary_0_X -> leaf_X，entity_summary链 entity_summary_0_X -> summary_0_X）")

    # 7. 保存树结构
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

    # 8. 构建与检索器兼容的实体索引
    logger.info("构建实体索引映射...")
    compatible_index = {}

    for entity in entities:
        entity_name = entity['name']
        if entity_name in entity_to_chunks:
            # 将实体映射到正确的leaf_id列表
            leaf_ids = []
            for chunk_info in entity_to_chunks[entity_name]:
                episode_name = chunk_info.get('episode_name', '')
                content = chunk_info.get('content', '')

                # 优先使用episode_name映射，如果没有则使用content匹配
                if episode_name in chunk_id_mapping:
                    leaf_id = chunk_id_mapping[episode_name]
                else:
                    # 通过content匹配找到对应的leaf_id
                    content_stripped = content.strip()
                    if content_stripped in text_chunks_to_leaf:
                        leaf_id = text_chunks_to_leaf[content_stripped]
                    else:
                        continue  # 跳过无法匹配的chunk

                if leaf_id not in leaf_ids:  # 避免重复
                    leaf_ids.append(leaf_id)

            if leaf_ids:
                compatible_index[entity_name] = leaf_ids
                logger.debug(f"实体 {entity_name} 映射到 {len(leaf_ids)} 个文本块: {leaf_ids}")

    # 为了向后兼容，如果没有实体数据，使用传统索引
    if not compatible_index and index:
        logger.info("使用传入的原始索引作为备用")
        compatible_index = index

    # 9. 保存聚类索引到缓存
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(compatible_index, f, ensure_ascii=False, indent=4)
    logger.info(f"聚类索引已保存到: {index_path}")

    build_time = time.time() - start_time
    logger.info(f"树构建完成，共 {len(cache)} 个节点，{len(compatible_index)} 个实体索引，耗时 {build_time:.2f} 秒")

    return cache, build_time, compatible_index

# ===================== 同步包装函数 =====================

def build_tree_with_clustering_sync(
    text_chunks: List[str],
    llm_config: dict,
    cache_folder: str,
    graphiti_driver,
    group_id: str,
    language: str = "zh",
    cluster_threshold: float = 0.1,
    merge_num: int = 5,
    index: dict = None,
    text_edge_mapping: dict = None,
    original_text: List = None  # 原始文本（包含full_text和user_only_text的字典列表）
) -> Tuple[dict, float, dict]:
    """
    同步版本的基于聚类的树构建函数
    """
    # 直接调用同步版本的函数
    return build_tree_with_clustering(
        text_chunks=text_chunks,
        llm_config=llm_config,
        cache_folder=cache_folder,
        graphiti_driver=graphiti_driver,
        group_id=group_id,
        language=language,
        cluster_threshold=cluster_threshold,
        merge_num=merge_num,
        index=index,
        text_edge_mapping=text_edge_mapping,
        original_text=original_text
    )

# ===================== 接口兼容性包装函数 =====================

def build_tree(
    text_chunks: List[str],
    llm_config: dict,
    cache_folder: str,
    merge_num: int = 5,
    language: str = "zh"
) -> Tuple[dict, float]:
    """
    与原始build_tree完全兼容的接口包装函数

    Args:
        text_chunks: 文本块列表
        llm_config: LLM配置
        cache_folder: 缓存文件夹路径
        merge_num: 合并数量
        language: 语言设置

    Returns:
        (树结构字典, 耗时) - 与原始build_tree相同的返回格式
    """
    logger.info("使用基于聚类的树构建方法（兼容模式）")

    # 检查是否有Graphiti相关的环境变量或配置
    graphiti_driver = None
    group_id = None

    try:
        # 尝试从环境或全局状态获取Graphiti配置
        # 这里需要根据实际项目的配置方式调整
        import os
        from pathlib import Path

        # 检查是否存在图数据
        graph_cache_path = os.path.join(cache_folder, "graph.json")
        if os.path.exists(graph_cache_path):
            logger.info("检测到图数据缓存，尝试使用聚类方法")

            # 导入Graphiti相关模块
            try:
                from graphiti_extractor import GraphitiExtractor
                # 这里需要根据实际的配置获取driver
                # 暂时使用None，会回退到传统方法
                group_id = "default_group"
            except ImportError:
                logger.warning("无法导入Graphiti模块，回退到传统方法")
        else:
            logger.info("未找到图数据缓存，回退到传统方法")

    except Exception as e:
        logger.warning(f"获取Graphiti配置失败: {e}，回退到传统方法")

    # 如果无法获取Graphiti配置，回退到传统方法
    if graphiti_driver is None:
        logger.info("回退到传统build_tree方法")
        try:
            import importlib.util

            # 动态导入原始build_tree模块，避免循环导入
            spec = importlib.util.find_spec("build_tree")
            if spec is not None and spec.origin != __file__:
                build_tree_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(build_tree_module)
                return build_tree_module.build_tree(text_chunks, llm_config, cache_folder, merge_num, language)
            else:
                logger.error("无法找到原始build_tree模块，使用简化树构建")
                cache, build_time, _ = _build_simple_tree(text_chunks, llm_config, cache_folder, merge_num, language)
                return cache, build_time
        except Exception as e:
            logger.error(f"回退到传统方法失败: {e}，使用简化树构建")
            cache, build_time, _ = _build_simple_tree(text_chunks, llm_config, cache_folder, merge_num, language)
            return cache, build_time

    # 使用聚类方法
    try:
        cache, build_time, index = build_tree_with_clustering_sync(
            text_chunks=text_chunks,
            llm_config=llm_config,
            cache_folder=cache_folder,
            graphiti_driver=graphiti_driver,
            group_id=group_id,
            language=language,
            merge_num=merge_num
        )

        # 返回与原始接口兼容的格式（只返回cache和build_time）
        logger.info(f"聚类树构建成功，实体索引包含 {len(index)} 个条目")
        return cache, build_time

    except Exception as e:
        logger.error(f"聚类树构建失败: {e}，回退到传统方法")
        try:
            import importlib.util

            # 动态导入原始build_tree模块，避免循环导入
            spec = importlib.util.find_spec("build_tree")
            if spec is not None and spec.origin != __file__:
                build_tree_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(build_tree_module)
                return build_tree_module.build_tree(text_chunks, llm_config, cache_folder, merge_num, language)
            else:
                logger.error("无法找到原始build_tree模块，使用简化树构建")
                cache, build_time, _ = _build_simple_tree(text_chunks, llm_config, cache_folder, merge_num, language)
                return cache, build_time
        except Exception as fallback_e:
            logger.error(f"最终回退也失败: {fallback_e}，使用简化树构建")
            cache, build_time, _ = _build_simple_tree(text_chunks, llm_config, cache_folder, merge_num, language)
            return cache, build_time
