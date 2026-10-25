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
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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

# 导入token计算工具
try:
    import tiktoken
except ImportError:
    tiktoken = None
    logger.warning("tiktoken未安装，将使用字符数估算token数")

# ===================== 统一的LLM调用函数（支持vLLM和MT API） =====================

def _call_llm_unified(messages: List[dict], llm_config: dict) -> str:
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
        ) or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

# ===================== GMM聚类功能 =====================

def perform_clustering(embeddings: np.ndarray, dim: int = 2, threshold: float = 0.1, verbose: bool = True, min_clusters: int = None) -> List[np.ndarray]:
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
    if len(embeddings) <= 5:
        # 如果实体太少（<=5），全部归为一类，跳过UMAP降维（避免光谱初始化失败）
        if verbose:
            logger.info(f"实体数量为 {len(embeddings)}，太少无法进行稳定的UMAP降维，直接使用原始embeddings")
        # 直接使用原始embeddings进行聚类
        reduced_embeddings = embeddings
    else:
        if umap is None:
            raise ImportError("umap-learn is required for temporal entity clustering")
        # UMAP降维 - 只在实体数量足够时执行
        if verbose:
            logger.info(f"UMAP降维: {embeddings.shape} -> ({len(embeddings)}, {dim})")

        # 确保参数合理
        n_neighbors = max(2, min(15, len(embeddings) - 1))
        n_components = min(dim, len(embeddings) - 1, 50)  # 限制最大维度

        # 修复: 使用random初始化避免光谱初始化的问题
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            init='random',  # 使用random而不是spectral初始化
            random_state=42  # 确保可重现
        )

        try:
            reduced_embeddings = reducer.fit_transform(embeddings)
            if verbose:
                logger.info(f"UMAP降维成功: {embeddings.shape} -> {reduced_embeddings.shape}")
        except Exception as e:
            logger.warning(f"UMAP降维失败: {e}，使用原始embeddings进行聚类")
            reduced_embeddings = embeddings

    # 确定最优聚类数（使用BIC）
    max_clusters = min(50, len(embeddings))
    min_clusters_val = min_clusters if min_clusters is not None else 1

    # 对于小数据集，限制聚类数量并增加正则化
    if len(embeddings) <= 5:
        n_clusters_range = range(1, min(2, len(embeddings) + 1))  # 最多2个聚类
    else:
        n_clusters_range = range(max(min_clusters_val, 1), min(max_clusters, len(embeddings)))

    bics = []

    # 确保数据类型为float64以提高数值稳定性
    if reduced_embeddings.dtype != np.float64:
        reduced_embeddings = reduced_embeddings.astype(np.float64)

    for n in n_clusters_range:
        try:
            # 增加正则化参数以提高数值稳定性
            reg_covar = 1e-6 if len(embeddings) > 10 else 1e-3
            gm = GaussianMixture(
                n_components=n,
                random_state=42,
                n_init=3,
                reg_covar=reg_covar,  # 添加正则化
                covariance_type='diag'  # 使用对角协方差矩阵，更稳定
            )
            gm.fit(reduced_embeddings)
            bic = gm.bic(reduced_embeddings)
            bics.append(bic)

            # 早停条件
            if len(bics) > 1 and bics[-1] > bics[-2]:
                break

        except Exception as e:
            if verbose:
                logger.warning(f"聚类数{n}失败: {e}，尝试下一个")
            continue

    if len(bics) == 0:
        # 如果所有聚类都失败，返回单一聚类
        optimal_clusters = 1
        if verbose:
            logger.warning("所有GMM聚类尝试都失败，使用单一聚类")
    else:
        optimal_clusters = n_clusters_range[np.argmin(bics)]

    # 如果指定了最小聚类数，确保不小于最小值
    if min_clusters is not None and optimal_clusters < min_clusters_val:
        optimal_clusters = min_clusters_val

    if verbose:
        logger.info(f"最优聚类数: {optimal_clusters}")

    # 执行最终GMM聚类
    try:
        reg_covar = 1e-6 if len(embeddings) > 10 else 1e-3
        gm = GaussianMixture(
            n_components=optimal_clusters,
            random_state=42,
            n_init=5,
            reg_covar=reg_covar,
            covariance_type='diag'
        )
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

        if verbose:
            logger.info(f"聚类完成，返回{len(labels)}个实体的聚类标签")

        return labels

    except Exception as e:
        # 如果最终聚类也失败，返回所有实体归为一类
        if verbose:
            logger.warning(f"最终GMM聚类失败: {e}，所有实体归为一类")
        return [np.array([0]) for _ in range(len(embeddings))]

# ===================== Persona生成功能 =====================

def generate_persona_summary(
    texts: List[str],
    entities: List[str],
    llm_config: dict,
    language: str = "zh"
) -> str:
    """
    生成用户persona摘要（无长度限制，但有超长预警）

    Args:
        texts: 文本块列表
        entities: 实体名称列表
        llm_config: LLM配置
        language: 语言设置

    Returns:
        生成的persona摘要
    """
    # 计算总文本长度
    total_text = '\n\n'.join(texts)
    total_length = len(total_text)

    # 超长预警阈值
    WARNING_THRESHOLD = 10000  # 字符数
    CRITICAL_THRESHOLD = 50000  # 字符数

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

# ===================== 时间相关辅助函数 =====================

def parse_iso_datetime(dt_str: str) -> Optional[datetime]:
    """
    解析ISO格式的日期时间字符串

    Args:
        dt_str: ISO格式的日期时间字符串

    Returns:
        datetime对象，如果解析失败返回None
    """
    if not dt_str:
        return None

    try:
        # 尝试解析ISO格式
        if isinstance(dt_str, str):
            # 处理Neo4j DateTime对象的字符串表示
            if 'T' in dt_str:
                # 处理带纳秒的格式：2023-01-01T00:00:00.000000000+00:00
                # Python的fromisoformat不支持纳秒，需要移除纳秒部分
                if '.' in dt_str:
                    # 分离日期时间和时区部分
                    if '+' in dt_str:
                        # 格式：2023-01-01T00:00:00.000000000+00:00
                        time_part, tz_part = dt_str.split('+')
                        time_without_ns = time_part.split('.')[0]  # 移除纳秒
                        dt_str = f"{time_without_ns}+{tz_part}"
                    elif dt_str.count('-') > 2:
                        # 格式：2023-01-01T00:00:00.000000000-05:00
                        # 找到最后一个'-'，它前面是纳秒，后面是时区
                        last_dash_idx = dt_str.rfind('-')
                        if last_dash_idx > 10:  # 确保不是日期中的'-'
                            time_part = dt_str[:last_dash_idx].split('.')[0]
                            tz_part = dt_str[last_dash_idx:]
                            dt_str = f"{time_part}{tz_part}"
                    elif dt_str.endswith('Z'):
                        # 格式：2023-01-01T00:00:00.000000000Z
                        time_without_ns = dt_str.split('.')[0]
                        dt_str = f"{time_without_ns}+00:00"
                    else:
                        # 没有时区信息，只移除纳秒
                        dt_str = dt_str.split('.')[0]
                elif dt_str.endswith('Z'):
                    dt_str = dt_str[:-1] + '+00:00'

                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return None
    except Exception as e:
        logger.warning(f"解析日期时间失败: {dt_str}, 错误: {e}")
        return None

def get_time_range_from_graph(graphiti_driver, group_id: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    查询整个图的valid_at时间范围

    Args:
        graphiti_driver: Neo4j驱动
        group_id: 组ID

    Returns:
        (最早时间, 最晚时间)，如果查询失败返回(None, None)
    """
    # 先检查边的数量和有valid_at的边的数量
    check_query = """
    MATCH (src)-[e:RELATES_TO]->(dst)
    WHERE e.group_id = $group_id
    RETURN count(e) as total_edges,
           sum(CASE WHEN e.valid_at IS NOT NULL THEN 1 ELSE 0 END) as edges_with_valid_at
    """

    # 也检查节点的valid_at（如果边没有valid_at，可能节点有）
    check_node_query = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id
    RETURN count(n) as total_nodes,
           sum(CASE WHEN n.valid_at IS NOT NULL THEN 1 ELSE 0 END) as nodes_with_valid_at
    """

    # 查询边的valid_at时间范围
    query = """
    MATCH (src)-[e:RELATES_TO]->(dst)
    WHERE e.group_id = $group_id AND e.valid_at IS NOT NULL
    RETURN min(e.valid_at) as min_time, max(e.valid_at) as max_time
    """

    # 如果边没有valid_at，尝试查询节点的valid_at
    node_query = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id AND n.valid_at IS NOT NULL
    RETURN min(n.valid_at) as min_time, max(n.valid_at) as max_time
    """

    import asyncio
    try:
        # 先检查边的统计信息
        check_result = graphiti_driver.execute_query(check_query, group_id=group_id)
        if asyncio.iscoroutine(check_result):
            loop = asyncio.get_event_loop()
            check_result = loop.run_until_complete(check_result)

        if hasattr(check_result, 'records') and len(check_result.records) > 0:
            check_record = check_result.records[0]
            total_edges = check_record.get('total_edges', 0)
            edges_with_valid_at = check_record.get('edges_with_valid_at', 0)
            logger.info(f"[DEBUG] group_id={group_id}: 总边数={total_edges}, 有valid_at的边数={edges_with_valid_at}")

        # 检查节点的统计信息
        check_node_result = graphiti_driver.execute_query(check_node_query, group_id=group_id)
        if asyncio.iscoroutine(check_node_result):
            loop = asyncio.get_event_loop()
            check_node_result = loop.run_until_complete(check_node_result)

        if hasattr(check_node_result, 'records') and len(check_node_result.records) > 0:
            check_node_record = check_node_result.records[0]
            total_nodes = check_node_record.get('total_nodes', 0)
            nodes_with_valid_at = check_node_record.get('nodes_with_valid_at', 0)
            logger.info(f"[DEBUG] group_id={group_id}: 总节点数={total_nodes}, 有valid_at的节点数={nodes_with_valid_at}")

        # 查询时间范围（优先从边查询）
        result = graphiti_driver.execute_query(query, group_id=group_id)
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)

        # 如果边没有valid_at，尝试从节点查询
        if hasattr(result, 'records') and len(result.records) > 0:
            record = result.records[0]
            min_time = record.get('min_time')
            max_time = record.get('max_time')
            if min_time is None and max_time is None:
                logger.info(f"[DEBUG] group_id={group_id}: 边没有valid_at，尝试从节点查询...")
                result = graphiti_driver.execute_query(node_query, group_id=group_id)
                if asyncio.iscoroutine(result):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(result)
    except Exception as e:
        logger.error(f"查询时间范围失败: {e}", exc_info=True)
        return None, None

    if not hasattr(result, 'records') or len(result.records) == 0:
        logger.warning(f"[DEBUG] group_id={group_id}: 未找到有效的时间范围记录")
        return None, None

    record = result.records[0]
    min_time = record.get('min_time')
    max_time = record.get('max_time')

    logger.info(f"[DEBUG] group_id={group_id}: 原始查询结果 - min_time={min_time} (type={type(min_time)}), max_time={max_time} (type={type(max_time)})")

    # 处理Neo4j DateTime对象
    if min_time:
        try:
            # 优先尝试直接转换为datetime（如果已经是datetime对象）
            if isinstance(min_time, datetime):
                min_time = min_time
            elif hasattr(min_time, 'iso_format'):
                # Neo4j DateTime对象
                iso_str = min_time.iso_format()
                min_time = parse_iso_datetime(iso_str)
            elif hasattr(min_time, 'isoformat'):
                # Python datetime对象
                iso_str = min_time.isoformat()
                min_time = parse_iso_datetime(iso_str)
            elif isinstance(min_time, str):
                min_time = parse_iso_datetime(min_time)
            else:
                logger.warning(f"[DEBUG] 无法处理min_time类型: {type(min_time)}, 值: {min_time}")
                min_time = None
        except Exception as e:
            logger.error(f"[DEBUG] 处理min_time失败: {e}", exc_info=True)
            min_time = None

    if max_time:
        try:
            # 优先尝试直接转换为datetime（如果已经是datetime对象）
            if isinstance(max_time, datetime):
                max_time = max_time
            elif hasattr(max_time, 'iso_format'):
                # Neo4j DateTime对象
                iso_str = max_time.iso_format()
                max_time = parse_iso_datetime(iso_str)
            elif hasattr(max_time, 'isoformat'):
                # Python datetime对象
                iso_str = max_time.isoformat()
                max_time = parse_iso_datetime(iso_str)
            elif isinstance(max_time, str):
                max_time = parse_iso_datetime(max_time)
            else:
                logger.warning(f"[DEBUG] 无法处理max_time类型: {type(max_time)}, 值: {max_time}")
                max_time = None
        except Exception as e:
            logger.error(f"[DEBUG] 处理max_time失败: {e}", exc_info=True)
            max_time = None

    logger.info(f"图时间范围: {min_time} 到 {max_time}")
    return min_time, max_time

def split_time_range_by_month(start_time: datetime, end_time: datetime) -> List[Tuple[datetime, datetime]]:
    """
    将时间范围按月切分

    Args:
        start_time: 开始时间
        end_time: 结束时间

    Returns:
        时间窗口列表，每个窗口是(start, end)的元组
    """
    windows = []
    current = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while current < end_time:
        # 计算当前月的结束时间（下个月的第一天）
        next_month = current + relativedelta(months=1)
        window_end = min(next_month, end_time)

        windows.append((current, window_end))
        current = next_month

    logger.info(f"时间范围按月切分为 {len(windows)} 个窗口")
    for i, (s, e) in enumerate(windows):
        logger.info(f"窗口 {i+1}: {s.isoformat()} 到 {e.isoformat()}")

    return windows

# ===================== 主要功能函数 =====================

def get_entities_with_embeddings(
    graphiti_driver,
    group_id: str,
    time_window: Optional[Tuple[datetime, datetime]] = None
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    从Neo4j读取指定时间窗口内的实体及其嵌入向量

    Args:
        graphiti_driver: Neo4j驱动
        group_id: 组ID
        time_window: 时间窗口 (start_time, end_time)，如果为None则返回所有实体

    Returns:
        (实体列表, 嵌入向量列表)
    """
    # 构建查询：通过边的时间窗口来过滤实体
    if time_window:
        start_time, end_time = time_window
        query = """
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE e.group_id = $group_id
          AND e.valid_at IS NOT NULL
          AND e.valid_at >= datetime($start_time)
          AND e.valid_at < datetime($end_time)
        WITH DISTINCT src as entity
        MATCH (entity)
        WHERE entity.group_id = $group_id
        RETURN entity.uuid as uuid,
               entity.name as name,
               entity.summary as summary,
               entity.entity_type as type,
               entity.name_embedding as embedding
        UNION
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE e.group_id = $group_id
          AND e.valid_at IS NOT NULL
          AND e.valid_at >= datetime($start_time)
          AND e.valid_at < datetime($end_time)
        WITH DISTINCT dst as entity
        MATCH (entity)
        WHERE entity.group_id = $group_id
        RETURN entity.uuid as uuid,
               entity.name as name,
               entity.summary as summary,
               entity.entity_type as type,
               entity.name_embedding as embedding
        """
    else:
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
        # 构建查询参数
        params = {"group_id": group_id}
        if time_window:
            params["start_time"] = start_time.isoformat()
            params["end_time"] = end_time.isoformat()

        # 尝试同步调用
        result = graphiti_driver.execute_query(query, **params)
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

def get_entity_chunk_mapping(
    graphiti_driver,
    group_id: str,
    time_window: Optional[Tuple[datetime, datetime]] = None
) -> Dict[str, List[Dict]]:
    """
    获取指定时间窗口内实体与文本块(episodes)的映射关系

    Args:
        graphiti_driver: Neo4j驱动
        group_id: 组ID
        time_window: 时间窗口 (start_time, end_time)，如果为None则返回所有映射

    Returns:
        entity_name -> chunk信息列表的映射
    """
    if time_window:
        start_time, end_time = time_window
        # 通过边的时间窗口来过滤episode
        query = """
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE e.group_id = $group_id
          AND e.valid_at IS NOT NULL
          AND e.valid_at >= datetime($start_time)
          AND e.valid_at < datetime($end_time)
        UNWIND e.episodes AS ep_uuid
        MATCH (ep:Episodic {uuid: ep_uuid})-[:MENTIONS]->(n:Entity)
        WHERE n.group_id = $group_id
          AND (ep.valid_at IS NULL OR (ep.valid_at >= datetime($start_time) AND ep.valid_at < datetime($end_time)))
        RETURN DISTINCT n.name as entity_name,
               n.uuid as entity_uuid,
               ep.name as episode_name,
               ep.content as chunk_content,
               ep.uuid as episode_uuid
        """
    else:
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
        # 构建查询参数
        params = {"group_id": group_id}
        if time_window:
            params["start_time"] = start_time.isoformat()
            params["end_time"] = end_time.isoformat()

        # 尝试同步调用
        result = graphiti_driver.execute_query(query, **params)
        # 如果返回的是协程，需要等待
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)
    except Exception as e:
        logger.error(f"Neo4j查询失败: {e}")
        return {}

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


def get_edges_with_episodes(
    graphiti_driver,
    group_id: str,
    time_window: Optional[Tuple[datetime, datetime]] = None,
) -> List[Dict]:
    """
    获取指定时间窗口内的 RELATES_TO 边及其 episodes 列表，用于后续按 cluster 聚合 episode 文本。
    只依赖边的 valid_at 与 e.episodes，不依赖 :MENTIONS 关系，避免“有实体但拿不到chunk”的情况。
    """
    if time_window:
        start_time, end_time = time_window
        query = """
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE e.group_id = $group_id
          AND e.valid_at IS NOT NULL
          AND e.valid_at >= datetime($start_time)
          AND e.valid_at < datetime($end_time)
          AND e.episodes IS NOT NULL
        RETURN src.uuid as src_uuid, dst.uuid as dst_uuid, e.episodes as episodes
        """
        params = {
            "group_id": group_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }
    else:
        query = """
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE e.group_id = $group_id
          AND e.episodes IS NOT NULL
        RETURN src.uuid as src_uuid, dst.uuid as dst_uuid, e.episodes as episodes
        """
        params = {"group_id": group_id}

    import asyncio
    try:
        result = graphiti_driver.execute_query(query, **params)
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)
    except Exception as e:
        logger.error(f"Neo4j查询边episodes失败: {e}", exc_info=True)
        return []

    edges: List[Dict] = []
    if hasattr(result, "records"):
        for record in result.records:
            edges.append({
                "src_uuid": record.get("src_uuid"),
                "dst_uuid": record.get("dst_uuid"),
                "episodes": record.get("episodes") or [],
            })
    return edges


def get_episode_contents(
    graphiti_driver,
    group_id: str,
    episode_uuids: List[str],
) -> Dict[str, Dict]:
    """
    批量拉取 Episodic 内容，返回 uuid -> {content, name}。
    episode_uuids 会在调用方去重；这里仍做一次去重保护。
    """
    if not episode_uuids:
        return {}

    unique_uuids = sorted({u for u in episode_uuids if isinstance(u, str) and u})
    query = """
    MATCH (ep:Episodic)
    WHERE ep.group_id = $group_id AND ep.uuid IN $episode_uuids
    RETURN ep.uuid as uuid, ep.content as content, ep.name as name
    """
    params = {"group_id": group_id, "episode_uuids": unique_uuids}

    import asyncio
    try:
        result = graphiti_driver.execute_query(query, **params)
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)
    except Exception as e:
        logger.error(f"Neo4j批量查询Episodic失败: {e}", exc_info=True)
        return {}

    mapping: Dict[str, Dict] = {}
    if hasattr(result, "records"):
        for record in result.records:
            uuid = record.get("uuid")
            if not uuid:
                continue
            mapping[uuid] = {
                "content": record.get("content") or "",
                "name": record.get("name") or "",
            }
    return mapping

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
    original_text: List = None,  # 原始文本（包含full_text和user_only_text的字典列表）
    snapshots_only: bool = False,  # 仅重生成 snapshots.json，不覆盖 tree/index
    tree_only: bool = False,  # 仅重生成 tree/index，不覆盖 snapshots.json
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

    # Keep the complete source corpus separate from the per-window working set.
    # The original implementation reassigns ``text_chunks`` inside every
    # temporal window.  That is useful for window-local clustering, but it
    # must not become the source of truth for the final dense tree.
    source_text_chunks = list(text_chunks)

    # 检查缓存
    tree_path = os.path.join(cache_folder, "tree.json")
    index_path = os.path.join(cache_folder, "cluster_index.json")
    snapshots_path = os.path.join(cache_folder, "snapshots.json")

    # 如果 tree 存在但 snapshots 不存在：默认进入“只重生成 snapshots”模式，避免覆盖现有 tree/index
    if os.path.exists(tree_path) and not os.path.exists(snapshots_path) and not snapshots_only and not tree_only:
        logger.info("检测到 tree.json 存在但 snapshots.json 缺失，将进入 snapshots_only 模式重生成 snapshots（不覆盖 tree/index）")
        snapshots_only = True

    # tree_only 模式：加载现有 snapshots，只重新构建 tree/index
    existing_snapshots: Optional[List[dict]] = None
    if tree_only:
        if not os.path.exists(snapshots_path):
            raise FileNotFoundError(f"tree_only模式需要已存在的snapshots.json，但未找到: {snapshots_path}")
        logger.info("tree_only模式：加载现有 snapshots（将只重写 tree/index）")
        with open(snapshots_path, 'r', encoding='utf-8') as f:
            existing_snapshots = json.load(f)
        logger.info(f"加载了 {len(existing_snapshots)} 个现有snapshots")

    # 如果snapshots_only=True，跳过缓存检查，强制重新生成snapshots
    logger.info(f"build_tree_clustering_time: snapshots_only={snapshots_only}, tree_only={tree_only}, tree_exists={os.path.exists(tree_path)}, snapshots_exists={os.path.exists(snapshots_path)}")
    if os.path.exists(tree_path) and os.path.exists(snapshots_path) and not snapshots_only and not tree_only:
        logger.info("从缓存加载树结构和snapshots")
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

    existing_cache_tree: Optional[dict] = None
    existing_cluster_index: dict = {}
    if snapshots_only:
        if not os.path.exists(tree_path):
            raise FileNotFoundError(f"snapshots_only模式需要已存在的tree.json，但未找到: {tree_path}")
        logger.info("snapshots_only模式：加载现有 tree/index（将只重写 snapshots.json）")
        with open(tree_path, 'r', encoding='utf-8') as f:
            existing_cache_tree = json.load(f)
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    existing_cluster_index = json.load(f)
            except Exception as e:
                logger.warning(f"snapshots_only模式：加载cluster_index.json失败，将使用空索引: {e}")
                existing_cluster_index = {}


    # 0. 查询整个图的时间范围
    logger.info("查询整个图的时间范围...")
    min_time, max_time = get_time_range_from_graph(graphiti_driver, group_id)

    if min_time is None or max_time is None:
        logger.warning("无法获取时间范围，使用全图模式")
        time_windows = [None]  # None表示不使用时间窗口
    else:
        # 按月切分时间窗口
        logger.info(f"时间范围: {min_time.isoformat()} 到 {max_time.isoformat()}")
        time_windows = split_time_range_by_month(min_time, max_time)

    # 存储所有snapshot
    all_snapshots = existing_snapshots if tree_only else []
    all_caches = {} if not snapshots_only else (existing_cache_tree or {})
    all_cluster_indices = {} if not snapshots_only else (existing_cluster_index or {})

    # Stable source-to-leaf identity.  Leaf IDs must be global across all
    # monthly windows; otherwise ``all_caches.update(cache)`` silently
    # overwrites leaves from earlier windows.
    episode_uuid_to_leaf: Dict[str, str] = {}
    source_leaf_texts: Dict[str, str] = {}
    source_content_to_leaf: Dict[str, str] = {}
    for i, chunk_content in enumerate(source_text_chunks):
        leaf_id = f"leaf_{i}"
        leaf_text = chunk_content
        if original_text and i < len(original_text):
            original_chunk = original_text[i]
            if isinstance(original_chunk, dict) and "full_text" in original_chunk:
                leaf_text = original_chunk["full_text"]
        source_leaf_texts[leaf_id] = leaf_text
        source_content_to_leaf.setdefault(str(chunk_content).strip(), leaf_id)
    next_extra_leaf_index = len(source_leaf_texts)

    # 对每个时间窗口生成独立的snapshot
    for window_idx, time_window in enumerate(time_windows):
        # 初始化window_start和window_end，避免在创建snapshot时未定义
        if time_window:
            window_start, window_end = time_window
            logger.info(f"\n{'='*60}")
            logger.info(f"处理时间窗口 {window_idx + 1}/{len(time_windows)}: {window_start.isoformat()} 到 {window_end.isoformat()}")
            logger.info(f"{'='*60}")
        else:
            window_start, window_end = None, None  # 全图模式，时间窗口为None
            logger.info(f"\n{'='*60}")
            logger.info(f"处理全图模式（无时间窗口）")
            logger.info(f"{'='*60}")

        # 1. 从Neo4j读取指定时间窗口内的实体和嵌入
        logger.info("从Neo4j读取实体和嵌入...")
        entities, entity_embeddings = get_entities_with_embeddings(graphiti_driver, group_id, time_window)
        logger.info(f"读取到 {len(entities)} 个实体")

        if len(entities) == 0:
            logger.warning(f"时间窗口 {window_idx + 1} 未找到实体，跳过")
            continue

        # 获取实体到文本块(episodes)的映射（用于后续按聚类组织chunk）
        # 之前这里缺失初始化会导致 NameError: entity_to_chunks is not defined
        try:
            logger.info("从Neo4j读取实体到文本块的映射关系...")
            entity_to_chunks = get_entity_chunk_mapping(graphiti_driver, group_id, time_window)
        except Exception as e:
            logger.warning(f"读取entity_to_chunks失败，将使用空映射继续: {e}", exc_info=True)
            entity_to_chunks = {}

        # 基于边的 episodes 直接收集窗口内的文本（按 episode_uuid 去重）
        logger.info("从Neo4j读取时间窗口内边的episodes并批量拉取Episodic内容...")
        window_edges = get_edges_with_episodes(graphiti_driver, group_id, time_window)
        all_episode_uuids: List[str] = []
        for e in window_edges:
            eps = e.get("episodes") or []
            if isinstance(eps, list):
                all_episode_uuids.extend([x for x in eps if isinstance(x, str)])
        episode_content_map = get_episode_contents(graphiti_driver, group_id, all_episode_uuids)
        logger.info(
            f"窗口内边数={len(window_edges)}, episodes去重后={len(set(all_episode_uuids))}, "
            f"成功拉取Episodic内容={len(episode_content_map)}"
        )

        # Resolve each window episode to a stable global leaf.  Graphiti
        # episodes created by TSM are named chunk_<global_index>; use that
        # identity when available.  The UUID fallback handles compatible
        # graphs whose episode names do not follow that convention.
        window_episode_leaf_ids: Dict[str, str] = {}
        for episode_uuid, entry in episode_content_map.items():
            name = str(entry.get("name") or "")
            leaf_id = None
            if name.startswith("chunk_"):
                try:
                    source_index = int(name[len("chunk_"):])
                    if 0 <= source_index < len(source_text_chunks):
                        if str(source_text_chunks[source_index]).strip() != (entry.get("content") or "").strip():
                            raise RuntimeError(f"Episode/source mismatch: {name} ({episode_uuid})")
                        leaf_id = f"leaf_{source_index}"
                except ValueError:
                    pass
            if leaf_id is None:
                leaf_id = episode_uuid_to_leaf.get(episode_uuid)
            if leaf_id is None:
                leaf_id = source_content_to_leaf.get((entry.get("content") or "").strip())
            if leaf_id is None:
                leaf_id = f"leaf_{next_extra_leaf_index}"
                next_extra_leaf_index += 1
            episode_uuid_to_leaf[episode_uuid] = leaf_id
            window_episode_leaf_ids[episode_uuid] = leaf_id

            # Preserve a source text for compatible graphs without a
            # chunk_<index> name.  For normal TSM episodes the source text is
            # already populated above from the original input.
            if leaf_id not in source_leaf_texts:
                source_leaf_texts[leaf_id] = (entry.get("content") or "").strip()
            content_key = (entry.get("content") or "").strip()
            if content_key:
                source_content_to_leaf.setdefault(content_key, leaf_id)

        # 从episode_content_map中提取所有文本块（用于统计和后续处理）
        window_text_chunks = []
        for ep_uuid, entry in episode_content_map.items():
            content = (entry.get("content") or "").strip()
            if content:
                window_text_chunks.append(content)
        # 去重（防止不同uuid内容相同）
        text_chunks = list(dict.fromkeys(window_text_chunks))

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

        # 检测超长聚类并重新聚类
        # 阈值：100000 tokens，留出安全余量（模型最大context length是128000 tokens）
        MAX_CLUSTER_TOKENS = 100000
        max_recluster_iterations = 3  # 最多重新聚类3次

        # 获取模型名称用于token计算
        model_name = llm_config.get("model", "gpt-4o-mini")

        # 初始化tokenizer
        tokenizer = None
        if tiktoken:
            try:
                # 尝试获取模型的tokenizer
                tokenizer = tiktoken.encoding_for_model(model_name)
            except (KeyError, ValueError):
                # 如果模型不支持，使用cl100k_base（GPT-4使用的编码）
                try:
                    tokenizer = tiktoken.get_encoding("cl100k_base")
                except:
                    logger.warning(f"无法初始化tokenizer，将使用字符数估算token数")

        def count_tokens(text: str) -> int:
            """计算文本的token数"""
            if tokenizer:
                try:
                    return len(tokenizer.encode(text))
                except:
                    # 如果编码失败，使用字符数估算（中文约1.5字符=1token，英文约4字符=1token）
                    # 这里使用保守估算：3字符=1token
                    return len(text) // 3
            else:
                # 如果没有tokenizer，使用字符数估算
                return len(text) // 3

        for iteration in range(max_recluster_iterations):
            # 计算每个聚类的token数
            cluster_tokens = {}
            for cluster_id, cluster_data in clusters.items():
                cluster_entity_uuids = set()
                for item in cluster_data:
                    ent = item.get("entity") or {}
                    if ent.get("uuid"):
                        cluster_entity_uuids.add(ent["uuid"])

                # 通过窗口内边->episodes计算token数
                cluster_episode_uuids = set()
                if cluster_entity_uuids:
                    for edge in window_edges:
                        src_uuid = edge.get("src_uuid")
                        dst_uuid = edge.get("dst_uuid")
                        if src_uuid in cluster_entity_uuids or dst_uuid in cluster_entity_uuids:
                            eps = edge.get("episodes") or []
                            if isinstance(eps, list):
                                for ep_uuid in eps:
                                    if isinstance(ep_uuid, str) and ep_uuid:
                                        cluster_episode_uuids.add(ep_uuid)

                # 收集所有文本内容
                cluster_texts = []
                for ep_uuid in cluster_episode_uuids:
                    entry = episode_content_map.get(ep_uuid) or {}
                    content = (entry.get("content") or "").strip()
                    if content:
                        cluster_texts.append(content)

                # 计算token数（考虑prompt的额外开销）
                if cluster_texts:
                    # 将所有文本合并，加上prompt的估计开销（约500 tokens）
                    combined_text = '\n\n'.join(cluster_texts)
                    text_tokens = count_tokens(combined_text)
                    # 加上prompt模板的估计token数（约500 tokens）
                    total_tokens = text_tokens + 500
                else:
                    total_tokens = 0

                cluster_tokens[cluster_id] = total_tokens

            # 找出超长的聚类
            oversized_clusters = {cid: tokens for cid, tokens in cluster_tokens.items() if tokens > MAX_CLUSTER_TOKENS}

            if not oversized_clusters:
                # 没有超长聚类，退出循环
                if iteration > 0:
                    logger.info(f"重新聚类完成，第 {iteration + 1} 次迭代后所有聚类都在阈值内")
                break

            logger.warning(f"检测到 {len(oversized_clusters)} 个超长聚类（第 {iteration + 1} 次检测）:")
            for cid, tokens in oversized_clusters.items():
                logger.warning(f"  聚类 {cid}: {tokens} tokens (阈值: {MAX_CLUSTER_TOKENS} tokens)")

            # 收集需要重新聚类的实体
            entities_to_recluster = []
            entity_indices_to_recluster = []
            oversized_cluster_ids = set(oversized_clusters.keys())

            for i, entity in enumerate(entities):
                entity_cluster_ids = set(int(cid) for cid in cluster_labels[i])
                # 如果实体属于超长聚类，需要重新聚类
                if entity_cluster_ids.intersection(oversized_cluster_ids):
                    entities_to_recluster.append(entity)
                    entity_indices_to_recluster.append(i)

            if not entities_to_recluster:
                logger.warning("未找到需要重新聚类的实体，跳过重新聚类")
                break

            logger.info(f"准备重新聚类 {len(entities_to_recluster)} 个实体，当前聚类数: {len(clusters)}")

            # 获取这些实体的embeddings
            recluster_embeddings = embeddings_matrix[entity_indices_to_recluster]

            # 计算新的最小聚类数：当前超长聚类的数量 * 2（增加聚类数量）
            current_oversized_count = len(oversized_clusters)
            new_min_clusters = max(current_oversized_count * 2, len(entities_to_recluster) // 10)  # 至少是原来的2倍，或者每10个实体一个聚类

            logger.info(f"重新聚类参数: 实体数={len(entities_to_recluster)}, 最小聚类数={new_min_clusters}")

            # 重新聚类
            new_cluster_labels = perform_clustering(
                recluster_embeddings,
                dim=2,
                threshold=cluster_threshold,
                verbose=True,
                min_clusters=new_min_clusters
            )

            # 获取当前最大的cluster_id，新的聚类从这之后开始编号
            max_existing_cluster_id = max(clusters.keys()) if clusters else -1
            new_cluster_start_id = max_existing_cluster_id + 1

            # 从原clusters中移除超长聚类的实体
            for cid in list(oversized_cluster_ids):
                if cid in clusters:
                    del clusters[cid]

            # 更新cluster_labels，将重新聚类的实体的标签替换为新的标签
            for idx, orig_idx in enumerate(entity_indices_to_recluster):
                # 将新的聚类标签加上偏移量，避免与现有聚类ID冲突
                new_labels = np.array([int(cid) + new_cluster_start_id for cid in new_cluster_labels[idx]])
                cluster_labels[orig_idx] = new_labels

            # 重新组织clusters
            for idx, orig_idx in enumerate(entity_indices_to_recluster):
                entity = entities[orig_idx]
                entity_name = entity['name']
                for new_cluster_id in cluster_labels[orig_idx]:
                    cluster_id = int(new_cluster_id)
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    # 检查是否已存在，避免重复添加
                    entity_uuid = entity.get('uuid', '')
                    existing_uuids = {item.get('entity', {}).get('uuid', '') for item in clusters[cluster_id]}
                    if entity_uuid not in existing_uuids:
                        clusters[cluster_id].append({
                            'entity': entity,
                            'chunks': entity_to_chunks.get(entity_name, [])
                        })

            logger.info(f"重新聚类完成，新聚类数: {len(clusters)}")

        # 打印聚类统计信息
        print(f"\n===== 聚类统计信息（时间窗口 {window_idx + 1}）=====")
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

        # 4. 初始化缓存结构：snapshots_only 复用现有 tree，不再重建/覆盖；tree_only 需要重建 tree
        cache = (existing_cache_tree or {}) if snapshots_only else {}

        # 建立窗口内容/episode到稳定leaf的映射。这里不再按窗口局部
        # enumerate 生成 leaf_0, leaf_1, ...。
        all_chunks = {}  # episode_name -> content 的映射
        chunk_id_mapping = {}  # episode_name -> leaf_id 的映射

        # snapshots_only 不需要 leaf 映射/leaf 节点重建；tree_only 需要重建
        text_chunks_to_leaf = {}
        if not snapshots_only:
            for episode_uuid, entry in episode_content_map.items():
                content = (entry.get("content") or "").strip()
                leaf_id = window_episode_leaf_ids.get(episode_uuid)
                if not content or leaf_id is None:
                    continue
                text_chunks_to_leaf.setdefault(content, leaf_id)
                episode_name = str(entry.get("name") or episode_uuid)
                all_chunks[episode_name] = content
                chunk_id_mapping[episode_name] = leaf_id

        # 5. 为每个聚类生成两条并行的一层摘要：persona摘要与实体总结摘要
        first_layer_summaries = []
        first_layer_entity_summaries = []
        cluster_stats = {}

        # 定义处理单个聚类的函数（必须在循环内部，因为使用了window_edges和episode_content_map）
        def process_single_cluster(args):
            cluster_id, cluster_data = args

            # 收集聚类中的实体
            cluster_entities: List[str] = []
            cluster_entity_uuids: List[str] = []
            for item in cluster_data:
                ent = item.get("entity") or {}
                if ent.get("name"):
                    cluster_entities.append(ent["name"])
                if ent.get("uuid"):
                    cluster_entity_uuids.append(ent["uuid"])

            # 去重实体名称
            cluster_entities = list(set(cluster_entities))
            cluster_entity_uuids = list(set(cluster_entity_uuids))

            # 通过“窗口内边->episodes”按 cluster 实体集合聚合 episode 文本（按 episode_uuid 去重）
            cluster_episode_uuids: set = set()
            if cluster_entity_uuids:
                cluster_entity_uuid_set = set(cluster_entity_uuids)
                for edge in window_edges:
                    src_uuid = edge.get("src_uuid")
                    dst_uuid = edge.get("dst_uuid")
                    if src_uuid in cluster_entity_uuid_set or dst_uuid in cluster_entity_uuid_set:
                        eps = edge.get("episodes") or []
                        if isinstance(eps, list):
                            for ep_uuid in eps:
                                if isinstance(ep_uuid, str) and ep_uuid:
                                    cluster_episode_uuids.add(ep_uuid)

            cluster_texts: List[str] = []
            cluster_episode_names: List[str] = []
            for ep_uuid in cluster_episode_uuids:
                entry = episode_content_map.get(ep_uuid) or {}
                content = (entry.get("content") or "").strip()
                if content:
                    cluster_texts.append(content)
                    # 尝试获取episode名称，如果没有则使用uuid
                    episode_name = entry.get("name") or entry.get("episode_name") or ep_uuid
                    cluster_episode_names.append(episode_name)

            # 对内容再去重一次（防止不同uuid内容相同）
            cluster_texts = list(dict.fromkeys(cluster_texts))
            cluster_episode_names = list(dict.fromkeys(cluster_episode_names))

            # 统计信息
            total_chars = sum(len(text) for text in cluster_texts)
            stats = {
                'entity_count': len(cluster_entities),
                'chunk_count': len(cluster_texts),  # 这里的chunk_count表示用于persona的episode文本段数量
                'total_chars': total_chars
            }

            # 同时基于实体summary生成实体总结摘要（不依赖chunk文本）
            entity_level_summary = generate_entity_cluster_summary(
                [item['entity'] for item in cluster_data],
                cluster_id,
                llm_config,
                language
            )

            # 生成persona摘要：若该窗口内拿不到chunk文本，则跳过LLM生成，用占位文本避免整个月份snapshot缺失
            if len(cluster_texts) == 0:
                logger.warning(
                    f"聚类 {cluster_id} 在该时间窗口内未匹配到可用chunk文本（episodes映射为空），"
                    f"将跳过persona生成并使用占位摘要，以保证按月snapshot连续性。"
                )
                persona_summary = "（该时间窗口内缺少可用于生成persona的文本片段）"
            else:
                persona_summary = generate_persona_summary(
                    cluster_texts,
                    cluster_entities,
                    llm_config,
                    language
                )

            return {
                'cluster_id': cluster_id,
                'stats': stats,
                'persona_summary': persona_summary,
                'entity_level_summary': entity_level_summary,
                'cluster_texts_list': cluster_texts,
                'cluster_episode_names_list': cluster_episode_names
            }

        # 使用线程池并发执行
        max_workers = min(10, len(clusters))  # 限制最大并发数
        logger.info(f"时间窗口 {window_idx + 1}: 准备开始并发生成聚类摘要，聚类数={len(clusters)}, 并发数={max_workers}")
        if len(clusters) == 0:
            logger.warning(f"时间窗口 {window_idx + 1}: 聚类数为0，跳过摘要生成")
        else:
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
                    # 只在非snapshots_only模式下执行，因为snapshots_only模式不需要leaf映射
                    if not snapshots_only and 'cluster_texts_list' in result and 'cluster_episode_names_list' in result:
                        for content, episode_name in zip(result['cluster_texts_list'], result['cluster_episode_names_list']):
                            if episode_name and episode_name not in chunk_id_mapping:
                                content_stripped = content.strip()
                                if content_stripped in text_chunks_to_leaf:
                                    chunk_id_mapping[episode_name] = text_chunks_to_leaf[content_stripped]
                                    all_chunks[episode_name] = content

                    # 注意：按需求 A，不再把 persona/topic 写入树（cache/tree.json），避免被稠密检索当作证据。
                    # 但仍然把 persona/topic 结果收集到 snapshots.json 中（persona_summaries/topic_summaries）。
                    first_layer_summaries.append(result['persona_summary'])
                    first_layer_entity_summaries.append(result['entity_level_summary'])

                except Exception as e:
                    logger.error(f"聚类 {cluster_id} 处理失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        if not snapshots_only:
            # 仅把当前窗口涉及的episode写入snapshot cache，但使用全局
            # 稳定leaf ID，避免不同窗口覆盖彼此的数据。
            logger.info(f"创建窗口叶子节点: {len(window_episode_leaf_ids)}")
            for episode_uuid, leaf_id in window_episode_leaf_ids.items():
                leaf_text = source_leaf_texts.get(
                    leaf_id,
                    (episode_content_map.get(episode_uuid, {}).get("content") or "").strip(),
                )
                cache[leaf_id] = {
                    "text": leaf_text,
                    "children": None,
                    "parent": None
                }
            logger.info(f"叶子节点创建完成，共 {len([k for k in cache.keys() if k.startswith('leaf_')])} 个leaf节点")

        # 打印汇总统计（仅在有cluster_stats时打印）
        if cluster_stats:
            print(f"\n===== 聚类汇总（时间窗口 {window_idx + 1}）=====")
            avg_entities = np.mean([s['entity_count'] for s in cluster_stats.values()])
            avg_chunks = np.mean([s['chunk_count'] for s in cluster_stats.values()])
            avg_chars = np.mean([s['total_chars'] for s in cluster_stats.values()])
            max_chars = max([s['total_chars'] for s in cluster_stats.values()])

            print(f"平均每聚类实体数: {avg_entities:.1f}")
            print(f"平均每聚类文本块数: {avg_chunks:.1f}")
            print(f"平均每聚类字符数: {avg_chars:.0f}")
            print(f"最大聚类字符数: {max_chars}")

        # 6. 按需求 A：tree.json 只保留 leaf 节点（真实 chunk），persona/topic 只写 snapshots.json
        logger.info("树结构已构建完成（tree.json 仅包含 leaf 节点；persona/topic 写入 snapshots.json）")

        # 8. 实体索引：snapshots_only 复用已有 cluster_index；否则优先使用传入的 index
        if snapshots_only:
            compatible_index = existing_cluster_index or {}
        else:
            compatible_index = index or {}

        # 收集topic和persona摘要
        persona_summaries = []
        topic_summaries = []

        # 从并行结果中提取 persona/topic 摘要（不再从 cache_tree 中提取）
        # 说明：node_id 仍保持旧命名，便于向后兼容与调试定位。
        for idx, text in enumerate(first_layer_summaries):
            persona_summaries.append({
                "node_id": f"summary_0_{idx}",
                "text": text,
                "stats": cluster_stats.get(idx, {}) if cluster_stats else {}
            })
        for idx, text in enumerate(first_layer_entity_summaries):
            topic_summaries.append({
                "node_id": f"entity_summary_0_{idx}",
                "text": text
            })

        # 创建snapshot（tree_only模式跳过，复用现有snapshots）
        # 注意：即使没有聚类结果（cluster_stats为空），也应该创建snapshot以保证时间窗口的连续性
        if not tree_only:
            snapshot = {
                "snapshot_id": f"snapshot_{window_idx}",
                "valid_from": window_start.isoformat() if window_start else None,
                "valid_to": window_end.isoformat() if window_end else None,
                "persona_summaries": persona_summaries,
                "topic_summaries": topic_summaries,
                "cache": cache,
                "cluster_index": compatible_index,
                "entity_count": len(entities),
                "cluster_count": len(clusters)
            }
            all_snapshots.append(snapshot)
        if not snapshots_only:
            all_caches.update(cache)  # 合并所有cache
            all_cluster_indices.update(compatible_index)  # 合并所有索引

        logger.info(f"时间窗口 {window_idx + 1} 完成，生成 {len(persona_summaries)} 个persona摘要，{len(topic_summaries)} 个topic摘要")

    # 保存所有snapshots（tree_only模式不覆盖）
    if all_snapshots and not tree_only:
        with open(snapshots_path, "w", encoding="utf-8") as f:
            json.dump(all_snapshots, f, ensure_ascii=False, indent=4)
        logger.info(f"保存了 {len(all_snapshots)} 个snapshots到 {snapshots_path}")

    # snapshots_only/tree_only：不覆盖对应的文件
    if not snapshots_only:
        # Episodes without a temporal RELATES_TO edge must remain searchable
        # as raw source chunks.  Add the complete source corpus once after
        # window processing; this is separate from snapshot-local caches.
        for leaf_id, leaf_text in source_leaf_texts.items():
            all_caches.setdefault(leaf_id, {
                "text": leaf_text,
                "children": None,
                "parent": None,
            })
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(all_caches, f, ensure_ascii=False, indent=4)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(all_cluster_indices, f, ensure_ascii=False, indent=4)
        logger.info(f"聚类索引已保存到: {index_path}")

    build_time = time.time() - start_time
    if snapshots_only:
        logger.info(
            f"snapshots_only完成：生成 {len(all_snapshots)} 个snapshots，耗时 {build_time:.2f} 秒（未覆盖tree/index）"
        )
        return existing_cache_tree or {}, build_time, existing_cluster_index or {}

    if tree_only:
        logger.info(
            f"tree_only完成：重新构建 tree/index，耗时 {build_time:.2f} 秒（未覆盖snapshots.json，保留了 {len(existing_snapshots) if existing_snapshots else 0} 个snapshots）"
        )

    logger.info(f"树构建完成，共 {len(all_caches)} 个节点，{len(all_cluster_indices)} 个实体索引，{len(all_snapshots)} 个snapshots，耗时 {build_time:.2f} 秒")
    return all_caches, build_time, all_cluster_indices

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
    original_text: List = None,  # 原始文本（包含full_text和user_only_text的字典列表）
    snapshots_only: bool = False,
    tree_only: bool = False,
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
        original_text=original_text,
        snapshots_only=snapshots_only,
        tree_only=tree_only,
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
