# -*- coding: utf-8 -*-
"""
第三层消融：基于chunk聚类的树构建模块
直接对文本chunk进行聚类，生成传统summary作为树的第一层
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

# 导入现有模块
from utils import Timer
from prompt_dict import Prompts

logger = logging.getLogger(__name__)

# 从build_tree_clustering.py复制LLM调用函数
def _call_llm_unified(messages: List[dict], llm_config: dict, op_type: str | None = None) -> str:
    """Call the configured public OpenAI-compatible backend."""
    try:
        from provider.public_llm_provider import call_llm
        return call_llm(
            messages=messages,
            model=llm_config.get("backend_model", llm_config["model"]),
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 8192),
            base_url=llm_config.get("base_url"),
            api_key_env=llm_config.get("api_key_env", "OPENAI_API_KEY"),
            max_retries=llm_config.get("max_retries", 3),
            timeout=llm_config.get("timeout", 120),
            op_type=op_type,
        )
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return f"LLM调用失败: {str(e)}"

# 从build_tree_clustering.py复制聚类函数
def perform_clustering(embeddings: np.ndarray, dim: int = 2, threshold: float = 0.1, verbose: bool = True) -> List[np.ndarray]:
    """
    执行GMM聚类（复制自build_tree_clustering.py）

    参数:
        embeddings: 输入的embedding矩阵
        dim: UMAP降维目标维度
        threshold: 软聚类阈值
        verbose: 是否输出详细日志

    返回:
        每个实体的聚类标签列表
    """
    if len(embeddings) == 0:
        logger.warning("输入的embeddings为空")
        return []

    logger.info(f"开始聚类，输入维度: {embeddings.shape}")

    # 确定最优聚类数（使用BIC）
    n_entities = len(embeddings)
    # 设置最小聚类数：如果实体数较多，至少分成多个聚类
    # 对于大量实体（>100），至少分成5-10个聚类；对于中等数量（>50），至少3-5个；对于少量（>20），至少2个
    if n_entities > 100:
        min_clusters = min(10, n_entities // 10)
        max_clusters = min(25, n_entities // 4)
    elif n_entities > 50:
        min_clusters = min(5, n_entities // 10)
        max_clusters = min(15, n_entities // 3)
    elif n_entities > 20:
        min_clusters = 3
        max_clusters = min(10, n_entities // 2)
    elif n_entities > 10:
        min_clusters = 2
        max_clusters = min(5, n_entities // 2)
    else:
        min_clusters = 1
        max_clusters = min(3, n_entities)

    if verbose:
        logger.info(f"聚类数搜索范围: {min_clusters} 到 {min(max_clusters, len(embeddings))} (实体数: {len(embeddings)})")

    # 使用BIC选择最优聚类数
    bics = []
    max_clusters = min(max_clusters, len(embeddings))

    for n in range(min_clusters, max_clusters + 1):
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, max_iter=100)
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            bics.append(bic)
            if verbose:
                logger.debug(f"聚类数 {n}: BIC = {bic:.2f}")
        except Exception as e:
            logger.warning(f"GMM聚类失败 (n={n}): {e}")
            bics.append(float('inf'))

    if not bics or all(bic == float('inf') for bic in bics):
        logger.warning("所有聚类尝试都失败，使用单聚类")
        optimal_clusters = 1
    else:
        optimal_clusters = min_clusters + np.argmin(bics)
        # 确保至少是最小聚类数
        optimal_clusters = max(optimal_clusters, min_clusters)
        if verbose:
            logger.info(f"最优聚类数: {optimal_clusters} (BIC范围: {min(bics):.2f} - {max(bics):.2f})")

    # 执行GMM聚类
    gmm = GaussianMixture(n_components=optimal_clusters, random_state=42, max_iter=100)
    gmm.fit(embeddings)

    # 获取软聚类概率
    probabilities = gmm.predict_proba(embeddings)

    # 根据阈值确定每个实体所属的聚类
    cluster_labels = []
    for i, probs in enumerate(probabilities):
        assigned_clusters = np.where(probs >= threshold)[0]
        if len(assigned_clusters) == 0:
            # 如果没有聚类超过阈值，选择概率最高的
            assigned_clusters = [np.argmax(probs)]
        cluster_labels.append(assigned_clusters)

    if verbose:
        logger.info(f"聚类完成，生成 {optimal_clusters} 个聚类")

    return cluster_labels

def generate_chunk_cluster_summary(chunks: List[str], cluster_id: int, llm_config: dict) -> str:
    """
    基于聚类内的文本chunks生成传统摘要

    参数:
        chunks: 聚类中的文本块列表
        cluster_id: 聚类ID
        llm_config: LLM配置

    返回:
        生成的聚类摘要
    """
    if not chunks:
        logger.warning(f"聚类 {cluster_id} 文本块为空，生成占位摘要")
        return f"聚类 {cluster_id} 无有效文本内容"

    # 合并所有文本块
    combined_text = "\n\n".join(chunks)

    # 检查文本长度，如果过长需要截断
    max_length = 15000  # 设置最大长度
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length] + "...[文本已截断]"
        logger.warning(f"聚类 {cluster_id} 文本过长，已截断到 {max_length} 字符")

    # 生成传统摘要的prompt
    prompt = f"""请为以下文本内容生成一个简洁的摘要，重点概括主要信息和关键内容：

文本内容：
{combined_text}

要求：
1. 摘要应该简洁明了，长度控制在200-500字
2. 重点提取文本中的核心信息和主题
3. 保持客观中性的语调
4. 如果文本包含多个主题，请分别概括

摘要："""

    messages = [{"role": "user", "content": prompt}]

    try:
        summary = _call_llm_unified(messages, llm_config, op_type="summary_generation")
        if summary and len(summary.strip()) > 10:
            return summary.strip()
        else:
            logger.warning(f"聚类 {cluster_id} 摘要生成结果为空或过短")
            return f"聚类 {cluster_id} 的文本摘要（共{len(chunks)}个文本块）"
    except Exception as e:
        logger.error(f"聚类 {cluster_id} 摘要生成失败: {e}")
        return f"聚类 {cluster_id} 摘要生成失败"

def build_tree_with_chunk_clustering(
    text_chunks: List[str],
    llm_config: dict,
    cache_folder: str,
    merge_num: int = 5,
    language: str = "en",
    max_workers: int = 8,
    cluster_threshold: float = 0.1,
    original_text: Optional[List] = None,
    **kwargs
) -> Tuple[Dict, float]:
    """
    基于文本chunk聚类构建层次化树结构（第三层消融）

    参数:
        text_chunks: 输入的文本块列表
        llm_config: LLM配置
        cache_folder: 缓存文件夹
        merge_num: 合并数量
        language: 语言
        max_workers: 最大并发数
        cluster_threshold: 聚类阈值
        original_text: 原始文本（用于保存完整turn）

    返回:
        (树结构字典, 构建耗时)
    """
    timer = Timer()
    timer.start("total")

    logger.info("开始基于chunk聚类的树构建...")

    # 1. 对文本chunks进行embedding
    logger.info("对文本chunks进行embedding...")
    from provider.public_embedder import PublicEmbedder

    embedder = PublicEmbedder.from_config(kwargs.get("embedding_config"))

    # 异步获取embeddings
    async def get_embeddings():
        embeddings = await embedder.embed_texts(text_chunks)
        return np.array(embeddings, dtype='float32')

    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        chunk_embeddings = loop.run_until_complete(get_embeddings())
    except RuntimeError:
        chunk_embeddings = asyncio.run(get_embeddings())

    logger.info(f"获得 {len(chunk_embeddings)} 个chunk的embeddings")

    # 2. 执行聚类
    logger.info("执行chunk聚类...")
    cluster_labels = perform_clustering(
        chunk_embeddings,
        threshold=cluster_threshold,
        verbose=True
    )

    # 3. 根据聚类组织文本块
    logger.info("根据聚类结果组织文本块...")
    clusters = defaultdict(list)

    for chunk_idx, cluster_assignments in enumerate(cluster_labels):
        for cluster_id in cluster_assignments:
            clusters[cluster_id].append(chunk_idx)

    logger.info(f"形成 {len(clusters)} 个聚类")

    # 打印聚类统计信息
    print(f"\n===== Chunk聚类统计信息 =====")
    print(f"总聚类数: {len(clusters)}")
    for cluster_id, chunk_indices in clusters.items():
        print(f"聚类 {cluster_id}: {len(chunk_indices)} 个文本块")

    # 4. 为每个聚类生成传统摘要
    logger.info("开始并发生成聚类摘要...")

    def process_single_cluster(args):
        cluster_id, chunk_indices = args

        # 收集聚类中的所有文本
        cluster_chunks = [text_chunks[idx] for idx in chunk_indices]

        # 生成传统摘要
        summary = generate_chunk_cluster_summary(cluster_chunks, cluster_id, llm_config)

        return {
            "cluster_id": cluster_id,
            "summary": summary,
            "chunk_indices": chunk_indices,
            "chunk_count": len(chunk_indices)
        }

    cluster_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有聚类处理任务
        future_to_cluster = {
            executor.submit(process_single_cluster, (cluster_id, chunk_indices)): cluster_id
            for cluster_id, chunk_indices in clusters.items()
        }

        # 收集结果
        for future in tqdm(as_completed(future_to_cluster), total=len(future_to_cluster), desc="生成聚类摘要"):
            try:
                result = future.result()
                if result:
                    cluster_id = result["cluster_id"]
                    cluster_results[cluster_id] = result

                    print(f"\n聚类 {cluster_id}:")
                    print(f"  文本块数: {result['chunk_count']}")
                    print(f"  摘要: {result['summary'][:100]}...")
                else:
                    logger.warning(f"聚类处理结果为空，跳过")
            except Exception as e:
                cluster_id = future_to_cluster[future]
                logger.error(f"聚类 {cluster_id} 处理失败: {e}")

    # 5. 构建树结构
    logger.info("构建最终树结构...")
    tree_structure = {}

    # 为每个原始文本块创建leaf节点
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"leaf_{i}"

        # 如果有原始文本，使用原始文本；否则使用chunk文本
        if original_text and i < len(original_text):
            if isinstance(original_text[i], dict):
                # 保存完整的turn信息
                tree_structure[chunk_id] = {
                    "text": original_text[i].get("full_text", chunk_text),
                    "children": [],
                    "parent": None,
                    "metadata": {
                        "user_only_text": original_text[i].get("user_only_text", ""),
                        "session_id": original_text[i].get("session_id", ""),
                        "turn_index": original_text[i].get("turn_index", i)
                    }
                }
            else:
                tree_structure[chunk_id] = {
                    "text": original_text[i] if isinstance(original_text[i], str) else chunk_text,
                    "children": [],
                    "parent": None
                }
        else:
            tree_structure[chunk_id] = {
                "text": chunk_text,
                "children": [],
                "parent": None
            }

    # 为每个聚类创建summary节点
    for cluster_id, result in cluster_results.items():
        summary_id = f"summary_{cluster_id}"
        child_ids = [f"leaf_{idx}" for idx in result["chunk_indices"]]

        tree_structure[summary_id] = {
            "text": result["summary"],
            "children": child_ids,
            "parent": None,
            "metadata": {
                "type": "traditional_summary",
                "cluster_id": cluster_id,
                "chunk_count": result["chunk_count"]
            }
        }

        # 更新子节点的parent信息
        for child_id in child_ids:
            if child_id in tree_structure:
                tree_structure[child_id]["parent"] = summary_id

    # 6. 创建根节点
    summary_ids = [f"summary_{cluster_id}" for cluster_id in cluster_results.keys()]
    if summary_ids:
        tree_structure["root"] = {
            "text": f"文档根节点，包含{len(summary_ids)}个主题聚类的传统摘要",
            "children": summary_ids,
            "parent": None,
            "metadata": {
                "type": "root",
                "clustering_method": "chunk_clustering",
                "total_clusters": len(summary_ids)
            }
        }

        # 更新summary节点的parent信息
        for summary_id in summary_ids:
            tree_structure[summary_id]["parent"] = "root"

    # 7. 保存结果到缓存
    cache_path = os.path.join(cache_folder, "tree.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(tree_structure, f, ensure_ascii=False, indent=2)

    total_time = timer.stop("total")

    logger.info(f"基于chunk聚类的树构建完成，耗时: {total_time:.2f}秒")
    logger.info(f"生成了 {len(cluster_results)} 个聚类摘要，{len(text_chunks)} 个文本块")

    return tree_structure, total_time
