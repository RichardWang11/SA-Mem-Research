"""
密集检索索引预构建模块
用于在预处理阶段构建和保存FAISS索引，提高检索启动速度
"""

import os
import json
import time
import hashlib
import logging
import numpy as np
import faiss
import asyncio
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
from provider.public_embedder import PublicEmbedder

logger = logging.getLogger("E2GraphRAG")


def get_tree_hash(tree: Dict) -> str:
    """计算树结构的哈希值，用于验证缓存有效性"""
    tree_str = json.dumps(tree, sort_keys=True)
    return hashlib.md5(tree_str.encode()).hexdigest()


def extract_docs_from_tree(tree: Dict) -> Tuple[List[str], List[str]]:
    """从树结构中提取所有文档内容和ID"""
    docs = []
    doc_ids = []

    # 提取所有leaf和summary节点的文本
    for node_id, node_data in tree.items():
        if isinstance(node_data, dict) and "text" in node_data:
            docs.append(node_data["text"])
            doc_ids.append(node_id)

    return docs, doc_ids


def load_dense_index_cache(cache_folder: str) -> Tuple[Optional[faiss.Index], Optional[List[str]], Optional[str]]:
    """
    加载预构建的密集索引缓存

    Returns:
        (index, doc_ids, tree_hash) 或 (None, None, None) 如果缓存不存在
    """
    index_path = os.path.join(cache_folder, "dense_index.faiss")
    mapping_path = os.path.join(cache_folder, "dense_index_mapping.json")

    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        return None, None, None

    try:
        # 加载FAISS索引
        index = faiss.read_index(index_path)

        # 加载文档ID映射和元数据
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            doc_ids = mapping_data.get("doc_ids", [])
            tree_hash = mapping_data.get("tree_hash", "")

        logger.info(f"成功加载预构建索引，包含 {len(doc_ids)} 个文档")
        return index, doc_ids, tree_hash

    except Exception as e:
        logger.error(f"加载密集索引缓存失败: {e}")
        return None, None, None


def save_dense_index_cache(cache_folder: str, index: faiss.Index, doc_ids: List[str], tree_hash: str):
    """保存密集索引到缓存"""
    os.makedirs(cache_folder, exist_ok=True)

    index_path = os.path.join(cache_folder, "dense_index.faiss")
    mapping_path = os.path.join(cache_folder, "dense_index_mapping.json")

    try:
        # 保存FAISS索引
        faiss.write_index(index, index_path)

        # 保存文档ID映射和元数据
        mapping_data = {
            "doc_ids": doc_ids,
            "tree_hash": tree_hash,
            "created_at": time.time(),
            "num_docs": len(doc_ids),
            "index_type": "IndexFlatIP"  # Inner Product for similarity
        }

        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)

        logger.info(f"密集索引已保存: {len(doc_ids)} 个文档")

    except Exception as e:
        logger.error(f"保存密集索引失败: {e}")
        # 清理部分保存的文件
        for path in [index_path, mapping_path]:
            if os.path.exists(path):
                os.remove(path)
        raise


def validate_index_cache(cache_folder: str, current_tree: Dict) -> bool:
    """
    验证索引缓存是否有效

    Returns:
        True if cache is valid and can be used, False otherwise
    """
    _, _, cached_tree_hash = load_dense_index_cache(cache_folder)

    if cached_tree_hash is None:
        return False

    current_tree_hash = get_tree_hash(current_tree)

    if cached_tree_hash != current_tree_hash:
        logger.info("树结构已变化，需要重建索引")
        return False

    return True


def embed_documents_batch(embedder, docs: List[str], batch_size: int = 1024) -> np.ndarray:
    """批量编码文档 - 改为同步执行"""
    embeddings = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]

        if hasattr(embedder, 'embed_texts'):
            # Public async embedder interface, bridged into this synchronous stage.
            try:
                import asyncio
                # 在同步环境中运行异步函数
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    batch_embeds = loop.run_until_complete(embedder.embed_texts(batch))
                    embeddings.extend(batch_embeds)
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"批次 {i//batch_size} 编码失败: {e}")
                # 使用零向量作为fallback
                dimension = int(getattr(embedder, "dimension", 1536))
                embeddings.extend([[0.0] * dimension for _ in batch])
        else:
            # SentenceTransformer同步接口
            batch_embeds = embedder.encode(batch, batch_size=16)
            embeddings.extend(batch_embeds.tolist())

    return np.array(embeddings, dtype='float32')


def build_dense_index_task(args):
    """
    构建密集检索索引的任务函数

    Args:
        args: (tree, cache_folder, embedder_config, config_bundle)

    Returns:
        (index, doc_ids), time_cost
    """
    start_time = time.time()

    # 解析参数
    if len(args) == 4:
        tree, cache_folder, embedder_config, config_bundle = args
    else:
        raise ValueError(f"期望4个参数，收到{len(args)}个")

    logger.info(f"开始构建密集索引，缓存文件夹: {cache_folder}")

    # 检查是否已有有效缓存
    if validate_index_cache(cache_folder, tree):
        logger.info("使用已有的密集索引缓存")
        index, doc_ids, _ = load_dense_index_cache(cache_folder)
        return (index, doc_ids), -1

    try:
        # 从树结构提取文档
        docs, doc_ids = extract_docs_from_tree(tree)
        logger.info(f"从树结构提取了 {len(docs)} 个文档")

        if len(docs) == 0:
            logger.warning("没有文档需要索引")
            return (None, []), 0

        # 初始化embedder
        if embedder_config.get("type") == "PublicEmbedder":
            embedder = PublicEmbedder.from_config(embedder_config)

            # 同步编码 - 与原版保持一致
            logger.info("开始批量编码文档...")
            doc_embeds = embed_documents_batch(
                embedder,
                docs,
                batch_size=embedder_config.get("batch_size", 1024)
            )

        else:
            # 使用SentenceTransformer
            from sentence_transformers import SentenceTransformer

            device = embedder_config.get("device", "cuda:0")
            model_name = embedder_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            embedder = SentenceTransformer(model_name, device=device)
            logger.info(f"使用SentenceTransformer编码 {len(docs)} 个文档...")
            doc_embeds = embedder.encode(docs, batch_size=16)

        # 确保编码结果有效
        doc_embeds = np.ascontiguousarray(doc_embeds.astype('float32'))

        if doc_embeds.ndim != 2:
            raise ValueError(f"编码结果维度错误: {doc_embeds.shape}")

        # 构建FAISS索引
        logger.info(f"构建FAISS索引，向量维度: {doc_embeds.shape[1]}")
        index = faiss.IndexFlatIP(doc_embeds.shape[1])
        index.add(doc_embeds)

        # 保存索引到缓存
        tree_hash = get_tree_hash(tree)
        save_dense_index_cache(cache_folder, index, doc_ids, tree_hash)

        end_time = time.time()
        time_cost = end_time - start_time
        logger.info(f"密集索引构建完成，耗时: {time_cost:.2f}秒")

        return (index, doc_ids), time_cost

    except Exception as e:
        logger.error(f"构建密集索引失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 返回空索引，让系统回退到实时构建
        return (None, []), 0
    finally:
        # 清理资源
        if 'embedder' in locals():
            del embedder


def clean_dense_index_cache(cache_folder: str):
    """清理密集索引缓存文件"""
    files_to_remove = [
        "dense_index.faiss",
        "dense_index_mapping.json"
    ]

    for filename in files_to_remove:
        file_path = os.path.join(cache_folder, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"已删除缓存文件: {filename}")
            except Exception as e:
                logger.error(f"删除缓存文件失败 {filename}: {e}")


if __name__ == "__main__":
    # 测试代码

    if len(sys.argv) > 1:
        test_cache_folder = sys.argv[1]

        # 测试加载缓存
        index, doc_ids, tree_hash = load_dense_index_cache(test_cache_folder)

        if index is not None:
            print(f"成功加载索引:")
            print(f"  - 文档数量: {len(doc_ids)}")
            print(f"  - 索引大小: {index.ntotal}")
            print(f"  - 树哈希: {tree_hash}")
        else:
            print("没有找到有效的索引缓存")
