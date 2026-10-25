# -*- coding: utf-8 -*-
"""
混合检索集成模块

这个模块提供了与现有 E-2GraphRAG 系统的集成接口，
使得可以在不修改大量代码的情况下使用新的混合检索功能。
"""

import logging
from typing import Dict, Any, Optional
from hybrid_retriever import HybridRetriever, create_hybrid_retriever

logger = logging.getLogger(__name__)

def integrate_hybrid_retriever(retriever_instance, graphiti_client=None, group_id: str = None, config_bundle=None) -> HybridRetriever:
    """
    将现有的 Retriever 实例升级为 HybridRetriever

    Args:
        retriever_instance: 现有的 Retriever 实例
        graphiti_client: Graphiti 客户端实例 (可选)
        group_id: 图数据组ID (可选)

    Returns:
        HybridRetriever 实例
    """
    logger.info("开始集成混合检索器...")

    # 提取现有 Retriever 的属性
    cache_tree = retriever_instance.cache_tree
    G = retriever_instance.G
    index = retriever_instance.index
    appearance_count = retriever_instance.appearance_count
    nlp = retriever_instance.nlp  # 提取nlp模型

    # 提取其他配置参数
    kwargs = {
        'external_text_map': getattr(retriever_instance, 'external_text_map', {}),
        'device': getattr(retriever_instance, 'device', 'cuda:0'),
        'merge_num': getattr(retriever_instance, 'merge_num', 5),
        'min_count': getattr(retriever_instance, 'min_count', 2),
        'embedder': getattr(retriever_instance, 'embedder', None),
        'faiss_index': getattr(retriever_instance, 'faiss_index', None),
        'prebuilt_faiss_index': getattr(retriever_instance, 'faiss_index', None),
        'prebuilt_doc_ids': getattr(retriever_instance, 'prebuilt_doc_ids', None),
        'embedding_batch_size': getattr(retriever_instance, 'embedding_batch_size', 1024),
    }

    # 创建混合检索器
    hybrid_retriever = create_hybrid_retriever(
        cache_tree=cache_tree,
        G=G,
        index=index,
        appearance_count=appearance_count,
        nlp=nlp,
        graphiti_client=graphiti_client,
        group_id=group_id,
        config_bundle=config_bundle,
        **kwargs
    )

    logger.info(f"混合检索器集成完成，Cypher 支持: {'启用' if graphiti_client else '禁用'}")
    return hybrid_retriever


def patch_retriever_methods(retriever_instance, hybrid_retriever: HybridRetriever):
    """
    用混合检索器的方法替换现有检索器的方法

    Args:
        retriever_instance: 现有的 Retriever 实例
        hybrid_retriever: 新的 HybridRetriever 实例
    """
    logger.info("开始替换检索器方法...")

    # 替换关键方法
    retriever_instance.local_retrieval = hybrid_retriever.local_retrieval
    retriever_instance.dense_retrieval = hybrid_retriever.dense_retrieval

    # 如果有异步查询方法，也替换它
    if hasattr(hybrid_retriever, 'query'):
        retriever_instance.hybrid_query = hybrid_retriever.query

    # 保留原有方法的备份
    if not hasattr(retriever_instance, '_original_local_retrieval'):
        retriever_instance._original_local_retrieval = getattr(retriever_instance, 'graph_filter', None)

    logger.info("检索器方法替换完成")


class HybridQueryAdapter:
    """
    混合查询适配器，提供与原始查询接口兼容的功能
    """

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever

    async def query_async(self, query: str, entities: list, **kwargs) -> Dict[str, Any]:
        """
        异步查询方法

        Args:
            query: 查询字符串
            entities: 实体列表
            **kwargs: 其他参数

        Returns:
            查询结果字典
        """
        return await self.hybrid_retriever.async_query(query, entities, **kwargs)

    def query_sync(self, query: str, entities: list, **kwargs) -> Dict[str, Any]:
        """
        同步查询方法

        Args:
            query: 查询字符串
            entities: 实体列表
            **kwargs: 其他参数

        Returns:
            查询结果字典
        """
        import asyncio

        try:
            # 尝试在现有事件循环中运行
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.hybrid_retriever.async_query(query, entities, **kwargs)
                )
            else:
                return loop.run_until_complete(
                    self.hybrid_retriever.async_query(query, entities, **kwargs)
                )
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(
                self.hybrid_retriever.async_query(query, entities, **kwargs)
            )

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            'cypher_enabled': self.hybrid_retriever.use_cypher,
            'group_id': self.hybrid_retriever.group_id,
            'faiss_available': self.hybrid_retriever.faiss_index is not None,
            'embedder_available': self.hybrid_retriever.embedder is not None,
        }


def create_graphiti_context(config_bundle) -> Optional[tuple]:
    """
    从配置中创建 Graphiti 上下文

    Args:
        config_bundle: E-2GraphRAG 配置bundle

    Returns:
        (graphiti_client, group_id) 元组，如果失败返回 None
    """
    try:
        # 检查是否有 Graphiti 配置
        if not hasattr(config_bundle, 'graphiti_config'):
            logger.info("配置中未找到 Graphiti 配置，使用纯 NetworkX 模式")
            return None

        graphiti_config = config_bundle.graphiti_config

        # 检查是否启用图抽取方法为 graphiti
        extraction_method = getattr(config_bundle, 'extraction_method', 'spacy')
        if extraction_method != 'graphiti':
            logger.info(f"图抽取方法为 '{extraction_method}'，不使用 Graphiti 检索")
            return None

        # 尝试导入和初始化 Graphiti
        try:
            from graphiti_extractor import GraphitiExtractor, GRAPHITI_AVAILABLE

            if not GRAPHITI_AVAILABLE:
                logger.warning("Graphiti 不可用，回退到 NetworkX 模式")
                return None

            # 创建 GraphitiExtractor 实例来获取 Graphiti 客户端
            extractor = GraphitiExtractor(config_bundle)

            # 使用同步方式初始化Graphiti，避免事件循环冲突
            import asyncio
            import threading
            import concurrent.futures

            def init_graphiti_sync():
                """同步初始化Graphiti"""
                # 创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # 同步运行异步初始化
                    async def async_init():
                        await extractor._initialize_graphiti()
                        return extractor.graphiti

                    # 在新的事件循环中运行
                    graphiti_client = loop.run_until_complete(async_init())
                    return graphiti_client
                finally:
                    loop.close()

            # 使用线程池执行器来运行初始化
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_graphiti_sync)
                graphiti_client = future.result()

            # 生成组ID
            group_id = getattr(config_bundle, 'group_id', f"e2graphrag_{hash(str(config_bundle)) % 10000}")

            logger.info(f"成功创建 Graphiti 上下文，组ID: {group_id}")
            return graphiti_client, group_id

        except ImportError as e:
            logger.warning(f"无法导入 Graphiti 相关模块: {e}")
            return None

    except Exception as e:
        logger.error(f"创建 Graphiti 上下文失败: {e}")
        return None


def apply_hybrid_retrieval(retriever_instance, config_bundle, force_graphiti: bool = False) -> HybridRetriever:
    """
    应用混合检索功能到现有的检索器实例

    Args:
        retriever_instance: 现有的 Retriever 实例
        config_bundle: E-2GraphRAG 配置bundle
        force_graphiti: 是否强制使用 Graphiti（忽略配置检查）

    Returns:
        HybridRetriever 实例
    """
    logger.info("开始应用混合检索功能...")

    graphiti_client = None
    group_id = None

    # 尝试创建 Graphiti 上下文
    if force_graphiti or hasattr(config_bundle, 'graphiti_config'):
        graphiti_context = create_graphiti_context(config_bundle)
        if graphiti_context:
            graphiti_client, group_id = graphiti_context

    # 创建混合检索器
    hybrid_retriever = integrate_hybrid_retriever(
        retriever_instance,
        graphiti_client=graphiti_client,
        group_id=group_id,
        config_bundle=config_bundle
    )

    # 创建查询适配器
    query_adapter = HybridQueryAdapter(hybrid_retriever)

    # 添加适配器到混合检索器
    hybrid_retriever.query_adapter = query_adapter

    logger.info("混合检索功能应用完成")
    return hybrid_retriever


# ============ 便捷函数 ============

def quick_hybrid_upgrade(retriever_instance, config_bundle):
    """
    快速升级现有检索器为混合检索器

    Args:
        retriever_instance: 现有的 Retriever 实例
        config_bundle: E-2GraphRAG 配置bundle

    Returns:
        升级后的 HybridRetriever 实例
    """
    return apply_hybrid_retrieval(retriever_instance, config_bundle)


def create_hybrid_from_scratch(cache_tree, G, index, appearance_count, config_bundle, **kwargs):
    """
    从头创建混合检索器

    Args:
        cache_tree: 文档摘要树
        G: NetworkX 图
        index: 实体索引
        appearance_count: 实体出现计数
        config_bundle: E-2GraphRAG 配置bundle
        **kwargs: 其他参数

    Returns:
        HybridRetriever 实例
    """
    # 加载nlp模型
    from extract_graph import load_nlp
    nlp = load_nlp()

    # 尝试创建 Graphiti 上下文
    graphiti_context = create_graphiti_context(config_bundle)
    graphiti_client, group_id = graphiti_context if graphiti_context else (None, None)

    # 创建混合检索器
    return create_hybrid_retriever(
        cache_tree=cache_tree,
        G=G,
        index=index,
        appearance_count=appearance_count,
        nlp=nlp,
        graphiti_client=graphiti_client,
        group_id=group_id,
        **kwargs
    )