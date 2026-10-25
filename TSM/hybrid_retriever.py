# -*- coding: utf-8 -*-
"""
HybridRetriever: E-2GraphRAG 与 Graphiti 的混合检索系统

这个模块实现了一个结合 Cypher 查询和传统 Python 检索逻辑的混合检索器，
旨在利用 Graphiti 的图数据库优势同时保持 E-2GraphRAG 的灵活性。
"""

import logging
import asyncio
import time
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 强制启用DEBUG级别

class GraphitiGraphRetriever:
    """
    Graphiti 图检索器，负责处理所有基于 Cypher 的图查询
    """

    def __init__(self, graphiti_client, group_id: str, config_bundle):
        """
        初始化 Graphiti 图检索器

        Args:
            graphiti_client: Graphiti 客户端实例
            group_id: 图数据组ID，用于数据隔离
        """
        self.graphiti = graphiti_client
        self.group_id = group_id

        # 从配置中获取连接信息
        graphiti_config = config_bundle.graphiti_config
        if not graphiti_config:
            uri = 'bolt://localhost:7687'
            user = 'neo4j'
            password = '12345678'
        else:
            uri = graphiti_config.get('uri', 'bolt://localhost:7687')
            user = graphiti_config.get('user', 'neo4j')
            password = graphiti_config.get('password', '12345678')
        auth = (user, password)

        # 创建同步驱动
        from neo4j import GraphDatabase
        self.sync_driver = GraphDatabase.driver(uri, auth=auth)

    def shortest_path_cypher(self, entities: List[str], k: int = 4) -> List[Tuple[str, str]]:
        """
        使用同步 Cypher 查询实体间的最短路径

        Args:
            entities: 实体名称列表
            k: 最大路径长度

        Returns:
            实体对列表，表示路径长度 <= k 的连接
        """
        if not entities or len(entities) < 2:
            return []

        # 构建 Cypher 查询
        cypher_query = f"""
        MATCH path = shortestPath((e1:Entity)-[*..{k}]-(e2:Entity))
        WHERE e1.name IN $entities
          AND e2.name IN $entities
          AND e1.name < e2.name
          AND e1.group_id = $group_id
          AND e2.group_id = $group_id
          AND length(path) <= $k
        RETURN e1.name as source, e2.name as target, length(path) as distance
        ORDER BY distance
        """

        try:
            logger.info(f"执行同步 Cypher 最短路径查询: {len(entities)} 个实体, k={k}")
            logger.debug(f"查询实体: {entities}")
            logger.debug(f"完整 Cypher 查询: {cypher_query}")

            with self.sync_driver.session(database='neo4j') as session:
                result = session.run(
                    cypher_query,
                    entities=entities,
                    k=k,
                    group_id=self.group_id
                )

                paths = []
                for record in result:
                    paths.append((record.get('source'), record.get('target')))

                logger.debug(f"找到 {len(paths)} 条最短路径: {paths}")
                return paths

        except Exception as e:
            logger.error(f"同步 Cypher 最短路径查询失败: {e}")
            return []

    def entity_to_chunks_cypher(self, entities: List[str]) -> Dict[str, List[str]]:
        """
        使用同步 Cypher 查询实体到文本块的映射

        Args:
            entities: 实体名称列表

        Returns:
            实体名称到文本块UUID列表的映射
        """
        if not entities:
            return {}

        cypher_query = """
        MATCH (e:Entity)-[:MENTIONS]-(chunk:Episodic)
        WHERE e.name IN $entities AND e.group_id = $group_id AND chunk.group_id = $group_id
        RETURN e.name as entity, collect(DISTINCT chunk.uuid) as chunk_uuids
        """

        try:
            logger.info(f"执行同步 Cypher 实体索引查询: {len(entities)} 个实体")
            logger.debug(f"查询实体: {entities}")

            with self.sync_driver.session(database='neo4j') as session:
                result = session.run(
                    cypher_query,
                    entities=entities,
                    group_id=self.group_id
                )

                mapping = {}
                for record in result:
                    entity = record.get('entity')
                    chunk_uuids = record.get('chunk_uuids', [])
                    mapping[entity] = chunk_uuids

                logger.debug(f"实体到 chunks 映射: {mapping}")
                return mapping

        except Exception as e:
            logger.error(f"同步 Cypher 实体索引查询失败: {e}")
            return {}


class HybridRetriever:
    """
    混合检索器：结合 Cypher 图查询和传统 Python 检索逻辑
    """

    def __init__(self, cache_tree, G: nx.Graph, index, appearance_count: Dict[str, int],
                 nlp, graphiti_client=None, group_id: str = None, config_bundle=None, **kwargs):
        """
        初始化混合检索器

        Args:
            cache_tree: 文档摘要树
            G: NetworkX 图 (保留用于回退)
            index: 实体到文本块索引
            appearance_count: 实体出现计数
            graphiti_client: Graphiti 客户端实例
            group_id: 图数据组ID
            **kwargs: 其他参数
        """
        # 继承原有的 Retriever 属性
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.inverse_index = self.get_inverse_index()
        self.external_text_map = kwargs.get("external_text_map", {})
        self.nlp = nlp

        # 设置参数
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)

        # Graphiti 相关配置
        self.use_cypher = graphiti_client is not None
        self.group_id = group_id

        if self.use_cypher:
            self.graph_retriever = GraphitiGraphRetriever(graphiti_client, group_id, config_bundle)
            logger.info("混合检索器初始化：启用 Cypher 查询支持")
        else:
            self.graph_retriever = None
            logger.info("混合检索器初始化：仅使用 NetworkX 查询")

        # FAISS 索引设置 (保持不变)
        if kwargs.get("prebuilt_faiss_index") is not None:
            self.faiss_index = kwargs.get("prebuilt_faiss_index")
            self.prebuilt_doc_ids = kwargs.get("prebuilt_doc_ids", [])
            self._setup_embedder(kwargs)
        elif kwargs.get("embedder", kwargs.get("tokenizer")) is not None:
            self._setup_embedder(kwargs)
            self.faiss_index = self._build_faiss_index()
            self.prebuilt_doc_ids = None
        else:
            logger.warning("未设置 embedder，密集检索功能不可用")
            self.embedder = None
            self.faiss_index = None
            self.prebuilt_doc_ids = None

    def _setup_embedder(self, kwargs):
        """设置嵌入器"""
        try:
            from provider.public_embedder import PublicEmbedder

            self.embedding_batch_size = kwargs.get("embedding_batch_size", 128)

            self.embedder = PublicEmbedder.from_config(kwargs.get("embedding_config"))
            logger.info("成功初始化公开 embedding 后端")
        except ImportError:
            logger.warning("无法导入 embedding 后端，使用回退方案")
            self.embedder = None

    # ============ 核心混合检索方法 ============

    def hybrid_local_retrieval(self, entities: List[str], shortest_path_k: int = 4) -> Dict[str, List[str]]:
        """
        混合本地检索：优先使用 Cypher，回退到 NetworkX

        Args:
            entities: 实体列表
            shortest_path_k: 最短路径最大长度

        Returns:
            实体组合到文本块列表的映射
        """
        if not entities:
            return {}

        logger.info(f"开始混合本地检索: {len(entities)} 个实体, k={shortest_path_k}")

        # 优先使用 Cypher 查询
        if self.use_cypher and self.graph_retriever:
            try:
                return self._cypher_local_retrieval(entities, shortest_path_k)
            except Exception as cypher_error:
                logger.warning(f"Cypher 检索失败，回退到 NetworkX: {cypher_error}")

        # 回退到传统 NetworkX 方法
        return self._networkx_local_retrieval(entities, shortest_path_k)

    def _cypher_local_retrieval(self, entities: List[str], shortest_path_k: int) -> Dict[str, List[str]]:
        """使用同步 Cypher 的本地检索实现"""
        start_time = time.time()

        # 1. 获取最短路径
        shortest_paths = self.graph_retriever.shortest_path_cypher(entities, shortest_path_k)

        # 2. 获取实体到文本块的映射
        # entity_chunks = self.graph_retriever.entity_to_chunks_cypher(entities)
        entity_chunks = self.index_mapping(shortest_paths)  # 使用传统index_mapping方法

        # 3. 构建路径到文本块的映射
        path_chunks = {}
        all_chunks = set()

        for source, target in shortest_paths:
            # 合并源和目标的文本块
            source_chunks = entity_chunks.get(source, [])
            target_chunks = entity_chunks.get(target, [])
            combined = list(set(source_chunks + target_chunks))
            path_key = f"{source}->{target}"
            path_chunks[path_key] = combined
            all_chunks.update(combined)

        # 4. 处理未连接的实体
        connected_entities = set()
        for s, t in shortest_paths:
            connected_entities.add(s)
            connected_entities.add(t)

        for entity in entities:
            if entity not in connected_entities:
                path_chunks[entity] = entity_chunks.get(entity, [])

        cypher_time = time.time() - start_time
        logger.info(f"同步 Cypher 本地检索完成: {len(path_chunks)} 个路径, 耗时 {cypher_time:.3f}s")

        return path_chunks

    def _networkx_local_retrieval(self, entities: List[str], shortest_path_k: int) -> Dict[str, List[str]]:
        """使用 NetworkX 的本地检索实现（原始逻辑）"""
        start_time = time.time()

        # 使用原有的图过滤逻辑
        shortest_paths = self.graph_filter_networkx(entities, shortest_path_k)
        init_chunk_ids = self.index_mapping(shortest_paths)
        neighbor_nodes = self.merge_keys(init_chunk_ids)

        networkx_time = time.time() - start_time
        logger.info(f"NetworkX 本地检索完成: {len(neighbor_nodes)} 个路径, 耗时 {networkx_time:.3f}s")

        return neighbor_nodes

    # ============ 兼容性方法 ============

    def graph_filter_networkx(self, entities: List[str], k: int) -> List[Tuple[str, str]]:
        """原始的 NetworkX 图过滤方法"""
        shortest_path_pairs = []
        for head, tail in combinations(entities, 2):
            if head in self.G.nodes() and tail in self.G.nodes():
                try:
                    shortest_path = nx.shortest_path(self.G, head, tail)
                except nx.NetworkXNoPath:
                    continue
                if len(shortest_path) <= k:
                    shortest_path_pairs.append((head, tail))
        return shortest_path_pairs

    def local_retrieval(self, entities: List[str], shortest_path_k: int = 4) -> Dict[str, List[str]]:
        """
        保持与原始接口兼容的同步方法
        """
        if self.use_cypher:
            # 使用线程池执行器运行异步查询，避免事件循环冲突
            import asyncio
            import concurrent.futures

            def run_async_query():
                # 创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.hybrid_local_retrieval(entities, shortest_path_k)
                    )
                finally:
                    loop.close()

            # 使用线程池执行器
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_query)
                return future.result()
        else:
            # 直接使用 NetworkX 方法
            return self._networkx_local_retrieval(entities, shortest_path_k)

    # ============ 继承的原始方法 ============

    def _collapse_tree(self, cache_tree):
        """从原始 Retriever 继承的方法"""
        # 保持原有实现
        collapse_tree = []
        collapse_tree_ids = []

        def dfs(node_id, content=""):
            node = cache_tree.get(node_id, {})
            current_content = node.get("content", "")

            if node_id.startswith("leaf_"):
                collapse_tree.append(current_content)
                collapse_tree_ids.append(node_id)
            else:
                children = node.get("children", [])
                # 确保children是列表类型，防止None值导致TypeError
                if children is None:
                    children = []
                for child_id in children:
                    dfs(child_id, current_content)

        # 从根节点开始遍历
        if cache_tree:
            root_keys = [k for k in cache_tree.keys() if not k.startswith("leaf_")]
            if root_keys:
                dfs(root_keys[0])

        return collapse_tree, collapse_tree_ids

    def get_inverse_index(self):
        """从原始 Retriever 继承的方法"""
        inverse_index = defaultdict(list)
        for entity, chunks in self.index.items():
            for chunk in chunks:
                inverse_index[chunk].append(entity)
        return dict(inverse_index)

    def index_mapping(self, entities: List) -> Dict[str, List[str]]:
        """从原始 Retriever 继承的方法"""
        chunk_ids = {}

        for entity in entities:
            if isinstance(entity, str):
                if entity in self.index.keys():
                    chunk_ids[entity] = self.index[entity]
            elif isinstance(entity, tuple):
                chunk_ids_set = set()
                entity_key = "_".join(entity)
                for e in entity:
                    if e in self.index.keys():
                        if chunk_ids_set == set():
                            chunk_ids_set = set(self.index[e])
                        else:
                            chunk_ids_set = chunk_ids_set & set(self.index[e])
                chunk_ids[entity_key] = sorted(list(chunk_ids_set))

        return chunk_ids

    def merge_keys(self, init_chunk_ids: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """从原始 Retriever 继承的方法"""
        # 实现合并逻辑
        merged = defaultdict(list)

        for key, chunks in init_chunk_ids.items():
            # 简化版本的合并逻辑
            merged[key].extend(chunks)

        return dict(merged)

    def _build_faiss_index(self):
        """从原始 Retriever 继承的 FAISS 索引构建方法"""
        if not self.embedder:
            return None

        logger.info("构建 FAISS 索引...")

        try:
            import faiss

            # 获取文档
            docs = [text for text in self.collapse_tree if text.strip()]

            if not docs:
                logger.warning("没有有效文档用于构建 FAISS 索引")
                return None

            # 创建嵌入
            if hasattr(self.embedder, 'embed_texts'):
                # Public asynchronous embedding method.
                try:
                    loop = asyncio.get_event_loop()
                    import nest_asyncio
                    nest_asyncio.apply()
                    doc_embeds = loop.run_until_complete(self.embedder.embed_texts(docs))
                except RuntimeError:
                    doc_embeds = asyncio.run(self.embedder.embed_texts(docs))

                doc_embeds = np.ascontiguousarray(np.array(doc_embeds, dtype='float32'))
            else:
                # 同步方法
                doc_embeds = self.embedder.encode(docs, batch_size=16, device=self.device)

            # 创建 FAISS 索引
            if doc_embeds.ndim != 2 or doc_embeds.shape[0] == 0:
                logger.error(f"无效的嵌入形状: {doc_embeds.shape}")
                return None

            vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
            vector_database.add(doc_embeds)

            logger.info(f"FAISS 索引构建完成: {len(docs)} 个文档, {doc_embeds.shape[1]} 维")
            return vector_database

        except Exception as e:
            logger.error(f"构建 FAISS 索引失败: {e}")
            return None

    def dense_retrieval(self, query: str, k: int):
        """从原始 Retriever 继承的密集检索方法"""
        if self.faiss_index is None:
            logger.warning("FAISS 索引不可用，返回空结果")
            return {"": []}

        try:
            # 获取查询嵌入
            if hasattr(self.embedder, 'create'):
                query_embed = self.embedder.create([query.replace('\n', ' ')])
                query_embed = query_embed[0] if isinstance(query_embed, list) and len(query_embed) > 0 else query_embed
            else:
                query_embed = self.embedder.encode([query], device=self.device)[0]

            # 转换为 FAISS 格式
            query_embed = np.ascontiguousarray(np.array(query_embed, dtype='float32').reshape(1, -1))

            # 搜索
            _, candidate_indices = self.faiss_index.search(query_embed, k=k)
            candidate_indices = candidate_indices[0]

            # 获取候选文本块
            candidate_chunk_ids = [self.collapse_tree_ids[i] for i in candidate_indices if i < len(self.collapse_tree_ids)]

            return {"": candidate_chunk_ids}

        except Exception as e:
            logger.error(f"密集检索失败: {e}")
            return {"": []}

    # ============ 高级检索方法 ============

    def _count_chunks(self, res: Dict[str, List[str]]) -> int:
        """计算文本块总数"""
        return sum(len(chunks) for chunks in res.values())

    async def async_query(self, query: str, entities: List[str], **kwargs) -> Dict[str, Any]:
        """
        异步混合查询接口
        """
        start_time = time.time()

        # 本地检索
        local_result = self.hybrid_local_retrieval(entities, kwargs.get('shortest_path_k', 4))

        # 密集检索（如果启用）
        dense_result = {}
        if self.embedder and self.faiss_index:
            dense_result = self.dense_retrieval(query, kwargs.get('dense_k', 5))

        # 4. 格式化结果
        total_time = time.time() - start_time
        result = {"chunks": self.format_res(local_result)} # Use local_result here

        if kwargs.get("debug", True):
            result.update({
                "retrieval_type": "Hybrid Local", # Changed from "Hybrid Local (k=...)"
                "total_time": total_time,
                "entities_found": len(entities),
                "chunks_retrieved": self._count_chunks(local_result), # Use local_result here
                "cypher_enabled": self.use_cypher
            })

        logger.info(f"混合检索完成: Hybrid Local, 耗时 {total_time:.3f}s")
        return result

    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        同步查询接口：兼容传统Retriever接口

        Args:
            query: 查询字符串
            **kwargs: 其他参数

        Returns:
            查询结果字典
        """
        # 使用与传统Retriever相同的实体提取逻辑
        from extract_graph import naive_extract_graph
        entities = naive_extract_graph(query.split("\n")[0], self.nlp)
        entities = entities["nouns"]

        logger.info(f"同步查询接口: 从查询中提取到 {len(entities)} 个实体")

        # 调用异步方法并处理事件循环
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.async_query(query, entities, **kwargs))
            else:
                return loop.run_until_complete(self.async_query(query, entities, **kwargs))
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(self.async_query(query, entities, **kwargs))

    def _apply_entity_filter(self, dense_result: Dict[str, List[str]], entities: List[str]) -> Dict[str, List[str]]:
        """应用实体过滤逻辑 (简化版本)"""
        # 这里可以实现更复杂的实体感知过滤
        # 目前返回原结果
        return dense_result

    def format_res(self, res: Dict[str, List[str]]) -> str:
        """格式化检索结果"""
        res_str = ""
        for key, chunks in res.items():
            for chunk_id in chunks:
                # 获取文本块内容
                chunk_text = self._get_chunk_text(chunk_id)
                res_str += f"{chunk_text}\n"
        return res_str

    def _get_chunk_text(self, chunk_id: str) -> str:
        """获取文本块内容"""
        # 优先从外部文本映射获取
        if chunk_id in self.external_text_map:
            return self.external_text_map[chunk_id]

        # 从缓存树获取
        if chunk_id in self.cache_tree:
            return self.cache_tree[chunk_id].get("content", "")

        # 从折叠树获取
        try:
            if chunk_id.startswith("leaf_"):
                index = int(chunk_id.split("_")[1])
                if 0 <= index < len(self.collapse_tree):
                    return self.collapse_tree[index]
        except (ValueError, IndexError):
            pass

        return f"[未找到文本块内容: {chunk_id}]"


# ============ 兼容性函数 ============

def create_hybrid_retriever(cache_tree, G, index, appearance_count, nlp, graphiti_client=None, group_id: str = None, config_bundle=None, **kwargs) -> HybridRetriever:
    return HybridRetriever(
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
