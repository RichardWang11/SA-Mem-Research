from extract_graph import naive_extract_graph
# from build_tree import sequential_merge
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from itertools import combinations
import networkx as nx
import faiss
import spacy
from collections import defaultdict
import os
from provider.public_embedder import PublicEmbedder
from neo4j import GraphDatabase
try:
    import torch
except ImportError:  # Used only for optional CUDA cache cleanup.
    torch = None
import random
random.seed(1)
import copy
import numpy as np
import re
import asyncio
import logging
import dateparser
from datetime import datetime, timezone
# search_dates已被移除，因为误判率太高（如"do", "did", "me"等被误判为日期）

from temporal_utils import (
    TemporalConstraint,
    extract_temporal_constraints,
    parse_iso_datetime,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Retriever:
    def __init__(self, cache_tree, G:nx.Graph, index, appearance_count:Dict[str, int], nlp:spacy.Language, **kwargs) -> None:
        # cache_tree: summary tree of the document.
        # G: graph of the document.
        # index: noun to chunks index. another index will be built in the get_inverse_index function.
        # appearance_count: appearance count of the entities in the chunks.
        # nlp: spacy model.
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        logger.info(f"DEV版本Retriever初始化: appearance_count类型={type(appearance_count)}, 长度={len(appearance_count) if appearance_count else 0}")
        if appearance_count:
            sample_keys = list(appearance_count.keys())[:5]
            logger.info(f"DEV版本appearance_count示例keys: {sample_keys}")
            for key in sample_keys:
                logger.info(f"DEV版本appearance_count[{key}]: {appearance_count[key]}")
        # get the inverse index, i.e., chunk_id to noun index.
        self.inverse_index = self.get_inverse_index()
        self.nlp = nlp
        # Optional external text map (e.g., from Graphiti) for chunk_ids not present in cache_tree
        self.external_text_map = kwargs.get("external_text_map", {})
        logger.info(f"DEV版本Retriever初始化：external_text_map类型={type(self.external_text_map)}, 长度={len(self.external_text_map) if isinstance(self.external_text_map, dict) else 'N/A'}")
        # set up the parameters.
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.temporal_filter_enabled = kwargs.get("enable_temporal_filter", True)
        self.keep_untimed_chunks = kwargs.get("keep_untimed_chunks", True)
        self.chunks_only = kwargs.get("chunks_only", False)  # 消融实验：只使用chunk节点

        # DEBUG: 记录temporal_filter_enabled的值
        logger.info(f"DEBUG: Retriever初始化 - temporal_filter_enabled: {self.temporal_filter_enabled} (类型: {type(self.temporal_filter_enabled)})")
        logger.info(f"DEBUG: Retriever初始化 - chunks_only: {self.chunks_only}")
        if not self.temporal_filter_enabled:
            logger.info("DEBUG: 时间过滤功能已禁用（消融实验设置）")
        self.graphiti_config = kwargs.get("graphiti_config", {})
        self.graphiti_group_id = kwargs.get("graphiti_group_id")
        self.keep_untimed_edges = kwargs.get("keep_untimed_edges", self.keep_untimed_chunks)
        self._graphiti_database = kwargs.get("graphiti_database")
        self._graphiti_sync_driver = kwargs.get("graphiti_sync_driver")
        self._owns_graphiti_driver = False
        self.last_temporal_edges = []
        self._last_temporal_constraint: Optional[TemporalConstraint] = None
        self._last_temporal_order: Optional[str] = None
        self._last_reference_date: Optional[datetime] = None

        if self._graphiti_sync_driver is None and self.graphiti_config:
            neo4j_config = self.graphiti_config.get("neo4j", self.graphiti_config)
            uri = os.getenv(
                neo4j_config.get("uri_env", "NEO4J_URI"),
                neo4j_config.get("uri", "bolt://localhost:7687"),
            )
            user = os.getenv(
                neo4j_config.get("user_env", "NEO4J_USER"),
                neo4j_config.get("user", "neo4j"),
            )
            password = os.getenv(
                neo4j_config.get("password_env", "NEO4J_PASSWORD"),
                neo4j_config.get("password", ""),
            )
            database = os.getenv(
                neo4j_config.get("database_env", "NEO4J_DATABASE"),
                neo4j_config.get("database", "neo4j"),
            )
            if uri:
                try:
                    self._graphiti_sync_driver = GraphDatabase.driver(
                        uri,
                        auth=(user or "", password or ""),
                        max_connection_pool_size=50,
                    )
                    self._owns_graphiti_driver = True
                    self._graphiti_database = self._graphiti_database or database
                    logger.info("DEV版本：已初始化Graphiti Neo4j同步驱动")
                except Exception as exc:
                    logger.warning(f"DEV版本：初始化Graphiti驱动失败，将回退到本地时间过滤: {exc}")
                    self._graphiti_sync_driver = None

        if self._graphiti_database is None:
            neo4j_config = self.graphiti_config.get("neo4j", self.graphiti_config)
            self._graphiti_database = (
                os.getenv(
                    neo4j_config.get("database_env", "NEO4J_DATABASE"),
                    neo4j_config.get("database", "neo4j"),
                )
                if neo4j_config
                else None
            )

        # GPU加速FAISS索引
        self.use_gpu_faiss = kwargs.get("use_gpu_faiss", True)  # 默认启用GPU加速
        self.gpu_id = kwargs.get("gpu_id", 0)  # 默认使用GPU 0
        # self.overlap = kwargs.get("overlap", 100)
        # self.tokenizer = kwargs.get("tokenizer",kwargs.get("tokenizer"))
        # self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        # 检查是否有预构建的FAISS索引
        if kwargs.get("prebuilt_faiss_index") is not None:
            # 使用预构建的索引
            prebuilt_index = kwargs.get("prebuilt_faiss_index")
            self.prebuilt_doc_ids = kwargs.get("prebuilt_doc_ids", [])

            # 如果启用GPU且索引是CPU索引，尝试转移到GPU
            if self.use_gpu_faiss and hasattr(faiss, 'StandardGpuResources'):
                try:
                    # 检查索引是否已经是GPU索引（通过检查是否有device属性）
                    is_gpu_index = hasattr(prebuilt_index, 'device') or str(type(prebuilt_index)).find('Gpu') != -1
                    if not is_gpu_index:
                        gpu_res = faiss.StandardGpuResources()
                        self.faiss_index = faiss.index_cpu_to_gpu(gpu_res, self.gpu_id, prebuilt_index)
                        logger.info(f"DEV版本：预构建FAISS索引已转移到GPU {self.gpu_id}")
                    else:
                        self.faiss_index = prebuilt_index
                        logger.info(f"DEV版本：预构建FAISS索引已在GPU上")
                except Exception as e:
                    logger.warning(f"DEV版本：预构建索引GPU转移失败，使用CPU: {e}")
                    self.faiss_index = prebuilt_index
            else:
                self.faiss_index = prebuilt_index
                logger.info(f"DEV版本：使用预构建CPU FAISS索引")
            # 仍然需要初始化embedder用于查询编码
            if kwargs.get("embedder", kwargs.get("tokenizer")) is not None:
                self.embedding_batch_size = kwargs.get("embedding_batch_size", 1024)

                # 从统一配置读取embedding参数
                embedding_config = kwargs.get("embedding_config", {})
                self.embedder = PublicEmbedder.from_config(embedding_config)
            else:
                print("Warning: embedder is None, dense retrieval will not work properly")
                self.embedder = None
        elif kwargs.get("embedder", kwargs.get("tokenizer")) is not None:
            # 没有预构建索引，实时构建
            self.embedding_batch_size = kwargs.get("embedding_batch_size", 128)

            # 从统一配置读取embedding参数
            embedding_config = kwargs.get("embedding_config", {})
            self.embedder = PublicEmbedder.from_config(embedding_config)
            self.faiss_index = self._build_faiss_index()
            self.prebuilt_doc_ids = None
        else:
            print("Warning: the embedder is set to None, dense retrieval is not implemented.")
            self.embedder = None
            self.faiss_index = None
            self.prebuilt_doc_ids = None

    def __del__(self):
        """Ensure proper cleanup of resources
           avoid the memory leak.
        """
        try:
            if hasattr(self, 'embedder'):
                del self.embedder
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
            if getattr(self, '_owns_graphiti_driver', False) and getattr(self, '_graphiti_sync_driver', None):
                try:
                    self._graphiti_sync_driver.close()
                except Exception as exc:
                    logger.info(f"DEV版本：关闭Graphiti驱动失败: {exc}")
            if torch is not None:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during Retriever cleanup: {e}")

    def update(self, cache_tree, G, index, appearance_count):
        # update the retriever, from a document to another document.
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.inverse_index = self.get_inverse_index()
        self.docs = self.collapse_tree
        # 只有在没有预构建索引时才重新构建
        if self.embedder is not None and not hasattr(self, 'prebuilt_doc_ids'):
            # Async public embedders expose create_batch.
            if hasattr(self.embedder, 'create_batch'):
                self.faiss_index = self._build_faiss_index()
            else:
                print("Warning: embedder does not have create_batch method")
        elif hasattr(self, 'prebuilt_doc_ids') and self.prebuilt_doc_ids is not None:
            # 使用预构建的索引，需要确保文档ID映射正确
            if self.collapse_tree_ids != self.prebuilt_doc_ids:
                print("Warning: Document IDs mismatch between tree and prebuilt index, rebuilding...")
                if self.embedder is not None:
                    self.faiss_index = self._build_faiss_index()
                    self.prebuilt_doc_ids = None

    def get_inverse_index(self):
        # get the inverse index, i.e., chunk_id to noun index.
        inverse_index = {}
        for key, value in self.index.items():
            for chunk_id in value:
                inverse_index.setdefault(chunk_id, []).append(key)
        return inverse_index

    def _collapse_tree(self, cache_tree:Dict[str, Dict]) -> Dict[str, Dict]:
        # collapse the tree. for dense retrieval
        # return the collapsed tree.
        collapsed_tree = []
        collapsed_tree_ids = []
        for i, (key, value) in enumerate(self.cache_tree.items()):
            # 消融实验：如果启用chunks_only，只保留chunk节点，排除summary_开头的persona节点
            if getattr(self, 'chunks_only', False):
                # 只保留leaf_开头或UUID格式的chunk，明确排除summary_开头的persona节点
                if (key.startswith("leaf_") or self._is_graphiti_chunk_id(key)) and not key.startswith("summary_"):
                    collapsed_tree.append(value["text"])
                    collapsed_tree_ids.append(key)
                    logger.info(f"DEV版本chunks_only: 保留chunk节点 {key}")
                else:
                    logger.info(f"DEV版本chunks_only: 过滤非chunk节点 {key}")
            else:
                collapsed_tree.append(value["text"])
                collapsed_tree_ids.append(key)

        return collapsed_tree, collapsed_tree_ids

    def _detect_neighbor_nodes(self, keys:Set[str], chunk_id: str) -> List[str]:
        # detect the neighbor nodes of the chunk_id.
        # return the neighbor nodes.
        m = re.search(r"(\d+)$", chunk_id)
        if not m:
            # If chunk_id doesn't end with a number (e.g., UUID), we can't infer neighbors
            return [chunk_id]
        int_chunk_id = int(m.group(1))
        front = True
        back = True
        neighbor_nodes = [chunk_id]
        front_int_chunk_id = int_chunk_id
        back_int_chunk_id = int_chunk_id
        while front or back:
            if front:
                front_int_chunk_id = front_int_chunk_id - 1
                if front_int_chunk_id < 0 or set(self.inverse_index.get(f"leaf_{front_int_chunk_id}", [])) & keys != keys:
                    front = False
                str_chunk_id = "leaf_{}".format(front_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.index.get(key, []):
                        append = False
                        front = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)
            if back:
                back_int_chunk_id += 1
                if set(self.inverse_index.get(f"leaf_{back_int_chunk_id}", [])) & keys != keys:
                    back = False
                str_chunk_id = "leaf_{}".format(back_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.index.get(key, []):
                        append = False
                        back = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)
        return neighbor_nodes

    def _build_faiss_index(self):
        # build the faiss index.
        # only used when the dense retrieval is implemented.
        # return the faiss index.
        docs = self.collapse_tree

        # Check if docs is empty before proceeding
        if not docs:
            print(f"Warning: No documents to embed. collapse_tree is empty.")
            print(f"cache_tree keys: {list(self.cache_tree.keys()) if self.cache_tree else 'cache_tree is None/empty'}")
            # Return None to indicate no index was built
            return None

        if self.embedder is None:
            self.embedder = SentenceTransformer("BAAI/bge-m3", device=self.device)
            self.embedder.eval()
            print("the embedder is not set, using the default embedder BAAI/bge-m3.")

        # Check if the embedder has asynchronous batch methods.
        if hasattr(self.embedder, 'embed_texts'):
            # Use async batches to bound response sizes.
            batch_size = getattr(self, 'embedding_batch_size', 1024)

            async def _embed_all_batches():
                results = []
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i+batch_size]
                    batch_embeds = await self.embedder.embed_texts(batch)
                    results.extend(batch_embeds)
                return results

            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                # Run the async method synchronously
                import nest_asyncio
                nest_asyncio.apply()
                doc_embeds = loop.run_until_complete(_embed_all_batches())
            except RuntimeError:
                # No event loop, create a new one
                doc_embeds = asyncio.run(_embed_all_batches())
            # Ensure float32 dtype and contiguous memory for FAISS
            doc_embeds = np.ascontiguousarray(np.array(doc_embeds, dtype='float32'))
        else:
            # SentenceTransformer - use sync encode method
            doc_embeds = self.embedder.encode(docs, batch_size=16, device=self.device)

        # Check if doc_embeds is valid before creating faiss index
        if doc_embeds.ndim != 2:
            raise ValueError(f"doc_embeds should be 2D array, got shape: {doc_embeds.shape}")
        if doc_embeds.shape[0] == 0:
            raise ValueError("doc_embeds is empty")
        if doc_embeds.shape[1] == 0:
            raise ValueError("doc_embeds has 0 dimensions")

        # 创建FAISS索引，优先使用GPU加速
        vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])

        # 如果启用GPU且FAISS支持GPU，将索引转移到GPU
        if self.use_gpu_faiss and hasattr(faiss, 'StandardGpuResources'):
            try:
                gpu_res = faiss.StandardGpuResources()
                # 将CPU索引转移到GPU
                vector_database = faiss.index_cpu_to_gpu(gpu_res, self.gpu_id, vector_database)
                logger.info(f"DEV版本：FAISS索引已转移到GPU {self.gpu_id}，向量维度: {doc_embeds.shape[1]}, 向量数量: {doc_embeds.shape[0]}")
            except Exception as e:
                logger.warning(f"DEV版本：GPU索引创建失败，回退到CPU: {e}")
                # 如果GPU失败，使用CPU索引
                vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
        else:
            logger.info(f"DEV版本：使用CPU FAISS索引，向量维度: {doc_embeds.shape[1]}, 向量数量: {doc_embeds.shape[0]}")

        # 添加向量到索引（如果是GPU索引，会自动在GPU上操作）
        if hasattr(vector_database, 'add'):
            vector_database.add(doc_embeds)
        else:
            # GPU索引可能需要特殊处理
            vector_database.add(doc_embeds)

        return vector_database

    def index_mapping(self, entities:list) -> List[str]:
        # get the chunks from the cache tree.
        # for two types:
        # 1. entity is a list of strings, i.e., the entities are not related,
        logger.info(f"index_mapping输入entities: {entities}")
        chunk_ids = {}

        for entity in entities:
            logger.info(f"处理entity: {entity}, 类型: {type(entity)}")
            if isinstance(entity, str):
                if entity in self.index.keys():
                    chunk_ids[entity] = self.index[entity]
                    logger.info(f"找到单个实体 {entity} 的chunks: {len(self.index[entity])}个")
                else:
                    logger.info(f"单个实体 {entity} 不在index中")
            elif isinstance(entity, tuple):
                chunk_ids_set = set()
                entity_key = "_".join(entity)
                logger.info(f"处理实体元组: {entity}, 键: {entity_key}")
                for e in entity:
                    if e in self.index.keys():
                        logger.info(f"实体 {e} 在index中，有 {len(self.index[e])} 个chunks")
                        if chunk_ids_set == set():
                            chunk_ids_set = set(self.index[e])
                        else:
                            chunk_ids_set = chunk_ids_set & set(self.index[e])
                            logger.info(f"交集后chunk数量: {len(chunk_ids_set)}")
                    else:
                        logger.info(f"实体 {e} 不在index中")
                chunk_ids[entity_key] = sorted(list(chunk_ids_set))
                logger.info(f"实体元组 {entity_key} 最终chunk数量: {len(chunk_ids[entity_key])}")

        logger.info(f"index_mapping最终结果: {chunk_ids}")
        return chunk_ids

    def graph_filter(self, entities:List[str], k) -> List[str]:
        # get the shortest path between the entities.
        shortest_path_pairs = []
        logger.info(f"检查实体是否在图中: {entities}")
        for head, tail in combinations(entities, 2):
            head_in_graph = head in self.G.nodes()
            tail_in_graph = tail in self.G.nodes()
            logger.info(f"实体对 ({head}, {tail}): head在图中={head_in_graph}, tail在图中={tail_in_graph}")
            if head_in_graph and tail_in_graph:
                try:
                    shortest_path = nx.shortest_path(self.G, head, tail)
                    logger.info(f"找到路径: {head} -> {tail}, 路径长度: {len(shortest_path)}, 路径: {shortest_path}")
                except nx.NetworkXNoPath:
                    logger.info(f"未找到路径: {head} -> {tail}")
                    continue
                if len(shortest_path) <= k:
                    shortest_path_pairs.append((head, tail))
                    logger.info(f"添加路径对: ({head}, {tail})")

        # shortest_path_pairs = self.merge_tuples(shortest_path_pairs)
        logger.info(f"最终路径对数量: {len(shortest_path_pairs)}")
        return shortest_path_pairs

    def merge_tuples(self, lst):
        graph = defaultdict(set)

        for a, b in lst:
            graph[a].add(b)
            graph[b].add(a)

        visited = set()
        result = []

        def dfs(entity, cluster):
            if entity in visited:
                return
            visited.add(entity)
            cluster.add(entity)
            for neighbor in graph[entity]:
                dfs(neighbor, cluster)

        for a, b in lst:
            if a not in visited:
                cluster = set()
                dfs(a, cluster)
                result.append(tuple(sorted(cluster)))

        return result

    def validate_by_checking_father_chunks(self, init_chunk_ids:Dict[str, List[str]], min_count:int=2) -> Dict[str, List[str]]:
        # w, by input the shortest path pairs, get the father nodes.
        valid_child_ids = {}
        for key, chunk_ids in init_chunk_ids.items():
            father_nodes = {}
            for chunk_id in chunk_ids:
                father_chunk_id = self.cache_tree[chunk_id]["parent"]
                father_nodes.setdefault(father_chunk_id, []).append(chunk_id)
            valid_leaf_nodes = [child_id_list for _, child_id_list in father_nodes.items()
                               if len(child_id_list) >= min_count]

            valid_child_ids[key] = []
            if len(valid_leaf_nodes) > 0:
                for leaf_node in valid_leaf_nodes:
                    valid_child_ids[key].extend(leaf_node)

        return valid_child_ids

    def merge_keys(self, neighbor_nodes:Dict[str, List[str]]) -> Dict[str, List[str]]:
        # merge the nodes with different keys.
        # return with the same format.
        logger.info(f"merge_keys输入neighbor_nodes: {neighbor_nodes}")
        chunks_to_keys = defaultdict(set)
        for key, chunk_lists in neighbor_nodes.items():
            logger.info(f"处理键 {key}，包含 {len(chunk_lists)} 个chunks")
            for chunk in chunk_lists:
                chunks_to_keys[chunk].add(key)

        logger.info(f"chunks_to_keys: {dict(chunks_to_keys)}")
        merged_result = {}
        for chunk, keys in chunks_to_keys.items():
            # get the new key.
            if len(keys) > 1:
                all_entities = set()
                for key in keys:
                    all_entities.update(key.split("_"))
                new_key = "_".join(sorted(all_entities))
                logger.info(f"合并多个键 {keys} 为 {new_key}")
            else:
                new_key = keys.pop()
                logger.info(f"使用单个键: {new_key}")
            # add the chunk to the new key.
            if new_key in merged_result.keys():
                merged_result.setdefault(new_key, []).append(chunk)
            else:
                merged_result[new_key] = [chunk]
        logger.info(f"merge_keys最终结果: {merged_result}")
        return merged_result


    def _extract_first_datetime(self, text: str):
        """
        从文本中提取第一个日期时间。
        注意：此方法已禁用search_dates，因为其误判率较高。
        现在直接返回None，不再进行日期提取。
        """
        # search_dates已被移除，因为误判率太高（如"do", "did", "me"等被误判为日期）
        return None


    def _sort_chunk_ids_by_time(self, chunk_ids: List[str]) -> List[str]:
        if not chunk_ids:
            return chunk_ids

        enriched = []
        for idx, chunk_id in enumerate(chunk_ids):
            text = self.get_contiguous_chunks([chunk_id])
            dt = self._extract_first_datetime(text)
            if dt:
                logger.info(f"DEV版本时间排序: chunk {chunk_id} -> {dt.isoformat()}")
            else:
                logger.info(f"DEV版本时间排序: chunk {chunk_id} 未检测到时间信息")
            enriched.append((chunk_id, dt, idx))

        enriched.sort(key=lambda item: (0, item[1]) if item[1] is not None else (1, item[2]))
        return [chunk_id for chunk_id, _, _ in enriched]


    def sort_chunks_by_time(self, chunk_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        if not isinstance(chunk_map, dict):
            return chunk_map

        logger.info("DEV版本时间排序已临时关闭，保持原有顺序")
        return chunk_map


    def get_contiguous_chunks(self, leaf_nodes:List[str]) -> str:
        leaf_texts = []
        for leaf_node in leaf_nodes:
            leaf_text = None
            # First try cache_tree (built tree with leaf_*)
            # 消融实验：如果启用chunks_only，跳过summary_开头的persona节点
            if isinstance(self.cache_tree, dict) and leaf_node in self.cache_tree:
                if getattr(self, 'chunks_only', False) and leaf_node.startswith("summary_"):
                    leaf_text = None
                else:
                    try:
                        leaf_text = self.cache_tree[leaf_node]["text"]
                    except Exception:
                        leaf_text = None
            # Fallback to external text map (e.g., Graphiti UUID mapping)
            if leaf_text is None and isinstance(self.external_text_map, dict):
                entry = self.external_text_map.get(leaf_node)
                if isinstance(entry, dict):
                    leaf_text = entry.get("text")
                elif isinstance(entry, str):
                    leaf_text = entry
            # Final fallback: placeholder to avoid KeyError
            if leaf_text is None:
                leaf_text = f"[text missing for {leaf_node}]"

            leaf_texts.append(leaf_text)

        return '\n\n'.join(leaf_texts)


    def detect_contiguous_chunks(self, chunk_ids:List[str]) -> List[List[str]]:
        # Detect the contiguous chunks from the chunk_ids,
        # if there are contiguous chunks, return the list of the contiguous chunks.
        # otherwise, the only id will be a list.
        res = []
        current_chunk = []
        last_num = None

        # Build list of (chunk_id, trailing_number) where trailing_number is None if not found
        cid_nums = []
        for cid in chunk_ids:
            m = re.search(r"(\d+)$", cid)
            num = int(m.group(1)) if m else None
            cid_nums.append((cid, num))

        # Sort by numeric id when available; non-numeric IDs come last
        cid_nums.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else float('inf')))

        for cid, num in cid_nums:
            if num is None:
                # Flush any ongoing contiguous group and append this as its own group
                if current_chunk:
                    res.append(current_chunk)
                    current_chunk = []
                    last_num = None
                res.append([cid])
            else:
                if not current_chunk:
                    current_chunk = [cid]
                    last_num = num
                else:
                    if last_num is not None and num == last_num + 1:
                        current_chunk.append(cid)
                        last_num = num
                    else:
                        res.append(current_chunk)
                        current_chunk = [cid]
                        last_num = num

        if current_chunk:
            res.append(current_chunk)

        return res

    def format_res(self, res:Dict[str, List[str]]) -> str:
        res_str = ""
        sorted_res = self.sort_chunks_by_time(res)
        logger.info(f"DEV版本format_res输入: {res}")
        for key, chunks in sorted_res.items():
            logger.info(f"DEV版本处理key: {key}, chunks: {chunks}")
            #chunks = self.detect_contiguous_chunks(chunks)
            # for chunk_list in chunks:
            logger.info(f"DEV版本chunks数量: {len(chunks)}")
            str_of_list = self.get_contiguous_chunks(chunks)
            logger.info(f"DEV版本chunk内容: {str_of_list[:200]}...")  # 只显示前200字符
            res_str += "{}\n".format(str_of_list)
        logger.info(f"DEV版本format_res最终结果长度: {len(res_str)}")
        return res_str

    def str_chunkid_2_int_chunkid(self, str_chunk:str) -> int:
        return int(str_chunk.split("_")[-1])

    def local_retrieval(
        self,
        entities: List[str],
        shortest_path_k: int = 4,
        temporal_constraint: Optional[TemporalConstraint] = None,
        temporal_order: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        logger.info(f"开始局部检索，实体: {entities}, shortest_path_k: {shortest_path_k}")

        # initialize by shortest path
        shortest_path = self.graph_filter(entities, shortest_path_k)
        logger.info(f"graph_filter返回的路径对: {shortest_path}")
        # it returns the list of pairs existing shortest path shorter than k.

        # initialize the chunks.
        init_chunk_ids = self.index_mapping(shortest_path)
        logger.info(f"index_mapping返回的chunk_ids: {init_chunk_ids}")

        neighbor_nodes = self.merge_keys(init_chunk_ids)
        logger.info(f"merge_keys返回的neighbor_nodes: {neighbor_nodes}")
        neighbor_nodes = self.sort_chunks_by_time(neighbor_nodes)
        neighbor_nodes = self._apply_temporal_constraint(neighbor_nodes, temporal_constraint, temporal_order)
        logger.info(f"时间排序后的neighbor_nodes: {neighbor_nodes}")
        logger.info(f"局部检索最终结果chunk数量: {self._count_chunks(neighbor_nodes)}")
        return neighbor_nodes

    async def dense_retrieval(
        self,
        query,
        k,
        temporal_constraint: Optional[TemporalConstraint] = None,
        temporal_order: Optional[str] = None,
    ):
        # using dense retrieval to get the chunks.
        # Check if faiss_index is available
        if self.faiss_index is None:
            logger.warning("DEV版本：FAISS index不可用，返回空结果")
            return {"": []}

        logger.info(f"DEV版本开始稠密检索，query: {query}, k: {k}")

        query_embed = await self.embedder.create(query)
        if isinstance(query_embed, list) and len(query_embed) > 0:
            query_embed = query_embed
        else:
            logger.warning(f"DEV版本：embedding返回无效结果: {query_embed}")
            # 提供一个默认的随机向量
            query_embed = [0.0] * 1536  # 默认1536维向量

        # 转换为numpy数组并reshape
        if isinstance(query_embed, list):
            query_embed = np.array(query_embed, dtype='float32')
        # Ensure float32 and contiguous
        query_embed = np.ascontiguousarray(query_embed.reshape(1, -1).astype('float32'))

        _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = k)
        # the normal faiss index return the (1, k) shape. squeeze it to (k,).
        condidate_chunks_indexs = condidate_chunks_indexs[0]

        condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
        logger.info(f"DEV版本对应的chunk_ids: {condidate_chunk_ids}")

        # 记录每个chunk的内容和索引映射
        for i, chunk_id in enumerate(condidate_chunk_ids[:3]):  # 只记录前10个
            chunk_content = self.get_contiguous_chunks([chunk_id])
            faiss_index = condidate_chunks_indexs[i]
            logger.info(f"DEV版本chunk {i}: FAISS索引={faiss_index}, chunk_id={chunk_id}, 内容={chunk_content[:100]}...")

        res = {"": condidate_chunk_ids}
        res = self._apply_temporal_constraint(res, temporal_constraint, temporal_order)

        logger.info(f"DEV版本稠密检索最终结果（时间过滤后）: {res}")
        return res

    def _count_chunks(self, res:Dict[str, List[str]]) -> int:
        # count the chunks.
        count = 0
        for chunk_ids in res.values():
            count += len(chunk_ids)
        return count

    def _collect_chunk_datetimes(self, chunk_id: str) -> List[datetime]:
        datetimes: List[datetime] = []
        seen: Set[str] = set()

        def _add_datetime(value: Optional[str]):
            dt = parse_iso_datetime(value)
            if dt is None:
                return
            iso = dt.isoformat()
            if iso not in seen:
                seen.add(iso)
                datetimes.append(dt)

        if isinstance(self.external_text_map, dict):
            external_entry = self.external_text_map.get(chunk_id)
        else:
            external_entry = None

        potential_sources: List = []
        if isinstance(external_entry, dict):
            potential_sources.append(external_entry)
            metadata = external_entry.get("metadata")
            if isinstance(metadata, dict):
                potential_sources.append(metadata)

        for source in potential_sources:
            if not isinstance(source, dict):
                continue

            for key in ("reference_time", "valid_at", "invalid_at", "time", "start_time", "end_time"):
                value = source.get(key)
                if isinstance(value, str):
                    _add_datetime(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            _add_datetime(item)

            temporal_edges = source.get("temporal_edges")
            if isinstance(temporal_edges, list):
                for edge in temporal_edges:
                    if not isinstance(edge, dict):
                        continue
                    for key in ("time", "valid_at", "invalid_at", "reference_time", "start_time", "end_time"):
                        val = edge.get(key)
                        if isinstance(val, str):
                            _add_datetime(val)

        return datetimes

    def _chunk_matches_temporal_constraint(
        self,
        chunk_id: str,
        constraint: Optional[TemporalConstraint],
    ) -> bool:
        chunk_datetimes = self._collect_chunk_datetimes(chunk_id)

        # 如果有明确的时间约束，使用约束来匹配
        if constraint is not None and not constraint.is_empty():
            if not chunk_datetimes:
                if not self.keep_untimed_chunks:
                    logger.info(f"DEV版本时间过滤：chunk {chunk_id} 无时间信息，已过滤")
                    return False
                logger.info(f"DEV版本时间过滤：chunk {chunk_id} 无时间信息，保留")
                return True

            for dt in chunk_datetimes:
                if constraint.matches(dt):
                    logger.info(
                        f"DEV版本时间过滤：chunk {chunk_id} 匹配时间 {dt.isoformat()}，保留"
                    )
                    return True

            logger.info(f"DEV版本时间过滤：chunk {chunk_id} 未满足时间约束，过滤")
            return False

        # 如果没有明确的时间约束，但有问题日期，则过滤掉问题日期之后的数据
        if self._last_reference_date is not None:
            if not chunk_datetimes:
                # 如果没有时间信息，根据配置决定是否保留
                if not self.keep_untimed_chunks:
                    logger.info(f"DEV版本时间过滤（问题日期之前）：chunk {chunk_id} 无时间信息，已过滤")
                    return False
                logger.info(f"DEV版本时间过滤（问题日期之前）：chunk {chunk_id} 无时间信息，保留")
                return True

            # 检查是否有任何时间在问题日期之前
            for dt in chunk_datetimes:
                if dt <= self._last_reference_date:
                    logger.info(
                        f"DEV版本时间过滤（问题日期之前）：chunk {chunk_id} 时间 {dt.isoformat()} "
                        f"在问题日期 {self._last_reference_date.isoformat()} 之前，保留"
                    )
                    return True

            logger.info(
                f"DEV版本时间过滤（问题日期之前）：chunk {chunk_id} 所有时间都在问题日期 "
                f"{self._last_reference_date.isoformat()} 之后，过滤"
            )
            return False

        # 既没有明确的时间约束，也没有问题日期，不过滤
        return True

    def _graphiti_available(self) -> bool:
        return (
            self._graphiti_sync_driver is not None
            and isinstance(self.graphiti_group_id, str)
            and self.graphiti_group_id != ''
        )

    def _is_graphiti_chunk_id(self, chunk_id: str) -> bool:
        return isinstance(chunk_id, str) and chunk_id.count('-') == 4

    def _map_chunk_id_to_uuid(self, chunk_id: str) -> Optional[str]:
        """
        将 leaf_* 格式的 chunk_id 映射到 Graphiti UUID。
        通过比较文本内容在 text_edge_mapping 中查找匹配的 UUID。
        """
        if self._is_graphiti_chunk_id(chunk_id):
            return chunk_id

        if not isinstance(self.external_text_map, dict) or len(self.external_text_map) == 0:
            logger.info(f"DEV版本chunk映射：external_text_map为空或不是字典")
            return None

        # 获取 chunk 的文本内容
        # 消融实验：如果启用chunks_only，跳过summary_开头的persona节点
        chunk_text = None
        if isinstance(self.cache_tree, dict) and chunk_id in self.cache_tree:
            if getattr(self, 'chunks_only', False) and chunk_id.startswith("summary_"):
                chunk_text = None
            else:
                try:
                    chunk_text = self.cache_tree[chunk_id].get("text")
                except Exception as exc:
                    logger.info(f"DEV版本chunk映射：获取cache_tree文本失败 {chunk_id}: {exc}")
                pass

        if not chunk_text:
            logger.info(f"DEV版本chunk映射：无法获取chunk文本 {chunk_id}")
            return None

        best_match = None
        best_score = 0

        for uuid, entry in self.external_text_map.items():
            if not isinstance(entry, dict):
                continue

            entry_text = entry.get("text", "")
            if not entry_text:
                continue


            # 计算匹配分数
            if chunk_text == entry_text:
                best_match = uuid

        if best_match:
            logger.info(f"DEV版本chunk映射成功：{chunk_id} -> {best_match}")
            return best_match
        else:
            logger.info(f"DEV版本chunk映射失败：{chunk_id}")
            return None

    def _run_graphiti_query(self, query: str, **params) -> List[Dict[str, Any]]:
        if not self._graphiti_sync_driver:
            return []
        session_kwargs: Dict[str, Any] = {}
        if self._graphiti_database:
            session_kwargs["database"] = self._graphiti_database
        try:
            with self._graphiti_sync_driver.session(**session_kwargs) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
        except Exception as exc:
            logger.warning(f"DEV版本：执行Graphiti查询失败，将回退到本地过滤: {exc}")
            return []

    def _fetch_edges_for_episode_ids(
        self,
        episode_ids: Optional[Set[str]] = None,
        temporal_constraint: Optional[TemporalConstraint] = None,
        reference_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        if not self._graphiti_available():
            return []

        # 构建时间约束条件
        time_conditions = []
        query_params = {"group_id": self.graphiti_group_id}

        if temporal_constraint is not None and not temporal_constraint.is_empty():
            # 有明确的时间约束，构建时间范围条件
            # 检查 edge 的 valid_at, created_at 是否在约束范围内
            range_conditions = []
            for i, temp_range in enumerate(temporal_constraint.ranges):
                param_start = f"constraint_start_{i}"
                param_end = f"constraint_end_{i}"
                query_params[param_start] = temp_range.start.isoformat()
                query_params[param_end] = temp_range.end.isoformat()
                # 检查 valid_at 或 created_at 是否在范围内
                # 假设valid_at和created_at是datetime类型，可以直接比较
                range_conditions.append(
                    f"(e.valid_at IS NOT NULL AND e.valid_at >= datetime(${param_start}) AND e.valid_at <= datetime(${param_end})) OR "
                    f"(e.valid_at IS NULL AND e.created_at IS NOT NULL AND e.created_at >= datetime(${param_start}) AND e.created_at <= datetime(${param_end}))"
                )

            if range_conditions:
                time_condition = "(" + " OR ".join(range_conditions) + ")"
                if self.keep_untimed_edges:
                    # 如果允许无时间edge，则时间条件为：有时间且在范围内 OR 无时间
                    time_conditions.append(f"({time_condition} OR (e.valid_at IS NULL AND e.created_at IS NULL))")
                else:
                    # 不允许无时间edge，只保留在时间范围内的
                    time_conditions.append(time_condition)
        elif reference_date is not None:
            # 没有明确的时间约束，但有问题日期，过滤掉问题日期之后的edges
            param_ref_date = "reference_date"
            query_params[param_ref_date] = reference_date.isoformat()
            if self.keep_untimed_edges:
                # 允许无时间edge，或者时间在问题日期之前
                time_conditions.append(
                    f"(e.valid_at IS NULL AND e.created_at IS NULL) OR "
                    f"(e.valid_at IS NOT NULL AND e.valid_at <= datetime(${param_ref_date})) OR "
                    f"(e.valid_at IS NULL AND e.created_at IS NOT NULL AND e.created_at <= datetime(${param_ref_date}))"
                )
            else:
                # 不允许无时间edge，只保留时间在问题日期之前的
                time_conditions.append(
                    f"(e.valid_at IS NOT NULL AND e.valid_at <= datetime(${param_ref_date})) OR "
                    f"(e.valid_at IS NULL AND e.created_at IS NOT NULL AND e.created_at <= datetime(${param_ref_date}))"
                )
        # 如果没有时间约束也没有问题日期，不过滤（保留所有edge）

        # 构建完整的WHERE子句
        where_clauses = ["e.group_id = $group_id"]
        if time_conditions:
            where_clauses.append("(" + " OR ".join(time_conditions) + ")")

        where_clause = " AND ".join(where_clauses)

        # 查找边，直接在查询中加入时间约束
        # RELATES_TO只存在于Entity之间，所以可以省略节点标签
        # 同时查询episode的content，用于文本匹配
        # 注意：episode 查询也需要过滤 group_id 以确保数据隔离
        query = f"""
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE {where_clause}
        WITH e, src, dst, e.episodes AS episode_uuids
        UNWIND episode_uuids AS ep_uuid
        OPTIONAL MATCH (ep:Episodic {{uuid: ep_uuid}})
        WHERE ep.group_id = $group_id
        WITH e, src, dst, collect({{
            uuid: ep_uuid,
            name: ep.name,
            content: ep.content,
            created_at: ep.created_at,
            valid_at: ep.valid_at
        }}) AS episode_details
        RETURN
            e.uuid AS edge_uuid,
            e.fact AS fact,
            e.name AS relation,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            e.episodes AS episodes,
            episode_details AS episode_details,
            src.uuid AS source_uuid,
            src.name AS source_name,
            dst.uuid AS target_uuid,
            dst.name AS target_name
        """

        records = self._run_graphiti_query(
            query,
            **query_params,
        )
        constraint_info = "有时间约束" if temporal_constraint and not temporal_constraint.is_empty() else ("有问题日期" if reference_date else "无时间限制")
        logger.info(f"DEV版本边查询：查询到 {len(records)} 条边（{constraint_info}），包含episode详情")
        return records

    def _extract_edge_datetime(self, edge: Dict[str, Any]) -> Optional[datetime]:
        edge_uuid = edge.get("edge_uuid", edge.get("uuid", "unknown"))
        logger.info(f"DEV版本时间提取：开始提取edge时间，edge_uuid={edge_uuid}")

        for field in ("valid_at", "created_at", "invalid_at"):
            value = edge.get(field)
            if value is None:
                logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, field={field} 为None，跳过")
                continue

            logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, 找到field={field}, 原始值={value}, 类型={type(value)}")

            # 处理Neo4j的DateTime对象
            if hasattr(value, 'iso_format'):
                # Neo4j DateTime对象，转换为ISO格式字符串
                value = value.iso_format()
                logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, Neo4j DateTime转换为ISO: {value}")
            elif hasattr(value, 'isoformat'):
                # Python datetime对象，转换为ISO格式字符串
                value = value.isoformat()
                logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, Python datetime转换为ISO: {value}")
            elif not isinstance(value, str):
                # 其他类型，尝试转换为字符串
                value = str(value)
                logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, 其他类型转换为字符串: {value}")

            dt = parse_iso_datetime(value)
            if dt is not None:
                logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, 从field={field}成功提取时间: {dt.isoformat()}")
                return dt
            else:
                logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, field={field}解析失败，继续尝试下一个字段")

        logger.info(f"DEV版本时间提取：edge_uuid={edge_uuid}, 未找到有效时间字段，返回None")
        return None

    def _match_edges_to_chunks(
        self,
        edges: List[Dict[str, Any]],
        constraint: Optional[TemporalConstraint],  # 保留参数以兼容调用，但不再使用
        sort_order: Optional[str],  # 保留参数以兼容调用，但不再使用
        chunk_texts: Set[str],  # chunk文本集合（保留用于向后兼容）
        chunk_id_to_text: Dict[str, str],  # chunk_id到文本的映射（保留用于向后兼容）
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        找到边所在的chunk。
        方案2：通过episode name（chunk_{i}）和chunk_id（leaf_{i}）的对应关系匹配。
        episode name格式：chunk_{i}，对应chunk_id格式：leaf_{i}

        注意：这些edges已在Cypher查询中经过时间过滤，所以这里只需要找到chunk，不需要再次过滤。

        返回:
            matched_items: 匹配的edge信息列表，每个item包含edge和matched_chunk_ids
            chunk_edge_map: chunk_id到edge items的映射
        """
        logger.info(
            f"DEV版本边-chunk匹配（方案2：通过episode name匹配）：开始匹配edges到chunks，"
            f"输入edges数量={len(edges)}, "
            f"chunk_id_to_text数量={len(chunk_id_to_text)}"
        )

        if not edges:
            logger.info("DEV版本边-chunk匹配：输入edges为空，返回空结果")
            return [], {}

        chunk_edge_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # 存储匹配的edge信息：每个item包含edge、匹配的chunk_ids等
        matched_items: List[Dict[str, Any]] = []

        stats = {
            "total": len(edges),
            "matched": 0,
            "no_match": 0,
        }

        for edge in edges:
            edge_uuid = edge.get("edge_uuid", edge.get("uuid", "unknown"))
            episode_details = edge.get("episode_details", [])

            # 通过episode name匹配找到edge关联的chunk_id
            matched_chunk_ids = []

            for ep_detail in episode_details:
                if not isinstance(ep_detail, dict):
                    continue

                # 获取episode name（格式：chunk_{i}）
                ep_name = ep_detail.get("name", "")
                if not ep_name:
                    # 如果没有name，回退到文本匹配（向后兼容）
                    ep_content = ep_detail.get("content", "")
                    if ep_content:
                        ep_content_normalized = ep_content.strip()
                        if ep_content_normalized in chunk_texts:
                            for cid, ctext in chunk_id_to_text.items():
                                if ctext == ep_content_normalized:
                                    if cid not in matched_chunk_ids:
                                        matched_chunk_ids.append(cid)
                    continue

                # 从episode name提取索引：chunk_{i} -> leaf_{i}
                # episode name格式：chunk_{i}，对应chunk_id格式：leaf_{i}
                if ep_name.startswith("chunk_"):
                    try:
                        chunk_index = int(ep_name.replace("chunk_", ""))
                        chunk_id = f"leaf_{chunk_index}"

                        # 验证chunk_id是否存在于chunk_id_to_text中
                        if chunk_id in chunk_id_to_text:
                            if chunk_id not in matched_chunk_ids:
                                matched_chunk_ids.append(chunk_id)
                                logger.debug(f"DEV版本边-chunk匹配：通过episode name {ep_name} -> chunk_id {chunk_id}")
                        else:
                            logger.debug(f"DEV版本边-chunk匹配：episode name {ep_name} 对应的chunk_id {chunk_id} 不存在于chunk_id_to_text中")
                    except ValueError:
                        logger.debug(f"DEV版本边-chunk匹配：无法从episode name {ep_name} 提取索引")
                        continue

            if not matched_chunk_ids:
                stats["no_match"] += 1
                logger.debug(f"DEV版本边-chunk匹配：edge_uuid={edge_uuid}, 未找到匹配的chunk")
                continue

            stats["matched"] += 1
            logger.debug(
                f"DEV版本边-chunk匹配：edge_uuid={edge_uuid}, "
                f"找到{len(matched_chunk_ids)}个匹配的chunk: {matched_chunk_ids}"
            )

            # 构建匹配的item
            item = {
                "edge": edge,
                "matched_chunk_ids": matched_chunk_ids,
            }
            matched_items.append(item)

            # 构建chunk到edge的映射
            for chunk_id in matched_chunk_ids:
                chunk_edge_map[chunk_id].append(item)

        logger.info(
            f"DEV版本边-chunk匹配：统计 - "
            f"总数={stats['total']}, "
            f"匹配={stats['matched']}, "
            f"未匹配={stats['no_match']}, "
            f"匹配的chunk数量={len(chunk_edge_map)}"
        )

        return matched_items, chunk_edge_map

    def _apply_graphiti_temporal_filter(
        self,
        chunk_map: Dict[str, List[str]],
        constraint: Optional[TemporalConstraint],
        sort_order: Optional[str],
    ) -> Optional[Dict[str, List[str]]]:
        # 检查是否启用时间过滤（消融实验控制）
        temporal_filter_enabled = getattr(self, 'temporal_filter_enabled', True)
        logger.info(f"DEV版本Graphiti时间过滤：temporal_filter_enabled={temporal_filter_enabled}")
        if not temporal_filter_enabled:
            logger.info("DEV版本Graphiti时间过滤：时间过滤已禁用（消融实验），跳过Graphiti时间rerank")
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            return None

        # 当时间提取返回None时，不查询边，直接返回None
        if constraint is None or constraint.is_empty():
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            logger.info("DEV版本Graphiti时间过滤：时间提取返回None，不查询边，回退到本地过滤")
            return None
        if not self._graphiti_available():
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            logger.info("DEV版本Graphiti时间过滤：Graphiti不可用，回退到本地过滤")
            return None
        if not isinstance(chunk_map, dict) or len(chunk_map) == 0:
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            logger.info("DEV版本Graphiti时间过滤：chunk_map为空，回退到本地过滤")
            return None

        # 收集所有chunk的文本内容，用于文本匹配
        chunk_texts = set()
        chunk_id_to_text = {}  # chunk_id -> text的映射

        for ids in chunk_map.values():
            for cid in ids:
                # 获取chunk的文本内容
                # 消融实验：如果启用chunks_only，跳过summary_开头的persona节点
                chunk_text = None
                if isinstance(self.cache_tree, dict) and cid in self.cache_tree:
                    if getattr(self, 'chunks_only', False) and cid.startswith("summary_"):
                        chunk_text = None
                    else:
                        try:
                            chunk_text = self.cache_tree[cid].get("text", "").strip()
                        except Exception as exc:
                            logger.debug(f"DEV版本Graphiti时间过滤：获取chunk文本失败 {cid}: {exc}")
                            pass

                # 如果获取到了文本，添加到集合和映射中
                if chunk_text:
                    chunk_texts.add(chunk_text)
                    chunk_id_to_text[cid] = chunk_text
                else:
                    # 如果从cache_tree获取失败，尝试从external_text_map获取
                    if isinstance(self.external_text_map, dict):
                        external_entry = self.external_text_map.get(cid)
                        if isinstance(external_entry, dict):
                            chunk_text = external_entry.get("text", "").strip()
                        elif isinstance(external_entry, str):
                            chunk_text = external_entry.strip()

                        if chunk_text:
                            chunk_texts.add(chunk_text)
                            chunk_id_to_text[cid] = chunk_text

        logger.info(
            f"DEV版本Graphiti时间rerank：收集到{len(chunk_texts)}个chunk文本，"
            f"共{len(chunk_id_to_text)}个chunk_id，开始查询edges（在查询中直接加入时间约束）"
        )
        edges = self._fetch_edges_for_episode_ids(
            episode_ids=set(),  # 不再需要episode_ids参数
            temporal_constraint=constraint,
            reference_date=self._last_reference_date,
        )
        if not edges:
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            self.last_temporal_edges = []
            logger.info(f"DEV版本Graphiti时间rerank：未找到关联的edges，返回原始chunk_map")
            return chunk_map
        logger.info(f"DEV版本Graphiti时间rerank：找到{len(edges)}条edges（已在查询中过滤时间），开始匹配edges到chunks")

        matched_items, chunk_edge_map = self._match_edges_to_chunks(
            edges,
            constraint,
            sort_order,
            chunk_texts,  # 传入chunk文本集合
            chunk_id_to_text,  # 传入chunk_id到文本的映射
        )

        # 收集匹配的chunk_id（已经通过文本匹配得到），用于后续重新排序
        matched_chunk_ids_set: Set[str] = set()
        if matched_items:
            for item in matched_items:
                matched_chunk_ids = item.get("matched_chunk_ids", [])
                matched_chunk_ids_set.update(matched_chunk_ids)
            self.last_temporal_edges = [item["edge"] for item in matched_items]
        else:
            self.last_temporal_edges = []

        # 保存匹配的chunk_id集合，用于重新排序
        self._temporal_matched_chunk_ids = matched_chunk_ids_set

        logger.info(
            f"DEV版本Graphiti时间rerank："
            f"原始{self._count_chunks(chunk_map)}个chunk，"
            f"匹配了{len(matched_items)}条edges，"
            f"匹配的chunk_id数量: {len(matched_chunk_ids_set)}"
        )

        # 使用_reorder_chunks_by_temporal_match函数重新排序：将匹配的chunk移到前面
        reordered_chunk_map = self._reorder_chunks_by_temporal_match(chunk_map)

        return reordered_chunk_map

    def _apply_local_temporal_filter(
        self,
        chunk_map: Dict[str, List[str]],
        constraint: Optional[TemporalConstraint],
    ) -> Dict[str, List[str]]:
        if constraint is None or constraint.is_empty():
            return chunk_map

        filtered: Dict[str, List[str]] = {}
        removed = 0
        original = 0

        for key, chunk_ids in chunk_map.items():
            filtered_ids: List[str] = []
            for chunk_id in chunk_ids:
                original += 1
                if self._chunk_matches_temporal_constraint(chunk_id, constraint):
                    filtered_ids.append(chunk_id)
                else:
                    removed += 1
            if filtered_ids:
                filtered[key] = filtered_ids

        logger.info(
            f"DEV版本时间过滤（本地）：共检查 {original} 个chunk，过滤 {removed} 个，剩余 {self._count_chunks(filtered)} 个"
        )
        return filtered

    def _reorder_chunks_by_temporal_match(
        self,
        chunk_map: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        重新排序chunk，将时间约束匹配的chunk移到前面
        """
        # 检查是否启用时间过滤（消融实验控制）
        temporal_filter_enabled = getattr(self, 'temporal_filter_enabled', True)
        logger.info(f"DEV版本重新排序：temporal_filter_enabled={temporal_filter_enabled}")
        if not temporal_filter_enabled:
            logger.info("DEV版本重新排序：时间过滤已禁用（消融实验），跳过时间重排序")
            return chunk_map

        if not isinstance(chunk_map, dict) or not self._temporal_matched_chunk_ids:
            return chunk_map

        reordered_chunk_map: Dict[str, List[str]] = {}

        for key, chunk_ids in chunk_map.items():
            # 分离匹配和不匹配的chunk
            matched_chunks = []
            unmatched_chunks = []

            for chunk_id in chunk_ids:
                if chunk_id in self._temporal_matched_chunk_ids:
                    matched_chunks.append(chunk_id)
                else:
                    unmatched_chunks.append(chunk_id)

            # 将匹配的chunk放在前面，不匹配的保持原顺序
            reordered_chunk_map[key] = matched_chunks + unmatched_chunks

        logger.info(
            f"DEV版本重新排序：将{len(self._temporal_matched_chunk_ids)}个时间匹配的chunk移到前面"
        )
        return reordered_chunk_map

    def _apply_temporal_constraint(
        self,
        chunk_map: Dict[str, List[str]],
        constraint: Optional[TemporalConstraint],
        sort_order: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        if not isinstance(chunk_map, dict):
            return chunk_map

        # 当没有约束时，使用基于第一行日期的过滤
        if constraint is None or constraint.is_empty():
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            return chunk_map

        # 检查是否启用时间过滤（消融实验控制）
        if not getattr(self, 'temporal_filter_enabled', True):
            logger.info("时间过滤已禁用（消融实验），跳过时间rerank")
            self._temporal_matched_chunk_ids = set()  # 清空匹配的chunk_id
            return chunk_map

        # 有约束时，应该rerank而不是过滤
        graphiti_reranked = self._apply_graphiti_temporal_filter(chunk_map, constraint, sort_order)

        if graphiti_reranked is not None:
            # Graphiti方式可用，已经完成排序，直接返回
            return graphiti_reranked

        # Graphiti方式不可用，使用基于第一行日期的过滤
        return self.chunk_map

    def _infer_temporal_order(self, text: str) -> str:
        lowered = (text or "").lower()
        if any(token in lowered for token in ("first", "earliest", "beginning", "start", "initial")):
            return "asc"
        if any(token in lowered for token in ("last", "latest", "recent", "most recent", "newest", "final")):
            return "desc"
        return "asc"

    def entityaware_filter(
        self,
        candidate_chunks: Dict[str, List[str]],
        entities: List[str],
        temporal_constraint: Optional[TemporalConstraint] = None,
        temporal_order: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        # filter rules:
        # 1. the chunk includes more different entities, the priority is higher.
        # 2. if the chunk has longer neighbor nodes, the priority is higher.
        # 3. if the chunk includes the same number of entities, the higher the number of appearance of the entities is, the higher the priority is.
        # Initialize result dictionary
        chunks_info = []
        for key, value in candidate_chunks.items():
            for chunk_id in value:
                key_count = len(key.split("_"))
                set_key = set(key.split("_"))
                neighbor_nodes_count = len(self._detect_neighbor_nodes(keys=set_key, chunk_id=chunk_id))
                entity_count = 0
                for key_entity in key.split('_'):
                    entity_count += self.appearance_count[chunk_id].get(key_entity, 0)
                chunk_id_info = {
                    "chunk_id": chunk_id,
                    "key_count": key_count,
                    "neighbor_nodes_count": neighbor_nodes_count,
                    "entity_count": entity_count
                }
                chunks_info.append(chunk_id_info)
        # sort the chunks_info by the key_count, neighbor_nodes_count, and entity_count.
        sorted_chunks_info = sorted(chunks_info, key=lambda x: (x["key_count"], x["neighbor_nodes_count"], x["entity_count"]), reverse=True)
        # get the top 25 chunks.
        top_25_chunks = sorted_chunks_info[:25]
        # get the chunk_ids from the top_25_chunks.
        top_25_chunk_ids = [chunk["chunk_id"] for chunk in top_25_chunks]
        # return the result.
        filtered_res = {}
        for id in top_25_chunk_ids:
            for entity in entities:
                if id in self.index.get(entity, []):
                    filtered_res.setdefault(entity, []).append(id)
        filtered_res = self.merge_keys(filtered_res)
        filtered_res = self.sort_chunks_by_time(filtered_res)
        filtered_res = self._apply_temporal_constraint(filtered_res, temporal_constraint, temporal_order)
        print("filtered_res,",filtered_res)
        return filtered_res

    def temporl_filter(self, candidate_chunks:Dict[str, List[str]], entities:List[str]) -> Dict[str, List[str]]:
        # temporal filter.
        # extract time from the query using
        date = dateparser.parse(query)
        if date is None:
            return candidate_chunks
        else:
            return candidate_chunks

    def _check_children(self, chunk_id:str, entities:List[str], visited=None) -> int:
        if visited is None:
            visited = set()

        if chunk_id in visited:
            return 0

        visited.add(chunk_id)

        entity_count = 0

        children = self.cache_tree.get(chunk_id, {}).get("children", [])

        # 确保children是列表类型，防止None值导致TypeError
        if children is None:
            children = []

        for child in children:
            if not child.startswith("leaf_"):
                entity_count += self._check_children(child, entities, visited)
            else:
                chunk_appearance_stat = self.appearance_count.get(child, {})
                for entity in entities:
                    entity_count += chunk_appearance_stat.get(entity, 0)

        return entity_count

    def occurrence_ranking(self, candidate_chunk_ids:List[str], entities:List[str], max_chunks:int=25) -> Dict[str, List[str]]:
        # occurrence ranking.
        logger.info(f"DEV版本occurrence_ranking输入: candidate_chunk_ids={len(candidate_chunk_ids)}个, entities={entities}, max_chunks={max_chunks}")

        # 消融实验：如果启用chunks_only，过滤掉非chunk节点（排除summary_开头的persona节点）
        if getattr(self, 'chunks_only', False):
            original_count = len(candidate_chunk_ids)
            candidate_chunk_ids = [cid for cid in candidate_chunk_ids if (cid.startswith("leaf_") or self._is_graphiti_chunk_id(cid)) and not cid.startswith("summary_")]
            logger.info(f"消融实验chunks_only过滤: {original_count} -> {len(candidate_chunk_ids)} chunks")

        filtered_res = {}
        filtered_chunk_ids = []
        chunk_count = []

        for i, chunk_id in enumerate(candidate_chunk_ids):
            if not chunk_id.startswith("leaf_"):
                count = self._check_children(chunk_id, entities)
                chunk_count.append(count)
                logger.info(f"DEV版本chunk {i} ({chunk_id}): 非leaf节点，count={count}")
                continue

            chunk_appearance_stat = self.appearance_count.get(chunk_id, {})
            logger.info(f"DEV版本chunk {i} ({chunk_id}): appearance_count={chunk_appearance_stat}")
            this_chunk_count = 0
            for entity in entities:
                entity_count = chunk_appearance_stat.get(entity, 0)
                this_chunk_count += entity_count
                if entity_count > 0:
                    logger.info(f"DEV版本chunk {i} ({chunk_id}): 实体{entity}出现{entity_count}次")
            chunk_count.append(this_chunk_count)
            logger.info(f"DEV版本chunk {i} ({chunk_id}): 总count={this_chunk_count}")

        chunk_count = np.array(chunk_count)
        nonzero_indices = np.nonzero(chunk_count)[0]
        logger.info(f"DEV版本nonzero_indices数量: {len(nonzero_indices)}")

        if len(nonzero_indices) == 0:
            logger.info(f"DEV版本：所有chunk的count都为0，返回前{max_chunks}个")
            default_res = {"": candidate_chunk_ids[:max_chunks]}
            return self.sort_chunks_by_time(default_res)

        argsorted_chunk_ids = np.argsort(chunk_count)[::-1]
        filtered_chunk_ids = [candidate_chunk_ids[i] for i in argsorted_chunk_ids if chunk_count[i] > 0]
        logger.info(f"DEV版本过滤后chunk数量: {len(filtered_chunk_ids)}")

        if len(filtered_chunk_ids) > max_chunks:
            filtered_chunk_ids = filtered_chunk_ids[:max_chunks]
            logger.info(f"DEV版本：截断到{max_chunks}个chunk")

        for id in filtered_chunk_ids:
            for entity in entities:
                # 优先使用appearance_count中的chunk级别统计
                chunk_appearance = self.appearance_count.get(id, {})
                if entity in chunk_appearance and chunk_appearance[entity] > 0:
                    filtered_res.setdefault(entity, []).append(id)
                # 如果没有chunk级别统计，回退到原来的inverse_index逻辑
                elif entity in self.inverse_index.get(id, []):
                    filtered_res.setdefault(entity, []).append(id)

        filtered_res = self.merge_keys(filtered_res)
        filtered_res = self.sort_chunks_by_time(filtered_res)
        return filtered_res

    def query(self, query, **kwargs):
        import time
        start_time = time.time()
        logger.info(f"=== DEV版本检索开始 ===")
        logger.info(f"开始查询处理: {query[:50]}...")
        logger.info(f"检索参数: {kwargs}")

        # step 1: extract the Entities from the query.
        # 实体提取已注释：由于local retrieval已禁用，不再需要实体提取
        # entity_start = time.time()
        # # reuse the naive_extract_graph function, which is used in the graph building process.
        # entities = naive_extract_graph(query.split("\n")[0], self.nlp)#
        # entities = entities["nouns"]
        # entity_time = time.time() - entity_start
        # logger.info(f"DEV版本实体提取完成，耗时: {entity_time:.3f}秒，提取到{len(entities)}个实体")
        # logger.info(f"DEV版本提取到的实体: {entities}")
        entities = []  # 设为空列表，直接走稠密检索路径

        # step 2.0: set up the parameters.
        shortest_path_k = kwargs.get("shortest_path_k", 4)

        temporal_constraint: Optional[TemporalConstraint] = None
        if self.temporal_filter_enabled:
            reference_input = (
                kwargs.get("question_date")
                or kwargs.get("reference_time")
                or kwargs.get("reference_date")
            )
            logger.info(f"DEV版本时间提取：从kwargs获取的参考时间输入: {reference_input}")
            reference_date = parse_iso_datetime(reference_input) if reference_input else None
            if reference_input and reference_date is None:
                parsed_reference = dateparser.parse(reference_input)
                if parsed_reference is not None:
                    reference_date = parse_iso_datetime(parsed_reference.isoformat())
                    logger.info(f"DEV版本时间提取：dateparser.parse成功，解析后的参考时间: {reference_date.isoformat() if reference_date else None}")
                else:
                    logger.warning(f"DEV版本时间提取：dateparser.parse也失败，无法解析参考时间: {reference_input}")
            elif reference_date:
                logger.info(f"DEV版本时间提取：parse_iso_datetime成功，解析后的参考时间: {reference_date.isoformat()}")

            temporal_constraint = extract_temporal_constraints(
                query.split("\n")[0],
                nlp=self.nlp,
                reference_date=reference_date,
            )
            if temporal_constraint and not temporal_constraint.is_empty():
                ranges_info = [
                    f"{temp_range.start.isoformat()}~{temp_range.end.isoformat()}({temp_range.granularity})"
                    for temp_range in temporal_constraint.ranges
                ]
                logger.info(
                    f"DEV版本时间约束生成成功，参考时间: {reference_date}, 短语: {temporal_constraint.phrases}, 范围: {ranges_info}"
                )
            else:
                logger.info("DEV版本未检测到有效时间约束")

        temporal_order = self._infer_temporal_order(query)
        self._last_temporal_constraint = temporal_constraint
        self._last_temporal_order = temporal_order
        self._last_reference_date = reference_date if self.temporal_filter_enabled else None
        self.last_temporal_edges = []
        self._temporal_matched_chunk_ids = set()  # 用于存储时间约束匹配的chunk_id

        # step 2.1: 由于entities为空，直接使用稠密检索（调用封装好的dense_retrieval函数）
        if len(entities) == 0:
            dense_start = time.time()
            try:
                import asyncio
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                chunk_ids = loop.run_until_complete(self.dense_retrieval(
                    query,
                    kwargs.get("max_chunk_setting", 25),
                    temporal_constraint=temporal_constraint,
                    temporal_order=temporal_order,
                ))
            except RuntimeError:
                chunk_ids = asyncio.run(self.dense_retrieval(
                    query,
                    kwargs.get("max_chunk_setting", 25),
                    temporal_constraint=temporal_constraint,
                    temporal_order=temporal_order,
                ))
            dense_time = time.time() - dense_start
            logger.info(f"DEV版本稠密检索完成，耗时: {dense_time:.3f}秒，检索到{self._count_chunks(chunk_ids)}个chunk")
            logger.info(f"DEV版本稠密检索结果chunk_ids: {chunk_ids}")
            result = {"chunks":self.format_res(chunk_ids)}
            if kwargs.get("info", True):
                supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), self._count_chunks(chunk_ids), [])
                result.update(supplement_info)
                result["retrieval_type"] = "DEV-Dense Retrieval"
            logger.info(f"DEV版本最终结果: {result}")
            return result

        # step 2.2: initialize the chunks by wasd method.
        # 局部检索已注释，直接使用稠密检索
        # 以下代码已注释，改为使用上面的dense_retrieval函数调用
        # logger.info("DEV版本：局部检索已注释，直接使用稠密检索")
        # # if the chunk count is 0, dense retrieval (occurrence ranking已注释)
        # # if chunk_count == 0:
        # if True:  # 直接使用稠密检索
        #     logger.info("DEV版本：局部检索结果为0，使用稠密检索（occurrence ranking已禁用）")
        #     # Check if faiss_index is available
        #     if self.faiss_index is None:
        #         logger.warning("DEV版本：FAISS index不可用，返回空结果")
        #         result = {"chunks": ""}
        #         if kwargs.get("info", True):
        #             supplement_info = self._build_supplement_info({}, entities, {}, [], 0, chunk_counts_history)
        #             result.update(supplement_info)
        #             result["retrieval_type"] = "DEV-Empty Index - Dense Retrieval"
        #         return result
        #
        #     try:
        #         import asyncio
        #         import nest_asyncio
        #         nest_asyncio.apply()
        #         loop = asyncio.get_event_loop()
        #         query_embed = loop.run_until_complete(self.embedder.embed_texts([query]))
        #     except RuntimeError:
        #         query_embed = asyncio.run(self.embedder.embed_texts([query]))
        #     # Ensure float32 and contiguous for FAISS
        #     query_embed = np.ascontiguousarray(np.array(query_embed, dtype='float32').reshape(1, -1))
        #     # 先按向量相似度取 max_chunk_setting 个候选
        #     max_chunk_setting = kwargs.get("max_chunk_setting", 25)
        #     _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = max_chunk_setting)
        #     condidate_chunks_indexs = condidate_chunks_indexs[0]
        #     logger.info(f"DEV版本FAISS搜索返回的索引: {condidate_chunks_indexs}")
        #     logger.info(f"DEV版本condidate_chunks_indexs长度: {len(condidate_chunks_indexs)}")
        #
        #     condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
        #     logger.info(f"DEV版本对应的chunk_ids: {condidate_chunk_ids}")
        #     logger.info(f"DEV版本condidate_chunk_ids长度: {len(condidate_chunk_ids)}")
        #
        #     # DEV版本：打印FAISS索引到chunk_id的映射
        #     logger.info(f"DEV版本FAISS索引映射: {list(zip(condidate_chunks_indexs[:10], condidate_chunk_ids[:10]))}")
        #
        #     # 注释掉 occurrence_ranking，直接使用向量搜索的结果
        #     # # 然后对这 max_chunk_setting 个候选做 occurrence_ranking 排序
        #     # filtered_chunk_ids = self.occurrence_ranking(condidate_chunk_ids, entities, max_chunks=max_chunk_setting)
        #     # logger.info(f"DEV版本occurrence ranking结果: {len(filtered_chunk_ids)}个chunk")
        #     # logger.info(f"DEV版本filtered_chunk_ids: {filtered_chunk_ids}")
        #
        #     # 直接使用向量搜索的结果，不做 occurrence_ranking
        #     filtered_chunk_ids = {"": condidate_chunk_ids}
        #
        #     # 先应用时间约束（可能会过滤掉一些chunk，并记录匹配的chunk_id）
        #     filtered_chunk_ids = self._apply_temporal_constraint(
        #         filtered_chunk_ids,
        #         temporal_constraint,
        #         temporal_order,
        #     )
        #     logger.info(
        #         f"DEV版本：跳过occurrence_ranking，向量搜索原始chunk数量: {len(condidate_chunk_ids)}，时间过滤后: {self._count_chunks(filtered_chunk_ids)}"
        #     )
        #
        #     # 重新排序：将时间约束匹配的chunk移到前面（仅当有明确时间约束时）
        #     # 当时间提取返回None时，不进行边重排序
        #     if temporal_constraint is not None and not temporal_constraint.is_empty() and self._temporal_matched_chunk_ids:
        #         filtered_chunk_ids = self._reorder_chunks_by_temporal_match(filtered_chunk_ids)
        #         logger.info(
        #             f"DEV版本：重新排序完成，时间匹配的chunk数量: {len(self._temporal_matched_chunk_ids)}"
        #         )
        #
        #     # return the entity_entityB: [chunk_id1, chunk_id2, ...]
        #     res_str = self.format_res(filtered_chunk_ids)
        #
        #     result = {"chunks":res_str}
        #     if kwargs.get("info", True):
        #         max_chunk_setting = kwargs.get("max_chunk_setting", 25)
        #         supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), len(condidate_chunk_ids), chunk_counts_history)
        #         result.update(supplement_info)
        #         result["retrieval_type"] = "DEV-Dense Retrieval (Occurrence Ranking Disabled)"
        #         logger.info(f"DEV版本向量搜索最终结果: {result}")
        #
        #     return result

        # 局部检索相关代码已注释
        # logger.info(f"DEV版本：开始调整chunk_count，当前chunk_count={chunk_count}")
        # while chunk_count > kwargs.get("max_chunk_setting", 25):
        #     logger.info(f"DEV版本：chunk_count({chunk_count}) > max_chunk_setting({kwargs.get('max_chunk_setting', 25)})，调整shortest_path_k")
        #     prev_local_res = copy.deepcopy(local_res)
        #     # if the chunk count is larger than the max chunk setting
        #     # then change the setting, increase the min count and decrease the shortest path k.
        #     shortest_path_k -= 1
        #     logger.info(f"DEV版本：调整shortest_path_k为{shortest_path_k}")
        #     # update the result with new restrictions.
        #     local_res = self.local_retrieval(
        #         entities,
        #         shortest_path_k,
        #         temporal_constraint,
        #         temporal_order,
        #     )
        #     chunk_count = self._count_chunks(local_res)
        #     chunk_counts_history.append((shortest_path_k, chunk_count))
        #     logger.info(f"DEV版本：调整后chunk_count={chunk_count}")

        # # if the chunk count is 0, dense retrieval + entity filter.
        # # if the chunk count is not 0, return the result.
        # if chunk_count != 0:
        #     # format the result.
        #     logger.info(f"DEV版本BOTTOM2TOP: final chunk_count {chunk_count}")
        #     res_str = self.format_res(local_res)

        #     result = {"chunks":res_str}
        #     if kwargs.get("info", True):
        #         supplement_info = self._build_supplement_info(local_res, entities, local_res, list(local_res.keys()), chunk_count, chunk_counts_history)
        #         result.update(supplement_info)
        #         result["retrieval_type"] = f"DEV-Local, Loop for {len(chunk_counts_history)-1} times"
        #     logger.info(f"DEV版本Local检索最终结果: {result}")
        # else:
        #     # the previous local result is not empty, so we can use it as candidate chunks.
        #     logger.info("DEV版本：chunk_count为0，使用entityaware_filter")
        #     candidate_chunks = prev_local_res
        #     logger.info(f"DEV版本候选chunks: {candidate_chunks}")
        #     res_ids = self.entityaware_filter(
        #         candidate_chunks,
        #         entities,
        #         temporal_constraint,
        #         temporal_order,
        #     )
        #     chunk_count = self._count_chunks(res_ids)
        #     res_str = self.format_res(res_ids)
        #     result = {"chunks":res_str}
        #     result["chunk_counts_history"] = chunk_counts_history
        #     if kwargs.get("info", True):
        #         supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), 25, chunk_counts_history)
        #         result.update(supplement_info)
        #         result["retrieval_type"] = f"DEV-EntityAware Filter, Loop for {len(chunk_counts_history)-1} times"
        #     logger.info(f"DEV版本EntityAware Filter最终结果: {result}")

        total_time = time.time() - start_time
        logger.info(f"DEV版本查询处理完成，总耗时: {total_time:.3f}秒")
        logger.info(f"=== DEV版本检索结束 ===")
        logger.info(f"DEV版本最终result: {result}")

        return result

    def _build_supplement_info(self, chunk_ids, entities, neighbor_nodes, keys, len_chunks, chunk_counts_history):
        return {
            "chunk_ids": chunk_ids,
            "entities": entities,
            "neighbor_nodes": neighbor_nodes,
            "keys": keys,
            "len_chunks": len_chunks,
            "chunk_counts_history": chunk_counts_history
        }
