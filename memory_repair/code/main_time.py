import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import os
from pathlib import Path
import json
import numpy as np
import traceback
import sys
# 移除硬编码路径，使用相对路径
import argparse
import time
import gc
from datetime import datetime
import logging
from tqdm import tqdm
import hashlib
from typing import List
import asyncio
import nest_asyncio

# 全局日志配置 - 在任何可能设置日志级别的导入之前
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 强制刷新日志缓冲
import sys
logger = logging.getLogger("E2GraphRAG")
logger.info("日志系统初始化完成，强制刷新缓冲")

# 显式设置根logger级别为INFO
logging.getLogger().setLevel(logging.INFO)

# 设置第三方库日志级别，防止它们覆盖全局配置
logging.getLogger('faiss.loader').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('neo4j').setLevel(logging.WARNING)
logging.getLogger('neo4j.notifications').setLevel(logging.WARNING)

logger = logging.getLogger("E2GraphRAG")
logging.getLogger('E2GraphRAG').setLevel(logging.INFO)

# 导入模块 - 在日志配置之后
from extract_graph import load_nlp
from utils import Timer, load_dataset
from process_utils import build_tree_task, extract_graph_task, build_dense_index_wrapper
from query_refactored import Retriever
from hybrid_integration import apply_hybrid_retrieval
from prompt_dict import Prompts
from provider.public_llm_provider import OpenAICompatibleProvider, call_llm
from collections import namedtuple
from path_utils import get_answer_dir, get_result_file, get_cache_folder, get_query_time_file, get_config_filename
from path_utils_improved import get_organized_paths, get_organized_cache_folder, get_stable_cache_path, print_final_summary
from constants import DATASET_NOVELQA, DATASET_LONGMEMEVAL, DATASET_INFINITE_QA_LOADER, DATASET_LOCOMO, SPLIT_SEQUENTIAL, SPLIT_ROUND_BASED, LANGUAGE_ZH
from decorators import log_exceptions, ensure_dir, timeit
from toolkits.token_tracker import init_token_tracker, get_token_tracker, STAGE_EXTRACT, STAGE_SUMMARY, STAGE_UPDATE, STAGE_GENERATION
# ===================== 基础功能函数 =====================

# ===================== 恢复管理器 =====================
class ResumeManager:
    """管理测试中断后的恢复功能"""

    def __init__(self, config_bundle, paths=None):
        self.config_bundle = config_bundle
        self.paths = paths

        # 设置恢复状态文件路径
        if paths:
            self.resume_dir = os.path.join(paths['cache_dir'], 'resume_states')
        else:
            self.resume_dir = os.path.join(config_bundle.cache_path, 'resume_states')

        ensure_dir(self.resume_dir)
        self.main_state_file = os.path.join(self.resume_dir, 'resume_state.json')

    def get_resume_state(self):
        """获取当前的恢复状态"""
        if os.path.exists(self.main_state_file):
            try:
                with open(self.main_state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def save_resume_state(self, state):
        """保存当前的恢复状态"""
        with open(self.main_state_file, 'w') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)

    def get_piece_state_file(self, piece_name):
        """获取数据项状态文件路径"""
        return os.path.join(self.resume_dir, f'piece_{piece_name}_state.json')

    def get_piece_state(self, piece_name):
        """获取特定数据项的状态"""
        state_file = self.get_piece_state_file(piece_name)
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def save_piece_state(self, piece_name, state):
        """保存特定数据项的状态"""
        state_file = self.get_piece_state_file(piece_name)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)

    def is_piece_completed(self, piece_name):
        """检查数据项是否已完成"""
        piece_state = self.get_piece_state(piece_name)
        return piece_state.get('completed', False)

    def is_stage_completed(self, piece_name, stage):
        """检查特定阶段是否已完成"""
        piece_state = self.get_piece_state(piece_name)
        return piece_state.get('stages', {}).get(stage, False)

    def mark_stage_completed(self, piece_name, stage):
        """标记阶段为已完成"""
        piece_state = self.get_piece_state(piece_name)
        if 'stages' not in piece_state:
            piece_state['stages'] = {}
        piece_state['stages'][stage] = True
        piece_state['last_update'] = datetime.now().isoformat()
        self.save_piece_state(piece_name, piece_state)

    def mark_piece_completed(self, piece_name):
        """标记数据项为已完成"""
        piece_state = self.get_piece_state(piece_name)
        piece_state['completed'] = True
        piece_state['completion_time'] = datetime.now().isoformat()
        self.save_piece_state(piece_name, piece_state)

    def validate_cache_files(self, cache_folder, check_dense_index=False):
        """验证缓存文件的完整性"""
        required_files = ['tree.json', 'graph.json', 'index.json', 'appearance_count.json']
        # 可选检查密集索引文件
        if check_dense_index:
            required_files.extend(['dense_index.faiss', 'dense_index_mapping.json'])
        for file_name in required_files:
            file_path = os.path.join(cache_folder, file_name)
            if not os.path.exists(file_path):
                return False
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return False
        return True

    def get_completed_qa_indices(self, piece_name, result_file):
        """获取已完成的QA问题索引"""
        if not os.path.exists(result_file):
            return set()

        try:
            with open(result_file, 'r') as f:
                results = json.load(f)

            if str(piece_name) in results:
                piece_results = results[str(piece_name)]
                if isinstance(piece_results, list):
                    return set(range(len(piece_results)))
            return set()
        except (json.JSONDecodeError, FileNotFoundError):
            return set()

    def clean_incomplete_files(self, cache_folder, clean_dense_index=False):
        """清理不完整的中间文件"""
        temp_files = ['temp_tree.json', 'temp_graph.json', 'temp_index.json']
        # 可选清理密集索引文件
        if clean_dense_index:
            temp_files.extend(['temp_dense_index.faiss', 'temp_dense_index_mapping.json'])
        for temp_file in temp_files:
            temp_path = os.path.join(cache_folder, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"清理不完整文件: {temp_path}")

    def clean_all_cache_files(self, cache_folder):
        """清理所有缓存文件，包括临时文件和最终结果"""
        logger.info(f"清理缓存文件: {cache_folder}")
        if os.path.exists(cache_folder):
            for item_name in os.listdir(cache_folder):
                item_path = os.path.join(cache_folder, item_name)
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        logger.info(f"删除文件: {item_path}")
                    except Exception as e:
                        logger.warning(f"删除文件失败 {item_path}: {e}")
                elif os.path.isdir(item_path):
                    try:
                        os.rmdir(item_path)
                        logger.info(f"删除目录: {item_path}")
                    except Exception as e:
                        logger.warning(f"删除目录失败 {item_path}: {e}")
        else:
            logger.info(f"缓存目录不存在，无需清理: {cache_folder}")

    def log_resume_status(self, piece_name, skipped_stages):
        """记录恢复状态日志"""
        if skipped_stages:
            logger.info(f"恢复数据项 {piece_name}: 跳过阶段 {', '.join(skipped_stages)}")
        else:
            logger.info(f"开始处理数据项 {piece_name}: 无可跳过的阶段")
ConfigBundle = namedtuple("ConfigBundle", [
    "llm_api", "dataset_name", "dataset_path",
    "cluster_length", "cluster_overlap", "cluster_merge_num",
    "answer_path", "cache_path", "split_method", "temporal_enable", "language", "retriever_kwargs",
    "max_tokens", "extraction_method", "graphiti_config", "parallel_api_calls", "resume",
    "batch_size", "use_batch_api", "llm_parallel_config",
    "enable_entity_clustering", "cluster_threshold", "use_user_only",  # 新增聚类相关配置和user-only配置
    "embedding_config"  # 新增embedding统一配置
])

def extract_config_bundle(configs):
    return ConfigBundle(
        llm_api=get_llm_api_config(configs),
        dataset_name=configs["dataset"]["dataset_name"],
        dataset_path=configs["dataset"]["dataset_path"],
        cluster_length=configs["cluster"]["length"],
        cluster_overlap=configs["cluster"]["overlap"],
        cluster_merge_num=configs["cluster"]["merge_num"],
        answer_path=configs["paths"]["answer_path"],
        cache_path=configs["paths"]["cache_path"],
        split_method=configs.get("split_method", "sequential"),
        temporal_enable=configs.get("temporal", {}).get("enable", False),
        language=configs.get("language", "en"),
        retriever_kwargs=configs["retriever"]["kwargs"],
        max_tokens=configs["llm"].get("max_tokens", 1024),
        extraction_method=configs.get("extraction", {}).get("method", "spacy"),
        graphiti_config=configs.get("graphiti", {}),
        parallel_api_calls=configs.get("parallel_api_calls", 4),
        resume=configs.get("resume", {}),
        batch_size=configs.get("batch_size", 10),
        use_batch_api=configs.get("use_batch_api", False),
        llm_parallel_config=configs.get("llm_parallel", {
            "enable_parallel": True,
            "max_parallel_llm": 8,
            "llm_retry_count": 3,
            "llm_timeout": 60
        }),
        enable_entity_clustering=configs.get("entity_clustering", {}).get("enable", False),  # 新增
        cluster_threshold=configs.get("entity_clustering", {}).get("threshold", 0.1),  # 新增
        use_user_only=configs.get("use_user_only", False),  # 新增：是否只使用user-message构建图谱和persona
        embedding_config=get_embedding_config(configs)  # 新增：embedding统一配置（解析后的配置）
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--query-only", action="store_true", help="仅使用缓存重新回答问题，跳过索引构建")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重新抽取，不使用缓存")
    parser.add_argument("--rebuild-snapshots", action="store_true", help="强制重新构建snapshots.json，不覆盖tree/index")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["query_only"] = args.query_only  # 新增：将参数添加到配置中
    config["force_rebuild"] = args.force_rebuild # 新增：将参数添加到配置中
    config["rebuild_snapshots"] = args.rebuild_snapshots  # 新增：强制重新构建snapshots
    return config


def setup_logger():
    return logging.getLogger(__name__)

# 移除tokenizer加载，不再需要本地模型

def get_llm_api_config(configs):
    """Resolve a public OpenAI-compatible chat backend without loading credentials."""
    llm_config = configs["llm"]
    llm_parallel = configs.get("llm_parallel", {})
    logical_model = llm_config.get("api_model", llm_config.get("model", "gpt-4o-mini"))
    backend_model = llm_config.get("backend_model", logical_model)
    resolved = {
        "type": "openai_compatible",
        "model": logical_model,
        "backend_model": backend_model,
        "base_url": llm_config.get("base_url"),
        "api_key_env": llm_config.get("api_key_env", "OPENAI_API_KEY"),
        "max_tokens": llm_config.get("max_tokens", 1024),
        "temperature": llm_config.get("temperature", 0),
        "max_retries": llm_parallel.get("llm_retry_count", 3),
        "timeout": llm_parallel.get("llm_timeout", 120),
    }
    # Graphiti has a separate structured-output budget. Preserve the optional
    # override instead of silently dropping it during config normalization.
    if "graphiti_max_tokens" in llm_config:
        resolved["graphiti_max_tokens"] = llm_config["graphiti_max_tokens"]
    if "api_protocol" in llm_config:
        resolved["api_protocol"] = llm_config["api_protocol"]
    return resolved

def load_llm(configs):
    # 统一返回MT API配置
    return get_llm_api_config(configs)

def get_embedding_config(configs):
    """Resolve public embedding settings; secrets remain in environment variables."""
    embedding_config = configs.get("embedding", {})
    default_model = "text-embedding-3-small"
    default_dimension_map = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    model = embedding_config.get("model", default_model)
    return {
        "backend": embedding_config.get("backend", "openai_compatible"),
        "model": model,
        "dimension": embedding_config.get("dimension", default_dimension_map.get(model, 1536)),
        "base_url": embedding_config.get("base_url"),
        "api_key_env": embedding_config.get("api_key_env", "OPENAI_API_KEY"),
        "timeout": embedding_config.get("timeout", 120),
        "batch_size": embedding_config.get("batch_size", 128),
    }

def generate_config_hash(configs):
    key_configs = {
        "retriever": configs.get("retriever", {}),
        "cluster": configs.get("cluster", {}),
        "llm_model": configs["llm"].get("api_model", "gpt-4o-mini"),
        "split_method": configs.get("split_method", "sequential"),
        "temporal": configs.get("temporal", {}),
        "language": configs.get("language", "en")
    }
    config_str = json.dumps(key_configs, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config_hash

def load_llm_from_bundle(config_bundle):
    return OpenAICompatibleProvider.from_config(config_bundle.llm_api)

def generate_config_hash_from_bundle(config_bundle):
    key_configs = {
        "llm_model": config_bundle.llm_api.get("model", "gpt-4o-mini"),
        "split_method": config_bundle.split_method,
        "temporal_enable": config_bundle.temporal_enable,
        "language": config_bundle.language,
        "extraction_method": config_bundle.extraction_method,
        "enable_entity_clustering": config_bundle.enable_entity_clustering,
        "cluster_threshold": config_bundle.cluster_threshold
    }
    config_str = json.dumps(key_configs, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config_hash

def save_results(res, config_bundle, piece_name, paths=None):
    """保存结果到文件，支持新旧两种路径系统"""

    if paths:  # 使用新的路径系统
        result_file = paths['result_file']
    else:  # 使用旧的路径系统（向后兼容）
        config_hash = generate_config_hash_from_bundle(config_bundle)
        data_file = config_bundle.dataset_path.split("/")[-1].split(".")[0]
        desc_parts = []
        if config_bundle.split_method != SPLIT_SEQUENTIAL:
            desc_parts.append(f"split_{config_bundle.split_method}")
        desc = "_".join(desc_parts) if desc_parts else "default"
        config_filename = get_config_filename(data_file, desc, config_hash)
        dataset_name = config_bundle.dataset_name
        answer_dir = ensure_dir(get_answer_dir(config_bundle.answer_path, dataset_name))
        result_file = get_result_file(answer_dir, config_filename)

    # 读取现有结果（如果存在）
    if os.path.exists(result_file):
        try:
            with open(result_file, "r") as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = {}
    else:
        all_results = {}

    # 添加新结果
    all_results[str(piece_name)] = res

    # 添加配置信息（如果不存在）
    if "config_info" not in all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results["config_info"] = {
            "timestamp": timestamp,
            "config_hash": generate_config_hash_from_bundle(config_bundle) if not paths else "N/A",
            "description": config_bundle.split_method,
            "key_configs": {
                "llm_model": config_bundle.llm_api.get("model", "gpt-4o-mini"),
                "split_method": config_bundle.split_method,
                "temporal_enable": config_bundle.temporal_enable,
                "language": config_bundle.language,
                "extraction_method": config_bundle.extraction_method
            }
        }

    # 保存结果
    try:
        # 确保目录存在
        result_dir = os.path.dirname(result_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        with open(result_file, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        # 验证文件是否成功写入
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            logger.info(f"结果已保存到: {result_file}")
        else:
            logger.error(f"结果保存失败: 文件不存在或为空 - {result_file}")
            raise IOError(f"结果文件保存失败: {result_file}")
    except Exception as e:
        logger.error(f"保存结果时发生错误: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    return result_file


def simple_text_split(text: str, length: int, overlap: int) -> List[str]:
    """简化的字符级文本分割，替代基于tokenizer的分割"""
    chunks = []
    for i in range(0, len(text), length - overlap):
        chunk = text[i:i+length]
        chunks.append(chunk)
    return chunks


def chunk_text(data_piece, config_bundle):
    if config_bundle.split_method == SPLIT_ROUND_BASED:
        if "rounds" in data_piece:
            text = []
            use_user_only = getattr(config_bundle, 'use_user_only', False)
            # 优先使用数据中已有的session_id（如locomo数据）
            base_session_id = data_piece.get("session_id")

            for session_index, session_rounds in enumerate(data_piece["rounds"]):
                # 优先使用round_data中的session_id，其次使用数据中的session_id，最后使用枚举生成的
                if session_rounds and len(session_rounds) > 0:
                    first_round_session_id = session_rounds[0].get("session_id")
                    if first_round_session_id:
                        session_id = first_round_session_id
                    elif base_session_id:
                        session_id = base_session_id
                    else:
                        session_id = f"session_{session_index}"
                elif base_session_id:
                    session_id = base_session_id
                else:
                    session_id = f"session_{session_index}"

                for turn_index, round_data in enumerate(session_rounds):
                    user_content = round_data["user"]["content"]
                    assistant_content = round_data["assistant"]["content"]
                    date = round_data.get("date", "")
                    full_text = f"Date: {date}\nUser: {user_content}\nAssistant: {assistant_content}"
                    user_only_text = f"Date: {date}\nUser: {user_content}"
                    global_turn_index = len(text)

                    if use_user_only:
                        # 返回字典格式，包含完整文本、仅user文本及session元数据
                        text.append({
                            "full_text": full_text,
                            "user_only_text": user_only_text,
                            "session_index": session_index,
                            "session_id": session_id,
                            "session_turn_index": turn_index,
                            "global_turn_index": global_turn_index
                        })
                    else:
                        # 向后兼容：返回字符串格式
                        text.append(full_text)
        else:
            raise ValueError("数据格式不支持：缺少book字段")
    return text


@log_exceptions()
async def parallel_build_extract(text, config_bundle, cache_folder, length, overlap, merge_num, group_id=None, rebuild_snapshots=False):
    timer = Timer()
    enable_temporal = config_bundle.temporal_enable
    enable_clustering = config_bundle.enable_entity_clustering
    cluster_threshold = config_bundle.cluster_threshold

    # 保存原始text（如果use_user_only启用，用于后续保存完整的turn到leaf节点）
    use_user_only = getattr(config_bundle, 'use_user_only', False)
    original_text = text if use_user_only else None

    with timer.timer("total"):
        # ======= 第1步：先进行图抽取（必须先有实体数据）=======
        logger.info("第1步：抽取实体图...")
        # 设置抽取阶段
        tracker = get_token_tracker()
        if tracker:
            tracker.set_stage(STAGE_EXTRACT)
        extract_args = (
            text,
            cache_folder,
            config_bundle.language,
            False,
            None,  # 不需要llm_path
            config_bundle,
            group_id  # 传递group_id
        )
        extract_res, extract_time_cost = await asyncio.to_thread(extract_graph_task, extract_args)
        if tracker:
            tracker.clear_stage()

        # 解包图数据
        if len(extract_res) == 4:
            G, index, appearance_count, text_edge_mapping = extract_res
        else:
            G, index, appearance_count = extract_res
            text_edge_mapping = {}

        logger.info(f"图抽取完成，耗时: {extract_time_cost}秒")
        logger.info(f"提取到 {len(G.nodes()) if hasattr(G, 'nodes') else 0} 个实体节点")

        # ======= 第2步：基于实体聚类构建树 =======
        if enable_clustering and group_id:
            logger.info("第2步：基于实体聚类构建树...")
            # 设置总结阶段
            tracker = get_token_tracker()
            if tracker:
                tracker.set_stage(STAGE_SUMMARY)
            try:
                # 导入聚类树构建模块
                from build_tree_clustering_time import build_tree_with_clustering_sync
                from graphiti_extractor import GraphitiExtractor

                # 初始化Graphiti
                graphiti_extractor = GraphitiExtractor(config_bundle)

                # 如果启用use_user_only，准备只包含user-message的文本用于树构建
                text_for_build = text
                if use_user_only and original_text:
                    text_for_build = []
                    for chunk in original_text:
                        if isinstance(chunk, dict) and "user_only_text" in chunk:
                            text_for_build.append(chunk["user_only_text"])
                        else:
                            text_for_build.append(chunk)

                # 在单独的线程中运行Graphiti初始化和树构建
                def init_and_build():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(graphiti_extractor._initialize_graphiti())

                        # 使用同步版本的聚类树构建函数
                        # 如果rebuild_snapshots=True，则使用snapshots_only模式
                        snapshots_only = rebuild_snapshots
                        logger.info(f"parallel_build_extract: rebuild_snapshots={rebuild_snapshots}, snapshots_only={snapshots_only}")
                        build_res, build_time_cost, cluster_index = build_tree_with_clustering_sync(
                            text_chunks=text_for_build,
                            llm_config=config_bundle.llm_api,
                            cache_folder=cache_folder,
                            graphiti_driver=graphiti_extractor.graphiti.driver,
                            group_id=group_id,
                            language=config_bundle.language,
                            cluster_threshold=cluster_threshold,
                            merge_num=merge_num,
                            index=index,
                            text_edge_mapping=text_edge_mapping,
                            original_text=original_text,  # 传递原始text用于保存完整的turn
                            snapshots_only=snapshots_only  # 传递snapshots_only参数
                        )

                        loop.run_until_complete(graphiti_extractor.graphiti.close())

                        return build_res, build_time_cost
                    finally:
                        # 移除 loop.close()，让事件循环自然结束，避免影响后续的 build_dense_index
                        pass

                # 在单独的线程中运行
                build_res, build_time_cost = await asyncio.to_thread(init_and_build)
                logger.info(f"基于聚类的树构建完成，耗时: {build_time_cost}秒")
                if tracker:
                    tracker.clear_stage()

            except Exception as e:
                logger.error(f"基于聚类的树构建失败: {e}")
                raise RuntimeError(
                    "Temporal entity clustering failed; refusing to change the TSM method."
                ) from e
        else:
            # 使用传统的树构建方法
            logger.info("第2步：使用传统方法构建树...")
            # 设置总结阶段
            tracker = get_token_tracker()
            if tracker:
                tracker.set_stage(STAGE_SUMMARY)
            build_args = (
                None,  # 不需要llm_path
                "cpu",  # 不需要GPU设备
                text,
                cache_folder,
                None,  # 不需要llm_path
                length,
                overlap,
                merge_num,
                None,  # 不需要torch.float16
                config_bundle.language,
                config_bundle
            )
            build_res, build_time_cost = await asyncio.to_thread(build_tree_task, build_args)
            if tracker:
                tracker.clear_stage()

        logger.info(f"树构建完成，耗时: {build_time_cost}秒")

        # 构建密集检索索引（如果启用）- 改为同步执行
        dense_index_result = None
        dense_index_time_cost = -1
        if config_bundle.retriever_kwargs.get('enable_dense_retrieval', False) and \
           config_bundle.retriever_kwargs.get('prebuilt_index', False):
            logger.info("开始构建密集检索索引...")
            # 准备embedder配置（从统一配置读取）
            embedding_config = getattr(config_bundle, 'embedding_config', {})
            embedder_config = dict(embedding_config)
            embedder_config["type"] = "PublicEmbedder"
            embedder_config["batch_size"] = config_bundle.retriever_kwargs.get(
                "embedding_batch_size", embedding_config.get("batch_size", 128)
            )
            dense_index_args = (build_res, cache_folder, embedder_config, config_bundle)
            # 改为同步执行，避免事件循环影响顺序
            dense_index_result, dense_index_time_cost = build_dense_index_wrapper(dense_index_args)
            logger.info(f"密集索引构建完成，耗时: {dense_index_time_cost}秒")

        temporal_time_cost = -1
        if enable_temporal:
            logger.info("开始串行执行时间抽取任务...")
            if len(extract_res) == 4:
                G, index, appearance_count, text_edge_mapping = extract_res
            else:
                raise ValueError(f"extract_res格式不正确，期望4个元素，实际{len(extract_res)}个")
            # 统一使用MT API进行时间抽取
            llm_pipeline = config_bundle.llm_api
            from extract_graph import add_temporal_to_graph
            # 设置更新阶段
            tracker = get_token_tracker()
            if tracker:
                tracker.set_stage(STAGE_UPDATE)
            temporal_start_time = time.time()
            G_temporal, temporal_edges = await asyncio.to_thread(add_temporal_to_graph, text_edge_mapping, llm_pipeline, config_bundle.language)
            temporal_time_cost = time.time() - temporal_start_time
            extract_res = (G_temporal, index, appearance_count, temporal_edges)
            logger.info(f"时间抽取完成，耗时: {temporal_time_cost}秒")
            if tracker:
                tracker.clear_stage()

    # 添加异步等待函数
    async def ensure_tasks_completed():
        try:
            # 获取当前事件循环中的所有任务
            current_task = asyncio.current_task()
            all_tasks = [task for task in asyncio.all_tasks() if task != current_task]
            if all_tasks:
                logger.info(f"等待 {len(all_tasks)} 个待完成任务...")
                await asyncio.gather(*all_tasks, return_exceptions=True)
                logger.info("所有pending协程已等待完成")
        except Exception as e:
            logger.error(f"等待协程失败: {e}")

    # 调用等待函数
    await ensure_tasks_completed()

    gc.collect()
    logger.info("-" * 15)
    logger.info(f"total time: {timer['total']} seconds")
    logger.info(f"build time: {build_time_cost} seconds")
    logger.info(f"extract time: {extract_time_cost} seconds")
    if 'dense_index_time_cost' in locals() and dense_index_time_cost != -1:
        logger.info(f"dense index time: {dense_index_time_cost} seconds")
    if enable_temporal and temporal_time_cost != -1:
        logger.info(f"temporal time: {temporal_time_cost} seconds")
    logger.info("-" * 15)
    if extract_time_cost != -1 and build_time_cost != -1:
        with open(os.path.join(cache_folder, "time_cost.txt"), "w") as f:
            f.write(f"total time: ||{timer['total']}|| seconds\n")
            f.write(f"build time: ||{build_time_cost}|| seconds\n")
            f.write(f"extract time: ||{extract_time_cost}|| seconds\n")
            if 'dense_index_time_cost' in locals() and dense_index_time_cost != -1:
                f.write(f"dense index time: ||{dense_index_time_cost}|| seconds\n")
            if enable_temporal and temporal_time_cost != -1:
                f.write(f"temporal time: ||{temporal_time_cost}|| seconds\n")
    # 返回密集索引结果（如果有）
    if 'dense_index_result' in locals():
        return build_res, extract_res, dense_index_result
    return build_res, extract_res

def _get_qa_config_info(config_bundle):
    """提取QA处理的配置信息"""
    config_hash = generate_config_hash_from_bundle(config_bundle)
    desc_parts = []
    if config_bundle.split_method != SPLIT_SEQUENTIAL:
        desc_parts.append(f"split_{config_bundle.split_method}")
    desc = "_".join(desc_parts) if desc_parts else "default"
    data_file = config_bundle.dataset_path.split("/")[-1].split(".")[0]
    config_filename = get_config_filename(data_file, desc, config_hash)
    dataset_name = config_bundle.dataset_name
    use_api = config_bundle.llm_api["type"] == "openai_compatible"

    return {
        "config_hash": config_hash,
        "config_filename": config_filename,
        "dataset_name": dataset_name,
        "use_api": use_api
    }

def _extract_golden_evidence(qa_piece):
    """提取黄金证据"""
    golden_evidence = {}
    if "answer_rounds" in qa_piece and "answer_sessions" in qa_piece:
        golden_evidence = {
            "answer_rounds": qa_piece.get("answer_rounds", []),
            "answer_sessions": qa_piece.get("answer_sessions", [])
        }
    elif "evidence" in qa_piece:
        golden_evidence = {
            "evidence": qa_piece.get("evidence", [])
        }
    return golden_evidence


def _load_query_filter(path):
    """Load an optional QID allowlist without changing the memory corpus."""
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        qids = payload.get("selected_qids", payload.get("qids", []))
    else:
        qids = payload
    if not isinstance(qids, list) or not all(isinstance(item, str) for item in qids):
        raise ValueError(f"query filter must contain a list of string QIDs: {path}")
    return set(qids)


def _load_cached_dense_index(cache_folder, tree, config_bundle):
    """Load a validated FAISS cache for query-only and resumed runs."""
    if not config_bundle.retriever_kwargs.get("prebuilt_index", False):
        return None
    try:
        from build_dense_index import load_dense_index_cache, validate_index_cache

        if not validate_index_cache(cache_folder, tree):
            return None
        index, doc_ids, _ = load_dense_index_cache(cache_folder)
        if index is None or not doc_ids:
            return None
        logger.info(f"Loaded cached FAISS index for {cache_folder}: {len(doc_ids)} documents")
        return index, doc_ids
    except Exception as exc:
        logger.warning(f"Cached FAISS index unavailable for {cache_folder}: {exc}")
        return None

@log_exceptions()
async def _process_single_qa(qa_piece, j, retriever, config_bundle, llm, i, logger, config_info):
    """处理单个QA问题的核心逻辑"""
    # 设置生成阶段
    tracker = get_token_tracker()
    if tracker:
        tracker.set_stage(STAGE_GENERATION)

    try:
        question = qa_piece["question"]
        questtion_date = qa_piece.get('question_date', None)  # 使用get方法，避免KeyError
        answer = qa_piece["answer"]
        question_id = qa_piece.get("question_id", f"unknown_{j}")
        golden_evidence = _extract_golden_evidence(qa_piece)

        # 查询阶段
        query_start_time = time.time()
        logger.info(f"=== DEV版本主程序检索调用开始 ===")
        logger.info(f"问题: {question}")
        logger.info(f"检索参数: {config_bundle.retriever_kwargs}")
        # 将 question_date 传递给 query 方法
        query_kwargs = config_bundle.retriever_kwargs.copy()
        if questtion_date:
            query_kwargs['question_date'] = questtion_date
        model_supplement = retriever.query(question, **query_kwargs)
        query_end_time = time.time()
        query_time = query_end_time - query_start_time
        logger.info(f"=== DEV版本主程序检索调用结束，耗时: {query_time:.3f}秒 ===")
        logger.info(f"DEV版本检索结果: {model_supplement}")

        # 记录查询时间
        answer_dir = ensure_dir(get_answer_dir(config_bundle.answer_path, config_info["dataset_name"]))
        query_time_file = get_query_time_file(answer_dir, config_info["config_filename"])
        with open(query_time_file, "a") as f:
            f.write(f"piece_{i}_question_{j}: query time: {query_time}\n")

        evidences = model_supplement["chunks"]
        logger.info(f"检索耗时: {query_time:.3f}s")
        answer_start_time = time.time()

        # 生成答案
        if config_bundle.dataset_name.lower() in [DATASET_LONGMEMEVAL.lower(), DATASET_LOCOMO.lower()]:
            # 统一使用LongMemEval的prompt逻辑（QA_prompt_mirix_improved）
            # 之前的实验表明zep_locomo_prompt效果不佳
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": Prompts["QA_prompt_mirix_improved"].format(question_date=questtion_date, question=question, evidence=evidences)}
            ]

            # 保留旧逻辑作为注释，以备参考
            # if config_bundle.dataset_name.lower() == DATASET_LOCOMO.lower():
            #     # locomo数据集使用zep_locomo_prompt，传入context和question
            #     messages = [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": Prompts["zep_locomo_prompt"].format(context=evidences, question=question)}
            #     ]
            # else:
            #     # LongMemEval数据集使用QA_prompt_mirix_improved，传入question_date、question和evidence
            #     messages = [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": Prompts["QA_prompt_mirix_improved"].format(question_date=questtion_date, question=question, evidence=evidences)}
            #          #Prompts["QA_prompt_simple"].format(question_date=questtion_date, question=question, evidence=evidences)}
            #     ]

            if config_bundle.dataset_name.lower() == DATASET_LOCOMO.lower():
                logger.info("Locomo数据集：使用统一的QA_prompt_mirix_improved")
            # logger.info(f"input_text: {messages[1]['content']}")  # 移除冗长日志输出

            # 注释掉原来的复杂prompt代码（保留以备将来恢复）
            """
            input_text = Prompts["QA_prompt_mirix_improved"].format(question_date=questtion_date, question=question, evidence=evidences)
            # logger.info(f"input_text: {input_text}")  # 移除冗长日志输出
            if config_info["use_api"]:
                try:
                    messages = [{"role": "system", "content": Prompts["QA_prompt_answer_system"]},
                               {"role": "user", "content": input_text}]
            """
            if config_info["use_api"]:
                try:
                    for message in messages:
                        logger.info(f"{message['role']}: {message['content']}")

                    if hasattr(llm, 'chat_completion'):
                        model_for_call = config_bundle.llm_api.get(
                            "backend_model", config_bundle.llm_api.get("model", "gpt-4o-mini")
                        )

                        response = llm.chat_completion(
                            messages=messages,
                            model=model_for_call,
                            temperature=config_bundle.llm_api["temperature"],
                            max_tokens=config_bundle.llm_api["max_tokens"],
                            op_type=STAGE_GENERATION  # 传递操作类型
                        )
                        output_text = response.choices[0].message.content
                    else:
                        output_text = call_llm(
                            messages=messages,
                            model=llm["model"],
                            temperature=llm["temperature"],
                            max_tokens=llm["max_tokens"],
                            base_url=llm.get("base_url"),
                            api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"),
                            op_type=STAGE_GENERATION  # 传递操作类型
                        )

                    if not output_text:
                        logger.error("API调用返回None")
                        output_text = "无法生成答案"
                except Exception as e:
                    logger.error(f"API调用出错: {e}")
                    logger.error(traceback.format_exc())
                    output_text = "API调用失败"
            else:
                logger.error("未启用API且不支持本地直接调用，请检查配置")
                output_text = "配置错误：未启用API"
        else:
            raise ValueError(f"不支持的数据集: {config_bundle.dataset_name}")

        answer_time = time.time() - answer_start_time
        logger.info(f"问题 {j} 答案生成完成，耗时: {answer_time:.3f}秒")

        question_result = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "output_text": output_text,
            "hypothesis": output_text,
            "retrieved_evidences": evidences,
            "golden_evidence": golden_evidence,
            "query_time": query_time,
            "generation_time": answer_time
        }
        if model_supplement.get("retrieval_trace") is not None:
            question_result["retrieval_trace"] = model_supplement["retrieval_trace"]
        return question_result
    finally:
        # 清除生成阶段
        if tracker:
            tracker.clear_stage()

@log_exceptions()
async def process_qa(qa, retriever, config_bundle, llm, date, i, logger):
    """简化的QA处理函数，使用重构后的共同逻辑"""
    res = []
    config_info = _get_qa_config_info(config_bundle)

    for j, qa_piece in enumerate(qa):
        question_result = await _process_single_qa(qa_piece, j, retriever, config_bundle, llm, i, logger, config_info)
        res.append(question_result)

    return res


@log_exceptions()
async def process_qa_with_resume(qa, retriever, config_bundle, llm, date, i, logger,
                          completed_qa_indices, result_file, piece_name, paths):
    """
    支持恢复的QA处理函数，可以跳过已完成的问题，使用并行处理多个QA问题
    """
    res = []
    config_info = _get_qa_config_info(config_bundle)

    # 加载现有结果（如果存在）
    existing_results = []
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                all_results = json.load(f)
                if str(piece_name) in all_results and isinstance(all_results[str(piece_name)], list):
                    existing_results = all_results[str(piece_name)]
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []

    # 收集需要处理的问题
    tasks_to_process = []
    for j, qa_piece in enumerate(qa):
        # 如果此问题已完成，跳过并使用现有结果
        if j in completed_qa_indices and j < len(existing_results):
            res.append((j, existing_results[j]))  # 存储索引和结果
            logger.info(f"跳过已完成的问题 {j}: {qa_piece['question'][:50]}...")
            continue

        # 添加到待处理任务列表
        tasks_to_process.append((j, qa_piece))

    # 并行处理待处理的问题
    max_workers = config_bundle.llm_parallel_config.get("max_parallel_llm", 8)
    logger.info(f"开始并行处理 {len(tasks_to_process)} 个QA问题，使用 {max_workers} 个线程")

    def process_qa_task(j, qa_piece):
        """单个QA问题的处理任务"""
        try:
            question_start_time = time.time()
            logger.info(f"开始处理问题 {j}: {qa_piece['question'][:50]}...")

            # 使用同步版本的QA处理逻辑（将异步逻辑包装为同步）
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()

            async def async_process():
                return await _process_single_qa(qa_piece, j, retriever, config_bundle, llm, i, logger, config_info)

            # 在新的事件循环中运行异步任务
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                question_result = loop.run_until_complete(async_process())
            except RuntimeError:
                # 如果已有事件循环，使用同步等待
                question_result = asyncio.run(async_process())

            question_total_time = time.time() - question_start_time
            logger.info(f"问题 {j} 处理完成，总耗时: {question_total_time:.3f}秒")

            return j, question_result, question_total_time
        except Exception as e:
            logger.error(f"处理问题 {j} 时发生错误: {e}")
            logger.error(traceback.format_exc())
            # 返回错误结果
            return j, {
                "question_id": qa_piece.get("question_id", f"error_{j}"),
                "question": qa_piece["question"],
                "answer": qa_piece["answer"],
                "output_text": f"处理失败: {e}",
                "hypothesis": f"处理失败: {e}",
                "retrieved_evidences": "",
                "golden_evidence": _extract_golden_evidence(qa_piece),
                "query_time": 0
            }, 0

    # 使用线程池并行执行任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_qa_task, j, qa_piece)
            for j, qa_piece in tasks_to_process
        ]

        # 使用tqdm显示并行处理进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="并行QA处理进度"):
            j, question_result, question_total_time = future.result()
            res.append((j, question_result))

            # 立即保存单个问题的结果，以支持细粒度恢复
            save_single_qa_result(question_result, result_file, piece_name, j)

    # 按索引排序结果
    res.sort(key=lambda x: x[0])
    final_res = [result for _, result in res]

    # 计算平均时间统计
    if final_res:
        query_times = [r.get("query_time", 0) for r in final_res if isinstance(r.get("query_time"), (int, float))]
        generation_times = [r.get("generation_time", 0) for r in final_res if isinstance(r.get("generation_time"), (int, float))]

        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            logger.info(f"检索时间统计: 处理了{len(query_times)}个问题，平均检索时间: {avg_query_time:.3f}秒")
        else:
            avg_query_time = 0.0
            logger.info("检索时间统计: 未找到有效的检索时间记录")

        if generation_times:
            avg_generation_time = sum(generation_times) / len(generation_times)
            logger.info(f"生成时间统计: 处理了{len(generation_times)}个问题，平均生成时间: {avg_generation_time:.3f}秒")
        else:
            avg_generation_time = 0.0
            logger.info("生成时间统计: 未找到有效的生成时间记录")

        # 将统计信息添加到结果中（作为附加信息返回，但不修改原有的结果结构）
        logger.info(f"QA处理统计: 总问题数={len(final_res)}, 平均检索时间={avg_query_time:.3f}秒, 平均生成时间={avg_generation_time:.3f}秒")
    else:
        logger.info("QA处理统计: 无有效结果")

    logger.info(f"并行QA处理完成，共处理 {len(final_res)} 个问题")
    return final_res


def calculate_and_save_time_stats(result_file, config_bundle, paths=None):
    """计算并保存时间统计信息"""
    if not os.path.exists(result_file):
        logger.info("结果文件不存在，跳过时间统计")
        return

    try:
        with open(result_file, 'r') as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.warning(f"无法读取结果文件: {result_file}")
        return

    # 收集所有piece的时间统计
    all_query_times = []
    all_generation_times = []
    total_questions = 0

    for piece_key, piece_results in all_results.items():
        if not isinstance(piece_results, list):
            continue

        for qa_result in piece_results:
            if isinstance(qa_result, dict):
                query_time = qa_result.get("query_time")
                generation_time = qa_result.get("generation_time")

                if isinstance(query_time, (int, float)) and query_time > 0:
                    all_query_times.append(query_time)

                if isinstance(generation_time, (int, float)) and generation_time > 0:
                    all_generation_times.append(generation_time)

                total_questions += 1

    # 计算平均时间和中位数
    import statistics

    if all_query_times:
        avg_query_time = sum(all_query_times) / len(all_query_times)
        median_query_time = statistics.median(all_query_times)
        logger.info(f"全局检索时间统计: 处理了{len(all_query_times)}个问题，平均检索时间: {avg_query_time:.3f}秒，中位数: {median_query_time:.3f}秒")
    else:
        avg_query_time = 0.0
        median_query_time = 0.0
        logger.info("全局检索时间统计: 未找到有效的检索时间记录")

    if all_generation_times:
        avg_generation_time = sum(all_generation_times) / len(all_generation_times)
        median_generation_time = statistics.median(all_generation_times)
        logger.info(f"全局生成时间统计: 处理了{len(all_generation_times)}个问题，平均生成时间: {avg_generation_time:.3f}秒，中位数: {median_generation_time:.3f}秒")
    else:
        avg_generation_time = 0.0
        median_generation_time = 0.0
        logger.info("全局生成时间统计: 未找到有效的生成时间记录")

    # 保存统计信息到结果文件
    if "time_statistics" not in all_results:
        all_results["time_statistics"] = {}

    all_results["time_statistics"].update({
        "total_questions": total_questions,
        "valid_query_times": len(all_query_times),
        "valid_generation_times": len(all_generation_times),
        "avg_query_time": round(avg_query_time, 3),
        "avg_generation_time": round(avg_generation_time, 3),
        "median_query_time": round(median_query_time, 3),
        "median_generation_time": round(median_generation_time, 3),
        "total_query_time": round(sum(all_query_times), 3),
        "total_generation_time": round(sum(all_generation_times), 3),
        "timestamp": datetime.now().isoformat()
    })

    # 保存更新后的结果
    try:
        # 确保目录存在
        result_dir = os.path.dirname(result_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        with open(result_file, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        # 验证文件是否成功写入
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            logger.info(f"时间统计已保存到结果文件: {result_file}")
        else:
            logger.error(f"时间统计保存失败: 文件不存在或为空 - {result_file}")
            raise IOError(f"时间统计文件保存失败: {result_file}")
    except Exception as e:
        logger.error(f"保存时间统计时发生错误: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def save_single_qa_result(qa_result, result_file, piece_name, qa_index):
    """保存单个QA结果，支持增量更新"""
    # 读取现有结果
    if os.path.exists(result_file):
        try:
            with open(result_file, "r") as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = {}
    else:
        all_results = {}

    # 确保数据项结果列表存在
    piece_key = str(piece_name)
    if piece_key not in all_results:
        all_results[piece_key] = []

    # 扩展列表以容纳新的索引
    while len(all_results[piece_key]) <= qa_index:
        all_results[piece_key].append(None)

    # 更新特定问题的结果
    all_results[piece_key][qa_index] = qa_result

    # 保存结果
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)



def build_retriever(tree, graph, config_bundle, dense_index_result=None, group_id=None, cache_folder=None):
    # graph may include an auxiliary mapping (e.g., text_edge_mapping) as the 4th element
    logger.info(f"DEV版本build_retriever：graph长度={len(graph)}")
    if len(graph) == 4:
        G, index, appearance_count, aux = graph
        logger.info(f"DEV版本build_retriever：从graph中获取aux，类型={type(aux)}, 长度={len(aux) if isinstance(aux, dict) else 'N/A'}")
    else:
        G, index, appearance_count = graph
        aux = {}
        logger.warning(f"DEV版本build_retriever：graph只有{len(graph)}个元素，aux设置为空字典")

    # 传递预构建的密集索引（如果有）
    retriever_kwargs = config_bundle.retriever_kwargs.copy()
    if dense_index_result is not None and dense_index_result[0] is not None:
        retriever_kwargs['prebuilt_faiss_index'] = dense_index_result[0]
        retriever_kwargs['prebuilt_doc_ids'] = dense_index_result[1]

    # 传递embedding统一配置
    if hasattr(config_bundle, 'embedding_config'):
        retriever_kwargs['embedding_config'] = config_bundle.embedding_config

    if getattr(config_bundle, 'graphiti_config', {}):
        retriever_kwargs.setdefault('graphiti_config', config_bundle.graphiti_config)
        # 传递数据库名称给 Retriever，实现数据隔离
        neo4j_config = config_bundle.graphiti_config.get('neo4j', {})
        if neo4j_config:
            graphiti_database = os.getenv(
                neo4j_config.get('database_env', 'NEO4J_DATABASE'),
                neo4j_config.get('database', 'neo4j'),
            )
            retriever_kwargs.setdefault('graphiti_database', graphiti_database)
    if group_id is not None:
        retriever_kwargs.setdefault('graphiti_group_id', group_id)
    # 传递cache_folder，供 Retriever 自动加载 snapshots.json
    if cache_folder is not None:
        retriever_kwargs.setdefault('cache_folder', cache_folder)

    # 检查是否启用混合检索
    enable_hybrid = retriever_kwargs.get('enable_hybrid_retrieval', False)

    if enable_hybrid:
        logger.info(f"启用混合检索模式，group_id: {group_id}")
        # 创建传统检索器
        base_retriever = Retriever(tree, G, index, appearance_count, load_nlp(), external_text_map=aux, **retriever_kwargs)
        # 应用混合检索功能
        hybrid_retriever = apply_hybrid_retrieval(base_retriever, config_bundle)
        return hybrid_retriever
    else:
        logger.info("使用传统检索模式")
        # 使用传统检索器
        retriever = Retriever(tree, G, index, appearance_count, load_nlp(), external_text_map=aux, **retriever_kwargs)
        return retriever


@log_exceptions()
async def run_pipeline(configs, logger):
    # 记录开始时间
    start_time = time.time()

    # 检查是否为仅查询模式
    query_only = configs.get("query_only", False)  # 新增：获取query_only标志
    force_rebuild = configs.get("force_rebuild", False) # 新增：获取force_rebuild标志
    rebuild_snapshots = configs.get("rebuild_snapshots", False)  # 新增：强制重新构建snapshots
    query_filter_qids = _load_query_filter(configs.get("query_filter_path"))
    conversation_filter = set(configs.get("conversation_filter", []))
    if query_filter_qids is not None:
        logger.info(
            f"Query filter enabled: {len(query_filter_qids)} QIDs; "
            "memory history remains complete"
        )
    if conversation_filter:
        logger.info(f"Conversation smoke filter enabled: {sorted(conversation_filter)}")
    if query_only:
        logger.info("=== 仅查询模式：使用现有缓存重新回答问题 ===")
    if force_rebuild:
        logger.info("=== 强制重新构建模式：强制重新抽取数据并重建缓存 ===")
    if rebuild_snapshots:
        logger.info("=== 强制重建snapshots模式：重新生成snapshots.json，不覆盖tree/index ===")

    # 1. loaddata
    date = datetime.now().strftime("%Y%m%d")
    config_bundle = extract_config_bundle(configs)

    # 创建组织良好的输出目录
    use_improved_paths = configs.get("use_improved_paths", True)  # 默认使用新路径系统
    paths = None
    if use_improved_paths:
        paths = get_organized_paths(config_bundle)

    # 初始化恢复管理器
    resume_manager = ResumeManager(config_bundle, paths)

    # 初始化Token跟踪器
    model_name = config_bundle.llm_api.get("model", "gpt-4o-mini")
    enable_token_tracking = configs.get("enable_token_tracking", True)  # 默认启用
    token_tracker = init_token_tracker(model_name, enable_tracking=enable_token_tracking)
    if enable_token_tracking:
        logger.info(f"Token跟踪已启用，模型: {model_name}")

    llm = load_llm_from_bundle(config_bundle)  # 不需要tokenizer
    dataset = load_dataset(config_bundle.dataset_name, config_bundle.dataset_path)

    start_index = config_bundle.resume.get('start_index', 0)

    # 设置resume_index.txt的路径（保存在稳定位置，确保下次启动能找到）
    if paths:
        # 使用新的路径系统时，保存在稳定的outputs目录下，而不是带时间戳的运行目录
        stable_output_dir = os.path.join("./outputs", "resume_states")
        os.makedirs(stable_output_dir, exist_ok=True)
        # 根据数据集和配置生成唯一的resume文件名，避免不同配置冲突
        dataset_name = os.path.basename(config_bundle.dataset_path).split('.')[0]
        model_name = config_bundle.llm_api.get("model", "unknown").replace(".", "_")
        resume_filename = f"resume_index_{dataset_name}_{model_name}_{config_bundle.split_method}.txt"
        resume_file = os.path.join(stable_output_dir, resume_filename)
    else:
        # 使用旧的路径系统（兼容）
        resume_file = os.path.join(config_bundle.answer_path, 'resume_index.txt')
    os.makedirs(os.path.dirname(resume_file), exist_ok=True)
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            saved_index = int(f.read().strip())
            if saved_index > start_index:
                start_index = saved_index
                logger.info(f"从保存的索引 {start_index} 恢复，覆盖配置文件中的索引")
            else:
                logger.info(f"使用配置文件中的索引 {start_index}，忽略保存的索引 {saved_index}")
    else:
        logger.info(f"使用配置文件中的索引 {start_index}，未找到保存的索引文件")
        # #region agent log
        # #endregion

    # 修复切片问题：手动创建从start_index开始的索引列表
    total_items = len(dataset)
    remaining_indices = list(range(start_index, total_items))
    if conversation_filter:
        remaining_indices = [
            index
            for index in remaining_indices
            if dataset[index].get("group_id") in conversation_filter
            or dataset.available_ids[index] in conversation_filter
        ]
    # #region agent log
    # #endregion

    # #region agent log
    # #endregion

    for idx, actual_index in tqdm(enumerate(remaining_indices), desc="处理进度", ncols=80, total=len(remaining_indices)):
        # #region agent log
        # #endregion

        # 保存当前索引，以便在中断后可以恢复
        with open(resume_file, 'w') as f:
            f.write(str(actual_index))

        # #region agent log
        # #endregion

        # 获取当前数据项
        data_piece = dataset[actual_index]
        # 生成 group_id，优先使用数据项中的，否则根据配置生成
        base_group_id = data_piece.get("group_id", f"default_{actual_index}")

        # 如果配置了 group_id_prefix，则添加前缀以隔离不同模型的数据
        # 这样可以避免 vLLM 和闭源模型使用同一个 Neo4j 数据库时的数据冲突
        group_id_prefix = None
        if hasattr(config_bundle, 'graphiti_config') and isinstance(config_bundle.graphiti_config, dict):
            group_id_prefix = config_bundle.graphiti_config.get('group_id_prefix', None)

        if group_id_prefix:
            # 如果 base_group_id 已经有前缀，则不重复添加
            if not base_group_id.startswith(group_id_prefix):
                group_id = f"{group_id_prefix}{base_group_id}"
            else:
                group_id = base_group_id
        else:
            group_id = base_group_id

        logger.info(f"使用 group_id: {group_id} (base: {base_group_id}, prefix: {group_id_prefix})")
        piece_name = dataset.available_ids[actual_index]
        dense_index_result = None

        # 检查是否已完成此数据项（仅在非query_only模式下跳过）
        if not query_only and resume_manager.is_piece_completed(piece_name):
            logger.info(f"跳过已完成的数据项: {piece_name}")
            continue
        elif query_only and resume_manager.is_piece_completed(piece_name):
            logger.info(f"仅查询模式：重新处理已完成的数据项: {piece_name}")

        skipped_stages = []

        # 2. chunking (仅在非query_only时执行)
        if not query_only:
            chunks = chunk_text(data_piece, config_bundle)  # 不需要tokenizer
        else:
            logger.info("跳过chunking阶段（仅查询模式）")
            chunks = None  # 占位符，不会使用

        # 3. indexing
        if paths:
            # 生成配置哈希用于区分不同配置的缓存
            config_hash = generate_config_hash_from_bundle(config_bundle)
            cache_folder = get_organized_cache_folder(paths['cache_dir'],
                                                     config_bundle.dataset_path.split("/")[-1].split(".")[0],
                                                     piece_name, config_hash)
            stable_cache_folder = get_stable_cache_path(
                config_bundle.dataset_path.split("/")[-1].split(".")[0],
                piece_name,
                config_hash,
            )
        else:
            data_file = config_bundle.dataset_path.split("/")[-1].split(".")[0]
            cache_folder = get_cache_folder(config_bundle.cache_path, data_file, date, piece_name)
            if not os.path.exists(cache_folder):
                ensure_dir(cache_folder)
            stable_cache_folder = None

        # 修改：在query_only模式下，强制从缓存加载，不构建
        if query_only:
            logger.info(f"仅查询模式：尝试从缓存加载 {piece_name}")

            # 优先使用稳定缓存路径，如果不存在则尝试运行目录缓存
            cache_to_use = stable_cache_folder if stable_cache_folder and os.path.exists(os.path.join(stable_cache_folder, 'tree.json')) else cache_folder

            logger.info(f"使用缓存路径: {cache_to_use}")
            if stable_cache_folder and cache_to_use == stable_cache_folder:
                logger.info("使用稳定缓存路径（基于配置哈希）")
            else:
                logger.info("使用运行目录缓存路径")

            try:
                with open(os.path.join(cache_to_use, 'tree.json'), 'r') as f:
                    tree = json.load(f)
                with open(os.path.join(cache_to_use, 'graph.json'), 'r') as f:
                    graph_data = json.load(f)
                with open(os.path.join(cache_to_use, 'index.json'), 'r') as f:
                    index = json.load(f)
                with open(os.path.join(cache_to_use, 'appearance_count.json'), 'r') as f:
                    appearance_count = json.load(f)
                with open(os.path.join(cache_to_use, 'text_edge_mapping.json'), 'r') as f:
                    text_edge_mapping = json.load(f)

                # 将graph_data转换为NetworkX图对象
                from extract_graph import build_graph
                graph_nx = build_graph(graph_data)

                # 重构图数据结构
                graph = (graph_nx, index, appearance_count, text_edge_mapping)
                dense_index_result = _load_cached_dense_index(cache_to_use, tree, config_bundle)
                skipped_stages.append('indexing')
                logger.info(f"成功从缓存加载: {piece_name}")

                # 如果启用实体聚类且（缺少 snapshots.json 或 强制重建snapshots），则仅重生成 snapshots（不覆盖 tree/index）
                # 但在 query_only 模式下不应该重新生成任何内容
                rebuild_snapshots = configs.get("rebuild_snapshots", False)
                if config_bundle.enable_entity_clustering and group_id and not query_only:
                    snapshots_path = os.path.join(cache_folder, "snapshots.json")
                    if not os.path.exists(snapshots_path) or rebuild_snapshots:
                        reason = "缺失" if not os.path.exists(snapshots_path) else "强制重建"
                        logger.info(f"检测到 snapshots.json {reason}，开始 snapshots_only 重生成: {snapshots_path}")
                        try:
                            from build_tree_clustering_time import build_tree_with_clustering_sync
                            from graphiti_extractor import GraphitiExtractor

                            graphiti_extractor = GraphitiExtractor(config_bundle)

                            def init_and_refresh_snapshots():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    loop.run_until_complete(graphiti_extractor._initialize_graphiti())
                                    # text_chunks 在 snapshots_only 中不会用到
                                    build_tree_with_clustering_sync(
                                        text_chunks=[],
                                        llm_config=config_bundle.llm_api,
                                        cache_folder=cache_folder,
                                        graphiti_driver=graphiti_extractor.graphiti.driver,
                                        group_id=group_id,
                                        language=config_bundle.language,
                                        cluster_threshold=config_bundle.cluster_threshold,
                                        merge_num=config_bundle.cluster_merge_num,
                                        index=index,
                                        text_edge_mapping=text_edge_mapping,
                                        original_text=None,
                                        snapshots_only=True,
                                    )
                                    loop.run_until_complete(graphiti_extractor.graphiti.close())
                                finally:
                                    pass

                            await asyncio.to_thread(init_and_refresh_snapshots)
                            logger.info("snapshots_only 重生成完成")
                        except Exception as e:
                            logger.error(f"snapshots_only 重生成失败: {e}")
            except Exception as e:
                logger.error(f"缓存加载失败（仅查询模式）: {e}")
                logger.error(f"尝试的缓存路径: {cache_to_use}")
                if stable_cache_folder:
                    logger.error(f"稳定缓存路径: {stable_cache_folder}")
                    logger.error(f"运行目录缓存路径: {cache_folder}")
                logger.error("请确保缓存文件存在且有效，或运行完整模式构建缓存")
                continue  # 跳过此数据项，继续下一个
        elif force_rebuild:
            # 强制重新构建
            logger.info(f"强制重新构建模式：清理缓存并重新构建 {piece_name}")
            resume_manager.clean_all_cache_files(cache_folder)
            build_result = await parallel_build_extract(
                chunks, config_bundle, cache_folder,
                config_bundle.cluster_length, config_bundle.cluster_overlap,
                config_bundle.cluster_merge_num, group_id, rebuild_snapshots
            )
            # 解包返回值（可能包含密集索引）
            if len(build_result) == 3:
                tree, graph, dense_index_result = build_result
            else:
                tree, graph = build_result
                dense_index_result = None
            resume_manager.mark_stage_completed(piece_name, 'indexing')
        elif (resume_manager.is_stage_completed(piece_name, 'indexing') and
              resume_manager.validate_cache_files(cache_folder) and not force_rebuild):
            # 从缓存加载已构建的树和图
            try:
                with open(os.path.join(cache_folder, 'tree.json'), 'r') as f:
                    tree = json.load(f)
                with open(os.path.join(cache_folder, 'graph.json'), 'r') as f:
                    graph_data = json.load(f)
                with open(os.path.join(cache_folder, 'index.json'), 'r') as f:
                    index = json.load(f)
                with open(os.path.join(cache_folder, 'appearance_count.json'), 'r') as f:
                    appearance_count = json.load(f)

                # 加载text_edge_mapping（如果存在）
                text_edge_mapping = {}
                text_edge_mapping_path = os.path.join(cache_folder, 'text_edge_mapping.json')
                if os.path.exists(text_edge_mapping_path):
                    try:
                        with open(text_edge_mapping_path, 'r') as f:
                            text_edge_mapping = json.load(f)
                        logger.info(f"成功加载text_edge_mapping: {len(text_edge_mapping)} 条")
                    except Exception as e:
                        logger.warning(f"加载text_edge_mapping失败: {e}，使用空字典")
                else:
                    logger.info("text_edge_mapping.json不存在，使用空字典（向后兼容）")

                # 将graph_data转换为NetworkX图对象
                from extract_graph import build_graph
                graph_nx = build_graph(graph_data)

                # 重构图数据结构（包含text_edge_mapping作为第4个元素）
                graph = (graph_nx, index, appearance_count, text_edge_mapping)
                dense_index_result = _load_cached_dense_index(cache_folder, tree, config_bundle)
                skipped_stages.append('indexing')
                logger.info(f"从缓存加载索引: {piece_name}")

                # 如果启用实体聚类且（缺少 snapshots.json 或 强制重建snapshots），则仅重生成 snapshots（不覆盖 tree/index）
                # 在正常模式下可以重新生成snapshots，但在query_only模式下不应该
                rebuild_snapshots = configs.get("rebuild_snapshots", False)
                if config_bundle.enable_entity_clustering and group_id and not query_only:
                    snapshots_path = os.path.join(cache_folder, "snapshots.json")
                    if not os.path.exists(snapshots_path) or rebuild_snapshots:
                        reason = "缺失" if not os.path.exists(snapshots_path) else "强制重建"
                        logger.info(f"检测到 snapshots.json {reason}，开始 snapshots_only 重生成: {snapshots_path}")
                        try:
                            from build_tree_clustering_time import build_tree_with_clustering_sync
                            from graphiti_extractor import GraphitiExtractor

                            graphiti_extractor = GraphitiExtractor(config_bundle)

                            def init_and_refresh_snapshots():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    loop.run_until_complete(graphiti_extractor._initialize_graphiti())
                                    build_tree_with_clustering_sync(
                                        text_chunks=[],
                                        llm_config=config_bundle.llm_api,
                                        cache_folder=cache_folder,
                                        graphiti_driver=graphiti_extractor.graphiti.driver,
                                        group_id=group_id,
                                        language=config_bundle.language,
                                        cluster_threshold=config_bundle.cluster_threshold,
                                        merge_num=config_bundle.cluster_merge_num,
                                        index=index,
                                        text_edge_mapping=text_edge_mapping,
                                        original_text=None,
                                        snapshots_only=True,
                                    )
                                    loop.run_until_complete(graphiti_extractor.graphiti.close())
                                finally:
                                    pass

                            await asyncio.to_thread(init_and_refresh_snapshots)
                            logger.info("snapshots_only 重生成完成")
                        except Exception as e:
                            logger.error(f"snapshots_only 重生成失败: {e}")
            except Exception as e:
                logger.warning(f"缓存加载失败，重新构建: {e}")
                resume_manager.clean_incomplete_files(cache_folder)
                build_result = await parallel_build_extract(
                    chunks, config_bundle, cache_folder,
                    config_bundle.cluster_length, config_bundle.cluster_overlap,
                    config_bundle.cluster_merge_num, group_id, rebuild_snapshots
                )
                # 解包返回值（可能包含密集索引）
                if len(build_result) == 3:
                    tree, graph, dense_index_result = build_result
                else:
                    tree, graph = build_result
                    dense_index_result = None
                resume_manager.mark_stage_completed(piece_name, 'indexing')
        else:
            # 清理不完整的文件并重新构建
            resume_manager.clean_incomplete_files(cache_folder)
            build_result = await parallel_build_extract(
                chunks, config_bundle, cache_folder,
                config_bundle.cluster_length, config_bundle.cluster_overlap,
                config_bundle.cluster_merge_num, group_id, rebuild_snapshots
            )
            # 解包返回值（可能包含密集索引）
            if len(build_result) == 3:
                tree, graph, dense_index_result = build_result
            else:
                tree, graph = build_result
                dense_index_result = None
            resume_manager.mark_stage_completed(piece_name, 'indexing')

        # 4. retrieval
        # dense_index_result 已经在前面的 parallel_build_extract 中获取
        if 'dense_index_result' not in locals():
            dense_index_result = None
        retriever = build_retriever(tree, graph, config_bundle, dense_index_result, group_id, cache_folder=cache_folder)

        # === promote-cache（默认开启）：将本次 run-cache 回写到 stable-cache，供下次自动复用 ===
        # 说明：为了避免“永远读不到缓存”的体验，默认启用 promote；
        # 可在配置中通过 promote_cache: false 关闭（适合并行跑/不希望覆盖稳定缓存的场景）。
        promote_cache = configs.get("promote_cache", True)
        if promote_cache and stable_cache_folder and not query_only:
            try:
                # 只有当缓存文件完整时才回写，避免把半成品写入稳定缓存
                if resume_manager.validate_cache_files(cache_folder, check_dense_index=False):
                    import shutil
                    os.makedirs(stable_cache_folder, exist_ok=True)
                    logger.info(f"📌 promote-cache: {cache_folder} -> {stable_cache_folder}")
                    shutil.copytree(cache_folder, stable_cache_folder, dirs_exist_ok=True)
                    logger.info("✅ promote-cache 完成")
                else:
                    logger.info("⚠️ promote-cache 跳过：run-cache 未通过完整性校验")
            except Exception as e:
                logger.warning(f"⚠️ promote-cache 失败（不影响本次运行结果）: {e}")

        # 5. qa
        qa_list = data_piece["qa"]
        if query_filter_qids is not None:
            qa_before_filter = len(qa_list)
            qa_list = [
                qa_piece
                for qa_piece in qa_list
                if qa_piece.get("question_id") in query_filter_qids
            ]
            logger.info(
                f"Query filter for {piece_name}: selected {len(qa_list)} of "
                f"{qa_before_filter} QA items after full-memory preparation"
            )

        # 获取结果文件路径来检查已完成的QA
        if paths:
            result_file = paths['result_file']
        else:
            config_hash = generate_config_hash_from_bundle(config_bundle)
            data_file = config_bundle.dataset_path.split("/")[-1].split(".")[0]
            desc_parts = []
            if config_bundle.split_method != SPLIT_SEQUENTIAL:
                desc_parts.append(f"split_{config_bundle.split_method}")
            desc = "_".join(desc_parts) if desc_parts else "default"
            config_filename = get_config_filename(data_file, desc, config_hash)
            dataset_name = config_bundle.dataset_name
            answer_dir = ensure_dir(get_answer_dir(config_bundle.answer_path, dataset_name))
            result_file = get_result_file(answer_dir, config_filename)

        # 检查已完成的QA问题（仅在非query_only模式下跳过）
        completed_qa_indices = resume_manager.get_completed_qa_indices(piece_name, result_file)

        if not query_only and len(completed_qa_indices) == len(qa_list):
            # 所有QA都已完成（非query_only模式）
            skipped_stages.append('qa_processing')
            logger.info(f"跳过已完成的QA处理: {piece_name}")
        else:
            # 处理QA问题（包括query_only模式下的重新处理）
            if query_only:
                logger.info(f"仅查询模式：重新处理所有QA问题: {piece_name}")
                # 在query_only模式下，忽略已完成的QA，重新处理所有问题
                completed_qa_indices = set()

            res = await process_qa_with_resume(qa_list, retriever, config_bundle, llm, date,
                                        actual_index, logger, completed_qa_indices,
                                        result_file, piece_name, paths)
            resume_manager.mark_stage_completed(piece_name, 'qa_processing')

        # 记录恢复状态日志
        resume_manager.log_resume_status(piece_name, skipped_stages)

        # 标记数据项完成
        resume_manager.mark_piece_completed(piece_name)

    # 计算总用时
    elapsed_time = time.time() - start_time

    # 计算并保存时间统计信息
    if paths:
        result_file = paths['result_file']
        calculate_and_save_time_stats(result_file, config_bundle, paths)
    else:
        # 对于旧的路径系统，也尝试计算统计信息
        config_hash = generate_config_hash_from_bundle(config_bundle)
        data_file = config_bundle.dataset_path.split("/")[-1].split(".")[0]
        desc_parts = []
        if config_bundle.split_method != SPLIT_SEQUENTIAL:
            desc_parts.append(f"split_{config_bundle.split_method}")
        desc = "_".join(desc_parts) if desc_parts else "default"
        config_filename = get_config_filename(data_file, desc, config_hash)
        dataset_name = config_bundle.dataset_name
        answer_dir = ensure_dir(get_answer_dir(config_bundle.answer_path, dataset_name))
        result_file = get_result_file(answer_dir, config_filename)
        calculate_and_save_time_stats(result_file, config_bundle, None)

    # 保存并打印Token统计
    token_tracker = get_token_tracker()
    if token_tracker and token_tracker.enable_tracking:
        if paths:
            # 每次运行的 run_dir 唯一，直接把统计写到 logs 目录中
            token_stats_dir = paths.get('logs_dir') or paths.get('run_dir')
            token_stats_file = os.path.join(token_stats_dir, 'token_statistics.json')
        else:
            # 旧路径系统：把统计集中存到 answer_path/token_stats/ 下
            dataset_stub = Path(config_bundle.dataset_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            legacy_dir = os.path.join(config_bundle.answer_path, 'token_stats')
            os.makedirs(legacy_dir, exist_ok=True)
            token_stats_file = os.path.join(legacy_dir, f"{dataset_stub}_{timestamp}_token_statistics.json")

        token_tracker.save_statistics(token_stats_file)
        token_tracker.print_summary()
        logger.info(f"Token统计已保存到: {token_stats_file}")

    # 打印最终摘要
    if paths:
        print_final_summary(paths, elapsed_time)
    else:
        logger.info(f"\n运行完成！总用时: {elapsed_time:.2f} 秒")

    # 无需GPU内存清理

# ===================== 程序入口 =====================
@log_exceptions()
def main():
    try:
        configs = parse_args()
        asyncio.run(run_pipeline(configs, logger))
    except Exception as e:
        logger.error(f"Error occurred in main: {e}")
        logger.error(traceback.format_exc())
        for child in mp.active_children():
            child.terminate()
            child.join(timeout=3)
        # 无需GPU内存清理
        sys.exit(1)

if __name__ == "__main__":
    nest_asyncio.apply()
    mp.set_start_method("spawn", force=True)
    main()
