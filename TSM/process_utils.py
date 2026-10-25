import os
try:
    import torch
except ImportError:  # CPU-only public installations do not require PyTorch.
    torch = None
from build_tree import build_tree, load_cache_summary
from extract_graph import extract_graph, load_nlp, load_cache
import time
from extract_graph import add_temporal_to_graph
import logging
from decorators import ensure_dir
logger = logging.getLogger("E2GraphRAG")

def clean_cuda_memory(device_id):
    """清理指定GPU设备的缓存"""
    if torch is not None and torch.cuda.is_available():
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()

def build_tree_task(args):
    # 检查参数长度，适配新的config_bundle参数格式
    if len(args) == 10:
        # 原有参数格式（向后兼容）
        llm_path, llm_device, text, cache_folder, tokenizer_name, length, overlap, merge_num, torch_dtype, language = args
        use_api = False
        api_config = None
    elif len(args) == 11:
        # 检查是否是config_bundle格式
        if hasattr(args[10], 'llm_api'):
            # 新参数格式，传递config_bundle对象
            llm_path, llm_device, text, cache_folder, tokenizer_name, length, overlap, merge_num, torch_dtype, language, config_bundle = args
            # 从config_bundle中提取API配置
            use_api = config_bundle.llm_api["type"] == "openai_compatible"
            if use_api:
                api_config = config_bundle.llm_api
            else:
                api_config = None
        else:
            # 旧参数格式，传递configs对象
            llm_path, llm_device, text, cache_folder, tokenizer_name, length, overlap, merge_num, torch_dtype, language, configs = args
            # 从configs中提取API配置
            use_api = configs["llm"].get("use_api", False)
            if use_api:
                api_config = {
                    "type": "openai_compatible",
                    "model": configs["llm"].get("api_model", "gpt-4o-mini"),
                    "base_url": configs["llm"].get("base_url"),
                    "api_key_env": configs["llm"].get("api_key_env", "OPENAI_API_KEY"),
                    "max_tokens": configs["llm"].get("max_tokens", 8192),
                    "temperature": configs["llm"].get("temperature", 0)
                }
            else:
                api_config = None
    elif len(args) == 12:
        # 旧的新参数格式，包含单独的API配置（保持兼容性）
        llm_path, llm_device, text, cache_folder, tokenizer_name, length, overlap, merge_num, torch_dtype, language, use_api, api_config = args
    else:
        raise ValueError(f"参数数量不正确，期望10、11或12个参数，实际收到{len(args)}个")

    logger.info(f"开始构建树任务，缓存文件夹: {cache_folder}")
    if os.path.exists(os.path.join(cache_folder, "tree.json")):
        logger.info("从缓存加载树结构")
        return load_cache_summary(os.path.join(cache_folder, "tree.json")), -1

    try:
        # 如果启用了use_user_only，从chunk中提取user_only_text用于树构建（persona生成）
        use_user_only = False
        if config_bundle and hasattr(config_bundle, 'use_user_only'):
            use_user_only = config_bundle.use_user_only

        # 保存原始text用于后续保存到树结构
        original_text = text

        if use_user_only:
            # 提取user_only_text用于树构建（persona生成）
            text_for_build = []
            for chunk in text:
                if isinstance(chunk, dict) and "user_only_text" in chunk:
                    text_for_build.append(chunk["user_only_text"])
                else:
                    # 向后兼容：如果不是字典格式，直接使用
                    text_for_build.append(chunk)
            text = text_for_build
            logger.info(f"启用use_user_only模式：使用user-message进行树构建（persona生成），文本块数量: {len(text)}")

        logger.info(f"开始构建新的树结构，文本块数量: {len(text)}")
        device_id = int(llm_device.split(':')[1]) if ':' in llm_device else 0

        # 只加载分词器，不加载大模型
        #tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 根据配置选择LLM类型
        if use_api and api_config:
            logger.info("使用API模式进行树构建")
            llm_pipeline = api_config
        else:
            raise ValueError("The public release requires an OpenAI-compatible LLM configuration")

        # Process
        # 从config_bundle获取并行API调用配置
        parallel_api_calls = 4  # 默认值
        if config_bundle and hasattr(config_bundle, 'parallel_api_calls'):
            parallel_api_calls = getattr(config_bundle, 'parallel_api_calls', 4)

        # 传递原始text和use_user_only标志给build_tree函数
        # 注意：build_tree函数需要知道原始text以便保存完整的turn到leaf节点
        result, time_cost = build_tree(text, llm_pipeline, cache_folder, merge_num, language, parallel_api_calls, original_text=original_text if use_user_only else None)
        logger.info(f"树构建完成，耗时: {time_cost}秒")
        return result, time_cost

    except Exception as e:
        logger.error(f"build tree task error: {e}")
        import traceback
        logger.error(f"build tree task error stack:\n{traceback.format_exc()}")
        raise e
    finally:
        # Clean up
        # if 'tokenizer' in locals():
        #     del tokenizer
        # API模式不需要清理GPU内存
        if not use_api:
            clean_cuda_memory(device_id)

def extract_graph_task(args):
    # 检查参数长度，适配新的config_bundle参数格式
    group_id = None
    if len(args) == 5:
        # 原有参数格式（向后兼容）
        text, cache_folder, language, _, llm_path = args
        config_bundle = None
    elif len(args) == 6:
        # 检查是否是config_bundle格式
        if hasattr(args[5], 'llm_api'):
            # 新参数格式，传递config_bundle对象
            text, cache_folder, language, _, llm_path, config_bundle = args
        else:
            # 旧参数格式，传递configs对象
            text, cache_folder, language, _, llm_path, configs = args
            config_bundle = None
    elif len(args) == 7:
        # 检查第6个参数是否是config_bundle
        if hasattr(args[5], 'llm_api'):
            # 新参数格式，包含group_id
            text, cache_folder, language, _, llm_path, config_bundle, group_id = args
        else:
            # 旧的新参数格式，包含单独的API配置（保持兼容性）
            text, cache_folder, language, _, llm_path, use_api, api_config = args
            config_bundle = None
    else:
        raise ValueError(f"参数数量不正确，期望5、6或7个参数，实际收到{len(args)}个")

    # 根据配置选择图抽取方法
    extraction_method = "spacy"  # 默认方法
    if config_bundle and hasattr(config_bundle, 'extraction_method'):
        extraction_method = config_bundle.extraction_method

    # 如果启用了use_user_only，从chunk中提取user_only_text用于图抽取
    use_user_only = False
    original_text_for_db = None  # 保存原始text（字典格式），用于后续保存完整turn到数据库
    if config_bundle and hasattr(config_bundle, 'use_user_only'):
        use_user_only = config_bundle.use_user_only

    if use_user_only:
        # 保存原始text（字典格式），用于后续保存完整turn到数据库
        original_text_for_db = text
        # 提取user_only_text用于图抽取（实体抽取）
        text_for_extraction = []
        for chunk in text:
            if isinstance(chunk, dict) and "user_only_text" in chunk:
                text_for_extraction.append(chunk["user_only_text"])
            else:
                # 向后兼容：如果不是字典格式，直接使用
                text_for_extraction.append(chunk)
        text = text_for_extraction
        logger.info(f"启用use_user_only模式：使用user-message进行图抽取，文本块数量: {len(text)}")

    logger.info(f"开始图抽取任务，方法: {extraction_method}，缓存文件夹: {cache_folder}")

    # 检查缓存（两种方法共用缓存检查逻辑）
    if os.path.exists(os.path.join(cache_folder, "graph.json")):
        logger.info("从缓存加载图结构")
        # 从缓存加载基础数据
        graph, index, appearance_count, text_edge_mapping = load_cache(cache_folder)
        return (graph, index, appearance_count, text_edge_mapping), -1

    # 根据方法选择相应的抽取函数
    if extraction_method == "graphiti":
        # 为Graphiti方法传递完整参数，包括group_id和original_text_for_db
        graphiti_args = (text, cache_folder, language, False, llm_path, config_bundle)
        if group_id is not None:
            graphiti_args = graphiti_args + (group_id,)
        # 如果提供了original_text_for_db，添加到参数中（用于保存完整turn到数据库）
        if original_text_for_db is not None:
            graphiti_args = graphiti_args + (original_text_for_db,)
        return extract_graph_graphiti_wrapper(graphiti_args)
    elif extraction_method == "none":
        # 消融实验：禁用图抽取，返回空的图结构
        logger.info("图抽取方法设置为'none'，跳过图抽取，返回空图结构")
        import networkx as nx
        empty_graph = nx.Graph()
        empty_index = {}
        empty_appearance_count = {}
        empty_text_edge_mapping = {}

        # 不保存缓存，避免影响其他实验
        logger.info("消融实验模式：不保存空图缓存，避免影响其他实验")

        return (empty_graph, empty_index, empty_appearance_count, empty_text_edge_mapping), 0
    else:
        return extract_graph_spacy_wrapper(args)


def extract_graph_spacy_wrapper(args):
    """使用spaCy方法进行图抽取的包装函数"""
    # 解析参数
    if len(args) == 5:
        text, cache_folder, language, _, llm_path = args
    elif len(args) == 6:
        text, cache_folder, language, _, llm_path, config_bundle = args
    elif len(args) == 7:
        text, cache_folder, language, _, llm_path, use_api, api_config = args
    else:
        raise ValueError(f"参数数量不正确，期望5、6或7个参数，实际收到{len(args)}个")

    nlp = None
    try:
        logger.info(f"使用spaCy方法抽取图结构，文本块数量: {len(text)}")
        nlp = load_nlp(language)
        (result, index, count, text_edge_mapping), time_cost = extract_graph(
            text, cache_folder, nlp,
            language=language
        )
        logger.info(f"spaCy图抽取完成，耗时: {time_cost}秒")
        return (result, index, count, text_edge_mapping), time_cost
    finally:
        if nlp is not None:
            del nlp


def extract_graph_graphiti_wrapper(args):
    """使用Graphiti方法进行图抽取的包装函数"""
    try:
        # 导入Graphiti抽取函数
        from graphiti_extractor import extract_graph_graphiti

        logger.info("开始使用Graphiti方法进行图抽取...")
        logger.info(f"Graphiti图抽取参数: {len(args)}个参数")

        result = extract_graph_graphiti(args)
        logger.info("Graphiti图抽取完成")
        return result

    except ImportError as e:
        logger.error(f"无法导入Graphiti模块: {e}")
        raise RuntimeError(
            "Graphiti is required for extraction.method=graphiti; refusing to change the method."
        ) from e
    except Exception as e:
        logger.error(f"Graphiti图抽取失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(
            "Graphiti extraction failed; refusing to fall back to a different method."
        ) from e


def build_dense_index_wrapper(args):
    """构建密集检索索引的包装函数"""
    try:
        # 导入密集索引构建模块
        from build_dense_index import build_dense_index_task

        logger.info("开始构建密集检索索引")
        result = build_dense_index_task(args)
        logger.info("密集索引构建完成")
        return result

    except ImportError as e:
        logger.error(f"无法导入密集索引模块: {e}")
        # 返回空索引，系统将回退到实时构建
        return (None, []), 0
    except Exception as e:
        logger.error(f"密集索引构建失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 返回空索引，系统将回退到实时构建
        return (None, []), 0
