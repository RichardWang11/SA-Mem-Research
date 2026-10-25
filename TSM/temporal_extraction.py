"""
时间抽取模块 - 为 E-2GraphRAG 添加时间信息抽取功能
参考 Graphiti 的时间抽取实现，但适配 E-2GraphRAG 的架构
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import re
from prompt_dict import Prompts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 添加控制台处理器确保输出到控制台
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 确保此模块的日志级别为INFO


class TemporalEdge:
    """带时间信息的边"""
    def __init__(self, source: str, target: str, weight: int, rel: str,
                 valid_at: Optional[str] = None,
                 invalid_at: Optional[str] = None,
                 context: str = ""):
        self.source = source
        self.target = target
        self.rel = rel
        self.weight = weight
        self.valid_at = valid_at
        self.invalid_at = invalid_at
        self.context = context

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "relationship": self.rel,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            source=data["source"],
            target=data["target"],
            weight=data["weight"],
            rel=data['relationship'],
            valid_at=data.get("valid_at"),
            invalid_at=data.get("invalid_at"),
            context=data.get("context", "")
        )


def create_temporal_extraction_prompt(entity_pair: str, context: str, reference_timestamp: str, language: str = "en") -> str:
    """创建时间抽取的提示词"""
    if language == "zh":
        prompt_template = Prompts["temporal_extraction_prompt_zh"]
    else:
        prompt_template = Prompts["temporal_extraction_prompt_en"]

    return prompt_template.format(
        entity_pair=entity_pair,
        context=context,
        reference_timestamp=reference_timestamp
    )


def extract_temporal_info_from_text(text: str, nlp, language: str = "en") -> List[Dict[str, Any]]:
    """从文本中提取时间信息"""
    doc = nlp(text)
    temporal_info = []

    for sent in doc.sents:
        # 提取句子中的时间实体
        time_entities = []
        for ent in sent.ents:
            if ent.label_ in ["DATE", "TIME"]:
                time_entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        if time_entities:
            temporal_info.append({
                "sentence": sent.text,
                "time_entities": time_entities
            })

    return temporal_info


def parse_llm_temporal_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """解析LLM返回的时间信息"""
    try:
        # 尝试解析JSON响应 - 支持列表格式
        if "[" in response and "]" in response:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            json_str = response[json_start:json_end]
            data_list = json.loads(json_str)

            data_list_parsed = []
            for data in data_list:
                valid_at = data.get("valid_at")
                invalid_at = data.get("invalid_at")

                if valid_at and valid_at != "null":
                    try:
                        datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                    except ValueError:
                        valid_at = None

                if invalid_at and invalid_at != "null":
                    try:
                        datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                    except ValueError:
                        invalid_at = None

                u = data.get("entity1")
                v = data.get("entity2")
                relationship = data.get("relationship")
                data_list_parsed.append({
                    "u": u,
                    "v": v,
                    "relationship": relationship,
                    "valid_at": valid_at,
                    "invalid_at": invalid_at
                })

            return data_list_parsed
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM temporal response: {e}")

    return None


# 定义LLM pipeline的类型
LLMPipeline = Union[Dict[str, Any], Callable[..., Any]]


def extract_edge_temporal_info(entity_pair: str, context: str, llm_pipeline: LLMPipeline,
                              reference_timestamp: str, language: str = "en") -> Optional[List[Dict[str, Any]]]:
    """使用LLM提取边的时间信息"""
    # try:
    prompt = create_temporal_extraction_prompt(entity_pair, context, reference_timestamp, language)
    # 调用LLM - 支持API模式和vLLM模式
    if isinstance(llm_pipeline, dict):
        if llm_pipeline.get("type") == "openai_compatible":
            from provider.public_llm_provider import call_llm
            response_text = call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=llm_pipeline.get("backend_model", llm_pipeline["model"]),
                temperature=llm_pipeline.get("temperature", 0),
                max_tokens=llm_pipeline.get("max_tokens", 1024),
                base_url=llm_pipeline.get("base_url"),
                api_key_env=llm_pipeline.get("api_key_env", "OPENAI_API_KEY"),
                max_retries=llm_pipeline.get("max_retries", 3),
                timeout=llm_pipeline.get("timeout", 120),
                op_type="update"
            )
            if not response_text:
                logger.warning("API调用返回空响应")
                return None
        else:
            logger.error(f"不支持的llm_pipeline类型: {llm_pipeline.get('type')}")
            return None
    else:
        # 函数模式（兼容旧版本）
        if callable(llm_pipeline):
            response = llm_pipeline(prompt, max_new_tokens=200)
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get("generated_text", "")
            else:
                response_text = str(response)
        else:
            logger.error("llm_pipeline既不是字典也不是可调用对象")
            return None
    # 解析响应
    parsed_facts = parse_llm_temporal_response(response_text)

    # write prompt, response_text, parsed_facts to log file
    with open("temporal_extraction.log", "a") as f:
        f.write(f"prompt: {prompt}\n")
        f.write(f"response_text: {response_text}\n")
        f.write(f"parsed_facts: {parsed_facts}\n")


    return parsed_facts

    # except Exception as e:
    #     logger.error(f"Error extracting temporal info: {e}")
    #     return None


def save_temporal_edges(temporal_edges: List[TemporalEdge], cache_path: str):
    """保存带时间信息的边"""
    edges_data = [edge.to_dict() for edge in temporal_edges]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(edges_data, f, indent=4, ensure_ascii=False)


def load_temporal_edges(cache_path: str) -> List[TemporalEdge]:
    """加载带时间信息的边"""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            edges_data = json.load(f)
        return [TemporalEdge.from_dict(edge_data) for edge_data in edges_data]
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.error(f"Error loading temporal edges: {e}")
        return []
