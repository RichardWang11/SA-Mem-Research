import os
from typing import List
from pathlib import Path
import json
import spacy
import networkx as nx
from itertools import combinations
from typing import List, Tuple
import time
from datetime import datetime
"""
Temporal 相关能力已移除：
- 不再引入 temporal_extraction
- 不再提供 add_temporal_to_graph 等时间增强逻辑
"""
from tqdm import tqdm

def _finalize_nlp(nlp: spacy.Language) -> spacy.Language:
    """确保可用的句子边界：若无 parser/senter/sentencizer，则添加 sentencizer。"""
    try:
        pipes = set(nlp.pipe_names)
        if 'parser' not in pipes and 'senter' not in pipes and 'sentencizer' not in pipes:
            nlp.add_pipe('sentencizer')
    except Exception:
        # 安全兜底：即使添加失败也不影响后续流程
        pass
    return nlp

def load_nlp(language:str="en"):
    """加载spaCy NLP模型，支持离线模式和更好的错误处理"""
    # 优先尝试使用本地模型目录（无需联网）
    try:
        from spacy.util import set_data_path
        candidate_dirs = [
            os.environ.get("SPACY_MODEL_DIR"),
        ]
        for d in candidate_dirs:
            if d and os.path.isdir(d):
                try:
                    set_data_path(Path(d))
                except Exception:
                    pass
                break
    except Exception:
        pass
    if language == "en":
        try:
            nlp = spacy.load("en_core_web_lg")
            return _finalize_nlp(nlp)
        except OSError:
            print("en_core_web_lg model not found locally; trying smaller installed models...")
            try:
                nlp = spacy.load("en_core_web_sm")
                return _finalize_nlp(nlp)
            except OSError:
                try:
                    nlp = spacy.load("en_core_web_md")
                    return _finalize_nlp(nlp)
                except OSError:
                    print("No English model installed; using deterministic blank pipeline.")
                    nlp = spacy.blank("en")
                    return _finalize_nlp(nlp)
    elif language == "zh":
        try:
            nlp = spacy.load("zh_core_web_lg")
            return _finalize_nlp(nlp)
        except OSError:
            print("zh_core_web_lg model not found locally; using deterministic blank pipeline.")
            return _finalize_nlp(spacy.blank("zh"))
    else:
        # 默认使用英语基础模型
        try:
            nlp = spacy.load("en_core_web_sm")
            return _finalize_nlp(nlp)
        except:
            print("Using basic English model...")
            nlp = spacy.blank("en")
            return _finalize_nlp(nlp)

def naive_extract_graph(text:str, nlp:spacy.Language):
    # process the text
    doc = nlp(text)

    # noun pairs provide the edge.
    noun_pairs = {}

    # all_nouns saving the nodes.
    all_nouns = set()

    # process the name like John Brown
    double_nouns = {}
    appearance_count = {}

    for sent in doc.sents:
        sentence_terms = []

        ent_positions = set()
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                # handle the name like John Brown, John Brown Smith.
                name_parts = ent.text.split()
                if len(name_parts) >= 2:
                    for name in name_parts:
                        double_nouns[name] = name_parts
                    sentence_terms.extend(name_parts)
                    for name in name_parts:
                        appearance_count[name] = appearance_count.get(name, 0) + 1
                else:
                    sentence_terms.append(ent.text)
                    appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1

            # process the organization or country.
            elif ent.label_ in ["ORG", "GPE"]:
                sentence_terms.append(ent.text)
                appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
            for token in ent:
                ent_positions.add(token.i)

        for token in sent:
            if token.i in ent_positions:
                continue
            if token.pos_ == "NOUN" and token.lemma_.strip():
                sentence_terms.append(token.lemma_.lower())
                appearance_count[token.lemma_.lower()] = appearance_count.get(token.lemma_.lower(), 0) + 1
            elif token.pos_ == "PROPN" and token.text.strip():
                sentence_terms.append(token.lemma_.lower())
                appearance_count[token.lemma_.lower()] = appearance_count.get(token.lemma_.lower(), 0) + 1
            elif token.pos_ == "PROPN" and token.text.strip():
                sentence_terms.append(token.text)
                appearance_count[token.text] = appearance_count.get(token.text, 0) + 1

        all_nouns.update(sentence_terms)

        # Count the cooccurrence of terms
        for i in range(len(sentence_terms)):
            for j in range(i+1, len(sentence_terms)):
                term1, term2 = sorted([sentence_terms[i], sentence_terms[j]])
                pair = (term1, term2)
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1

    return {
        "nouns": list(all_nouns),
        "cooccurrence": noun_pairs,
        "double_nouns": double_nouns,
        "appearance_count": appearance_count
    }


def build_graph(triplets: List[Tuple[str, str, int]]) -> nx.Graph:
    '''
    build the graph from the triplets, merging weights of duplicate edges
    Args:
        triplets: List of [node1, node2, weight] List
    Returns:
        NetworkX graph with merged weights
    '''
    G = nx.Graph()

    # 创建字典来存储边的权重和
    edge_weights = {}
    edge_attributes = {}
    for item in triplets:
        n1, n2, weight = item[:3]
        # Graphiti caches store [source, target, attributes], while legacy
        # caches store [source, target, numeric_weight]. Preserve attributes
        # and keep the existing numeric aggregation rule for both formats.
        attributes = dict(weight) if isinstance(weight, dict) else {}
        if isinstance(weight, dict):
            weight = attributes.get('weight', 1)
        # 因为是无向图，所以(a,b)和(b,a)是相同的边
        edge = tuple(sorted([n1, n2]))
        edge_weights[edge] = edge_weights.get(edge, 0) + int(float(weight))
        edge_attributes.setdefault(edge, {}).update(attributes)

    # 将合并后的边添加到图中
    for (n1, n2), weight in edge_weights.items():
        attributes = edge_attributes[(n1, n2)]
        attributes['weight'] = weight
        G.add_edge(n1, n2, **attributes)

    return G


# DEPRECATED: 此函数已弃用，请使用 Graphiti 进行图抽取
def build_temporal_graph(temporal_edges):
    '''
    DEPRECATED: 此函数已弃用，请使用 Graphiti 进行图抽取
    '''
    raise NotImplementedError("此功能已弃用，请使用 Graphiti 进行图抽取")


def load_cache(cache_path:str):
    graph_file_path = os.path.join(cache_path, "graph.json")
    index_file_path = os.path.join(cache_path, "index.json")
    appearance_count_file_path = os.path.join(cache_path, "appearance_count.json")
    text_edge_mapping_file_path = os.path.join(cache_path, "text_edge_mapping.json")

    edges = json.load(open(graph_file_path, "r"))
    index = json.load(open(index_file_path, "r"))
    appearance_count = json.load(open(appearance_count_file_path, "r"))

    graph = build_graph(edges)

    # 尝试加载text_edge_mapping，如果不存在则返回空字典
    text_edge_mapping = {}
    if os.path.exists(text_edge_mapping_file_path):
        try:
            text_edge_mapping = json.load(open(text_edge_mapping_file_path, "r"))
        except:
            text_edge_mapping = {}

    return graph, index, appearance_count, text_edge_mapping


def load_temporal_cache(cache_path: str):
    """加载带时间信息的缓存"""
    graph_file_path = os.path.join(cache_path, "graph.json")
    index_file_path = os.path.join(cache_path, "index.json")
    appearance_count_file_path = os.path.join(cache_path, "appearance_count.json")
    temporal_edges_file_path = os.path.join(cache_path, "temporal_edges.json")

    edges = json.load(open(graph_file_path, "r"))
    index = json.load(open(index_file_path, "r"))
    appearance_count = json.load(open(appearance_count_file_path, "r"))

    # 加载时间信息
    temporal_edges = load_temporal_edges(temporal_edges_file_path)

    # 构建图（temporal 功能已弃用，统一使用基础图构建）
    graph = build_graph(edges)

    return graph, index, appearance_count, temporal_edges


def save_graph(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_index(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_appearance_count(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_text_edge_mapping(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def extract_graph(text:List[str], cache_folder:str, nlp:spacy.Language, use_cache=True, reextract=False, language="en"):
    extract_start_time = time.time()
    # extract_graph for a session
    # 检查缓存文件
    cache_files_exist = (
        os.path.exists(os.path.join(cache_folder, "graph.json")) and
        os.path.exists(os.path.join(cache_folder, "index.json")) and
        os.path.exists(os.path.join(cache_folder, "appearance_count.json")) and
        os.path.exists(os.path.join(cache_folder, "text_edge_mapping.json"))
    )

    if use_cache and cache_files_exist:
        return load_cache(cache_folder), -1
    else:
        graph_file_path = os.path.join(cache_folder, "graph.json")
        index_file_path = os.path.join(cache_folder, "index.json")
        appearance_count_file_path = os.path.join(cache_folder, "appearance_count.json")
        text_edge_mapping_file_path = os.path.join(cache_folder, "text_edge_mapping.json")
        edges = []
        index = {}
        appearance_count = {}
        text_edge_mapping = {}  # 新增映射字典，记录每段文本和对应的边

        for i, chunk in tqdm(enumerate(text), desc="抽取图谱", total=len(text), ncols=80):
            chunk_id = f"leaf_{i}"
            naive_result = naive_extract_graph(chunk, nlp)
            # not merge the entities.
            appearance_count["leaf_{}".format(i)] = naive_result["appearance_count"]
            appearance_count[chunk_id] = naive_result["appearance_count"]

            # 记录当前文本块的边
            chunk_edges = []

            for noun in naive_result["nouns"]:
                if noun not in index:
                    index[noun] = []
                index[noun].append("leaf_{}".format(i))

            for noun, count in naive_result["appearance_count"].items():
                appearance_count[noun] = appearance_count.get(noun, 0) + count

            # add the cooccurrence.
            for pair, weight in naive_result["cooccurrence"].items():
                head, tail = pair
                edge = [head, tail, weight]
                edges.append(edge)
                chunk_edges.append(edge)  # 记录当前文本块的边

            # 保存当前文本块和对应的边
            text_edge_mapping[chunk_id] = {
                "text": chunk,
                "edges": chunk_edges
            }

        # build the graph.
        G = build_graph(edges)

        if reextract == True:
            save_appearance_count(appearance_count, appearance_count_file_path)
            return (G, index, appearance_count, text_edge_mapping), -1

        # save the graph and index.
        save_graph(edges, graph_file_path)
        save_index(index, index_file_path)
        save_appearance_count(appearance_count, appearance_count_file_path)
        save_text_edge_mapping(text_edge_mapping, text_edge_mapping_file_path)
        extract_end_time = time.time()
        return (G, index, appearance_count, text_edge_mapping), extract_end_time - extract_start_time


def add_temporal_to_graph(text_edge_mapping, llm_pipeline, language="en"):
    """已移除 Temporal 能力：保留同名函数仅用于向后兼容，避免旧代码 import 直接崩溃。"""
    raise NotImplementedError("Temporal 功能已移除：add_temporal_to_graph 不再可用")


if __name__ == "__main__":

    # 测试基础图构建
    edges = [
        ('a', 'b', 1),
        ('a', 'b', 3),
        ('b', 'a', 2)
    ]

    G = build_graph(edges)
