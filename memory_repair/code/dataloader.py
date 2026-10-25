import os
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta

class NovelQALoader:
    '''
    data will be returned as a dictionary with :
    {
        "book": str,
        "qa": list of dictionaries with keys:
            "id": str,
            "question": str,
            "answer": str
    }
    question is well formatted.
    '''
    def __init__(self, path):
        self.parent_folder = path
        self.qapath = os.path.join(path, "Demonstration")
        self.docpath = os.path.join(path, "Demonstration")
        self.dataset = self._initialize_dataset()
        self.available_ids = list(self.dataset.keys())

    def _initialize_dataset(self):
        dataset = {}

        # 首先读取所有book文件
        print("Loading book files...")

        for filename in os.listdir(self.docpath):
            # for directory in dirs:
            #     # copyright protected and public domain
            # for filename in files:
            if filename == ".ipynb_checkpoints" or filename.endswith(".json") or filename == "Frankenstein.txt":
                continue
            with open(os.path.join(self.docpath, filename), "r") as infile:
                book_id = int(filename.split('.')[0][1:])
                dataset[book_id] = {}
                dataset[book_id]["book"] = infile.read()

        print(f"Loaded {len(dataset)} book files")

        # 然后读取所有qa文件
        print("Loading QA files...")
        qa_count = 0
        missing_book_count = 0

        for filename in os.listdir(self.qapath):
            # for directory in dirs:
            #     if directory == "CopyrightProtected" or directory == ".ipynb_checkpoints":
            #         continue
            # for filename in files:
            if filename == ".ipynb_checkpoints" or filename.endswith(".txt") or filename == "Frankenstein.json":
                continue
            with open(os.path.join(self.qapath, filename), "r") as infile:
                qa_id = int(filename.split('.')[0][1:])
                # dataset[qa_id]["qa"] = json.loads(infile.read())

                # 检查对应的book是否存在
                if qa_id not in dataset:
                    print(f"Warning: QA file {filename} (ID: {qa_id}) has no corresponding book file")
                    missing_book_count += 1
                    continue

                try:
                    dataset[qa_id]["qa"] = json.loads(infile.read())
                    qa_count += 1
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in QA file {filename}: {e}")
                    continue

        print(f"Loaded {qa_count} QA files")
        if missing_book_count > 0:
            print(f"Warning: {missing_book_count} QA files had no corresponding book files")


        return dataset

    def _format_qa(self, qa_dict):
        formatted_qa = []
        for qa_id, qa in qa_dict.items():
            question = qa["Question"]
            options = qa["Options"]
            answer = qa["Gold"]
            question_text = question + "\n"
            for option, text in options.items():
                question_text += option + ". " + text
                if option != "D":
                    question_text += "\n"
            formatted_qa.append({
                "id": qa_id,
                "question": question_text,
                "answer": answer,
                "evidence": qa["Evidences"]
            })
        return formatted_qa

    def __getitem__(self, index):
        index = self.available_ids[index]
        to_return = {}
        to_return["book"] = self.dataset[index]["book"]
        to_return["qa"] = self._format_qa(self.dataset[index]["qa"])
        return to_return

    def __len__(self):
        return len(self.dataset)

class InfiniteQALoader:
    def __init__(self, path = "./InfiniteBench/data/longbook_qa_eng.jsonl"):
        self.dataset, self.available_ids = self._initialize_dataset(path)

    def _initialize_dataset(self, path):
        dataset = {}
        context_id = -1
        prev_context = ""

        with open(path, "r", encoding='utf-8') as infile:  # Add encoding='utf-8'
            for line in infile:
                try:
                    data = json.loads(line.strip())  # Add strip() to remove whitespace/newlines
                    context = data["context"]
                    question = data["input"]
                    answer = data["answer"]

                    if context != prev_context:
                        context_id += 1
                        prev_context = context
                        dataset[context_id] = {
                            "book": context,
                            "qa": []
                        }

                    dataset[context_id]["qa"].append({
                        "question": question,
                        "answer": answer
                    })
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue

        return dataset, list(dataset.keys())

    def __getitem__(self, index):
        return self.dataset[self.available_ids[index]]

    def __len__(self):
        return len(self.available_ids)


class InfiniteChoiceLoader:
    def __init__(self, path = "./InfiniteBench/data/longbook_choice_eng.jsonl"):
        self.dataset, self.available_ids = self._initialize_dataset(path)

    def _initialize_dataset(self, path):
        dataset = {}
        with open(path, "r") as infile:
            prev_context = ""
            context_id = -1
            for line in infile:
                data = json.loads(line)
                context = data["context"]
                question = data["input"]
                answer = data["answer"]
                if context != prev_context:
                    context_id += 1
                    prev_context = context
                    dataset[context_id] = {}
                    dataset[context_id]["book"] = context
                    dataset[context_id]["qa"] = []
                qa_dict = {
                    "question": question,
                    "answer": answer,
                    "options": data["options"]
                }
                dataset[context_id]["qa"].append(self._format_qa(qa_dict))
        return dataset, list(dataset.keys())

    def _format_qa(self, qa_dict):
        res = {}
        formatted_question = ""
        formatted_question += qa_dict["question"] + "\n"
        option_name = ["A", "B", "C", "D"]
        for i in range(len(qa_dict["options"])):
            formatted_question += option_name[i] + ". " + qa_dict["options"][i] + "\n"
            if qa_dict["answer"][0] == qa_dict["options"][i]:
                res["answer"] = option_name[i]
        res["question"] = formatted_question
        return res

    def __getitem__(self, index):
        data = self.dataset[self.available_ids[index]]
        return data

    def __len__(self):
        return len(self.available_ids)


class longmemevalLoader:
    def __init__(self, path):
        print(f"初始化longmemevalLoader，路径: {path}")
        self.dataset, self.available_ids = self._initialize_dataset(path)
        print(f"数据加载完成，共加载 {len(self.available_ids)} 个文档")

    def _initialize_dataset(self, path):
        dataset = {}
        with open(path, "r") as infile:
            datas = json.load(infile)

        print(f"原始数据包含 {len(datas)} 个问题")

        # 直接处理每个问题，保持原始会话结构
        for item_idx, item in enumerate(datas):
            # 为每个问题创建一个文档，保持原始的会话信息
            doc_id = f"question_{item_idx}"
            print(f"处理问题 {item_idx}: {item.get('question_id', 'unknown')}")

            # 将对话数据按round分开存储
            rounds_data, answer_rounds, answer_sessions = self.process_conversation_data_by_rounds(item["haystack_sessions"],
                                                                   item["haystack_session_ids"],
                                                                   item["haystack_dates"],
                                                                   item["answer_session_ids"])

            dataset[doc_id] = {
                "group_id": str(item.get("question_id", doc_id)),  # 使用question_id作为group_id
                "rounds": rounds_data,  # 按round分开存储
                "qa": [{
                    "question": item["question"],
                    "answer": item["answer"],
                    "question_date": item.get("question_date", None),
                    "answer_sessions": answer_sessions,
                    "answer_rounds": answer_rounds,
                    "question_id": item.get("question_id", None),
                    "question_type": item.get("question_type", None),
                }],
                "haystack_dates": item.get("haystack_dates", None),
                "haystack_session_ids": item.get("haystack_session_ids", None),
                "question_info": {
                    "total_rounds": len(rounds_data),
                    "total_sessions": len(item.get("haystack_sessions", [])),
                    "question_id": item.get("question_id", None),
                    "question_type": item.get("question_type", None)
                }
            }

            print(f"  问题包含 {len(rounds_data)} 个rounds, {len(item.get('haystack_sessions', []))} 个会话")

        print(f"最终数据集包含 {len(dataset)} 个文档")
        return dataset, list(dataset.keys())

    def process_conversation_data_by_rounds(self, data: List[List[Dict[str, Any]]], session_ids: List[str], dates: List[str], answer_session_ids: List[str]) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        将对话数据按round为单位分开处理

        Args:
            data: 对话数据列表，每个元素为一个session，session中包含多个rounds

        Returns:
            按round分开的对话数据列表
        """
        all_rounds = []
        answer_sessions = [session for session_idx, session in enumerate(data) if session_ids[session_idx] in answer_session_ids]
        answer_rounds = []
        for session_idx, session in enumerate(data):
            session_rounds = []

            # 提取rounds（user-assistant对话对）
            i = 0
            round_idx = 0
            while i < len(session) - 1:
                # 检查是否是user-assistant对
                if (session[i]['role'] == 'user' and
                    session[i + 1]['role'] == 'assistant'):

                    round_data = {
                        "round_id": f"session_{session_idx}_round_{round_idx}",
                        "session_id": session_ids[session_idx],
                        "round_index": round_idx,
                        "user": {
                            "role": session[i]['role'],
                            "content": session[i]['content'],
                            "has_answer": session[i].get('has_answer', False)
                        },
                        "assistant": {
                            "role": session[i + 1]['role'],
                            "content": session[i + 1]['content'],
                            "has_answer": session[i + 1].get('has_answer', False)
                        },
                        "metadata": {
                            "session_index": session_idx,
                            "round_index": round_idx,
                            "total_rounds_in_session": len(session) // 2  # 估算
                        },
                        "date": dates[session_idx]
                    }
                    if session[i].get('has_answer', False):
                        answer_rounds.append(round_data)
                    if session[i + 1].get('has_answer', False):
                        answer_rounds.append(round_data)

                    session_rounds.append(round_data)
                    round_idx += 1
                    i += 2
                else:
                    i += 1
            all_rounds.append(session_rounds)
        return all_rounds, answer_rounds, answer_sessions

    def add_session(self, session_data: Dict[str, Any], session_id: Optional[str] = None):
        """
        增量添加新的会话数据

        Args:
            session_data: 包含haystack_sessions的会话数据
            session_id: 可选的会话ID，如果不提供则自动生成
        """
        if session_id is None:
            session_id = f"session_{len(self.dataset)}"

        # 处理新会话的rounds
        rounds_data, answer_rounds, answer_sessions = self.process_conversation_data_by_rounds(session_data["haystack_sessions"],
                                                               session_data["haystack_session_ids"],
                                                               session_data["haystack_dates"],
                                                               session_data.get("answer_session_ids", []))

        # 添加到数据集
        self.dataset[session_id] = {
            "rounds": rounds_data,
            "qa": session_data.get("qa", []),
            "haystack_dates": session_data.get("haystack_dates", None),
            "haystack_session_ids": session_data.get("haystack_session_ids", None),
            "answer_rounds": answer_rounds,
            "answer_sessions": answer_sessions,
            "session_info": {
                "total_rounds": sum(len(session_rounds) for session_rounds in rounds_data),
                "session_id": session_id
            }
        }

        # 更新可用ID列表
        if session_id not in self.available_ids:
            self.available_ids.append(session_id)

        print(f"成功添加新会话: {session_id}, 包含 {len(rounds_data)} 个rounds")
        return session_id

    def add_round_to_session(self, session_id: str, round_data: Dict[str, Any]):
        """
        向指定会话添加新的round

        Args:
            session_id: 会话ID
            round_data: round数据，包含user和assistant信息
        """
        if session_id not in self.dataset:
            print(f"错误: 会话 {session_id} 不存在")
            return False

        # 找到最后一个session的rounds列表
        if not self.dataset[session_id]["rounds"]:
            # 如果没有rounds，创建第一个session
            self.dataset[session_id]["rounds"].append([])

        last_session_rounds = self.dataset[session_id]["rounds"][-1]
        current_rounds = len(last_session_rounds)
        round_id = f"{session_id}_session_{len(self.dataset[session_id]['rounds'])-1}_round_{current_rounds}"

        # 添加round数据
        new_round = {
            "round_id": round_id,
            "session_id": session_id,
            "round_index": current_rounds,
            "user": round_data["user"],
            "assistant": round_data["assistant"],
            "metadata": {
                "session_index": len(self.dataset[session_id]["rounds"]) - 1,
                "round_index": current_rounds,
                "total_rounds_in_session": current_rounds + 1
            }
        }

        last_session_rounds.append(new_round)
        self.dataset[session_id]["session_info"]["total_rounds"] += 1

        print(f"成功向会话 {session_id} 添加新round: {round_id}")
        return True

    def get_session_rounds(self, session_id: str) -> List[List[Dict[str, Any]]]:
        """
        获取指定会话的所有rounds

        Args:
            session_id: 会话ID

        Returns:
            rounds列表（按session分组）
        """
        if session_id not in self.dataset:
            print(f"错误: 会话 {session_id} 不存在")
            return []

        return self.dataset[session_id]["rounds"]

    def get_round_by_id(self, round_id: str) -> Optional[Dict[str, Any]]:
        """
        根据round ID获取特定round

        Args:
            round_id: round ID

        Returns:
            round数据，如果不存在则返回None
        """
        for session_id in self.dataset:
            for session_rounds in self.dataset[session_id]["rounds"]:
                for round_data in session_rounds:
                    if round_data["round_id"] == round_id:
                        return round_data

        print(f"错误: round {round_id} 不存在")
        return None

    def _format_qa(self, qa_dict):
        # 兼容NovelQALoader格式，直接返回
        return qa_dict

    def __getitem__(self, index):
        return self.dataset[self.available_ids[index]]

    def __len__(self):
        return len(self.available_ids)


class locomoLoader:
    """
    Loader for locomo dataset.
    与 longmemeval 的区别：一个 session 对应多个问题（qa），而不是一个问题对应多个 sessions。
    还没检查过，不一定对
    """
    def __init__(self, path):
        print(f"初始化 locomoLoader，路径: {path}")
        self.dataset, self.available_ids = self._initialize_dataset(path)
        print(f"数据加载完成，共加载 {len(self.available_ids)} 个文档")

    def _initialize_dataset(self, path):
        dataset = {}
        with open(path, "r", encoding='utf-8') as infile:
            data_items = json.load(infile)

        print(f"原始数据包含 {len(data_items)} 个条目")

        doc_id_counter = 0

        # 遍历每个条目
        for item_idx, item in enumerate(data_items):
            conversation = item.get("conversation", {})
            qa_list = item.get("qa", [])

            # 提取所有 session 信息
            session_keys = [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")]
            session_keys.sort(key=lambda x: int(x.split("_")[1]))  # 按 session 编号排序

            print(f"处理条目 {item_idx}: 包含 {len(session_keys)} 个 sessions, {len(qa_list)} 个问题")

            # 将所有 sessions 合并到一个文档中
            all_rounds_data = []  # 存储所有sessions的rounds数据

            # ====== 关键：为 LoCoMo 生成“问题提出时间” ======
            # 需求：问题提出时间 = 最后一个 session 的时间戳 + 1 天
            # - session 时间戳来源：conversation["session_X_date_time"]
            # - 存储为 ISO 字符串，供 query_time.py 的 parse_iso_datetime/dateparser 解析
            from temporal_utils import parse_iso_datetime  # 局部导入，避免全局依赖问题

            question_date: Optional[str] = None
            last_session_dt = None
            if session_keys:
                # 优先取“最后一个 session”的时间戳；若缺失/不可解析，再向前回退
                for sk in reversed(session_keys):
                    dt = parse_iso_datetime(conversation.get(f"{sk}_date_time"))
                    if dt is not None:
                        last_session_dt = dt
                        break
            if last_session_dt is not None:
                question_date = (last_session_dt + timedelta(days=1)).isoformat()

            # 处理问题列表（去重）
            all_qa = []
            sample_id = item.get("sample_id", f"item_{item_idx}")
            for q_idx, qa_item in enumerate(qa_list):
                question_id = f"{sample_id}_{q_idx}"
                all_qa.append({
                    "question_id": question_id,
                    "question": qa_item.get("question", ""),
                    "answer": qa_item.get("answer", ""),
                    "question_date": question_date,
                    "evidence": qa_item.get("evidence", []),
                    "category": qa_item.get("category", None),
                    "adversarial_answer": qa_item.get("adversarial_answer", None)
                })

            for session_key in session_keys:
                session_num = int(session_key.split("_")[1])
                session_data = conversation[session_key]
                date_time_key = f"{session_key}_date_time"
                session_date_time = conversation.get(date_time_key, None)

                # 将 session 对话转换为 rounds 格式（类似 longmemeval）
                rounds_data = self._convert_session_to_rounds(session_data, session_num, session_date_time)
                all_rounds_data.append(rounds_data)  # 添加到总列表中

                print(f"  处理 session {session_key}: {len(rounds_data)} 个 rounds")

            # 创建一个包含所有sessions的文档
            doc_id = f"item_{item_idx}_all_sessions"
            dataset[doc_id] = {
                "group_id": str(item.get("sample_id", f"item_{item_idx}")),
                "rounds": all_rounds_data,  # 嵌套列表格式：每个元素是一个session的rounds列表
                "qa": all_qa,  # 所有问题
                "speaker_a": conversation.get("speaker_a", None),
                "speaker_b": conversation.get("speaker_b", None),
                "session_info": {
                    "total_sessions": len(session_keys),
                    "total_rounds": sum(len(rounds) for rounds in all_rounds_data),
                    "total_qa": len(all_qa)
                }
            }

            doc_id_counter += 1
            print(f"  创建文档 {doc_id}: {len(session_keys)} 个 sessions, 共 {sum(len(rounds) for rounds in all_rounds_data)} 个 rounds, {len(all_qa)} 个问题")

        print(f"最终数据集包含 {len(dataset)} 个文档")
        return dataset, list(dataset.keys())

    def _convert_session_to_rounds(self, session_data: List[Dict[str, Any]], session_num: int, session_date_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        将 session 对话数据转换为 rounds 格式（类似 longmemeval）

        Args:
            session_data: session 对话列表
            session_num: session 编号
            session_date_time: session 的日期时间字符串

        Returns:
            rounds 列表
        """
        rounds = []
        round_idx = 0
        i = 0

        # 提取 user-assistant 对话对（根据 speaker 判断）
        while i < len(session_data) - 1:
            current_msg = session_data[i]
            next_msg = session_data[i + 1] if i + 1 < len(session_data) else None

            # 检查是否是连续的对话对（不同 speaker）
            if next_msg and current_msg.get("speaker") != next_msg.get("speaker"):
                round_data = {
                    "round_id": f"session_{session_num}_round_{round_idx}",
                    "session_id": f"session_{session_num}",
                    "round_index": round_idx,
                    "date": session_date_time or "",  # 添加 date 字段
                    "user": {
                        "role": "user",
                        "content": current_msg.get("text", ""),
                        "speaker": current_msg.get("speaker", ""),
                        "dia_id": current_msg.get("dia_id", ""),
                        "img_url": current_msg.get("img_url", None),
                        "blip_caption": current_msg.get("blip_caption", None),
                        "query": current_msg.get("query", None)
                    },
                    "assistant": {
                        "role": "assistant",
                        "content": next_msg.get("text", ""),
                        "speaker": next_msg.get("speaker", ""),
                        "dia_id": next_msg.get("dia_id", ""),
                        "img_url": next_msg.get("img_url", None),
                        "blip_caption": next_msg.get("blip_caption", None),
                        "query": next_msg.get("query", None)
                    },
                    "metadata": {
                        "session_number": session_num,
                        "round_index": round_idx,
                        "total_rounds_in_session": len(session_data) // 2  # 估算
                    }
                }
                rounds.append(round_data)
                round_idx += 1
                i += 2
            else:
                i += 1

        return rounds

    def get_session_rounds(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定 session 的所有 rounds

        Args:
            session_id: session ID（文档ID）

        Returns:
            rounds 列表
        """
        if session_id not in self.dataset:
            print(f"错误: session {session_id} 不存在")
            return []

        return self.dataset[session_id]["rounds"]

    def get_round_by_id(self, round_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 round ID 获取特定 round

        Args:
            round_id: round ID

        Returns:
            round 数据，如果不存在则返回 None
        """
        for doc_id in self.dataset:
            for round_data in self.dataset[doc_id]["rounds"]:
                if round_data["round_id"] == round_id:
                    return round_data

        print(f"错误: round {round_id} 不存在")
        return None

    def _format_qa(self, qa_dict):
        # 兼容其他 Loader 格式，直接返回
        return qa_dict

    def __getitem__(self, index):
        return self.dataset[self.available_ids[index]]

    def __len__(self):
        return len(self.available_ids)


class MemoryBenchLoader:
    """
    Loader for MemoryBench dataset.
    MemoryBench 是一个记忆评估数据集，包含多个子数据集（如 Locomo, DialSim, HelloBench 等）。
    每个条目包含一个问题、答案、以及各种记忆系统的对话记录。
    """
    def __init__(self, path):
        """
        初始化 MemoryBenchLoader

        Args:
            path: 数据文件路径，可以是单个 JSON 文件或包含多个 JSON 文件的目录
        """
        print(f"初始化 MemoryBenchLoader，路径: {path}")
        self.path = path
        self.dataset, self.available_ids = self._initialize_dataset(path)
        print(f"数据加载完成，共加载 {len(self.available_ids)} 个问题")

    def _initialize_dataset(self, path):
        """
        初始化数据集

        Args:
            path: 数据文件路径

        Returns:
            dataset: 数据集字典
            available_ids: 可用的 ID 列表
        """
        dataset = {}

        # 判断是文件还是目录
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            # 如果是目录，查找所有 _test.json 文件
            files = [os.path.join(path, f) for f in os.listdir(path)
                    if f.endswith('_test.json')]
            files.sort()
        else:
            raise ValueError(f"路径不存在: {path}")

        print(f"找到 {len(files)} 个数据文件")

        global_idx = 0

        for file_path in files:
            dataset_name = os.path.basename(file_path).replace('_test.json', '')
            print(f"加载文件: {file_path} (数据集: {dataset_name})")

            try:
                with open(file_path, "r", encoding='utf-8') as infile:
                    data_items = json.load(infile)

                print(f"  文件包含 {len(data_items)} 个问题")

                for item in data_items:
                    # 解析 info 字段（可能是字符串或字典）
                    info = item.get("info", {})
                    if isinstance(info, str):
                        try:
                            info = json.loads(info)
                        except json.JSONDecodeError:
                            print(f"警告: 无法解析 info 字段: {info[:100]}...")
                            info = {}

                    # 解析各种 dialog 和 feedback 字段
                    dialogs = {}
                    feedbacks = {}

                    for key, value in item.items():
                        if key.startswith("dialog_"):
                            dialog_name = key.replace("dialog_", "")
                            if isinstance(value, str):
                                try:
                                    dialogs[dialog_name] = json.loads(value)
                                except (json.JSONDecodeError, ValueError):
                                    try:
                                        import ast
                                        dialogs[dialog_name] = ast.literal_eval(value)
                                    except:
                                        print(f"警告: 无法解析 dialog_{dialog_name}")
                                        dialogs[dialog_name] = []
                            else:
                                dialogs[dialog_name] = value

                        elif key.startswith("implicit_feedback_"):
                            feedback_name = key.replace("implicit_feedback_", "")
                            if isinstance(value, str):
                                try:
                                    feedbacks[feedback_name] = json.loads(value)
                                except (json.JSONDecodeError, ValueError):
                                    try:
                                        import ast
                                        feedbacks[feedback_name] = ast.literal_eval(value)
                                    except:
                                        print(f"警告: 无法解析 implicit_feedback_{feedback_name}")
                                        feedbacks[feedback_name] = []
                            else:
                                feedbacks[feedback_name] = value

                    # 创建文档 ID
                    doc_id = f"{dataset_name}_{item.get('test_idx', global_idx)}"

                    dataset[doc_id] = {
                        "group_id": str(item.get("test_idx", global_idx)),
                        "dataset_name": dataset_name,
                        "test_idx": item.get("test_idx", global_idx),
                        "qa": [{
                            "question": item.get("origin_question", ""),
                            "input_prompt": item.get("input_prompt", ""),
                            "answer": info.get("golden_answer", ""),
                            "category": info.get("category", None),
                            "evidence": info.get("evidence", []),
                            "question_id": item.get("test_idx", global_idx),
                        }],
                        "info": info,
                        "dialogs": dialogs,  # 各种记忆系统的对话记录
                        "feedbacks": feedbacks,  # 各种记忆系统的反馈记录
                        "lang": item.get("lang", "en"),
                        "question_info": {
                            "dataset_name": dataset_name,
                            "test_idx": item.get("test_idx", global_idx),
                            "category": info.get("category", None),
                            "has_dialogs": len(dialogs) > 0,
                            "has_feedbacks": len(feedbacks) > 0,
                        }
                    }

                    global_idx += 1

            except Exception as e:
                print(f"错误: 加载文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"最终数据集包含 {len(dataset)} 个文档")
        return dataset, list(dataset.keys())

    def get_dialog_by_system(self, doc_id: str, system_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        获取指定记忆系统的对话记录

        Args:
            doc_id: 文档 ID
            system_name: 记忆系统名称（如 'bm25', 'mem0', 'a_mem' 等）

        Returns:
            对话记录列表，如果不存在则返回 None
        """
        if doc_id not in self.dataset:
            print(f"错误: 文档 {doc_id} 不存在")
            return None

        dialogs = self.dataset[doc_id].get("dialogs", {})
        return dialogs.get(system_name, None)

    def get_feedback_by_system(self, doc_id: str, system_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        获取指定记忆系统的反馈记录

        Args:
            doc_id: 文档 ID
            system_name: 记忆系统名称

        Returns:
            反馈记录列表，如果不存在则返回 None
        """
        if doc_id not in self.dataset:
            print(f"错误: 文档 {doc_id} 不存在")
            return None

        feedbacks = self.dataset[doc_id].get("feedbacks", {})
        return feedbacks.get(system_name, None)

    def get_available_systems(self, doc_id: str) -> List[str]:
        """
        获取文档中可用的记忆系统列表

        Args:
            doc_id: 文档 ID

        Returns:
            记忆系统名称列表
        """
        if doc_id not in self.dataset:
            return []

        dialogs = self.dataset[doc_id].get("dialogs", {})
        return list(dialogs.keys())

    def _format_qa(self, qa_dict):
        # 兼容其他 Loader 格式，直接返回
        return qa_dict

    def convert_to_e2graphrag_format(self, doc_id: str = None, index: int = None) -> Dict[str, Any]:
        """
        将 MemoryBench 数据格式转换为 E-2GraphRAG 格式

        Args:
            doc_id: 文档 ID（可选）
            index: 数据索引（可选，与 doc_id 二选一）

        Returns:
            Dict: E-2GraphRAG 格式的数据，包含 "book" 和 "qa" 字段
        """
        # 获取数据项
        if doc_id:
            if doc_id not in self.dataset:
                raise ValueError(f"文档 ID {doc_id} 不存在")
            item = self.dataset[doc_id]
        elif index is not None:
            if index >= len(self.available_ids):
                raise ValueError(f"索引 {index} 超出范围")
            item = self.dataset[self.available_ids[index]]
        else:
            raise ValueError("必须提供 doc_id 或 index 之一")

        # 提取对话内容（优先使用 bm25_dialog，如果没有则使用第一个可用的 dialog）
        dialog = None
        dialog_systems = list(item.get("dialogs", {}).keys())

        # 优先级：bm25_dialog > embedder_dialog > 其他
        preferred_systems = ["bm25_dialog", "embedder_dialog", "a_mem", "mem0", "memoryos"]
        for system in preferred_systems:
            if system in item.get("dialogs", {}):
                dialog = item["dialogs"][system]
                break

        # 如果都没有，使用第一个可用的
        if not dialog and dialog_systems:
            dialog = item["dialogs"][dialog_systems[0]]

        # 如果没有对话，尝试从 input_prompt 构建
        if not dialog:
            # 从 input_prompt 中提取对话内容
            input_prompt = item.get("qa", [{}])[0].get("input_prompt", "")
            if input_prompt:
                # 创建一个简单的对话格式
                dialog = [
                    {"role": "user", "content": input_prompt}
                ]
            else:
                # 如果都没有，创建一个空对话
                dialog = []

        # 将对话转换为文档格式
        book_text = self._convert_dialog_to_book(dialog)

        # 构建 E-2GraphRAG 格式
        e2graphrag_data = {
            "book": book_text,
            "qa": item.get("qa", []),
            "metadata": {
                "dataset_name": item.get("dataset_name"),
                "test_idx": item.get("test_idx"),
                "group_id": item.get("group_id"),
                "original_dialog_systems": dialog_systems,
                "used_dialog_system": preferred_systems[0] if dialog_systems else None,
            }
        }

        return e2graphrag_data

    def _convert_dialog_to_book(self, dialog: List[Dict[str, Any]]) -> str:
        """
        将 MemoryBench 的对话格式转换为 E-2GraphRAG 的文档格式

        Args:
            dialog: 对话列表，格式为 [{"role": "user", "content": "..."}, ...]

        Returns:
            str: 格式化的文档文本
        """
        if not dialog:
            return ""

        texts = []
        for i, msg in enumerate(dialog):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # 提取时间信息（如果有）
            timestamp = msg.get("timestamp") or msg.get("date") or msg.get("time")

            # 提取原始内容（如果有）
            origin_content = msg.get("origin_content", "")

            # 格式化消息
            if timestamp:
                texts.append(f"[{timestamp}] {role}: {content}")
            else:
                texts.append(f"{role}: {content}")

            # 如果有原始内容且与当前内容不同，也添加
            if origin_content and origin_content != content:
                texts.append(f"  (原始内容: {origin_content})")

        return "\n\n".join(texts)

    def convert_all_to_e2graphrag_format(self) -> List[Dict[str, Any]]:
        """
        将所有数据转换为 E-2GraphRAG 格式

        Returns:
            List[Dict]: E-2GraphRAG 格式的数据列表
        """
        results = []
        for index in range(len(self.available_ids)):
            try:
                e2graphrag_data = self.convert_to_e2graphrag_format(index=index)
                results.append(e2graphrag_data)
            except Exception as e:
                print(f"转换索引 {index} 的数据时出错: {e}")
                continue
        return results

    def get_dialog_for_memory(self, doc_id: str = None, index: int = None, system_name: str = None) -> List[Dict[str, Any]]:
        """
        获取用于构建记忆系统的对话数据

        Args:
            doc_id: 文档 ID（可选）
            index: 数据索引（可选）
            system_name: 记忆系统名称（可选，如果不指定则使用优先级选择）

        Returns:
            List[Dict]: 对话列表
        """
        # 获取数据项
        if doc_id:
            if doc_id not in self.dataset:
                raise ValueError(f"文档 ID {doc_id} 不存在")
            item = self.dataset[doc_id]
        elif index is not None:
            if index >= len(self.available_ids):
                raise ValueError(f"索引 {index} 超出范围")
            item = self.dataset[self.available_ids[index]]
        else:
            raise ValueError("必须提供 doc_id 或 index 之一")

        # 如果指定了系统名称，使用指定的
        if system_name and system_name in item.get("dialogs", {}):
            return item["dialogs"][system_name]

        # 否则使用优先级选择
        preferred_systems = ["bm25_dialog", "embedder_dialog", "a_mem", "mem0", "memoryos"]
        for system in preferred_systems:
            if system in item.get("dialogs", {}):
                return item["dialogs"][system]

        # 如果都没有，返回第一个可用的
        dialog_systems = list(item.get("dialogs", {}).keys())
        if dialog_systems:
            return item["dialogs"][dialog_systems[0]]

        # 如果完全没有对话，返回空列表
        return []

    def __getitem__(self, index):
        return self.dataset[self.available_ids[index]]

    def __len__(self):
        return len(self.available_ids)


if __name__ == "__main__":
    # loader = NovelQALoader("NovelQA")
    # print(loader[0]["qa"][0])
    # print(loader[0]["qa"][0]["question"].split("\n")[0])
    # loader = InfiniteChoiceLoader()
    # print(loader[0].keys())
    # print(type(loader[0]["book"]))
    # print(loader[0]["qa"])
    # loader = InfiniteQALoader()
    # print(loader[0].keys())
    # print(type(loader[0]["book"]))
    # print(loader[0]["qa"])
    # print(len(loader))
    # loader = longmemevalLoader("data/longmemeval/longmemeval_s")
    # print("数据集结构:")
    # print(loader[0].keys())
    # print(f"会话包含 {len(loader[0]['rounds'])} 个sessions")
    # if loader[0]['rounds']:
    #     print(f"第一个session包含 {len(loader[0]['rounds'][0])} 个rounds")
    #     if loader[0]['rounds'][0]:
    #         print(f"第一个round的结构:")
    #         print(loader[0]['rounds'][0][0].keys())
    #         print(f"User: {loader[0]['rounds'][0][0]['user']['content'][:100]}...")
    #         print(f"Assistant: {loader[0]['rounds'][0][0]['assistant']['content'][:100]}...")

    # with open("dataloader_test.json", "w") as outfile:
    #     json.dump(loader[0], outfile, indent=4)

    # MemoryBenchLoader 测试示例
    # loader = MemoryBenchLoader("data/locomo/locomo.json")
    # print("MemoryBench 数据集结构:")
    # print(f"数据集大小: {len(loader)}")
    # if len(loader) > 0:
    #     print(f"第一个条目的键: {list(loader[0].keys())}")
    #     print(f"数据集名称: {loader[0]['dataset_name']}")
    #     print(f"问题: {loader[0]['qa'][0]['question']}")
    #     print(f"答案: {loader[0]['qa'][0]['answer']}")
    #     print(f"可用的记忆系统: {loader.get_available_systems(loader.available_ids[0])}")
    pass
