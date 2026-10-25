"""
Token统计跟踪模块
用于跟踪不同阶段的token使用情况
"""
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
from toolkits.token_monitor import CostStateManager, CostState

# 定义阶段类型
STAGE_EXTRACT = "extract"  # 抽取阶段
STAGE_SUMMARY = "summary"  # 总结阶段 (build_tree_clustering)
STAGE_UPDATE = "update"    # 更新阶段
STAGE_GENERATION = "generation"  # 生成阶段

class TokenTracker:
    """Token使用统计跟踪器"""

    def __init__(self, model_name: str, enable_tracking: bool = True):
        """
        初始化Token跟踪器

        Args:
            model_name: 模型名称
            enable_tracking: 是否启用跟踪
        """
        self.model_name = model_name
        self.enable_tracking = enable_tracking

        if enable_tracking:
            # 为每个阶段注册CostState
            # 使用字典形式支持多操作类型
            stage_states = {
                STAGE_EXTRACT: CostState(),
                STAGE_SUMMARY: CostState(),
                STAGE_UPDATE: CostState(),
                STAGE_GENERATION: CostState(),
            }
            CostStateManager.register(
                model=model_name,
                state=stage_states,
                exist_ok=True
            )

    def get_current_stage(self) -> Optional[str]:
        """获取当前阶段（通过线程局部变量）"""
        import threading
        return getattr(threading.current_thread(), '_token_tracker_stage', None)

    def set_stage(self, stage: str):
        """设置当前阶段"""
        import threading
        threading.current_thread()._token_tracker_stage = stage

    def clear_stage(self):
        """清除当前阶段"""
        import threading
        if hasattr(threading.current_thread(), '_token_tracker_stage'):
            delattr(threading.current_thread(), '_token_tracker_stage')

    def get_statistics(self) -> Dict[str, Any]:
        """获取所有阶段的统计信息"""
        if not self.enable_tracking:
            return {}

        try:
            states = CostStateManager.get(self.model_name)
            if isinstance(states, dict):
                return {
                    stage: state.to_dict()
                    for stage, state in states.items()
                }
            else:
                return {"default": states.to_dict()}
        except KeyError:
            return {}

    def save_statistics(self, filepath: str):
        """保存统计信息到文件"""
        if not self.enable_tracking:
            return

        stats = self.get_statistics()
        output = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "stages": stats
        }

        # 计算总计
        total_input_tokens = sum(s.get("input_tokens", 0) for s in stats.values())
        total_output_tokens = sum(s.get("output_tokens", 0) for s in stats.values())
        total_tokens = total_input_tokens + total_output_tokens
        total_time = sum(s.get("total_time", 0) for s in stats.values())
        total_count = sum(s.get("total_count", 0) for s in stats.values())

        output["summary"] = {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "total_count": total_count,
            "average_tokens_per_call": total_tokens / max(total_count, 1),
            "average_time_per_call": total_time / max(total_count, 1),
        }

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    def print_summary(self):
        """打印统计摘要"""
        if not self.enable_tracking:
            return

        stats = self.get_statistics()
        if not stats:
            print("暂无统计数据")
            return

        print("\n" + "="*60)
        print(f"Token使用统计 - 模型: {self.model_name}")
        print("="*60)

        for stage, stage_stats in stats.items():
            print(f"\n【{stage.upper()}阶段】")
            print(f"  调用次数: {stage_stats.get('total_count', 0)}")
            print(f"  输入Token: {stage_stats.get('input_tokens', 0):,}")
            print(f"  输出Token: {stage_stats.get('output_tokens', 0):,}")
            print(f"  总Token: {stage_stats.get('total_tokens', 0):,}")
            print(f"  总耗时: {stage_stats.get('total_time', 0):.2f}秒")
            if stage_stats.get('total_count', 0) > 0:
                print(f"  平均每次Token: {stage_stats.get('average_tokens_per_call', 0):.2f}")
                print(f"  平均每次耗时: {stage_stats.get('average_time_per_call', 0):.3f}秒")

        # 打印总计
        total_input = sum(s.get("input_tokens", 0) for s in stats.values())
        total_output = sum(s.get("output_tokens", 0) for s in stats.values())
        total_tokens = total_input + total_output
        total_time = sum(s.get("total_time", 0) for s in stats.values())
        total_count = sum(s.get("total_count", 0) for s in stats.values())

        print(f"\n【总计】")
        print(f"  总调用次数: {total_count}")
        print(f"  总输入Token: {total_input:,}")
        print(f"  总输出Token: {total_output:,}")
        print(f"  总Token: {total_tokens:,}")
        print(f"  总耗时: {total_time:.2f}秒")
        if total_count > 0:
            print(f"  平均每次Token: {total_tokens / total_count:.2f}")
            print(f"  平均每次耗时: {total_time / total_count:.3f}秒")
        print("="*60 + "\n")

# 全局跟踪器实例
_global_tracker: Optional[TokenTracker] = None

def init_token_tracker(model_name: str, enable_tracking: bool = True) -> TokenTracker:
    """初始化全局token跟踪器"""
    global _global_tracker
    _global_tracker = TokenTracker(model_name, enable_tracking)
    return _global_tracker

def get_token_tracker() -> Optional[TokenTracker]:
    """获取全局token跟踪器"""
    return _global_tracker
