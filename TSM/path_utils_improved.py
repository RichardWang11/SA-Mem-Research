#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的路径管理工具，提供更清晰的输出文件组织
"""

import os
from datetime import datetime
from pathlib import Path

def get_output_base_dir():
    """获取输出基础目录，在项目根目录创建outputs文件夹"""
    base_dir = Path("./outputs")
    base_dir.mkdir(exist_ok=True)
    return base_dir

def get_timestamped_run_dir(dataset_name, model_name, split_method):
    """
    创建带时间戳的运行目录
    格式: outputs/YYYYMMDD_HHMMSS_dataset_model_method/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 清理模型名称和方法名称，去除特殊字符
    model_short = model_name.replace("gpt-", "").replace("claude-", "c").replace(".", "_")
    method_short = split_method.replace("_", "")

    # 创建运行目录名称
    run_name = f"{timestamp}_{dataset_name}_{model_short}_{method_short}"
    run_dir = get_output_base_dir() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def get_latest_results_dir():
    """
    创建并返回latest链接目录，指向最新的结果
    """
    latest_dir = get_output_base_dir() / "latest"
    return latest_dir

def create_latest_symlink(run_dir):
    """
    创建指向最新运行结果的符号链接
    """
    latest_dir = get_latest_results_dir()
    # 如果存在旧的符号链接，先删除
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        elif latest_dir.is_dir():
            import shutil
            shutil.rmtree(latest_dir)

    # 创建新的符号链接（相对路径）
    try:
        # 使用相对路径创建符号链接
        relative_path = os.path.relpath(run_dir, latest_dir.parent)
        latest_dir.symlink_to(relative_path)
    except:
        # 如果符号链接失败（比如在Windows上），就复制README文件
        latest_dir.mkdir(exist_ok=True)
        readme_path = latest_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"最新的结果位于: {run_dir}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def get_organized_paths(config_bundle):
    """
    根据配置生成组织良好的输出路径
    返回一个包含所有路径的字典

    """
    # 从配置中提取关键信息
    dataset_name = config_bundle.dataset_name
    model_name = config_bundle.llm_api.get("model", "unknown")
    split_method = config_bundle.split_method
    extraction_method = config_bundle.extraction_method

    # 获取数据文件名（不含路径和扩展名）
    data_file = Path(config_bundle.dataset_path).stem

    # 创建运行目录
    run_dir = get_timestamped_run_dir(data_file, model_name, split_method)

    # 创建子目录
    results_dir = run_dir / "results"
    cache_dir = run_dir / "cache"
    logs_dir = run_dir / "logs"
    evaluation_dir = run_dir / "evaluation"

    # 创建所有目录
    for dir_path in [results_dir, cache_dir, logs_dir, evaluation_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 生成结果文件名（更简洁）
    result_filename = f"{data_file}_results.json"

    # 创建配置摘要文件
    config_summary_path = run_dir / "config_summary.json"
    import json
    config_summary = {
        "run_timestamp": datetime.now().isoformat(),
        "dataset": {
            "name": dataset_name,
            "file": data_file,
            "path": config_bundle.dataset_path
        },
        "model": {
            "name": model_name,
            "max_tokens": config_bundle.max_tokens
        },
        "method": {
            "split": split_method,
            "extraction": extraction_method,
            "temporal_enabled": config_bundle.temporal_enable
        },
        "parameters": {
            "cluster_length": config_bundle.cluster_length,
            "cluster_overlap": config_bundle.cluster_overlap,
            "cluster_merge_num": config_bundle.cluster_merge_num,
            "language": config_bundle.language
        }
    }

    with open(config_summary_path, 'w', encoding='utf-8') as f:
        json.dump(config_summary, f, indent=2, ensure_ascii=False)

    # 创建README文件
    readme_path = run_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"# E2GraphRAG 运行结果\n\n")
        f.write(f"## 运行信息\n")
        f.write(f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **数据集**: {dataset_name} ({data_file})\n")
        f.write(f"- **模型**: {model_name}\n")
        f.write(f"- **分割方法**: {split_method}\n")
        f.write(f"- **抽取方法**: {extraction_method}\n")
        f.write(f"- **时序分析**: {'开启' if config_bundle.temporal_enable else '关闭'}\n\n")
        f.write(f"## 目录结构\n")
        f.write(f"- `results/` - 主要结果文件\n")
        f.write(f"- `cache/` - 缓存文件\n")
        f.write(f"- `logs/` - 运行日志\n")
        f.write(f"- `evaluation/` - 评估结果\n")
        f.write(f"- `config_summary.json` - 配置摘要\n\n")
        f.write(f"## 如何查看结果\n")
        f.write(f"1. 主要结果: `results/{result_filename}`\n")
        f.write(f"2. 评估结果: 运行 `python run_evaluation.py results/{result_filename}`\n")

    # 更新最新结果链接
    create_latest_symlink(run_dir)

    # 返回路径字典
    paths = {
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "cache_dir": str(cache_dir),
        "logs_dir": str(logs_dir),
        "evaluation_dir": str(evaluation_dir),
        "result_file": str(results_dir / result_filename),
        "config_summary": str(config_summary_path),
        "readme": str(readme_path)
    }

    # 打印友好的输出信息
    print("\n" + "="*60)
    print("📁 输出目录已创建")
    print("="*60)
    print(f"🏠 主目录: {run_dir}")
    print(f"📊 结果文件: {paths['result_file']}")
    print(f"🔗 快速访问: outputs/latest/")
    print(f"📖 查看详情: {readme_path}")
    print("="*60 + "\n")

    return paths

def get_organized_cache_folder(cache_dir, data_file, piece_name, config_hash=None):
    """
    获取缓存文件夹路径。

    修改后的逻辑（2025-12-10）：
    为了保护历史实验记录不被覆盖，不再直接返回共享的稳定缓存路径，
    而是返回当前运行目录下的独立缓存路径。

    如果稳定缓存存在，会将其内容【复制】到当前目录，实现复用但不覆盖。

    Args:
        cache_dir: 运行目录下的缓存目录
        data_file: 数据文件名
        piece_name: 数据项名称
        config_hash: 配置哈希值
    """
    # 1. 确定稳定缓存路径（仅作为读取源）
    stable_cache_root = Path("./outputs/cache")
    if config_hash:
        stable_cache_path = stable_cache_root / data_file / config_hash / str(piece_name)
    else:
        stable_cache_path = stable_cache_root / data_file / str(piece_name)

    # 2. 确定当前运行的独立缓存路径（作为实际读写目标）
    run_cache_path = Path(cache_dir) / data_file / str(piece_name)
    run_cache_path.mkdir(parents=True, exist_ok=True)

    # 3. 尝试从稳定缓存复制数据（如果当前缓存为空且稳定缓存存在）
    # 检查run_cache_path是否为空
    is_empty = not any(run_cache_path.iterdir())

    if is_empty and stable_cache_path.exists():
        try:
            import shutil
            print(f"📋 [缓存复用] 正在从稳定缓存复制数据...")
            print(f"   源: {stable_cache_path}")
            print(f"   目标: {run_cache_path}")

            # 复制目录内容
            shutil.copytree(stable_cache_path, run_cache_path, dirs_exist_ok=True)
            print(f"✅ 缓存复制完成")
        except Exception as e:
            print(f"⚠️ 缓存复制失败: {e}")
            print("   将使用空目录开始新的构建")

    # 4. 返回当前运行的独立路径
    return str(run_cache_path)


def get_stable_cache_path(data_file, piece_name, config_hash=None):
    """
    返回稳定缓存路径（跨run复用的权威缓存位置）。
    约定：outputs/cache/<data_file>/<config_hash>/<piece_name>/
    """
    stable_cache_root = Path("./outputs/cache")
    if config_hash:
        return str(stable_cache_root / data_file / config_hash / str(piece_name))
    return str(stable_cache_root / data_file / str(piece_name))

def print_final_summary(paths, elapsed_time=None):
    """
    打印最终的结果摘要
    """
    print("\n" + "="*60)
    print("✅ 运行完成！")
    print("="*60)
    if elapsed_time:
        print(f"⏱️  总用时: {elapsed_time:.2f} 秒")
    print(f"📊 结果保存在: {paths['result_file']}")
    print(f"📁 完整输出: {paths['run_dir']}")
    print(f"🔗 快速访问: outputs/latest/")
    print("\n💡 提示:")
    print("  1. 查看结果: cat outputs/latest/results/*.json")
    print("  2. 运行评估: python run_evaluation.py outputs/latest/results/*.json")
    print("="*60 + "\n")
