import os
import sys
import torch
import numpy as np
import yaml  # 新增：用于读取 yml 配置
from oracle_eval.data_loader import SWABDataLoader
from LogME_official.LogME import LogME



def get_project_root():
    # 当前脚本所在的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 当前脚本所在的目录 (hybrid_eval)
    current_dir = os.path.dirname(current_file_path)
    # 项目根目录 (hybrid_eval 的父目录)
    root_dir = os.path.dirname(current_dir)
    return root_dir


def load_model_list(config_path):
    """
    从 yml 文件加载模型列表。
    支持格式：
    - ['RN50', 'openai']  -> 自动转换为 'RN50_openai'
    - 'ViT-B-32_openai'  -> 保持原样
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("models.yml must contain a list of models")

    model_list = []
    for item in data:
        if isinstance(item, list):
            # 处理 ['RN50', 'openai'] 格式
            model_name = "_".join(item)
            model_list.append(model_name)
        elif isinstance(item, str):
            # 处理 'RN50_openai' 格式
            model_list.append(item)
        else:
            print(f"Warning: Skipping invalid model entry: {item}")

    return model_list


def load_dataset_list(config_path):
    """
    从 txt 文件加载数据集列表。
    每行一个数据集名称，自动忽略空行和空白字符。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    dataset_list = []
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dataset_list.append(line)
    return dataset_list


# --- 主程序逻辑 ---

def run_experiment():
    # 1. 确定路径
    ROOT_DIR = get_project_root()  # 更新为项目根目录，确保数据加载路径一致
    model_config_path = os.path.join(ROOT_DIR, 'model', 'models.yml')
    dataset_config_path = os.path.join(ROOT_DIR, 'data', 'datasets_name.txt')

    # 2. 加载配置
    try:
        MODEL_LIST = load_model_list(model_config_path)
        DATASET_LIST = load_dataset_list(dataset_config_path)
        print(f"Loaded {len(MODEL_LIST)} models and {len(DATASET_LIST)} datasets.")
    except Exception as e:
        print(f"Error loading configurations: {e}")
        sys.exit(1)

    loader = SWABDataLoader(ROOT_DIR)
    results = {}

    print(f"{'Model':<25} | {'Dataset':<20} | {'LogME Score':<12} | {'GT Acc':<10}")
    print("-" * 75)

    for dataset in DATASET_LIST:
        for model in MODEL_LIST:
            try:
                # 1. 加载数据
                feats, logits, targets = loader.load_data(model, dataset)

                # 2. 计算 GT Accuracy
                acc = (logits.argmax(dim=1) == targets).float().mean().item()

                # 3. 计算 LogME (Oracle)
                logme = LogME(regression=False)
                # LogME 需要 numpy 格式
                score = logme.fit(feats.numpy(), targets.numpy())

                print(f"{model:<25} | {dataset:<20} | {score:.4f}       | {acc:.4f}")

                # 存储结果以便后续分析相关性
                key = (dataset, model)
                results[key] = {'logme': score, 'acc': acc}

            except Exception as e:
                # 打印具体错误以便调试
                print(f"{model:<25} | {dataset:<20} | FAILED ({str(e)})")

    # TODO: 这里可以计算 Kendall's Tau 相关性
    return results


if __name__ == "__main__":
    run_experiment()