import torch
import pickle
import os
import numpy as np

# 需要检查的模型列表
FAILED_MODELS = [
    "RN50_openai",
    "RN101_openai",
    "RN50x4_openai",
    "RN50x16_openai",
    "RN50x64_openai"
]

GT_BASE = "ptm_stats/stats_on_hist_task/img_feat"
MY_BASE = "ptm_stats_test/stats_on_hist_task/img_feat"
DATASET = "cifar100"

def check_models():
    print(f"🔍 深度验证 5 个 ResNet 模型的数值一致性 (Dataset: {DATASET})\n")
    print(f"{'Model':<20} | {'Max Diff':<10} | {'Correlation':<12} | {'Dot Product Sim'}")
    print("-" * 70)

    for model_name in FAILED_MODELS:
        gt_path = os.path.join(GT_BASE, f"{model_name}.pkl")
        my_path = os.path.join(MY_BASE, f"{model_name}.pkl")

        if not os.path.exists(my_path):
            print(f"{model_name:<20} | 文件不存在，跳过")
            continue

        with open(gt_path, 'rb') as f: gt_data = pickle.load(f)[DATASET]
        with open(my_path, 'rb') as f: my_data = pickle.load(f)[DATASET]

        # 取第一个类别的特征进行对比
        # 结构: Dict[classname] -> Tensor
        first_class = list(gt_data.keys())[0]
        gt_tensor = gt_data[first_class].float().cpu().numpy().flatten()
        my_tensor = my_data[first_class].float().cpu().numpy().flatten()

        # 1. 计算最大绝对误差
        max_diff = np.max(np.abs(gt_tensor - my_tensor))

        # 2. 计算皮尔逊相关系数 (应该接近 1.0)
        corr = np.corrcoef(gt_tensor, my_tensor)[0, 1]

        # 3. 计算余弦相似度 (应该接近 1.0)
        # 既然已经是归一化特征，点积就是余弦相似度
        norm_gt = gt_tensor / np.linalg.norm(gt_tensor)
        norm_my = my_tensor / np.linalg.norm(my_tensor)
        cos_sim = np.dot(norm_gt, norm_my)

        print(f"{model_name:<20} | {max_diff:.2e}   | {corr:.8f}   | {cos_sim:.8f}")

if __name__ == "__main__":
    check_models()