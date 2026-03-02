import torch
import pickle
import os
import numpy as np
import glob
from tqdm import tqdm

# ================= 配置区 =================
GT_BASE = "ptm_stats/stats_on_hist_task"
MY_BASE = "ptm_stats_test/stats_on_hist_task"


# ==========================================

def load_pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def check_tensor_diff(gt_tensor, my_tensor):
    # 确保转为 float32 CPU 进行对比
    gt = gt_tensor.float().cpu()
    my = my_tensor.float().cpu()

    if gt.shape != my.shape:
        return False, f"Shape mismatch: {gt.shape} vs {my.shape}"

    diff = (gt - my).abs()
    max_diff = diff.max().item()

    # 阈值判定：1e-4 允许一定的浮点误差
    if max_diff < 1e-4:
        return True, f"{max_diff:.2e}"
    else:
        return False, f"MaxDiff: {max_diff:.2e}"


def main():
    # 1. 自动扫描你跑出来的所有模型结果
    search_pattern = os.path.join(MY_BASE, "img_feat", "*.pkl")
    my_model_files = glob.glob(search_pattern)

    if not my_model_files:
        print(f"❌ 在 {search_pattern} 下没有找到任何 .pkl 文件！请检查路径或确认是否运行了生成脚本。")
        return

    print(f"🔍 发现 {len(my_model_files)} 个模型文件，开始批量验证...\n")
    print(f"{'Model Name':<40} | {'Dataset':<20} | {'Status':<10} | {'Info'}")
    print("-" * 90)

    stats = {"pass": 0, "fail": 0, "skip": 0}

    for my_file in my_model_files:
        # 获取文件名 (e.g., ViT-B-32_openai.pkl)
        filename = os.path.basename(my_file)
        model_name = filename.replace('.pkl', '')

        # 对应的 GT 文件路径
        gt_file = os.path.join(GT_BASE, "img_feat", filename)
        gt_text_file = os.path.join(GT_BASE, "text_classifier", filename)
        my_text_file = os.path.join(MY_BASE, "text_classifier", filename)

        if not os.path.exists(gt_file):
            print(f"{model_name:<40} | {'ALL':<20} | ⚠️ SKIP | GT missing")
            stats["skip"] += 1
            continue

        # 加载数据
        my_img_data = load_pkl(my_file)
        gt_img_data = load_pkl(gt_file)

        # 尝试加载 Text Classifier (如果有的话)
        has_text = os.path.exists(gt_text_file) and os.path.exists(my_text_file)
        my_text_data = load_pkl(my_text_file) if has_text else {}
        gt_text_data = load_pkl(gt_text_file) if has_text else {}

        if my_img_data is None or gt_img_data is None:
            continue

        # 遍历该模型下的所有数据集 (Dataset)
        # 注意：这里以你跑出来的数据集为准
        for dataset_name in my_img_data.keys():
            if dataset_name not in gt_img_data:
                print(f"{model_name:<40} | {dataset_name:<20} | ❓ UNK  | Dataset missing in GT")
                continue

            # --- 验证 1: 图像特征 (Image Feat) ---
            # 为了速度，只验证每个数据集的第一个 Class 的特征
            classes = list(my_img_data[dataset_name].keys())
            if not classes:
                continue
            first_class = classes[0]

            my_feat = my_img_data[dataset_name][first_class]
            gt_feat = gt_img_data[dataset_name][first_class]

            passed_img, msg_img = check_tensor_diff(gt_feat, my_feat)

            # --- 验证 2: 文本分类器 (Text Classifier) ---
            passed_txt = True
            msg_txt = "No Text"
            if has_text and dataset_name in my_text_data and dataset_name in gt_text_data:
                my_w = my_text_data[dataset_name]
                gt_w = gt_text_data[dataset_name]
                passed_txt, msg_txt = check_tensor_diff(gt_w, my_w)

            # --- 综合判定 ---
            if passed_img and passed_txt:
                print(f"{model_name:<40} | {dataset_name:<20} | ✅ PASS | Diff: {msg_img}")
                stats["pass"] += 1
            else:
                fail_reason = []
                if not passed_img: fail_reason.append(f"Img({msg_img})")
                if not passed_txt: fail_reason.append(f"Txt({msg_txt})")
                print(f"{model_name:<40} | {dataset_name:<20} | ❌ FAIL | {', '.join(fail_reason)}")
                stats["fail"] += 1

    print("-" * 90)
    print(f"验证完成: ✅ {stats['pass']} Passed, ❌ {stats['fail']} Failed, ⚠️ {stats['skip']} Skipped")


if __name__ == "__main__":
    main()