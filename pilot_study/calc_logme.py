import os
import pickle
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "LogME_official"))

# ================= 修复后的导入逻辑 =================
try:
    # 直接从下载的源码文件 LogME.py 中导入 LogME 类
    from LogME import LogME
except ImportError as e:
    raise ImportError(
        f"导入失败: {e}\n"
        "请确认你已在当前目录下运行了:\n"
        "git clone https://github.com/thuml/LogME.git LogME_official"
    )
# ================= 配置路径 =================
FEAT_DIR = "../ptm_stats_test/stats_on_hist_task/img_feat"
CALIB_DIR = "ptm_stats_test/stats_on_hist_task/calibration_metrics"
ACC_DIR = "ptm_stats_test/stats_on_hist_task/class_level_acc"
OUTPUT_CSV = "ptm_stats_test/logme_vs_ece_analysis.csv"


# ===========================================

def get_logme_score(features, labels):
    """
    计算 LogME 分数 (适配 Class API)
    :param features: [N, D] numpy array, float64
    :param labels: [N] numpy array, int
    :return: LogME score (float)
    """
    # 确保数据类型正确
    features = features.astype(np.float64)
    try:
        logme_model = LogME(regression=False)
    except TypeError:
        # 某些旧版本不需要 regression 参数
        logme_model = LogME()

    # 2. 调用 fit 方法
    score = logme_model.fit(features, labels)

    return score


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    # 检查输入目录
    if not os.path.exists(FEAT_DIR):
        print(f"Error: Feature directory not found: {FEAT_DIR}")
        return

    # 获取所有模型文件
    model_files = [f for f in os.listdir(FEAT_DIR) if f.endswith('.pkl')]
    model_files.sort()

    print(f"Found {len(model_files)} models to process.")

    all_results = []

    for m_file in tqdm(model_files, desc="Processing Models"):
        model_name = m_file.replace('.pkl', '')

        # 1. 加载特征数据
        feat_path = os.path.join(FEAT_DIR, m_file)
        try:
            feat_dict = load_pkl(feat_path)
        except Exception as e:
            print(f"Skipping {model_name}: Failed to load features ({e})")
            continue

        # 2. 加载 ECE 数据
        calib_path = os.path.join(CALIB_DIR, m_file)
        calib_data = {}
        if os.path.exists(calib_path):
            try:
                calib_data = load_pkl(calib_path)
            except Exception as e:
                print(f"Warning: Failed to load calibration data for {model_name}: {e}")

        # 3. 加载 Accuracy 数据
        acc_path = os.path.join(ACC_DIR, m_file)
        acc_data = {}
        if os.path.exists(acc_path):
            try:
                acc_raw = load_pkl(acc_path)
                for d_name, d_cls in acc_raw.items():
                    if d_cls:
                        acc_data[d_name] = np.mean(list(d_cls.values()))
            except Exception as e:
                print(f"Warning: Failed to load accuracy data for {model_name}: {e}")

        # 4. 遍历该模型下的每个数据集
        for dataset_name, class_data in feat_dict.items():
            if not class_data:
                continue

            # 重组数据为 (X, y) 格式
            X_list = []
            y_list = []
            sorted_classes = sorted(class_data.keys())

            total_samples = 0
            for label_idx, class_name in enumerate(sorted_classes):
                feats = class_data[class_name]
                if isinstance(feats, torch.Tensor):
                    feats = feats.cpu().numpy()

                if feats.size == 0 or feats.shape[0] == 0:
                    continue

                X_list.append(feats)
                labels = np.full(feats.shape[0], label_idx, dtype=np.int32)
                y_list.append(labels)
                total_samples += feats.shape[0]

            if total_samples == 0:
                continue

            try:
                # 拼接并转换为 float64（LogME 要求）
                X = np.concatenate(X_list, axis=0).astype(np.float64)
                y = np.concatenate(y_list, axis=0)

                # 计算 LogME
                logme_score = get_logme_score(X, y)

                # 获取 ECE 和 Accuracy
                ece_val = calib_data.get(dataset_name, {}).get('ece', np.nan)
                acc_val = acc_data.get(dataset_name, np.nan)

                all_results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'LogME': logme_score,
                    'ECE': ece_val,
                    'Accuracy': acc_val
                })

            except Exception as e:
                print(f"  Failed LogME for {model_name} on {dataset_name}: {type(e).__name__}: {e}")

    # ================= 保存结果 =================
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Results saved to: {OUTPUT_CSV}")
        print("Columns: Model, Dataset, LogME, ECE, Accuracy")

        # 打印相关性
        clean_df = df.dropna(subset=['LogME', 'ECE'])
        if not clean_df.empty:
            corr = clean_df['LogME'].corr(clean_df['ECE'])
            print(f"Global Correlation (LogME vs ECE): {corr:.4f} (N={len(clean_df)})")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()