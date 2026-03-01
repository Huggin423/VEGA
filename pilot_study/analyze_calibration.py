import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置路径
BASE_DIR = "../ptm_stats_test/stats_on_hist_task"
CALIB_DIR = os.path.join(BASE_DIR, "calibration_metrics")
ACC_DIR = os.path.join(BASE_DIR, "class_level_acc")
OUTPUT_DIR = "../analysis_plots"

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data():
    """读取所有 pkl 文件并将数据整理成 DataFrame"""
    data_list = []

    # 获取所有模型文件名
    model_files = [f for f in os.listdir(CALIB_DIR) if f.endswith('.pkl')]

    print(f"Found {len(model_files)} calibration files.")

    for filename in model_files:
        model_name = filename.replace('.pkl', '')

        # 1. 读取 Calibration 数据 (ECE/ACE)
        try:
            with open(os.path.join(CALIB_DIR, filename), 'rb') as f:
                calib_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading calibration for {model_name}: {e}")
            continue

        # 2. 读取 Accuracy 数据 (为了画散点图)
        # 注意：class_level_acc 存的是 {dataset: {class: acc}}，我们需要算出平均准确率
        acc_data = None
        acc_file_path = os.path.join(ACC_DIR, filename)
        if os.path.exists(acc_file_path):
            with open(acc_file_path, 'rb') as f:
                acc_raw = pickle.load(f)
                # 计算每个数据集的平均准确率 (Macro-Average)
                acc_data = {}
                for d_name, d_classes in acc_raw.items():
                    # d_classes 是 {class_name: acc}
                    if len(d_classes) > 0:
                        acc_data[d_name] = np.mean(list(d_classes.values()))
                    else:
                        acc_data[d_name] = 0.0

        # 3. 整合数据
        for dataset, metrics in calib_data.items():
            # 获取准确率
            accuracy = np.nan
            if acc_data and dataset in acc_data:
                accuracy = acc_data[dataset]

            data_list.append({
                'Model': model_name,
                'Dataset': dataset,
                'ECE': metrics.get('ece', np.nan),
                'ACE': metrics.get('ace', np.nan),
                'Accuracy': accuracy
            })

    return pd.DataFrame(data_list)


def plot_heatmap(df):
    """画 ECE 热力图：模型 vs 数据集"""
    plt.figure(figsize=(20, 12))

    # 转换成矩阵形式: 行=Dataset, 列=Model
    pivot_table = df.pivot(index="Dataset", columns="Model", values="ECE")

    # 绘制热力图
    sns.heatmap(pivot_table, cmap="viridis", annot=False, cbar_kws={'label': 'ECE (Lower is Better)'})

    plt.title("ECE Heatmap across Models and Datasets", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "ece_heatmap.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Heatmap to {save_path}")


def plot_scatter_acc_vs_ece(df):
    """画散点图：Accuracy vs ECE"""
    plt.figure(figsize=(10, 8))

    # 为了区分不同系列的模型，我们可以简单提取模型前缀
    # 例如 'ViT-B-32_openai' -> 'ViT'
    df['Model_Family'] = df['Model'].apply(lambda x: x.split('_')[0].split('-')[0])

    sns.scatterplot(
        data=df,
        x="Accuracy",
        y="ECE",
        hue="Model_Family",
        alpha=0.6,
        palette="tab10"
    )

    # 添加参考线
    plt.title("Correlation between Accuracy and ECE (Raw Logits)", fontsize=16)
    plt.xlabel("Class-Level Accuracy", fontsize=12)
    plt.ylabel("ECE (Expected Calibration Error)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 添加一段说明文字
    plt.figtext(0.5, -0.05, "Note: High ECE implies under-confidence due to raw cosine similarity scaling.",
                ha="center", fontsize=10, style='italic')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "scatter_acc_ece.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Scatter Plot to {save_path}")


def plot_model_comparison(df):
    """画柱状图：每个模型的平均 ECE"""
    plt.figure(figsize=(15, 8))

    # 按平均 ECE 排序
    order = df.groupby('Model')['ECE'].mean().sort_values().index

    sns.barplot(
        data=df,
        x="Model",
        y="ECE",
        order=order,
        palette="magma"
    )

    plt.title("Average ECE per Model (Sorted)", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Average ECE", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "model_avg_ece.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Bar Plot to {save_path}")


def main():
    print("Loading data...")
    df = load_data()

    if df.empty:
        print("No data found! Please check the pickle paths.")
        return

    print(f"Loaded {len(df)} records.")

    # 保存 CSV 供导师查看原始数据
    csv_path = os.path.join(OUTPUT_DIR, "calibration_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    print("\nGenerating plots...")
    plot_heatmap(df)
    plot_scatter_acc_vs_ece(df)
    plot_model_comparison(df)

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    main()