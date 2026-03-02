# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def parse_log(file_path):
    data = []
    # 正则匹配日志行: "ModelName | DatasetName | Score | Acc"
    # 示例: RN50_openai | cars | 1.2818 | 0.5425
    pattern = re.compile(r'([\w\-\.]+)\s+\|\s+([\w\_]+)\s+\|\s+([\-\d\.]+)\s+\|\s+([\d\.]+)')

    with open(file_path, 'r') as f:
        for line in f:
            if "FAILED" in line:
                continue
            match = pattern.search(line)
            if match:
                model, dataset, score, acc = match.groups()
                data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'LogME': float(score),
                    'Accuracy': float(acc)
                })
    return pd.DataFrame(data)


def analyze(df):
    if df.empty:
        print("No valid data parsed!")
        return

    # 1. 计算每个数据集的 Tau
    datasets = df['Dataset'].unique()
    tau_results = []

    print(f"\n{'Dataset':<25} | {'Tau':<8} | {'Models'}")
    print("-" * 45)

    for ds in datasets:
        sub_df = df[df['Dataset'] == ds]
        if len(sub_df) < 3:  # 样本太少无法计算相关性
            continue

        tau, p_value = kendalltau(sub_df['LogME'], sub_df['Accuracy'])
        tau_results.append(tau)
        print(f"{ds:<25} | {tau:.4f}   | {len(sub_df)}")

    avg_tau = np.mean(tau_results)
    print("-" * 45)
    print(f"Average Kendall's Tau (Oracle): {avg_tau:.4f}")
    print(f"Total Datasets Analyzed: {len(tau_results)}")

    # 2. 画图 (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Accuracy', y='LogME', hue='Dataset', legend=False, alpha=0.7)
    plt.title(f'Oracle LogME vs. Ground Truth Accuracy\nAvg Tau = {avg_tau:.4f}')
    plt.xlabel('Ground Truth Accuracy')
    plt.ylabel('LogME Score (Oracle)')
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = 'oracle_analysis.png'
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    # 请先将你刚才的输出保存为 oracle_log.txt
    log_file = "oracle_log.txt"

    # 如果你没有保存文件，我这里直接把你的日志作为字符串处理（演示用）
    # 但实际操作建议你保存文件
    import os

    if not os.path.exists(log_file):
        print(f"Please save your output content to {log_file} first!")
    else:
        df = parse_log(log_file)
        analyze(df)