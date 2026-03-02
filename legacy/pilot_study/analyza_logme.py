import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
df = pd.read_csv("ptm_stats_test/logme_vs_ece_analysis.csv")

# 清洗：移除无效值和 ImageNet1k（无 ECE）
df = df[df['Dataset'] != 'imagenet1k'].dropna(subset=['ECE', 'LogME'])

print(f"📊 有效数据点: {len(df)}")
print(f"   LogME 范围: [{df['LogME'].min():.3f}, {df['LogME'].max():.3f}]")
print(f"   ECE 范围: [{df['ECE'].min():.3f}, {df['ECE'].max():.3f}]")

# ==================== 2. 散点图：ECE vs LogME ====================
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='ECE', y='LogME', alpha=0.6, s=40, color='steelblue')

# 添加趋势线
sns.regplot(data=df, x='ECE', y='LogME', scatter=False, color='red', line_kws={'linewidth': 2})

# 标注相关系数
corr = df['ECE'].corr(df['LogME'])
plt.title(f'ECE vs LogME (r = {corr:.3f})', fontsize=14, fontweight='bold')
plt.xlabel('ECE ↓', fontsize=12)
plt.ylabel('LogME ↑', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ece_vs_logme_scatter.png", dpi=300, bbox_inches='tight')
plt.show()

# ==================== 3. 按数据集着色 ====================
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='ECE', y='LogME', hue='Dataset', palette='tab20', alpha=0.7, s=50)
plt.title('ECE vs LogME (按数据集着色)', fontsize=14, fontweight='bold')
plt.xlabel('ECE ↓', fontsize=12)
plt.ylabel('LogME ↑', fontsize=12)
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ece_vs_logme_by_dataset.png", dpi=300, bbox_inches='tight')
plt.show()

# ==================== 4. 按模型家族着色 ====================
# 提取模型家族
def get_family(model_name):
    if 'ViT' in model_name or 'vit' in model_name:
        return 'ViT'
    elif 'RN' in model_name or 'resnet' in model_name.lower():
        return 'ResNet'
    elif 'convnext' in model_name.lower():
        return 'ConvNeXt'
    elif 'beit' in model_name.lower():
        return 'BEiT'
    elif 'blip' in model_name.lower():
        return 'BLIP'
    elif 'coca' in model_name.lower():
        return 'CoCa'
    else:
        return 'Other'

df['Family'] = df['Model'].apply(get_family)

plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='ECE', y='LogME', hue='Family', palette='Set2', alpha=0.7, s=60)
plt.title('ECE vs LogME (按模型家族着色)', fontsize=14, fontweight='bold')
plt.xlabel('ECE ↓', fontsize=12)
plt.ylabel('LogME ↑', fontsize=12)
plt.legend(title='Model Family', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ece_vs_logme_by_family.png", dpi=300, bbox_inches='tight')
plt.show()

# ==================== 5. 输出相关系数 ====================
print("\n" + "="*50)
print(f"📊 ECE 与 LogME 的 Pearson 相关系数: {corr:.4f}")
print("="*50)
if corr > 0:
    print("→ 正相关：ECE 越高，LogME 越高（校准差的模型迁移能力反而强）")
else:
    print("→ 负相关：ECE 越低，LogME 越高（校准好的模型迁移能力更强）")
print("="*50)