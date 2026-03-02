# VLM Model Selection Framework

基于视觉-语言模型(VLM)的模型选择实验框架，用于毕业论文研究。

## 项目概述

本项目旨在研究和改进VLM模型选择方法，主要基于VEGA框架进行优化。

### 核心目标

1. **基线复现**: 复现LogME和VEGA基线方法
2. **创新优化**: 基于置信度估计和语义拓扑优化模型选择
3. **实验验证**: 在多个数据集和模型上进行验证

## 目录结构

```
e:/SWAB/
├── configs/                    # 配置文件
│   ├── dataset_config.py      # 数据集配置
│   └── model_config.py        # 模型配置
├── data/                       # 数据加载
│   └── data_loader.py         # 统一数据加载接口
├── methods/                    # 核心方法实现
│   └── baseline/              # 基线方法
│       ├── logme.py          # LogME (wrapper of official)
│       └── vega.py           # VEGA实现
├── evaluation/                 # 评估模块
│   └── metrics.py             # 评估指标(Rank Correlation等)
├── experiments/                # 实验脚本
│   ├── run_baselines.py       # 运行基线实验
│   └── evaluate.py            # 评估工具
├── oracle_eval/                # Oracle评估代码
├── LogME_official/             # 官方LogME实现
├── utils/                      # 工具函数
├── legacy/                     # 旧代码存档
│   ├── LOVM/                  # LOVM相关
│   ├── Original_SWAB_Core_Code/# SWAB原始代码
│   ├── pilot_study/           # 初步实验
│   └── ...
└── doc/                        # 文档
    ├── important_notes.md     # 重要笔记
    └── opening_report.txt     # 开题报告
```

## 快速开始

### 安装依赖

```bash
pip install numpy torch scipy tqdm numba
```

### 运行基线实验

```python
from experiments import run_logme_experiment, run_vega_experiment
from data.data_loader import load_model_data

# 加载数据
features_dict, logits_dict, labels = load_model_data('dataset_name')

# 运行LogME
logme_metrics = run_logme_experiment(features_dict, labels, ground_truth_acc)

# 运行VEGA
vega_metrics = run_vega_experiment(features_dict, text_embeddings, logits_dict, ground_truth_acc)
```

### 使用单独的方法

```python
from methods.baseline import LogME, VEGAScorer

# LogME
logme = LogME()
score = logme.fit(features, labels)

# VEGA
vega = VEGAScorer(k_neighbors=10)
score = vega.compute_score(features, text_embeddings, logits)
```

## 实验进度

- [x] 代码框架重组
- [x] LogME基线实现
- [x] VEGA基线框架
- [ ] VEGA完整实现（需参考论文）
- [ ] 置信度估计模块
- [ ] 语义拓扑构建模块
- [ ] 完整实验验证

## 方法说明

### LogME

LogME (Log Maximum Evidence) 是一种基于贝叶斯线性回归的迁移性评估方法，通过计算特征对标签的拟合能力来评估模型。

**优点**: 计算快速，无需训练
**缺点**: 未考虑视觉-语言模型的语义结构

### VEGA

VEGA 基于图匹配方法构建视觉图和语义图，通过图匹配评分评估模型适配性。

**核心思想**:
- 构建视觉特征图（图像之间的相似度）
- 构建语义图（类别之间的相似度）
- 计算图匹配分数

## 数据说明

### 可用数据集 (ptm_stats/logits/)

约50个VLM模型在22个数据集上的zero-shot预测结果：

- **图像分类**: ImageNet, CIFAR100, Cars, Flowers, DTD等
- **细粒度任务**: diabetic_retinopathy, eurosat, fer2013等
- **CLEVR任务**: count, distance等

### 数据格式

- `logits/{model}_{dataset}.npy`: 模型输出的logits [N, C]
- `features/{model}_{dataset}.npy`: 图像特征 [N, D]
- `labels/{dataset}.npy`: 真实标签 [N]

## 参考文献

1. LogME: You et al., "LogME: Practical Assessment of Pre-trained Models for Transfer Learning", ICML 2021
2. VEGA: [待补充]
3. LOVM: [待补充]
4. SWAB: [待补充]

## 作者

毕业论文实验项目