#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据探索脚本：确认 ptm_stats 文件夹中的数据结构
运行环境：实验室服务器 /root/mxy/SWAB
"""

import os
import torch
import pickle
import numpy as np
from pathlib import Path

def explore_logits(data_dir, model_name='RN50_openai', dataset_name='cars'):
    """探索 logits 文件结构"""
    logits_path = os.path.join(data_dir, 'ptm_stats/logits', f'{model_name}__{dataset_name}.pth')
    
    print("=" * 60)
    print(f"探索 Logits 文件: {logits_path}")
    print("=" * 60)
    
    if not os.path.exists(logits_path):
        print(f"文件不存在: {logits_path}")
        return None
    
    logits = torch.load(logits_path)
    print(f"类型: {type(logits)}")
    
    if isinstance(logits, dict):
        print(f"Keys: {logits.keys()}")
        for k, v in logits.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, np.ndarray)):
                print(f"  {k}: len={len(v)}")
            else:
                print(f"  {k}: type={type(v)}")
    elif isinstance(logits, (list, tuple)):
        print(f"Length: {len(logits)}")
        for i, item in enumerate(logits):
            if hasattr(item, 'shape'):
                print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  [{i}]: type={type(item)}")
    elif hasattr(logits, 'shape'):
        print(f"Shape: {logits.shape}")
        print(f"Dtype: {logits.dtype}")
        print(f"Min: {logits.min():.4f}, Max: {logits.max():.4f}")
    
    return logits


def explore_img_feat(data_dir, model_name='RN50_openai'):
    """探索图像特征文件结构"""
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索图像特征文件: {feat_path}")
    print("=" * 60)
    
    if not os.path.exists(feat_path):
        print(f"文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        img_feat = pickle.load(f)
    
    print(f"类型: {type(img_feat)}")
    
    if isinstance(img_feat, dict):
        print(f"数据集数量: {len(img_feat)}")
        print(f"数据集列表: {list(img_feat.keys())}")
        
        # 检查第一个数据集的结构
        first_key = list(img_feat.keys())[0]
        first_val = img_feat[first_key]
        print(f"\n示例数据集 '{first_key}':")
        if hasattr(first_val, 'shape'):
            print(f"  Shape: {first_val.shape}")
            print(f"  Dtype: {first_val.dtype}")
        elif isinstance(first_val, dict):
            print(f"  Keys (前5个): {list(first_val.keys())[:5]}")
        elif isinstance(first_val, (list, np.ndarray)):
            print(f"  Length: {len(first_val)}")
    
    return img_feat


def explore_text_feat(data_dir, model_name='RN50_openai'):
    """探索文本特征文件结构"""
    # caption_text_feat 可能是类别名称的 embeddings
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索文本特征文件 (caption_text_feat): {feat_path}")
    print("=" * 60)
    
    if not os.path.exists(feat_path):
        print(f"文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        text_feat = pickle.load(f)
    
    print(f"类型: {type(text_feat)}")
    
    if isinstance(text_feat, dict):
        print(f"数据集数量: {len(text_feat)}")
        print(f"数据集列表: {list(text_feat.keys())}")
        
        # 检查第一个数据集的结构
        first_key = list(text_feat.keys())[0]
        first_val = text_feat[first_key]
        print(f"\n示例数据集 '{first_key}':")
        if hasattr(first_val, 'shape'):
            print(f"  Shape: {first_val.shape}")
            print(f"  Dtype: {first_val.dtype}")
            print(f"  特征维度: {first_val.shape[-1] if len(first_val.shape) > 1 else 'N/A'}")
        elif isinstance(first_val, dict):
            print(f"  Keys (前5个): {list(first_val.keys())[:5]}")
        elif isinstance(first_val, (list, np.ndarray)):
            print(f"  Length: {len(first_val)}")
    
    return text_feat


def explore_class_level_acc(data_dir, model_name='RN50_openai'):
    """探索类别级别准确率"""
    acc_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/class_level_acc', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索类别级别准确率: {acc_path}")
    print("=" * 60)
    
    if not os.path.exists(acc_path):
        print(f"文件不存在: {acc_path}")
        return None
    
    with open(acc_path, 'rb') as f:
        acc = pickle.load(f)
    
    print(f"类型: {type(acc)}")
    
    if isinstance(acc, dict):
        print(f"数据集数量: {len(acc)}")
        
        # 检查 cars 数据集
        if 'cars' in acc:
            cars_acc = acc['cars']
            print(f"\nCars 数据集:")
            print(f"  类别数量: {len(cars_acc)}")
            # 计算平均准确率
            avg_acc = np.mean(list(cars_acc.values()))
            print(f"  平均准确率: {avg_acc:.4f}")
            print(f"  前3个类别: {list(cars_acc.items())[:3]}")
    
    return acc


def list_available_models(data_dir):
    """列出所有可用的模型"""
    logits_dir = os.path.join(data_dir, 'ptm_stats/logits')
    
    print("\n" + "=" * 60)
    print("可用的模型列表")
    print("=" * 60)
    
    if not os.path.exists(logits_dir):
        print(f"目录不存在: {logits_dir}")
        return []
    
    # 获取所有 .pth 文件
    files = [f for f in os.listdir(logits_dir) if f.endswith('.pth')]
    
    # 提取模型名称
    models = set()
    datasets = set()
    for f in files:
        # 格式: model__dataset.pth
        parts = f.replace('.pth', '').split('__')
        if len(parts) == 2:
            models.add(parts[0])
            datasets.add(parts[1])
    
    models = sorted(list(models))
    datasets = sorted(list(datasets))
    
    print(f"模型数量: {len(models)}")
    print(f"数据集数量: {len(datasets)}")
    print(f"\n前10个模型:")
    for m in models[:10]:
        print(f"  - {m}")
    print(f"\n所有数据集:")
    for d in datasets:
        print(f"  - {d}")
    
    return models, datasets


def main():
    # 数据目录（服务器上的 SWAB 项目路径）
    data_dir = '/root/mxy/SWAB'
    
    # 如果在本地测试，使用当前目录
    if not os.path.exists(data_dir):
        print(f"警告: 服务器路径不存在，使用当前目录")
        data_dir = '.'
    
    print("=" * 60)
    print("VEGA 数据探索")
    print("=" * 60)
    
    # 1. 列出可用模型
    models, datasets = list_available_models(data_dir)
    
    # 2. 探索 logits 结构
    logits = explore_logits(data_dir, 'RN50_openai', 'cars')
    
    # 3. 探索图像特征
    img_feat = explore_img_feat(data_dir, 'RN50_openai')
    
    # 4. 探索文本特征
    text_feat = explore_text_feat(data_dir, 'RN50_openai')
    
    # 5. 探索类别级别准确率
    acc = explore_class_level_acc(data_dir, 'RN50_openai')
    
    print("\n" + "=" * 60)
    print("探索完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()