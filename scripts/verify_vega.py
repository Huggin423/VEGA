#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VEGA 实现正确性验证脚本

该脚本逐步验证 VEGA 算法的每个步骤，输出详细的中间结果。
"""

import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.baseline.vega import VEGAScorer


# ==================== 配置 ====================
DATA_ROOT = "/root/mxy/SWAB"  # 服务器上的数据路径
# 本地测试时使用符号链接路径
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = str(Path(__file__).parent.parent / "ptm_stats")
    if not os.path.exists(DATA_ROOT):
        # 尝试上级目录
        DATA_ROOT = "E:/VEGA"

# 测试配置
TEST_MODELS = [
    "ViT-L-14_openai",  # 高性能模型
    "ViT-B-32_openai",  # 中等性能模型
    "RN50_openai",      # 较低性能模型
]
TEST_DATASET = "cifar100"  # 使用 cifar100 作为验证数据集


def print_separator(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def load_logits(model_name, dataset_name):
    """加载 logits 文件"""
    logits_dir = os.path.join(DATA_ROOT, "ptm_stats", "logits")
    
    # 尝试不同的文件名格式
    possible_names = [
        f"{model_name}__{dataset_name}.pth",
        f"{model_name}_{dataset_name}.pth",
    ]
    
    for name in possible_names:
        path = os.path.join(logits_dir, name)
        if os.path.exists(path):
            print(f"[加载] {path}")
            data = torch.load(path, map_location='cpu')
            return data
    
    print(f"[错误] 找不到 logits 文件: {model_name} @ {dataset_name}")
    return None


def load_image_features(model_name, dataset_name):
    """加载图像特征"""
    feat_path = os.path.join(DATA_ROOT, "ptm_stats", "stats_on_hist_task", "img_feat", f"{model_name}.pkl")
    
    if not os.path.exists(feat_path):
        print(f"[错误] 找不到图像特征文件: {feat_path}")
        return None
    
    print(f"[加载] {feat_path}")
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
    
    if dataset_name in data:
        return data[dataset_name]
    else:
        print(f"[错误] 图像特征中没有数据集 {dataset_name}")
        print(f"  可用数据集: {list(data.keys())[:5]}...")
        return None


def load_text_features(model_name, dataset_name):
    """加载文本特征"""
    feat_path = os.path.join(DATA_ROOT, "ptm_stats", "class_text_feat", f"{model_name}.pkl")
    
    if not os.path.exists(feat_path):
        print(f"[错误] 找不到文本特征文件: {feat_path}")
        return None
    
    print(f"[加载] {feat_path}")
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
    
    if dataset_name in data:
        return data[dataset_name]
    else:
        print(f"[错误] 文本特征中没有数据集 {dataset_name}")
        return None


def verify_data_loading(model_name, dataset_name):
    """步骤 1: 验证数据加载"""
    print_separator(f"步骤 1: 数据加载验证 - {model_name}")
    
    # 加载 logits
    logits_data = load_logits(model_name, dataset_name)
    if logits_data is None:
        return None, None, None, None
    
    logits = logits_data['logits']
    labels = logits_data.get('labels', None)
    acc1 = logits_data.get('acc1', None)
    
    print(f"\n[Logits 信息]")
    print(f"  Shape: {logits.shape}")
    print(f"  Dtype: {logits.dtype}")
    print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  准确率: {acc1:.4f}" if acc1 else "  准确率: 未记录")
    
    # 加载图像特征
    img_features = load_image_features(model_name, dataset_name)
    if img_features is None:
        return None, None, None, None
    
    # 图像特征通常是 dict {class_name: features}
    if isinstance(img_features, dict):
        # 取第一个类别的特征查看维度
        first_key = list(img_features.keys())[0]
        first_feat = img_features[first_key]
        if isinstance(first_feat, torch.Tensor):
            feat_dim = first_feat.shape[-1] if len(first_feat.shape) > 1 else first_feat.shape[0]
            num_classes = len(img_features)
            print(f"\n[图像特征信息]")
            print(f"  类别数: {num_classes}")
            print(f"  特征维度: {feat_dim}")
            print(f"  第一个类别: {first_key}, 特征 shape: {first_feat.shape}")
        else:
            print(f"\n[图像特征信息] 未知的特征格式: {type(first_feat)}")
    else:
        print(f"\n[图像特征信息] 未知的数据结构: {type(img_features)}")
    
    # 加载文本特征
    text_features = load_text_features(model_name, dataset_name)
    if text_features is None:
        return None, None, None, None
    
    if isinstance(text_features, dict):
        num_text_classes = len(text_features)
        first_key = list(text_features.keys())[0]
        first_text = text_features[first_key]
        if isinstance(first_text, torch.Tensor):
            text_dim = first_text.shape[-1] if len(first_text.shape) > 1 else first_text.shape[0]
            print(f"\n[文本特征信息]")
            print(f"  类别数: {num_text_classes}")
            print(f"  特征维度: {text_dim}")
            print(f"  第一个类别: {first_key}")
        else:
            print(f"\n[文本特征信息] 未知的特征格式: {type(first_text)}")
    
    return logits_data, img_features, text_features, acc1


def verify_vega_step_by_step(model_name, dataset_name, logits_data, img_features, text_features):
    """逐步验证 VEGA 计算"""
    print_separator(f"步骤 2-6: VEGA 计算验证 - {model_name}")
    
    logits = logits_data['logits']
    
    # 准备数据
    # 图像特征需要转换为 tensor [N, D]
    if isinstance(img_features, dict):
        # 合并所有类别的特征
        all_features = []
        all_labels = []
        for class_name, feats in img_features.items():
            if isinstance(feats, torch.Tensor):
                if len(feats.shape) == 1:
                    all_features.append(feats.unsqueeze(0))
                else:
                    all_features.append(feats)
        if all_features:
            features = torch.cat(all_features, dim=0)
        else:
            print("[错误] 无法合并图像特征")
            return None
    else:
        features = img_features
    
    # 文本特征需要转换为 tensor [K, D]
    if isinstance(text_features, dict):
        text_list = []
        for class_name, feat in text_features.items():
            if isinstance(feat, torch.Tensor):
                if len(feat.shape) == 1:
                    text_list.append(feat.unsqueeze(0))
                else:
                    text_list.append(feat)
        if text_list:
            text_embeddings = torch.cat(text_list, dim=0)
        else:
            print("[错误] 无法合并文本特征")
            return None
    else:
        text_embeddings = text_features
    
    print(f"\n[数据准备完成]")
    print(f"  图像特征: {features.shape}")
    print(f"  文本特征: {text_embeddings.shape}")
    print(f"  Logits: {logits.shape}")
    
    # 创建 VEGA scorer
    vega = VEGAScorer(
        temperature=0.05,
        use_pca=True,
        pca_dim=256,
        pca_whiten=True
    )
    
    # 计算 VEGA 分数，返回详细信息
    try:
        result = vega.compute_score(
            features=features,
            text_embeddings=text_embeddings,
            logits=logits,
            return_details=True
        )
        
        if isinstance(result, dict):
            score = result.get('score', result.get('vega_score', 0))
            node_sim = result.get('node_similarity', 0)
            edge_sim = result.get('edge_similarity', 0)
            pca_dim = result.get('pca_dim', 'N/A')
            
            print(f"\n[VEGA 计算结果]")
            print(f"  PCA 维度: {pca_dim}")
            print(f"  节点相似度 (s_n): {node_sim:.4f}")
            print(f"  边相似度 (s_e): {edge_sim:.4f}")
            print(f"  VEGA 总分 (s_n + s_e): {score:.4f}")
            print(f"  分数范围检查: 0 <= {score:.4f} <= 2 ? {0 <= score <= 2}")
            
            return {
                'score': score,
                'node_similarity': node_sim,
                'edge_similarity': edge_sim,
                'pca_dim': pca_dim
            }
        else:
            print(f"\n[VEGA 计算结果] 分数: {result:.4f}")
            return {'score': result}
            
    except Exception as e:
        print(f"\n[错误] VEGA 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_verification():
    """运行完整验证"""
    print_separator("VEGA 实现正确性验证")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"测试数据集: {TEST_DATASET}")
    print(f"测试模型: {TEST_MODELS}")
    
    results = {}
    
    for model_name in TEST_MODELS:
        print(f"\n{'='*60}")
        print(f" 验证模型: {model_name}")
        print(f"{'='*60}")
        
        # 步骤 1: 数据加载
        logits_data, img_features, text_features, acc1 = verify_data_loading(model_name, TEST_DATASET)
        
        if logits_data is None:
            print(f"[跳过] {model_name} 数据加载失败")
            continue
        
        # 步骤 2-6: VEGA 计算
        vega_result = verify_vega_step_by_step(model_name, TEST_DATASET, logits_data, img_features, text_features)
        
        if vega_result:
            results[model_name] = {
                'accuracy': acc1,
                'vega_score': vega_result.get('score', 0),
                'node_similarity': vega_result.get('node_similarity', 0),
                'edge_similarity': vega_result.get('edge_similarity', 0)
            }
    
    # 汇总结果
    print_separator("验证结果汇总")
    
    if results:
        print(f"\n{'模型':<25} {'准确率':<12} {'VEGA分数':<12} {'s_n':<12} {'s_e':<12}")
        print("-" * 75)
        
        for model_name, res in results.items():
            print(f"{model_name:<25} {res['accuracy']:<12.4f} {res['vega_score']:<12.4f} "
                  f"{res['node_similarity']:<12.4f} {res['edge_similarity']:<12.4f}")
        
        # 检查排序一致性
        print("\n[排序分析]")
        sorted_by_acc = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        sorted_by_vega = sorted(results.items(), key=lambda x: x[1]['vega_score'], reverse=True)
        
        print(f"  按准确率排序: {[x[0] for x in sorted_by_acc]}")
        print(f"  按 VEGA 排序: {[x[0] for x in sorted_by_vega]}")
        
        # 计算简单的排序匹配
        acc_order = [x[0] for x in sorted_by_acc]
        vega_order = [x[0] for x in sorted_by_vega]
        
        if acc_order == vega_order:
            print("  ✓ 排序完全一致！")
        else:
            # 计算 Kendall's tau
            from scipy.stats import kendalltau
            acc_ranks = [acc_order.index(m) for m in results.keys()]
            vega_ranks = [vega_order.index(m) for m in results.keys()]
            tau, p_value = kendalltau(acc_ranks, vega_ranks)
            print(f"  Kendall's τ: {tau:.4f} (p={p_value:.4f})")
    else:
        print("没有成功验证的模型")


if __name__ == "__main__":
    run_verification()