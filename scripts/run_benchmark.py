#!/usr/bin/env python
"""
VEGA vs LogME 基准测试脚本
对比两种方法在模型选择任务上的表现

运行环境：实验室服务器 /root/mxy/VEGA（符号链接到 SWAB 数据）
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from methods.baseline.vega import VEGA
from methods.baseline.logme import LogME


def load_logits(data_dir: str, model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """加载模型的 logits"""
    logits_path = os.path.join(data_dir, 'ptm_stats/logits', f'{model_name}__{dataset_name}.pth')
    
    if not os.path.exists(logits_path):
        return None
    
    logits = torch.load(logits_path, map_location='cpu')
    
    # 转换为 numpy 数组
    if isinstance(logits, torch.Tensor):
        return logits.numpy()
    elif isinstance(logits, np.ndarray):
        return logits
    elif isinstance(logits, dict):
        # 可能包含 'logits' 键
        if 'logits' in logits:
            return logits['logits'].numpy() if isinstance(logits['logits'], torch.Tensor) else logits['logits']
        # 或者直接取第一个 tensor 值
        for v in logits.values():
            if isinstance(v, torch.Tensor):
                return v.numpy()
    elif isinstance(logits, (list, tuple)):
        # 可能是 [features, logits] 或其他格式
        for item in logits:
            if hasattr(item, 'shape') and len(item.shape) == 2:
                return item.numpy() if isinstance(item, torch.Tensor) else item
    
    return None


def load_image_features(data_dir: str, model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """加载模型的图像特征"""
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', f'{model_name}.pkl')
    
    if not os.path.exists(feat_path):
        return None
    
    with open(feat_path, 'rb') as f:
        img_feats = pickle.load(f)
    
    if isinstance(img_feats, dict) and dataset_name in img_feats:
        feat = img_feats[dataset_name]
        if isinstance(feat, torch.Tensor):
            return feat.numpy()
        elif isinstance(feat, np.ndarray):
            return feat
    
    return None


def load_text_features(data_dir: str, model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """加载模型的文本特征（类别嵌入）"""
    # 尝试 caption_text_feat
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', f'{model_name}.pkl')
    
    if os.path.exists(feat_path):
        with open(feat_path, 'rb') as f:
            text_feats = pickle.load(f)
        
        if isinstance(text_feats, dict) and dataset_name in text_feats:
            feat = text_feats[dataset_name]
            if isinstance(feat, torch.Tensor):
                return feat.numpy()
            elif isinstance(feat, np.ndarray):
                return feat
    
    # 尝试 syn_text_feat
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/syn_text_feat', f'{model_name}.pkl')
    
    if os.path.exists(feat_path):
        with open(feat_path, 'rb') as f:
            text_feats = pickle.load(f)
        
        if isinstance(text_feats, dict) and dataset_name in text_feats:
            feat = text_feats[dataset_name]
            if isinstance(feat, torch.Tensor):
                return feat.numpy()
            elif isinstance(feat, np.ndarray):
                return feat
    
    return None


def load_ground_truth_accuracy(data_dir: str, model_name: str, dataset_name: str) -> Optional[float]:
    """加载模型在数据集上的真实准确率"""
    # 从 class_level_acc 计算平均准确率
    acc_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/class_level_acc', f'{model_name}.pkl')
    
    if not os.path.exists(acc_path):
        return None
    
    with open(acc_path, 'rb') as f:
        acc_data = pickle.load(f)
    
    if isinstance(acc_data, dict) and dataset_name in acc_data:
        class_acc = acc_data[dataset_name]
        if isinstance(class_acc, dict):
            return np.mean(list(class_acc.values()))
        elif isinstance(class_acc, (list, np.ndarray)):
            return np.mean(class_acc)
    
    return None


def compute_vega_score(logits: np.ndarray, text_features: np.ndarray) -> float:
    """计算 VEGA 分数"""
    if logits is None or text_features is None:
        return None
    
    try:
        vega = VEGA()
        score = vega.compute_score(logits, text_features)
        return score
    except Exception as e:
        print(f"VEGA 计算错误: {e}")
        return None


def compute_logme_score(features: np.ndarray, pseudo_labels: np.ndarray) -> float:
    """计算 LogME 分数"""
    if features is None or pseudo_labels is None:
        return None
    
    try:
        logme = LogME()
        score = logme.logme(features, pseudo_labels)
        return score
    except Exception as e:
        print(f"LogME 计算错误: {e}")
        return None


def compute_metrics(predicted_scores: Dict[str, float], 
                   ground_truth: Dict[str, float]) -> Dict[str, float]:
    """计算评估指标"""
    # 过滤有效数据
    common_models = set(predicted_scores.keys()) & set(ground_truth.keys())
    common_models = [m for m in common_models if predicted_scores[m] is not None and ground_truth[m] is not None]
    
    if len(common_models) < 3:
        return {'error': 'Insufficient data for metrics'}
    
    pred = np.array([predicted_scores[m] for m in common_models])
    gt = np.array([ground_truth[m] for m in common_models])
    
    # Kendall's tau
    tau, p_value = stats.kendalltau(pred, gt)
    
    # Spearman correlation
    spearman, sp_pvalue = stats.spearmanr(pred, gt)
    
    # Pearson correlation
    pearson, pp_pvalue = stats.pearsonr(pred, gt)
    
    # Top-5 Recall
    top5_gt = set(np.argsort(gt)[-5:])
    top5_pred = set(np.argsort(pred)[-5:])
    top5_recall = len(top5_gt & top5_pred) / 5
    
    # Top-1 Accuracy (预测最佳的模型的实际准确率)
    top1_pred_model = common_models[np.argmax(pred)]
    top1_accuracy = ground_truth[top1_pred_model]
    
    # Oracle (最佳模型的准确率)
    oracle_accuracy = np.max(gt)
    
    return {
        'kendall_tau': tau,
        'kendall_p': p_value,
        'spearman': spearman,
        'spearman_p': sp_pvalue,
        'pearson': pearson,
        'pearson_p': pp_pvalue,
        'top5_recall': top5_recall,
        'top1_accuracy': top1_accuracy,
        'oracle_accuracy': oracle_accuracy,
        'num_models': len(common_models)
    }


def run_single_dataset_benchmark(data_dir: str, dataset_name: str, 
                                  model_list: List[str]) -> Dict:
    """在单个数据集上运行基准测试"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    
    vega_scores = {}
    logme_scores = {}
    ground_truth = {}
    
    for model_name in model_list:
        # 加载数据
        logits = load_logits(data_dir, model_name, dataset_name)
        img_feat = load_image_features(data_dir, model_name, dataset_name)
        text_feat = load_text_features(data_dir, model_name, dataset_name)
        gt_acc = load_ground_truth_accuracy(data_dir, model_name, dataset_name)
        
        if gt_acc is not None:
            ground_truth[model_name] = gt_acc
        
        # 计算 VEGA 分数
        if logits is not None and text_feat is not None:
            score = compute_vega_score(logits, text_feat)
            if score is not None:
                vega_scores[model_name] = score
        
        # 计算 LogME 分数
        if img_feat is not None and logits is not None:
            # 使用 logits 的 argmax 作为伪标签
            pseudo_labels = np.argmax(logits, axis=1)
            score = compute_logme_score(img_feat, pseudo_labels)
            if score is not None:
                logme_scores[model_name] = score
        
        # 打印进度
        status = []
        if logits is not None:
            status.append("logits✓")
        if img_feat is not None:
            status.append("img_feat✓")
        if text_feat is not None:
            status.append("text_feat✓")
        if gt_acc is not None:
            status.append(f"acc={gt_acc:.3f}")
        
        print(f"  {model_name}: {', '.join(status) if status else 'N/A'}")
    
    # 计算评估指标
    print(f"\n计算评估指标...")
    
    vega_metrics = compute_metrics(vega_scores, ground_truth)
    logme_metrics = compute_metrics(logme_scores, ground_truth)
    
    return {
        'dataset': dataset_name,
        'vega_scores': vega_scores,
        'logme_scores': logme_scores,
        'ground_truth': ground_truth,
        'vega_metrics': vega_metrics,
        'logme_metrics': logme_metrics
    }


def print_results(results: Dict):
    """打印结果"""
    print(f"\n{'='*60}")
    print(f"数据集: {results['dataset']}")
    print(f"{'='*60}")
    
    vega_m = results['vega_metrics']
    logme_m = results['logme_metrics']
    
    if 'error' in vega_m:
        print(f"VEGA: {vega_m['error']}")
    else:
        print(f"\nVEGA:")
        print(f"  Kendall τ: {vega_m['kendall_tau']:.4f} (p={vega_m['kendall_p']:.4f})")
        print(f"  Spearman: {vega_m['spearman']:.4f}")
        print(f"  Pearson: {vega_m['pearson']:.4f}")
        print(f"  Top-5 Recall: {vega_m['top5_recall']:.2f}")
        print(f"  Top-1 Accuracy: {vega_m['top1_accuracy']:.4f} (Oracle: {vega_m['oracle_accuracy']:.4f})")
    
    if 'error' in logme_m:
        print(f"\nLogME: {logme_m['error']}")
    else:
        print(f"\nLogME:")
        print(f"  Kendall τ: {logme_m['kendall_tau']:.4f} (p={logme_m['kendall_p']:.4f})")
        print(f"  Spearman: {logme_m['spearman']:.4f}")
        print(f"  Pearson: {logme_m['pearson']:.4f}")
        print(f"  Top-5 Recall: {logme_m['top5_recall']:.2f}")
        print(f"  Top-1 Accuracy: {logme_m['top1_accuracy']:.4f} (Oracle: {logme_m['oracle_accuracy']:.4f})")


def main():
    # 数据目录
    data_dir = '/root/mxy/SWAB'
    if not os.path.exists(data_dir):
        # 尝试符号链接路径
        data_dir = '/root/mxy/VEGA/ptm_stats'
        if not os.path.exists(data_dir):
            print("错误: 数据目录不存在")
            print("请在服务器上运行此脚本")
            return
    
    # 定义测试模型（选择代表性模型）
    test_models = [
        'RN50_openai',
        'RN101_openai',
        'ViT-B-32_openai',
        'ViT-B-16_openai',
        'ViT-L-14_openai',
        'ViT-B-32_laion2b_s34b_b79k',
        'ViT-B-16_laion400m_e32',
        'BLIP_retrieval_base_coco',
        'BEIT3_retrieval_base_coco',
    ]
    
    # 测试数据集
    test_datasets = [
        'cars',
        'cifar100',
        'flowers',
        'pets',
        'dtd',
    ]
    
    print("=" * 60)
    print("VEGA vs LogME 基准测试")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"测试模型数: {len(test_models)}")
    print(f"测试数据集: {test_datasets}")
    
    # 运行基准测试
    all_results = []
    
    for dataset in test_datasets:
        result = run_single_dataset_benchmark(data_dir, dataset, test_models)
        all_results.append(result)
        print_results(result)
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    
    avg_vega_tau = np.mean([r['vega_metrics'].get('kendall_tau', np.nan) 
                           for r in all_results if 'error' not in r['vega_metrics']])
    avg_logme_tau = np.mean([r['logme_metrics'].get('kendall_tau', np.nan) 
                            for r in all_results if 'error' not in r['logme_metrics']])
    
    avg_vega_top5 = np.mean([r['vega_metrics'].get('top5_recall', np.nan) 
                            for r in all_results if 'error' not in r['vega_metrics']])
    avg_logme_top5 = np.mean([r['logme_metrics'].get('top5_recall', np.nan) 
                             for r in all_results if 'error' not in r['logme_metrics']])
    
    print(f"\n平均 Kendall τ:")
    print(f"  VEGA: {avg_vega_tau:.4f}")
    print(f"  LogME: {avg_logme_tau:.4f}")
    
    print(f"\n平均 Top-5 Recall:")
    print(f"  VEGA: {avg_vega_top5:.2f}")
    print(f"  LogME: {avg_logme_top5:.2f}")


if __name__ == '__main__':
    main()