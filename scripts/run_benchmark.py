#!/usr/bin/env python
"""
VEGA vs LogME 基准测试脚本（改进版）
对比两种方法在模型选择任务上的表现

运行环境：实验室服务器 /root/mxy/VEGA（符号链接到 SWAB 数据）
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from methods.baseline.vega import VEGA
from methods.baseline.logme import LogME


# ============================================================================
# 数据加载函数
# ============================================================================

def load_logits_data(data_dir: str, model_name: str, dataset_name: str) -> Optional[Dict]:
    """
    加载模型的 logits 数据
    
    返回:
        dict: {
            'logits': np.ndarray [N, K],
            'labels': np.ndarray [N],
            'acc1': float
        }
    """
    logits_path = os.path.join(data_dir, 'ptm_stats/logits', f'{model_name}__{dataset_name}.pth')
    
    if not os.path.exists(logits_path):
        return None
    
    data = torch.load(logits_path, map_location='cpu')
    
    result = {}
    
    if isinstance(data, dict):
        if 'logits' in data:
            logits = data['logits']
            result['logits'] = logits.numpy() if isinstance(logits, torch.Tensor) else logits
        if 'labels' in data:
            labels = data['labels']
            result['labels'] = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        if 'acc1' in data:
            result['acc1'] = data['acc1']
    elif isinstance(data, torch.Tensor):
        result['logits'] = data.numpy()
    elif isinstance(data, np.ndarray):
        result['logits'] = data
    
    return result if 'logits' in result else None


def load_image_features(data_dir: str, model_name: str, dataset_name: str, verbose: bool = False) -> Optional[np.ndarray]:
    """
    加载模型的图像特征
    
    数据结构: pickle 文件包含多个数据集的特征
    每个数据集的特征是 dict: {class_name: feature_array}
    
    返回:
        np.ndarray: [N, D] 图像特征矩阵
    """
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', f'{model_name}.pkl')
    
    if not os.path.exists(feat_path):
        if verbose:
            print(f"    [!] 图像特征文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        img_feats = pickle.load(f)
    
    if not isinstance(img_feats, dict) or dataset_name not in img_feats:
        if verbose:
            print(f"    [!] 数据集 {dataset_name} 不在图像特征中")
        return None
    
    dataset_feats = img_feats[dataset_name]
    
    # 数据集特征是 {class_name: feature_array} 格式
    if isinstance(dataset_feats, dict):
        all_features = []
        for class_name, feat in dataset_feats.items():
            if isinstance(feat, torch.Tensor):
                feat = feat.numpy()
            if isinstance(feat, np.ndarray):
                if len(feat.shape) == 1:
                    # 单个特征向量
                    all_features.append(feat)
                elif len(feat.shape) == 2:
                    # 多个特征向量 [n, D]
                    all_features.extend(feat)
                elif len(feat.shape) == 3:
                    # 可能是 [n, D, 1] 或类似
                    all_features.extend(feat.reshape(-1, feat.shape[-1]))
        if all_features:
            return np.array(all_features)
    elif isinstance(dataset_feats, (torch.Tensor, np.ndarray)):
        feat = dataset_feats.numpy() if isinstance(dataset_feats, torch.Tensor) else dataset_feats
        if len(feat.shape) == 2:
            return feat
        elif len(feat.shape) == 3:
            return feat.reshape(-1, feat.shape[-1])
    
    return None


def load_text_features(data_dir: str, model_name: str, dataset_name: str, verbose: bool = False) -> Optional[np.ndarray]:
    """
    加载模型的文本特征（类别嵌入）
    
    数据结构: pickle 文件包含多个数据集
    每个数据集是 {class_name: text_embedding}
    
    返回:
        np.ndarray: [K, D] 文本嵌入矩阵
    """
    # 尝试 caption_text_feat
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', f'{model_name}.pkl')
    
    if not os.path.exists(feat_path):
        # 尝试 syn_text_feat
        feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/syn_text_feat', f'{model_name}.pkl')
    
    if not os.path.exists(feat_path):
        if verbose:
            print(f"    [!] 文本特征文件不存在")
        return None
    
    with open(feat_path, 'rb') as f:
        text_feats = pickle.load(f)
    
    if not isinstance(text_feats, dict) or dataset_name not in text_feats:
        if verbose:
            print(f"    [!] 数据集 {dataset_name} 不在文本特征中")
        return None
    
    dataset_feats = text_feats[dataset_name]
    
    # 数据集特征是 {class_name: text_embedding} 格式
    if isinstance(dataset_feats, dict):
        embeddings = []
        for class_name, emb in dataset_feats.items():
            if isinstance(emb, torch.Tensor):
                emb = emb.numpy()
            if isinstance(emb, np.ndarray):
                if len(emb.shape) == 1:
                    embeddings.append(emb)
                elif len(emb.shape) == 2 and emb.shape[0] == 1:
                    embeddings.append(emb.flatten())
        if embeddings:
            return np.array(embeddings)
    elif isinstance(dataset_feats, (torch.Tensor, np.ndarray)):
        emb = dataset_feats.numpy() if isinstance(dataset_feats, torch.Tensor) else dataset_feats
        if len(emb.shape) == 2:
            return emb
    
    return None


def load_ground_truth_accuracy(data_dir: str, model_name: str, dataset_name: str) -> Optional[float]:
    """
    加载模型在数据集上的真实准确率
    """
    # 首先尝试从 logits 文件获取
    logits_data = load_logits_data(data_dir, model_name, dataset_name)
    if logits_data and 'acc1' in logits_data:
        return logits_data['acc1']
    
    # 从 class_level_acc 计算
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


# ============================================================================
# 分数计算函数
# ============================================================================

def compute_vega_score_detailed(
    img_features: np.ndarray, 
    text_features: np.ndarray, 
    logits: np.ndarray,
    model_name: str = ""
) -> Tuple[float, Dict]:
    """
    计算 VEGA 分数（带详细输出）
    
    返回:
        (score, details)
    """
    print(f"\n    计算 VEGA 分数:")
    print(f"      - 图像特征: {img_features.shape}")
    print(f"      - 文本特征: {text_features.shape}")
    print(f"      - Logits: {logits.shape}")
    
    try:
        vega = VEGA(temperature=0.05)
        result = vega.compute_score(
            features=img_features,
            text_embeddings=text_features,
            logits=logits,
            return_details=True
        )
        
        print(f"      - 节点相似度 (s_n): {result['node_similarity']:.4f}")
        print(f"      - 边相似度 (s_e): {result['edge_similarity']:.4f}")
        print(f"      - 有效类别数: {result['valid_classes']}")
        print(f"      - VEGA 总分: {result['score']:.4f}")
        
        return result['score'], result
        
    except Exception as e:
        print(f"      [!] VEGA 计算错误: {e}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


def compute_logme_score_detailed(
    features: np.ndarray, 
    pseudo_labels: np.ndarray,
    model_name: str = ""
) -> Tuple[float, Dict]:
    """
    计算 LogME 分数（带详细输出）
    """
    print(f"\n    计算 LogME 分数:")
    print(f"      - 特征: {features.shape}")
    print(f"      - 伪标签: {pseudo_labels.shape}, 唯一类别数: {len(np.unique(pseudo_labels))}")
    
    try:
        logme = LogME()
        score = logme.logme(features, pseudo_labels)
        
        print(f"      - LogME 分数: {score:.4f}")
        
        return score, {'score': score}
        
    except Exception as e:
        print(f"      [!] LogME 计算错误: {e}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


# ============================================================================
# 评估指标计算
# ============================================================================

def compute_metrics(predicted_scores: Dict[str, float], 
                   ground_truth: Dict[str, float],
                   verbose: bool = True) -> Dict[str, float]:
    """
    计算评估指标
    
    指标说明:
    - Kendall's τ: 排序相关性，范围 [-1, 1]，1 表示完全一致
    - Spearman: 斯皮尔曼相关系数，范围 [-1, 1]
    - Pearson: 皮尔逊相关系数，范围 [-1, 1]
    - Top-5 Recall: 预测的前5个模型中有多少在真实前5中
    - Top-1 Accuracy: 预测最佳模型的实际准确率
    - Oracle: 所有模型中最佳准确率
    """
    # 过滤有效数据
    common_models = set(predicted_scores.keys()) & set(ground_truth.keys())
    common_models = [m for m in common_models 
                    if predicted_scores[m] is not None and ground_truth[m] is not None]
    
    if len(common_models) < 3:
        if verbose:
            print(f"  [!] 有效模型数不足 ({len(common_models)} < 3)")
        return {'error': f'Insufficient data: only {len(common_models)} common models'}
    
    pred = np.array([predicted_scores[m] for m in common_models])
    gt = np.array([ground_truth[m] for m in common_models])
    
    # 打印排序详情
    if verbose:
        print(f"\n  模型排序详情 (共 {len(common_models)} 个模型):")
        
        # 按预测分数排序
        pred_order = np.argsort(pred)[::-1]  # 降序
        print(f"\n  按 VEGA 分数排序:")
        for rank, idx in enumerate(pred_order):
            model = common_models[idx]
            print(f"    {rank+1}. {model}: pred={pred[idx]:.4f}, gt_acc={gt[idx]:.4f}")
        
        # 按真实准确率排序
        gt_order = np.argsort(gt)[::-1]
        print(f"\n  按真实准确率排序:")
        for rank, idx in enumerate(gt_order):
            model = common_models[idx]
            print(f"    {rank+1}. {model}: gt_acc={gt[idx]:.4f}, pred={pred[idx]:.4f}")
    
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
    
    # Top-1 Accuracy
    top1_pred_idx = np.argmax(pred)
    top1_pred_model = common_models[top1_pred_idx]
    top1_accuracy = gt[top1_pred_idx]
    
    # Oracle
    oracle_idx = np.argmax(gt)
    oracle_accuracy = gt[oracle_idx]
    
    return {
        'kendall_tau': tau,
        'kendall_p': p_value,
        'spearman': spearman,
        'spearman_p': sp_pvalue,
        'pearson': pearson,
        'pearson_p': pp_pvalue,
        'top5_recall': top5_recall,
        'top1_accuracy': top1_accuracy,
        'top1_model': top1_pred_model,
        'oracle_accuracy': oracle_accuracy,
        'oracle_model': common_models[oracle_idx],
        'num_models': len(common_models)
    }


# ============================================================================
# 主基准测试函数
# ============================================================================

def run_single_dataset_benchmark(data_dir: str, dataset_name: str, 
                                  model_list: List[str],
                                  verbose: bool = True) -> Dict:
    """
    在单个数据集上运行基准测试
    """
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")
    
    vega_scores = {}
    logme_scores = {}
    ground_truth = {}
    
    # 存储详细数据用于调试
    debug_data = {}
    
    for model_name in model_list:
        print(f"\n  处理模型: {model_name}")
        
        # 1. 加载 logits
        logits_data = load_logits_data(data_dir, model_name, dataset_name)
        if logits_data is None:
            print(f"    [!] 无法加载 logits")
            continue
        
        logits = logits_data['logits']
        print(f"    Logits: {logits.shape}")
        
        # 2. 加载图像特征
        img_feat = load_image_features(data_dir, model_name, dataset_name, verbose=True)
        if img_feat is None:
            print(f"    [!] 无法加载图像特征")
            continue
        print(f"    图像特征: {img_feat.shape}")
        
        # 3. 加载文本特征
        text_feat = load_text_features(data_dir, model_name, dataset_name, verbose=True)
        if text_feat is None:
            print(f"    [!] 无法加载文本特征")
            continue
        print(f"    文本特征: {text_feat.shape}")
        
        # 4. 加载真实准确率
        gt_acc = load_ground_truth_accuracy(data_dir, model_name, dataset_name)
        if gt_acc is None:
            print(f"    [!] 无法加载准确率")
            continue
        ground_truth[model_name] = gt_acc
        print(f"    真实准确率: {gt_acc:.4f}")
        
        # 检查维度匹配
        n_samples, n_classes = logits.shape
        n_text_classes = text_feat.shape[0]
        
        if n_text_classes != n_classes:
            print(f"    [!] 类别数不匹配: logits={n_classes}, text_feat={n_text_classes}")
            # 尝试使用较小的类别数
            min_classes = min(n_classes, n_text_classes)
            logits = logits[:, :min_classes]
            text_feat = text_feat[:min_classes]
            print(f"    使用前 {min_classes} 个类别")
        
        # 存储 debug 数据
        debug_data[model_name] = {
            'logits_shape': logits.shape,
            'img_feat_shape': img_feat.shape,
            'text_feat_shape': text_feat.shape,
            'gt_acc': gt_acc
        }
        
        # 5. 计算 VEGA 分数
        # VEGA 需要: visual_features [N, D], text_embeddings [K, D], logits [N, K]
        if img_feat.shape[0] == logits.shape[0]:
            vega_score, vega_details = compute_vega_score_detailed(
                img_feat, text_feat, logits, model_name
            )
            if vega_score is not None:
                vega_scores[model_name] = vega_score
        else:
            print(f"    [!] 样本数不匹配: img_feat={img_feat.shape[0]}, logits={logits.shape[0]}")
            # 尝试使用较少的样本
            min_samples = min(img_feat.shape[0], logits.shape[0])
            vega_score, vega_details = compute_vega_score_detailed(
                img_feat[:min_samples], text_feat, logits[:min_samples], model_name
            )
            if vega_score is not None:
                vega_scores[model_name] = vega_score
        
        # 6. 计算 LogME 分数
        pseudo_labels = np.argmax(logits, axis=1)
        
        if img_feat.shape[0] == len(pseudo_labels):
            logme_score, logme_details = compute_logme_score_detailed(
                img_feat, pseudo_labels, model_name
            )
            if logme_score is not None:
                logme_scores[model_name] = logme_score
        else:
            min_samples = min(img_feat.shape[0], len(pseudo_labels))
            logme_score, logme_details = compute_logme_score_detailed(
                img_feat[:min_samples], pseudo_labels[:min_samples], model_name
            )
            if logme_score is not None:
                logme_scores[model_name] = logme_score
    
    # 计算评估指标
    print(f"\n{'='*70}")
    print(f"评估结果: {dataset_name}")
    print(f"{'='*70}")
    
    vega_metrics = compute_metrics(vega_scores, ground_truth, verbose=verbose)
    logme_metrics = compute_metrics(logme_scores, ground_truth, verbose=verbose)
    
    return {
        'dataset': dataset_name,
        'vega_scores': vega_scores,
        'logme_scores': logme_scores,
        'ground_truth': ground_truth,
        'vega_metrics': vega_metrics,
        'logme_metrics': logme_metrics,
        'debug_data': debug_data
    }


def print_final_results(all_results: List[Dict]):
    """打印最终汇总结果"""
    print(f"\n{'='*70}")
    print("汇总结果")
    print(f"{'='*70}")
    
    # 收集有效的结果
    valid_vega = [r for r in all_results if 'error' not in r['vega_metrics']]
    valid_logme = [r for r in all_results if 'error' not in r['logme_metrics']]
    
    # VEGA 统计
    if valid_vega:
        avg_vega_tau = np.mean([r['vega_metrics']['kendall_tau'] for r in valid_vega])
        avg_vega_spearman = np.mean([r['vega_metrics']['spearman'] for r in valid_vega])
        avg_vega_top5 = np.mean([r['vega_metrics']['top5_recall'] for r in valid_vega])
        
        print(f"\nVEGA (在 {len(valid_vega)} 个数据集上):")
        print(f"  平均 Kendall τ: {avg_vega_tau:.4f}")
        print(f"  平均 Spearman: {avg_vega_spearman:.4f}")
        print(f"  平均 Top-5 Recall: {avg_vega_top5:.2f}")
    else:
        print(f"\nVEGA: 无有效结果")
    
    # LogME 统计
    if valid_logme:
        avg_logme_tau = np.mean([r['logme_metrics']['kendall_tau'] for r in valid_logme])
        avg_logme_spearman = np.mean([r['logme_metrics']['spearman'] for r in valid_logme])
        avg_logme_top5 = np.mean([r['logme_metrics']['top5_recall'] for r in valid_logme])
        
        print(f"\nLogME (在 {len(valid_logme)} 个数据集上):")
        print(f"  平均 Kendall τ: {avg_logme_tau:.4f}")
        print(f"  平均 Spearman: {avg_logme_spearman:.4f}")
        print(f"  平均 Top-5 Recall: {avg_logme_top5:.2f}")
    else:
        print(f"\nLogME: 无有效结果")
    
    # 每个数据集的详细结果
    print(f"\n各数据集详细结果:")
    print(f"{'数据集':<25} {'VEGA τ':>10} {'LogME τ':>10} {'VEGA Top5':>10} {'LogME Top5':>10}")
    print("-" * 70)
    
    for r in all_results:
        dataset = r['dataset']
        vega_tau = r['vega_metrics'].get('kendall_tau', float('nan'))
        logme_tau = r['logme_metrics'].get('kendall_tau', float('nan'))
        vega_top5 = r['vega_metrics'].get('top5_recall', float('nan'))
        logme_top5 = r['logme_metrics'].get('top5_recall', float('nan'))
        
        print(f"{dataset:<25} {vega_tau:>10.4f} {logme_tau:>10.4f} {vega_top5:>10.2f} {logme_top5:>10.2f}")


def main():
    """主函数"""
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
    
    print("=" * 70)
    print("VEGA vs LogME 基准测试（改进版）")
    print("=" * 70)
    print(f"数据目录: {data_dir}")
    print(f"测试模型数: {len(test_models)}")
    print(f"测试数据集: {test_datasets}")
    
    # 运行基准测试
    all_results = []
    
    for dataset in test_datasets:
        result = run_single_dataset_benchmark(data_dir, dataset, test_models, verbose=True)
        all_results.append(result)
    
    # 打印最终结果
    print_final_results(all_results)


if __name__ == '__main__':
    main()