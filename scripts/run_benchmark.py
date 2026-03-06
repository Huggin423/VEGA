#!/usr/bin/env python
"""
VEGA vs LogME 基准测试脚本（改进版）
对比两种方法在模型选择任务上的表现

运行环境：实验室服务器 /root/mxy/VEGA（符号链接到 SWAB 数据）

更新日志:
- 2026-03-06: 使用基础 VEGA（符合论文），添加进度条显示
- 2026-03-06: 修复 LogME 调用方式（使用 LogME_official.fit）
- 2026-03-06: 添加缓存系统，避免重复计算
- 2026-03-06: 添加详细进度日志，显示每个计算步骤
- 2026-03-06: 添加 VEGA 内部进度显示（边相似度计算）
- 2026-03-06: 改进错误处理，捕获并记录失败的模型
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
import time
from datetime import datetime
import json
import hashlib
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# 配置
# ============================================================================

# 缓存目录
CACHE_DIR = project_root / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 是否启用缓存
ENABLE_CACHE = True

# ============================================================================
# 进度显示工具
# ============================================================================

class ProgressBar:
    """简单的进度条显示"""
    
    def __init__(self, total: int, desc: str = "Progress", width: int = 50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, n: int = 1, info: str = ""):
        self.current += n
        elapsed = time.time() - self.start_time
        
        # 计算进度
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # 计算预计剩余时间
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        
        # 构建进度条字符串
        status = f"{self.desc}: |{bar}| {self.current}/{self.total} [{progress*100:.1f}%] {eta_str}"
        if info:
            status += f" | {info}"
        
        # 打印进度条（覆盖上一行）
        print(f"\r{status}", end='', flush=True)
        
        # 完成时换行
        if self.current >= self.total:
            print(f" | 完成: {elapsed:.1f}s")
    
    def close(self):
        if self.current < self.total:
            print()  # 未完成时换行


def print_header(title: str, width: int = 70):
    """打印标题头"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 70):
    """打印子标题"""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


def print_status(msg: str, level: str = "INFO"):
    """打印状态信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def print_detail(msg: str, indent: int = 2):
    """打印详细信息"""
    print(" " * indent + msg)


# ============================================================================
# 缓存系统
# ============================================================================

def get_cache_key(model_name: str, dataset_name: str, method: str) -> str:
    """生成缓存键"""
    key_str = f"{model_name}_{dataset_name}_{method}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(model_name: str, dataset_name: str, method: str) -> Path:
    """获取缓存文件路径"""
    cache_key = get_cache_key(model_name, dataset_name, method)
    return CACHE_DIR / f"{cache_key}.pkl"


def save_cache(model_name: str, dataset_name: str, method: str, data: Dict):
    """保存缓存"""
    if not ENABLE_CACHE:
        return
    
    cache_path = get_cache_path(model_name, dataset_name, method)
    cache_data = {
        'model': model_name,
        'dataset': dataset_name,
        'method': method,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print_detail(f"[缓存] 已保存 {method} 结果", indent=4)


def load_cache(model_name: str, dataset_name: str, method: str) -> Optional[Dict]:
    """加载缓存"""
    if not ENABLE_CACHE:
        return None
    
    cache_path = get_cache_path(model_name, dataset_name, method)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print_detail(f"[缓存] 命中 {method} 结果 (缓存时间: {cache_data['timestamp']})", indent=4)
        return cache_data['data']
    except Exception as e:
        print_detail(f"[缓存] 加载失败: {e}", indent=4)
        return None


def clear_cache():
    """清空所有缓存"""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        print_status("缓存已清空")


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
    """
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', f'{model_name}.pkl')
    
    if not os.path.exists(feat_path):
        if verbose:
            print_detail(f"[!] 图像特征文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        img_feats = pickle.load(f)
    
    if not isinstance(img_feats, dict) or dataset_name not in img_feats:
        if verbose:
            print_detail(f"[!] 数据集 {dataset_name} 不在图像特征中")
        return None
    
    dataset_feats = img_feats[dataset_name]
    
    # 数据集特征是 {class_name: feature_array} 格式
    if isinstance(dataset_feats, dict):
        all_features = []
        for class_name, feat in dataset_feats.items():
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            if isinstance(feat, np.ndarray):
                if len(feat.shape) == 1:
                    all_features.append(feat)
                elif len(feat.shape) == 2:
                    all_features.extend(feat)
                elif len(feat.shape) == 3:
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
    """
    # 尝试多个路径（按优先级）
    search_paths = [
        os.path.join(data_dir, 'ptm_stats/class_text_feat', f'{model_name}.pkl'),
        os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', f'{model_name}.pkl'),
        os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/syn_text_feat', f'{model_name}.pkl'),
    ]
    
    feat_path = None
    for path in search_paths:
        if os.path.exists(path):
            feat_path = path
            break
    
    if feat_path is None:
        if verbose:
            print_detail(f"[!] 无法加载文本特征")
        return None
    
    with open(feat_path, 'rb') as f:
        text_feats = pickle.load(f)
    
    if not isinstance(text_feats, dict) or dataset_name not in text_feats:
        if verbose:
            print_detail(f"[!] 数据集 {dataset_name} 不在文本特征中")
        return None
    
    dataset_feats = text_feats[dataset_name]
    
    # 情况 1: 直接是 [K, D] 数组
    if isinstance(dataset_feats, (torch.Tensor, np.ndarray)):
        emb = dataset_feats.cpu().numpy() if isinstance(dataset_feats, torch.Tensor) else dataset_feats
        if verbose:
            print_detail(f"文本特征 (数组格式): {emb.shape}", indent=4)
        return emb
    
    # 情况 2: {class_name: text_embedding} 格式
    if isinstance(dataset_feats, dict):
        embeddings = []
        for class_name, emb in dataset_feats.items():
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            if isinstance(emb, np.ndarray):
                if len(emb.shape) == 1:
                    embeddings.append(emb)
                elif len(emb.shape) == 2 and emb.shape[0] == 1:
                    embeddings.append(emb.flatten())
        if embeddings:
            result = np.array(embeddings)
            if verbose:
                print_detail(f"文本特征 (字典格式): {result.shape}", indent=4)
            return result
    
    return None


def load_ground_truth_accuracy(data_dir: str, model_name: str, dataset_name: str) -> Optional[float]:
    """加载模型在数据集上的真实准确率"""
    logits_data = load_logits_data(data_dir, model_name, dataset_name)
    if logits_data and 'acc1' in logits_data:
        return logits_data['acc1']
    
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
# VEGA 分数计算（带进度日志）
# ============================================================================

def compute_vega_score_with_progress(
    img_features: np.ndarray, 
    text_features: np.ndarray, 
    logits: np.ndarray,
    model_name: str = ""
) -> Tuple[Optional[float], Dict]:
    """
    计算 VEGA 分数（带详细进度日志）
    
    使用 VEGA 论文的方法：
    1. 构建文本图：节点=类别嵌入，边=余弦相似度
    2. 构建视觉图：节点=类别高斯分布，边=Bhattacharyya距离
    3. 节点相似度：s_n = (1/K) * sum_k sim_k * N_k
    4. 边相似度：s_e = (PearsonCorr + 1) / 2
    5. 最终分数：s = s_n + s_e
    """
    print_detail("计算 VEGA 分数:", indent=4)
    print_detail(f"- 图像特征: {img_features.shape}", indent=6)
    print_detail(f"- 文本特征: {text_features.shape}", indent=6)
    print_detail(f"- Logits: {logits.shape}", indent=6)
    
    n_samples, n_classes = logits.shape
    feature_dim = img_features.shape[1]
    
    start_time = time.time()
    
    try:
        # Step 1: 生成伪标签
        print_detail("[1/6] 生成伪标签...", indent=4)
        pseudo_labels = np.argmax(logits, axis=1)
        
        # Step 2: 构建文本图
        print_detail("[2/6] 构建文本图...", indent=4)
        text_embeddings = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
        
        # 文本边矩阵（余弦相似度）
        text_edge_matrix = text_embeddings @ text_embeddings.T
        print_detail(f"  文本边矩阵: {text_edge_matrix.shape}", indent=6)
        
        # Step 3: 构建视觉图节点（每个类别的高斯分布）
        print_detail("[3/6] 构建视觉图节点（类别分布）...", indent=4)
        
        class_means = []
        class_covs = []
        class_counts = []
        valid_classes = []
        
        for k in range(n_classes):
            class_mask = pseudo_labels == k
            class_count = np.sum(class_mask)
            
            if class_count < 2:
                continue
            
            class_features = img_features[class_mask]
            class_mean = np.mean(class_features, axis=0)
            
            # 计算协方差（添加正则化以保证数值稳定）
            if class_count > feature_dim:
                class_cov = np.cov(class_features.T) + 1e-6 * np.eye(feature_dim)
            else:
                # 样本数不足，使用对角协方差
                class_cov = np.diag(np.var(class_features, axis=0) + 1e-6)
            
            class_means.append(class_mean)
            class_covs.append(class_cov)
            class_counts.append(class_count)
            valid_classes.append(k)
        
        if len(valid_classes) == 0:
            print_detail("[!] 无有效类别", indent=4)
            return None, {'error': 'No valid classes'}
        
        K = len(valid_classes)
        print_detail(f"  有效类别数: {K}", indent=6)
        
        # Step 4: 计算节点相似度
        print_detail("[4/6] 计算节点相似度...", indent=4)
        
        node_similarities = []
        temperature = 0.05
        
        for i, k in enumerate(valid_classes):
            class_mask = pseudo_labels == k
            class_features = img_features[class_mask]
            
            # 计算每个样本与所有文本嵌入的相似度
            similarities = class_features @ text_embeddings.T
            similarities = similarities / temperature
            exp_sim = np.exp(similarities - similarities.max(axis=1, keepdims=True))
            probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)
            
            # 取该类别对应的概率
            class_probs = probs[:, k]
            avg_prob = np.mean(class_probs)
            
            node_similarities.append(avg_prob * class_counts[i])
        
        s_n = np.sum(node_similarities) / np.sum(class_counts)
        print_detail(f"  节点相似度 (s_n): {s_n:.4f}", indent=6)
        
        # Step 5: 计算边相似度（Bhattacharyya 距离）
        print_detail("[5/6] 计算边相似度（Bhattacharyya 距离）...", indent=4)
        print_detail(f"  需要计算 {K*(K-1)//2} 对类别间的距离", indent=6)
        
        visual_edge_matrix = np.zeros((K, K))
        computed_pairs = 0
        total_pairs = K * (K - 1) // 2
        last_progress = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                # 计算 Bhattacharyya 距离
                mean_i, mean_j = class_means[i], class_means[j]
                cov_i, cov_j = class_covs[i], class_covs[j]
                
                # 使用简化的计算方式
                try:
                    cov_avg = (cov_i + cov_j) / 2
                    diff = mean_i - mean_j
                    
                    # 使用伪逆来避免奇异矩阵问题
                    try:
                        cov_inv = np.linalg.pinv(cov_avg)
                    except:
                        cov_inv = np.eye(feature_dim) * 0.1
                    
                    bh_dist = 0.125 * diff @ cov_inv @ diff
                    bh_dist = max(0, min(bh_dist, 10))  # 限制范围
                except:
                    bh_dist = 1.0  # 默认值
                
                # 转换为相似度
                visual_edge_matrix[i, j] = bh_dist
                visual_edge_matrix[j, i] = bh_dist
                
                computed_pairs += 1
                
                # 每 100 对或完成时显示进度
                progress = int(computed_pairs / total_pairs * 100)
                if progress > last_progress and (progress % 10 == 0 or computed_pairs == total_pairs):
                    print_detail(f"  边计算进度: {computed_pairs}/{total_pairs} ({progress}%)", indent=6)
                    last_progress = progress
        
        # 计算 Pearson 相关系数
        # 展平上三角矩阵（不含对角线）
        text_edges = []
        visual_edges = []
        for i in range(K):
            for j in range(i + 1, K):
                text_edges.append(text_edge_matrix[i, j])
                visual_edges.append(visual_edge_matrix[i, j])
        
        if len(text_edges) > 1:
            corr, _ = stats.pearsonr(text_edges, visual_edges)
            s_e = (corr + 1) / 2
        else:
            s_e = 0.5
        
        print_detail(f"  边相似度 (s_e): {s_e:.4f}", indent=6)
        
        # Step 6: 计算总分
        print_detail("[6/6] 计算总分...", indent=4)
        vega_score = s_n + s_e
        
        elapsed = time.time() - start_time
        print_detail(f"VEGA 总分: {vega_score:.4f} (耗时: {elapsed:.2f}s)", indent=4)
        
        result = {
            'score': vega_score,
            'node_similarity': s_n,
            'edge_similarity': s_e,
            'valid_classes': K,
            'computation_time': elapsed
        }
        
        return vega_score, result
        
    except Exception as e:
        print_detail(f"[!] VEGA 计算错误: {e}", indent=4)
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


def compute_logme_score(
    features: np.ndarray, 
    pseudo_labels: np.ndarray
) -> Tuple[Optional[float], Dict]:
    """
    计算 LogME 分数
    
    使用 LogME_official 库，API 为 logme.fit(features, labels)
    """
    print_detail("计算 LogME 分数:", indent=4)
    print_detail(f"- 特征: {features.shape}", indent=6)
    print_detail(f"- 伪标签: {pseudo_labels.shape}, 唯一类别数: {len(np.unique(pseudo_labels))}", indent=6)
    
    start_time = time.time()
    
    try:
        # 使用 LogME_official 库
        from LogME_official import LogME as LogMEOfficial
        
        logme = LogMEOfficial(regression=False)
        score = logme.fit(features.astype(np.float64), pseudo_labels.astype(np.int64))
        
        elapsed = time.time() - start_time
        print_detail(f"LogME 分数: {score:.4f} (耗时: {elapsed:.2f}s)", indent=4)
        
        return score, {'score': score, 'computation_time': elapsed}
        
    except Exception as e:
        print_detail(f"[!] LogME 计算错误: {e}", indent=4)
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


# ============================================================================
# 评估指标计算
# ============================================================================

def compute_metrics(predicted_scores: Dict[str, float], 
                   ground_truth: Dict[str, float],
                   verbose: bool = True) -> Dict[str, float]:
    """计算评估指标"""
    # 过滤有效数据
    common_models = set(predicted_scores.keys()) & set(ground_truth.keys())
    common_models = [m for m in common_models 
                    if predicted_scores[m] is not None and ground_truth[m] is not None]
    
    if len(common_models) < 3:
        if verbose:
            print_detail(f"[!] 有效模型数不足 ({len(common_models)} < 3)", indent=2)
        return {'error': f'Insufficient data: only {len(common_models)} common models'}
    
    pred = np.array([predicted_scores[m] for m in common_models])
    gt = np.array([ground_truth[m] for m in common_models])
    
    # 打印排序详情
    if verbose:
        print(f"\n  模型排序详情 (共 {len(common_models)} 个模型):")
        
        # 按预测分数排序
        pred_order = np.argsort(pred)[::-1]
        print(f"\n  按预测分数排序:")
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
    """在单个数据集上运行基准测试"""
    print_subheader(f"数据集: {dataset_name}")
    
    vega_scores = {}
    logme_scores = {}
    ground_truth = {}
    
    failed_models = []  # 记录失败的模型
    debug_data = {}
    
    # 创建模型级别的进度条
    pbar = ProgressBar(len(model_list), desc=f"  {dataset_name}")
    
    for idx, model_name in enumerate(model_list):
        pbar.update(1, f"处理 {model_name[:25]}...")
        
        print(f"\n  处理模型: {model_name}")
        
        # 尝试从缓存加载
        cached_vega = load_cache(model_name, dataset_name, 'vega')
        cached_logme = load_cache(model_name, dataset_name, 'logme')
        
        # 1. 加载数据
        print_detail("加载数据...", indent=4)
        
        logits_data = load_logits_data(data_dir, model_name, dataset_name)
        if logits_data is None:
            print_detail(f"[!] 无法加载 logits", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no logits'})
            continue
        
        logits = logits_data['logits']
        print_detail(f"Logits: {logits.shape}", indent=6)
        
        img_feat = load_image_features(data_dir, model_name, dataset_name)
        if img_feat is None:
            print_detail(f"[!] 无法加载图像特征", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no image features'})
            continue
        print_detail(f"图像特征: {img_feat.shape}", indent=6)
        
        text_feat = load_text_features(data_dir, model_name, dataset_name)
        if text_feat is None:
            print_detail(f"[!] 无法加载文本特征", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no text features'})
            continue
        print_detail(f"文本特征: {text_feat.shape}", indent=6)
        
        gt_acc = load_ground_truth_accuracy(data_dir, model_name, dataset_name)
        if gt_acc is None:
            print_detail(f"[!] 无法加载准确率", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no accuracy'})
            continue
        ground_truth[model_name] = gt_acc
        print_detail(f"真实准确率: {gt_acc:.4f}", indent=6)
        
        # 检查维度匹配
        n_samples, n_classes = logits.shape
        n_text_classes = text_feat.shape[0]
        
        if n_text_classes != n_classes:
            print_detail(f"[!] 类别数不匹配: logits={n_classes}, text_feat={n_text_classes}", indent=4)
            min_classes = min(n_classes, n_text_classes)
            logits = logits[:, :min_classes]
            text_feat = text_feat[:min_classes]
            print_detail(f"使用前 {min_classes} 个类别", indent=6)
        
        # 处理样本数不匹配
        if img_feat.shape[0] != logits.shape[0]:
            print_detail(f"[!] 样本数不匹配: img_feat={img_feat.shape[0]}, logits={logits.shape[0]}", indent=4)
            min_samples = min(img_feat.shape[0], logits.shape[0])
            img_feat = img_feat[:min_samples]
            logits = logits[:min_samples]
            print_detail(f"使用前 {min_samples} 个样本", indent=6)
        
        # 2. 计算 VEGA 分数（使用缓存或重新计算）
        if cached_vega is not None:
            vega_score = cached_vega.get('score')
            vega_details = cached_vega
        else:
            vega_score, vega_details = compute_vega_score_with_progress(
                img_feat, text_feat, logits, model_name
            )
            if vega_score is not None:
                save_cache(model_name, dataset_name, 'vega', vega_details)
        
        if vega_score is not None:
            vega_scores[model_name] = vega_score
        
        # 3. 计算 LogME 分数（使用缓存或重新计算）
        pseudo_labels = np.argmax(logits, axis=1)
        
        if cached_logme is not None:
            logme_score = cached_logme.get('score')
            logme_details = cached_logme
        else:
            logme_score, logme_details = compute_logme_score(img_feat, pseudo_labels)
            if logme_score is not None:
                save_cache(model_name, dataset_name, 'logme', logme_details)
        
        if logme_score is not None:
            logme_scores[model_name] = logme_score
    
    # 打印失败模型汇总
    if failed_models:
        print(f"\n  失败模型汇总 ({len(failed_models)}/{len(model_list)}):")
        for fail in failed_models:
            print(f"    - {fail['model']}: {fail['reason']}")
    
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
        'failed_models': failed_models
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
    
    # 失败模型汇总
    print(f"\n失败模型汇总:")
    for r in all_results:
        if r.get('failed_models'):
            print(f"  {r['dataset']}: {len(r['failed_models'])} 个失败")
            for fail in r['failed_models']:
                print(f"    - {fail['model']}: {fail['reason']}")


def main():
    """主函数"""
    # 数据目录
    data_dir = '/root/mxy/SWAB'
    if not os.path.exists(data_dir):
        data_dir = '/root/mxy/VEGA/ptm_stats'
        if not os.path.exists(data_dir):
            print("错误: 数据目录不存在")
            print("请在服务器上运行此脚本")
            return
    
    # 定义测试模型
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
    print(f"缓存目录: {CACHE_DIR}")
    print(f"缓存启用: {ENABLE_CACHE}")
    print(f"测试模型数: {len(test_models)}")
    print(f"测试数据集: {test_datasets}")
    
    # 运行基准测试
    all_results = []
    
    for dataset in test_datasets:
        result = run_single_dataset_benchmark(data_dir, dataset, test_models, verbose=True)
        all_results.append(result)
    
    # 打印最终结果
    print_final_results(all_results)
    
    # 保存结果到文件
    result_file = CACHE_DIR / "benchmark_results.json"
    with open(result_file, 'w') as f:
        # 转换为可序列化的格式
        serializable_results = []
        for r in all_results:
            sr = {
                'dataset': r['dataset'],
                'vega_scores': r['vega_scores'],
                'logme_scores': r['logme_scores'],
                'ground_truth': r['ground_truth'],
                'vega_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                for k, v in r['vega_metrics'].items()},
                'logme_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                 for k, v in r['logme_metrics'].items()},
                'failed_models': r.get('failed_models', [])
            }
            serializable_results.append(sr)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()