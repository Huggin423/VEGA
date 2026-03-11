"""
VEGA SNR Version: Visual-Textual Graph Alignment with SNR-based Node Similarity
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

SNR版本 - 使用信号噪声比(Signal-to-Noise Ratio)进行 Node Similarity 计算

核心改进 (相对于 VEGACalibratedScorer):

1. 【问题诊断】
   - 加载的 `logits` 实际上是未缩放的原始 cosine similarities (值域 [-1, 1])
   - 对原始相似度应用 Softmax 隐式强制温度 T=1.0
   - 这导致严重的概率坍塌 (所有 s_n ≈ 1/K = 0.01)
   - `logit_scale` 在保存的数据中丢失
   - 原始 cosine 幅度严重受架构偏差影响 (ResNet vs. ViT)

2. 【解决方案: SNR-based Node Similarity ($s_n$)】
   - 采用无参数、尺度不变的置信度度量
   - 使用样本内 Z-score of maximum logit (Signal-to-Noise Ratio)
   - 计算:
     max_logits = logits.max(dim=1)    # [N]
     mean_logits = logits.mean(dim=1)  # [N]
     std_logits = logits.std(dim=1)    # [N]
     snr = (max_logits - mean_logits) / (std_logits + EPS)
     s_n = snr.mean()
   - 优势: 完全不受架构偏差影响，无需温度参数

3. 【权重配置】
   - 默认: node_weight=1.0, edge_weight=0.0
   - 隔离 SNR 节点相似度作为主要预测器

4. 【保留视觉图逻辑】
   - PCA、Shrinkage 协方差、向量化 Bhattacharyya 系数保持不变
   - $s_e$ 仍在 return_details 中输出，用于后续分析

实验动机:
- 原始 cosine 相似度的绝对值受架构影响
- ResNet 特征紧凑，cosine 值高
- ViT 特征分散，cosine 值低但边缘更好
- SNR 度量的是相对置信度，而非绝对值
- 因此不受架构偏差影响

更新日志:
- 2026-03-11: 创建 SNR 版本，使用 Signal-to-Noise Ratio
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Dict, List, Tuple
from scipy.stats import pearsonr
import logging
import sys
import os
import time
from pathlib import Path

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 全局详细日志开关
VERBOSE_LOGGING = os.environ.get('VEGA_VERBOSE', '1') == '1'

# 数值稳定性常数
EPS = 1e-8


def progress_print(msg: str, level: str = "INFO"):
    """打印进度信息"""
    if VERBOSE_LOGGING or level == "WARNING":
        print(f"[VEGA-SNR] {msg}", flush=True)


def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA-SNR] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA-SNR] {msg}", flush=True)


class VEGASNRScorer:
    """
    VEGA SNR版本评分器
    
    核心改进:
    
    1. **SNR-based Node Similarity ($s_n$)**
       - 无参数、尺度不变的置信度度量
       - 使用样本内 Z-score of maximum logit
       - s_n = mean((max_logits - mean_logits) / std_logits)
       - 不受架构偏差影响，无需温度参数
    
    2. **权重配置**
       - 默认 node_weight=1.0, edge_weight=0.0
       - 隔离 SNR 节点相似度
       - Edge Similarity 因架构偏差问题被禁用
    
    3. **保留视觉图逻辑**
       - PCA、Shrinkage、Bhattacharyya 系数保持不变
       - s_e 仍在 return_details 中输出
    
    使用方法:
        vega = VEGASNRScorer(pca_dim=64, shrinkage_alpha=0.1, 
                             node_weight=1.0, edge_weight=0.0)
        score = vega.compute_score(visual_features, text_embeddings, logits)
    """
    
    def __init__(
        self, 
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        node_weight: float = 1.0,
        edge_weight: float = 0.0
    ):
        """
        初始化 VEGA SNR版本评分器
        
        Args:
            min_samples_per_class: 每个类别最少样本数
            pca_dim: PCA 降维后的维度 (默认 64)
            shrinkage_alpha: Shrinkage 正则化参数 α (默认 0.1)
            node_weight: 节点相似度权重 (默认 1.0)
            edge_weight: 边相似度权重 (默认 0.0，因架构偏差问题)
        """
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        
        progress_print(f"初始化 VEGASNRScorer:")
        progress_print(f"  PCA 维度: {pca_dim}")
        progress_print(f"  Shrinkage alpha: {shrinkage_alpha}")
        progress_print(f"  Node weight: {node_weight}")
        progress_print(f"  Edge weight: {edge_weight}")
        progress_print(f"  Node Similarity: SNR-based (无参数，尺度不变)")
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """将输入转换为 PyTorch Tensor"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float() if x.dtype != torch.float32 else x
    
    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """将输入转换为 NumPy 数组"""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2 归一化特征向量"""
        return F.normalize(features, p=2, dim=1)
    
    def _apply_pca(
        self, 
        features: torch.Tensor,
        pca_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用 PCA 降维
        
        保留自 VEGAPerfectScorer - 运行完美
        """
        n_samples, n_features = features.shape
        
        if n_features <= pca_dim:
            progress_print(f"    原始维度 {n_features} <= 目标维度 {pca_dim}，跳过 PCA")
            return features, None, None
        
        progress_print(f"    应用 PCA: {n_features} -> {pca_dim}")
        
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        
        U, S, V = torch.pca_lowrank(centered, q=pca_dim, center=False)
        reduced_features = U * S
        
        total_var = (centered ** 2).sum()
        explained_var = (S ** 2).sum()
        explained_ratio = explained_var / total_var
        
        progress_print(f"    PCA 解释方差比例: {explained_ratio:.2%}")
        
        return reduced_features, V, S
    
    def compute_pseudo_labels_from_logits(
        self, 
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        从 logits 计算伪标签
        
        Args:
            logits: 模型预测 [N, K] (原始 cosine similarities)
            
        Returns:
            伪标签 [N]
        """
        return logits.argmax(dim=1)
    
    def build_textual_graph(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建文本图
        
        节点: 文本特征 (L2 归一化)
        边: Cosine 相似度矩阵 [K, K]
        """
        progress_print("  构建文本图...")
        
        text_embeddings = self._normalize_features(text_embeddings)
        nodes = text_embeddings
        edges = nodes @ nodes.T  # Cosine similarity [K, K]
        
        progress_print(f"    文本图节点: {nodes.shape}, 边矩阵: {edges.shape}")
        
        return nodes, edges
    
    def _compute_shrunk_covariance(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Shrinkage 正则化后的完整协方差矩阵
        
        保留自 VEGAPerfectScorer - 运行完美
        """
        n_samples, n_features = features.shape
        device = features.device
        dtype = features.dtype
        
        if n_samples < 2:
            return torch.eye(n_features, device=device, dtype=dtype)
        
        cov = torch.cov(features.T, correction=0)
        
        alpha = self.shrinkage_alpha
        trace_cov = torch.trace(cov)
        target_diag = trace_cov / n_features
        T = target_diag * torch.eye(n_features, device=device, dtype=dtype)
        
        shrunk_cov = (1 - alpha) * cov + alpha * T
        
        return shrunk_cov
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], torch.Tensor]:
        """
        构建视觉图 (在 PCA 降维后的特征上)
        
        返回:
            (class_means, class_covs, class_counts, edge_matrix) 元组
            edge_matrix: Bhattacharyya 系数矩阵 [K, K] (相似度)
        """
        progress_print("  构建视觉图 (PCA + 完整协方差 + Shrinkage)...")
        start_time = time.time()
        
        device = visual_features.device
        dtype = visual_features.dtype
        n_samples, n_features = visual_features.shape
        
        progress_print(f"    原始维度: {n_features}, 样本数: {n_samples}, 类别数: {n_classes}")
        
        # PCA 降维
        pca_start = time.time()
        reduced_features, _, _ = self._apply_pca(visual_features, self.pca_dim)
        pca_time = time.time() - pca_start
        progress_print(f"    PCA 降维完成: {visual_features.shape} -> {reduced_features.shape}, 耗时 {pca_time:.2f}s")
        
        # L2 归一化
        reduced_features = self._normalize_features(reduced_features)
        
        # 计算每个类别的统计量
        class_means_list = []
        class_covs_list = []
        class_counts = {}
        valid_classes = []
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                continue
            
            class_features = reduced_features[indices]
            class_counts[k] = len(indices)
            valid_classes.append(k)
            
            class_mean = class_features.mean(dim=0)
            class_means_list.append(class_mean)
            
            class_cov = self._compute_shrunk_covariance(class_features)
            class_covs_list.append(class_cov)
        
        n_valid = len(valid_classes)
        progress_print(f"    有效类别数: {n_valid}")
        
        if n_valid < 2:
            logger.warning("有效类别数不足，无法构建视觉图")
            return None, None, class_counts, None
        
        class_means = torch.stack(class_means_list, dim=0)
        class_covs = torch.stack(class_covs_list, dim=0)
        
        progress_print(f"    均值矩阵: {class_means.shape}, 协方差张量: {class_covs.shape}")
        
        # 向量化计算 Bhattacharyya 系数矩阵 (相似度!)
        progress_print("    计算 Bhattacharyya 系数矩阵 (向量化)...")
        
        edge_matrix = self._compute_bhattacharyya_coefficient_vectorized(
            class_means, class_covs, valid_classes, n_classes, device, dtype
        )
        
        elapsed = time.time() - start_time
        progress_print(f"    视觉图构建完成: 边矩阵 {edge_matrix.shape}, 总耗时 {elapsed:.2f}s")
        
        return class_means, class_covs, class_counts, edge_matrix
    
    def _compute_bhattacharyya_coefficient_vectorized(
        self,
        class_means: torch.Tensor,
        class_covs: torch.Tensor,
        valid_classes: List[int],
        n_classes: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        向量化计算 Bhattacharyya 系数矩阵 (相似度)
        
        【保留自 VEGAPerfectScorer】
        
        数学推导:
        D_B = 1/8 * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2) + 1/2 * ln(|Σ_avg| / sqrt(|Σ_1||Σ_2|))
        BC = exp(-D_B) ∈ [0, 1]
        
        Args:
            class_means: [K_valid, pca_dim]
            class_covs: [K_valid, pca_dim, pca_dim]
            
        Returns:
            Bhattacharyya 系数矩阵 [K, K] (相似度，对角线=1.0)
        """
        n_valid = len(valid_classes)
        n_features = class_means.shape[1]
        
        progress_print(f"      在 {n_features} 维空间计算 Bhattacharyya 系数...")
        
        # 准备广播张量
        means_i = class_means.unsqueeze(1).unsqueeze(-1)  # [K, 1, pca_dim, 1]
        means_j = class_means.unsqueeze(0).unsqueeze(-1)  # [1, K, pca_dim, 1]
        covs_i = class_covs.unsqueeze(1)  # [K, 1, pca_dim, pca_dim]
        covs_j = class_covs.unsqueeze(0)  # [1, K, pca_dim, pca_dim]
        
        # Σ_avg = (Σ_1 + Σ_2) / 2
        sigma_avg = (covs_i + covs_j) / 2  # [K, K, pca_dim, pca_dim]
        
        # Term1: 1/8 * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2)
        mu_diff = means_i - means_j  # [K, K, pca_dim, 1]
        sigma_avg_inv = torch.linalg.inv(sigma_avg)  # [K, K, pca_dim, pca_dim]
        quad_form = sigma_avg_inv @ mu_diff  # [K, K, pca_dim, 1]
        term1 = 0.125 * (mu_diff.transpose(-2, -1) @ quad_form).squeeze(-1).squeeze(-1)
        
        # Term2: 1/2 * ln(|Σ_avg|) - 1/4 * ln(|Σ_1|) - 1/4 * ln(|Σ_2|)
        sign_avg, logdet_avg = torch.linalg.slogdet(sigma_avg)
        sign_i, logdet_i = torch.linalg.slogdet(class_covs)
        sign_j, logdet_j = torch.linalg.slogdet(class_covs)
        
        term2 = 0.5 * logdet_avg - 0.25 * logdet_i.unsqueeze(1) - 0.25 * logdet_j.unsqueeze(0)
        
        # Bhattacharyya 距离
        bh_distance = term1 + term2
        
        # 确保非负
        bh_distance = torch.clamp(bh_distance, min=0.0)
        
        # 转换为 Bhattacharyya 系数 (相似度)
        # BC = exp(-D_B) ∈ [0, 1]
        bh_coefficient = torch.exp(-bh_distance)  # [K_valid, K_valid]
        
        # 对角线设为 1.0 (同一类别的相似度为 1)
        diagonal_mask = torch.eye(n_valid, device=device, dtype=dtype)
        bh_coefficient = bh_coefficient * (1 - diagonal_mask) + diagonal_mask
        
        # 填充到完整矩阵
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        for i_idx, i_class in enumerate(valid_classes):
            for j_idx, j_class in enumerate(valid_classes):
                edge_matrix[i_class, j_class] = bh_coefficient[i_idx, j_idx]
        
        return edge_matrix
    
    def compute_node_similarity_snr(
        self,
        logits_tensor: torch.Tensor
    ) -> float:
        """
        计算节点相似度 (SNR-based 版本)
        
        【核心改进: Signal-to-Noise Ratio】
        
        问题诊断:
        - 加载的 `logits` 是未缩放的原始 cosine similarities (值域 [-1, 1])
        - 对原始相似度应用 Softmax 隐式强制温度 T=1.0
        - 这导致严重的概率坍塌 (所有 s_n ≈ 1/K = 0.01)
        - `logit_scale` 在保存的数据中丢失
        - 原始 cosine 幅度严重受架构偏差影响 (ResNet vs. ViT)
        
        解决方案:
        - 采用无参数、尺度不变的置信度度量
        - 使用样本内 Z-score of maximum logit (Signal-to-Noise Ratio)
        
        计算:
        - max_logits = logits.max(dim=1)    # [N]
        - mean_logits = logits.mean(dim=1)  # [N]
        - std_logits = logits.std(dim=1)    # [N]
        - snr = (max_logits - mean_logits) / (std_logits + EPS)
        - s_n = snr.mean()
        
        优势:
        - 完全不受架构偏差影响
        - 无需温度参数
        - 度量的是相对置信度，而非绝对值
        
        Args:
            logits_tensor: 原始 cosine similarities [N, K]
            
        Returns:
            节点相似度分数 [SNR 值，通常 > 0]
        """
        progress_print("  计算节点相似度 (SNR-based 版本)...")
        
        n_samples = logits_tensor.shape[0]
        
        # 【核心改进】计算 Signal-to-Noise Ratio
        # 这是无参数、尺度不变的置信度度量
        
        # 每个样本的最大 logit (预测类别的置信度)
        max_logits = logits_tensor.max(dim=1)[0]  # [N]
        
        # 每个样本的均值和标准差
        mean_logits = logits_tensor.mean(dim=1)  # [N]
        std_logits = logits_tensor.std(dim=1)    # [N]
        
        # Signal-to-Noise Ratio (Z-score of maximum)
        # 度量最大值相对于分布中心的偏离程度
        snr = (max_logits - mean_logits) / (std_logits + EPS)
        
        # s_n 是所有样本的平均 SNR
        node_similarity = snr.mean().item()
        
        # 输出详细统计信息
        progress_print(f"    Logits 统计:")
        progress_print(f"      max_logits: mean={max_logits.mean().item():.4f}, std={max_logits.std().item():.4f}")
        progress_print(f"      mean_logits: mean={mean_logits.mean().item():.4f}, std={mean_logits.std().item():.4f}")
        progress_print(f"      std_logits: mean={std_logits.mean().item():.4f}, std={std_logits.std().item():.4f}")
        progress_print(f"    SNR-based Node Similarity = {node_similarity:.4f}")
        progress_print(f"    (无参数，尺度不变，不受架构偏差影响)")
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算边相似度
        
        【保留用于分析】
        
        注意: 由于 CNN 和 ViT 的结构流形拓扑差异，
        Edge Similarity 会引入架构偏差，默认权重为 0。
        但仍计算并在 return_details 中输出，用于后续分析。
        """
        progress_print("  计算边相似度 (保留用于分析)...")
        
        textual_edges = self._to_numpy(textual_edges)
        visual_edges = self._to_numpy(visual_edges)
        
        n = textual_edges.shape[0]
        
        # 提取上三角元素
        triu_indices = np.triu_indices(n, k=1)
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            progress_print("    边数不足，返回默认值 0.5", level="WARNING")
            return 0.5, 0.0
        
        try:
            corr, _ = pearsonr(textual_vec, visual_vec)
            
            if np.isnan(corr):
                progress_print("    Pearson 相关系数为 NaN，返回默认值 0.5", level="WARNING")
                return 0.5, 0.0
            
            # s_e = (corr + 1) / 2 映射到 [0, 1]
            edge_similarity = (corr + 1) / 2
            
            progress_print(f"    边相似度 = {edge_similarity:.4f} (Pearson corr = {corr:.4f})")
            
            return edge_similarity, corr
            
        except Exception as e:
            logger.warning(f"Pearson 相关系数计算失败: {e}")
            return 0.5, 0.0
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        计算 VEGA 迁移性分数（SNR版本）
        
        算法流程:
        1. 计算 SNR-based Node Similarity
           - max_logits = logits.max(dim=1)
           - mean_logits = logits.mean(dim=1)
           - std_logits = logits.std(dim=1)
           - snr = (max_logits - mean_logits) / (std_logits + EPS)
           - s_n = snr.mean()
        2. 构建文本图和视觉图 (保留用于分析)
        3. 计算 s_e (保留用于分析)
        4. 返回 s = node_weight * s_n + edge_weight * s_e
        
        Args:
            features: 图像特征 [N, D]
            text_embeddings: 文本嵌入 [K, D]
            logits: 原始 cosine similarities [N, K]
            pseudo_labels: 伪标签 [N]
            return_details: 是否返回详细信息
            
        Returns:
            VEGA 分数，或包含详细信息的字典
        """
        total_start = time.time()
        
        progress_print("=" * 60)
        progress_print("开始计算 VEGA 分数 (SNR版本)")
        progress_print(f"  PCA 维度: {self.pca_dim}")
        progress_print(f"  Shrinkage alpha: {self.shrinkage_alpha}")
        progress_print(f"  Node weight: {self.node_weight}")
        progress_print(f"  Edge weight: {self.edge_weight}")
        progress_print("  Node Similarity: SNR-based (无参数，尺度不变)")
        progress_print("  s_n = mean((max_logits - mean_logits) / std_logits)")
        progress_print("  s = node_weight * s_n + edge_weight * s_e")
        progress_print("=" * 60)
        
        # 转换为 Tensor
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        # 【关键】logits 必须提供
        if logits is None:
            raise ValueError("VEGASNRScorer 需要 logits 参数 (SNR-based Node Similarity)")
        
        logits_tensor = self._to_tensor(logits)
        
        n_samples, n_features = visual_features.shape
        n_classes = text_embeddings.shape[0]
        
        progress_print(f"数据维度: 样本数={n_samples}, 特征维度={n_features}, 类别数={n_classes}")
        
        # 获取伪标签
        if pseudo_labels is None:
            pseudo_labels = self.compute_pseudo_labels_from_logits(logits_tensor)
        else:
            pseudo_labels = self._to_tensor(pseudo_labels).long()
        
        # 步骤 1: 计算 SNR-based Node Similarity
        node_similarity = self.compute_node_similarity_snr(logits_tensor)
        
        # 步骤 2-4: 构建文本图和视觉图 (保留用于分析)
        textual_nodes, textual_edges = self.build_textual_graph(text_embeddings)
        
        visual_normalized = self._normalize_features(visual_features)
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_normalized, pseudo_labels, n_classes
        )
        
        # 步骤 5: 计算边相似度 (保留用于分析)
        edge_similarity = 0.0
        pearson_corr = 0.0
        
        if visual_edges is not None and len(class_counts) >= 2:
            edge_similarity, pearson_corr = self.compute_edge_similarity(textual_edges, visual_edges)
        
        # 步骤 6: 加权融合
        vega_score = self.node_weight * node_similarity + self.edge_weight * edge_similarity
        
        total_time = time.time() - total_start
        
        progress_print("-" * 60)
        progress_print(f"VEGA 最终分数 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f} (weight={self.node_weight})")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f} (weight={self.edge_weight})")
        progress_print(f"  s = {self.node_weight} * {node_similarity:.4f} + {self.edge_weight} * {edge_similarity:.4f}")
        progress_print(f"  Pearson corr = {pearson_corr:.4f}")
        progress_print(f"总耗时: {total_time:.2f}s")
        progress_print("=" * 60)
        
        if return_details:
            return {
                'score': vega_score,
                'node_similarity': node_similarity,
                'edge_similarity': edge_similarity,
                'pearson_correlation': pearson_corr,
                'valid_classes': len(class_counts) if class_counts else 0,
                'class_counts': class_counts,
                'compute_time': total_time,
                'pca_dim': self.pca_dim,
                'original_dim': n_features,
                'full_covariance': True,
                'shrinkage_alpha': self.shrinkage_alpha,
                'node_weight': self.node_weight,
                'edge_weight': self.edge_weight,
                'bhattacharyya_coefficient': True,
                'snr_based_node_similarity': True
            }
        
        return vega_score


def compute_vega_score_snr(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor],
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    node_weight: float = 1.0,
    edge_weight: float = 0.0
) -> float:
    """
    计算 VEGA 分数的便捷函数（SNR版本）
    
    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        logits: 原始 cosine similarities [N, K] (必须提供)
        pseudo_labels: 伪标签 [N]
        pca_dim: PCA 降维维度 (默认 64)
        shrinkage_alpha: Shrinkage 正则化参数 (默认 0.1)
        node_weight: 节点相似度权重 (默认 1.0)
        edge_weight: 边相似度权重 (默认 0.0)
        
    Returns:
        VEGA 分数
    """
    vega = VEGASNRScorer(
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        node_weight=node_weight,
        edge_weight=edge_weight
    )
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# 别名
VEGASNR = VEGASNRScorer