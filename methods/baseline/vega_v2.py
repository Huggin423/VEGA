"""
VEGA Optimized Version: Visual-Textual Graph Alignment with Mathematical Corrections
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

优化版本 - 严格遵循 VEGA 框架，修复数学和工程缺陷

核心优化:

1. 【鲁棒视觉图 (Dimensionality & Stability)】
   - PCA 降维 (pca_dim=64) 解决 O(D³) 维度诅咒
   - Ledoit-Wolf Shrinkage 正则化防止奇异矩阵 (NaN)
   - 向量化 batched slogdet 实现加速

2. 【边相似度度量修正 ($s_e$)】
   - 原论文问题: Pearson 相关性计算在 Cosine Similarity 和 Bhattacharyya Distance 之间
   - 这导致负相关 (距离 vs 相似度的悖论)
   - 优化: 将 Bhattacharyya 距离转换为相似度系数: bh_coeff = exp(-D_B)
   - 现在 Pearson 相关性正确度量拓扑对齐
   - s_e = (corr + 1) / 2

3. 【节点相似度的自适应温度缩放 ($s_n$)】
   - 固定温度 (t=0.05) 不公平地惩罚不同架构的模型
   - 优化: 在 Softmax 前应用实例级标准化
   - scaled_cos = (cosine_similarity - mean) / (std + EPS)
   - 这使得跨架构的尺度不变

4. 【自然融合】
   - 回到原始融合方法: vega_score = s_n + s_e
   - 默认权重: node_weight=1.0, edge_weight=1.0

更新日志:
- 2026-03-11: 创建优化版本，修复 VEGA 框架的数学缺陷
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Dict, List, Tuple
from scipy.stats import pearsonr, kendalltau
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
        print(f"[VEGA-Optimized] {msg}", flush=True)


class VEGAOptimizedScorer:
    """
    VEGA 优化版本评分器
    
    严格遵循 VEGA 框架，修复数学和工程缺陷
    
    框架流程:
    1. 构建文本图 (Textual Graph)
       - 节点: 文本嵌入 (L2 归一化)
       - 边: Cosine 相似度矩阵 [K, K]
    
    2. 构建视觉图 (Visual Graph)
       - PCA 降维解决维度诅咒
       - Shrinkage 正则化防止奇异矩阵
       - 节点: 类别均值
       - 边: Bhattacharyya 系数矩阵 (相似度!)
    
    3. 计算节点相似度 ($s_n$)
       - 自适应温度缩放 (实例级标准化)
       - Softmax 获取概率分布
       - s_n = 平均伪标签置信度
    
    4. 计算边相似度 ($s_e$)
       - Pearson 相关性 (文本边 vs 视觉边)
       - 两者都是相似度度量，相关性有意义
       - s_e = (corr + 1) / 2
    
    5. 融合
       - vega_score = node_weight * s_n + edge_weight * s_e
    
    使用方法:
        vega = VEGAOptimizedScorer(pca_dim=64, shrinkage_alpha=0.1)
        score = vega.compute_score(visual_features, text_embeddings)
    """
    
    def __init__(
        self, 
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        node_weight: float = 1.0,
        edge_weight: float = 1.0
    ):
        """
        初始化 VEGA 优化版本评分器
        
        Args:
            min_samples_per_class: 每个类别最少样本数
            pca_dim: PCA 降维后的维度 (默认 64)
            shrinkage_alpha: Shrinkage 正则化参数 α (默认 0.1)
            node_weight: 节点相似度权重 (默认 1.0)
            edge_weight: 边相似度权重 (默认 1.0)
        """
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        
        progress_print(f"初始化 VEGAOptimizedScorer:")
        progress_print(f"  PCA 维度: {pca_dim}")
        progress_print(f"  Shrinkage alpha: {shrinkage_alpha}")
        progress_print(f"  Node weight: {node_weight}")
        progress_print(f"  Edge weight: {edge_weight}")
        progress_print(f"  节点相似度: 自适应温度缩放 (实例级标准化)")
        progress_print(f"  边相似度: Bhattacharyya 系数 (相似度度量)")
    
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
    
    # =========================================================================
    # 优化 1: 鲁棒视觉图 (Dimensionality & Stability)
    # =========================================================================
    
    def _apply_pca(
        self, 
        features: torch.Tensor,
        pca_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用 PCA 降维
        
        解决 O(D³) 维度诅咒问题
        
        Args:
            features: 输入特征 [N, D]
            pca_dim: 目标维度
            
        Returns:
            (reduced_features, V, S) 元组
        """
        n_samples, n_features = features.shape
        
        if n_features <= pca_dim:
            progress_print(f"    原始维度 {n_features} <= 目标维度 {pca_dim}，跳过 PCA")
            return features, None, None
        
        progress_print(f"    应用 PCA: {n_features} -> {pca_dim}")
        
        # 中心化
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        
        # 使用 torch.pca_lowrank 进行高效 PCA
        U, S, V = torch.pca_lowrank(centered, q=pca_dim, center=False)
        reduced_features = U * S
        
        # 计算解释方差比例
        total_var = (centered ** 2).sum()
        explained_var = (S ** 2).sum()
        explained_ratio = explained_var / total_var
        
        progress_print(f"    PCA 解释方差比例: {explained_ratio:.2%}")
        
        return reduced_features, V, S
    
    def _compute_shrunk_covariance(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Ledoit-Wolf Shrinkage 正则化后的协方差矩阵
        
        防止奇异矩阵 (NaN) 错误
        
        数学公式:
        Σ_shrunk = (1 - α) * Σ + α * T
        其中 T = (tr(Σ) / d) * I 是缩放的单位矩阵
        
        Args:
            features: 输入特征 [N, d]
            
        Returns:
            正则化后的协方差矩阵 [d, d]
        """
        n_samples, n_features = features.shape
        device = features.device
        dtype = features.dtype
        
        if n_samples < 2:
            return torch.eye(n_features, device=device, dtype=dtype)
        
        # 计算样本协方差矩阵
        cov = torch.cov(features.T, correction=0)
        
        # Ledoit-Wolf Shrinkage
        alpha = self.shrinkage_alpha
        trace_cov = torch.trace(cov)
        target_diag = trace_cov / n_features
        T = target_diag * torch.eye(n_features, device=device, dtype=dtype)
        
        # 正则化协方差
        shrunk_cov = (1 - alpha) * cov + alpha * T
        
        return shrunk_cov
    
    # =========================================================================
    # 文本图构建
    # =========================================================================
    
    def build_textual_graph(
        self, 
        text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建文本图
        
        节点: 文本嵌入 (L2 归一化)
        边: Cosine 相似度矩阵 [K, K]
        
        Args:
            text_embeddings: 文本嵌入 [K, D]
            
        Returns:
            (nodes, edges) 元组
            nodes: [K, D] 归一化文本嵌入
            edges: [K, K] Cosine 相似度矩阵
        """
        progress_print("  构建文本图...")
        
        # L2 归一化
        nodes = self._normalize_features(text_embeddings)
        
        # Cosine 相似度作为边
        edges = nodes @ nodes.T  # [K, K]
        
        progress_print(f"    文本图节点: {nodes.shape}, 边矩阵: {edges.shape}")
        progress_print(f"    边值范围: [{edges.min().item():.4f}, {edges.max().item():.4f}]")
        
        return nodes, edges
    
    # =========================================================================
    # 视觉图构建 (带 PCA 和 Shrinkage)
    # =========================================================================
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], torch.Tensor]:
        """
        构建视觉图 (在 PCA 降维后的特征上)
        
        步骤:
        1. PCA 降维解决维度诅咒
        2. 计算每个类别的统计量 (均值, 协方差)
        3. Shrinkage 正则化防止奇异矩阵
        4. 计算 Bhattacharyya 系数矩阵 (相似度!)
        
        Args:
            visual_features: 视觉特征 [N, D]
            pseudo_labels: 伪标签 [N]
            n_classes: 类别数 K
            
        Returns:
            (class_means, class_covs, class_counts, edge_matrix) 元组
            edge_matrix: Bhattacharyya 系数矩阵 [K, K] (相似度)
        """
        progress_print("  构建视觉图 (PCA + 完整协方差 + Shrinkage)...")
        start_time = time.time()
        
        device = visual_features.device
        dtype = visual_features.dtype
        n_samples, n_features = visual_features.shape
        
        progress_print(f"    原始维度: {n_features}, 样本数: {n_samples}, 类别数: {n_classes}")
        
        # 步骤 1: PCA 降维
        pca_start = time.time()
        reduced_features, _, _ = self._apply_pca(visual_features, self.pca_dim)
        pca_time = time.time() - pca_start
        progress_print(f"    PCA 降维完成: {visual_features.shape} -> {reduced_features.shape}, 耗时 {pca_time:.2f}s")
        
        # L2 归一化
        reduced_features = self._normalize_features(reduced_features)
        
        # 步骤 2: 计算每个类别的统计量
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
            
            # 类别均值
            class_mean = class_features.mean(dim=0)
            class_means_list.append(class_mean)
            
            # 步骤 3: Shrinkage 正则化协方差
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
        
        # 步骤 4: 计算 Bhattacharyya 系数矩阵 (相似度!)
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
        
        【优化 2: 度量修正】
        
        原论文问题:
        - 计算的是 Bhattacharyya 距离 D_B (距离度量，越大越不相似)
        - 文本边是 Cosine 相似度 (相似度度量，越大越相似)
        - Pearson 相关性在距离和相似度之间计算，导致负相关
        
        优化:
        - 将 Bhattacharyya 距离转换为系数: BC = exp(-D_B)
        - BC ∈ [0, 1]，是相似度度量
        - 现在 Pearson 相关性正确度量拓扑对齐
        
        数学推导:
        D_B = 1/8 * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2) + 1/2 * ln(|Σ_avg| / sqrt(|Σ_1||Σ_2|))
        BC = exp(-D_B) ∈ [0, 1]
        
        Args:
            class_means: [K_valid, pca_dim]
            class_covs: [K_valid, pca_dim, pca_dim]
            valid_classes: 有效类别列表
            n_classes: 总类别数
            device: 设备
            dtype: 数据类型
            
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
        # 使用 slogdet 防止数值溢出
        sign_avg, logdet_avg = torch.linalg.slogdet(sigma_avg)
        sign_i, logdet_i = torch.linalg.slogdet(class_covs)
        sign_j, logdet_j = torch.linalg.slogdet(class_covs)
        
        term2 = 0.5 * logdet_avg - 0.25 * logdet_i.unsqueeze(1) - 0.25 * logdet_j.unsqueeze(0)
        
        # Bhattacharyya 距离
        bh_distance = term1 + term2
        
        # 确保非负
        bh_distance = torch.clamp(bh_distance, min=0.0)
        
        # 【关键优化】转换为 Bhattacharyya 系数 (相似度)
        # BC = exp(-D_B) ∈ [0, 1]
        bh_coefficient = torch.exp(-bh_distance)  # [K_valid, K_valid]
        
        # 对角线设为 1.0 (同一类别的相似度为 1)
        diagonal_mask = torch.eye(n_valid, device=device, dtype=dtype)
        bh_coefficient = bh_coefficient * (1 - diagonal_mask) + diagonal_mask
        
        progress_print(f"      Bhattacharyya 系数范围: [{bh_coefficient.min().item():.4f}, {bh_coefficient.max().item():.4f}]")
        
        # 填充到完整矩阵
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        for i_idx, i_class in enumerate(valid_classes):
            for j_idx, j_class in enumerate(valid_classes):
                edge_matrix[i_class, j_class] = bh_coefficient[i_idx, j_idx]
        
        return edge_matrix
    
    # =========================================================================
    # 优化 3: 自适应温度缩放的节点相似度
    # =========================================================================
    
    def compute_node_similarity(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> float:
        """
        计算节点相似度 (自适应温度缩放版本)
        
        【优化 3: 自适应温度缩放】
        
        问题:
        - 固定温度 (t=0.05) 不公平地惩罚不同架构的模型
        - ResNet 特征紧凑，cosine 值高
        - ViT 特征分散，cosine 值低
        
        解决方案:
        - 在 Softmax 前应用实例级标准化
        - scaled_cos = (cosine_similarity - mean) / (std + EPS)
        - 这使得跨架构的尺度不变
        
        计算:
        1. 计算 cosine_similarity = visual_features @ text_embeddings.T
        2. 实例级标准化: scaled_cos = (cos - mean) / std
        3. Softmax 获取概率分布
        4. s_n = 平均伪标签置信度
        
        Args:
            visual_features: 视觉特征 [N, D]
            text_embeddings: 文本嵌入 [K, D]
            pseudo_labels: 伪标签 [N]
            
        Returns:
            节点相似度分数 [范围 (0, 1)]
        """
        progress_print("  计算节点相似度 (自适应温度缩放)...")
        
        n_samples = visual_features.shape[0]
        n_classes = text_embeddings.shape[0]
        
        # 步骤 1: 计算 Cosine 相似度
        visual_normalized = self._normalize_features(visual_features)
        text_normalized = self._normalize_features(text_embeddings)
        cosine_similarity = visual_normalized @ text_normalized.T  # [N, K]
        
        progress_print(f"    Cosine 相似度范围: [{cosine_similarity.min().item():.4f}, {cosine_similarity.max().item():.4f}]")
        
        # 步骤 2: 实例级标准化 (自适应温度)
        # 这是关键优化 - 使得跨架构尺度不变
        mean_cos = cosine_similarity.mean(dim=1, keepdim=True)  # [N, 1]
        std_cos = cosine_similarity.std(dim=1, keepdim=True)    # [N, 1]
        
        # 标准化后的相似度
        scaled_cos = (cosine_similarity - mean_cos) / (std_cos + EPS)
        
        progress_print(f"    标准化后范围: [{scaled_cos.min().item():.4f}, {scaled_cos.max().item():.4f}]")
        
        # 步骤 3: Softmax 获取概率分布
        probs = F.softmax(scaled_cos, dim=1)  # [N, K]
        
        # 步骤 4: 计算伪标签的平均置信度
        # s_n = mean(probs[i, pseudo_label[i]])
        node_similarity = probs[torch.arange(n_samples), pseudo_labels].mean().item()
        
        progress_print(f"    自适应温度 Node Similarity = {node_similarity:.4f}")
        progress_print(f"    (实例级标准化，跨架构尺度不变)")
        
        return node_similarity
    
    # =========================================================================
    # 边相似度计算 (修正后的度量)
    # =========================================================================
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算边相似度
        
        【优化 2: 度量修正】
        
        现在:
        - textual_edges: Cosine 相似度 [K, K] (相似度度量)
        - visual_edges: Bhattacharyya 系数 [K, K] (相似度度量)
        
        两者都是相似度，Pearson 相关性正确度量拓扑对齐!
        
        Args:
            textual_edges: 文本边矩阵 [K, K]
            visual_edges: 视觉边矩阵 [K, K]
            
        Returns:
            (edge_similarity, pearson_corr) 元组
            edge_similarity: s_e = (corr + 1) / 2 ∈ [0, 1]
            pearson_corr: 原始 Pearson 相关系数
        """
        progress_print("  计算边相似度...")
        
        textual_edges = self._to_numpy(textual_edges)
        visual_edges = self._to_numpy(visual_edges)
        
        n = textual_edges.shape[0]
        
        # 提取上三角元素 (排除对角线)
        triu_indices = np.triu_indices(n, k=1)
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            progress_print("    边数不足，返回默认值 0.5", level="WARNING")
            return 0.5, 0.0
        
        progress_print(f"    文本边统计: mean={textual_vec.mean():.4f}, std={textual_vec.std():.4f}")
        progress_print(f"    视觉边统计: mean={visual_vec.mean():.4f}, std={visual_vec.std():.4f}")
        
        try:
            # Pearson 相关系数
            corr, p_value = pearsonr(textual_vec, visual_vec)
            
            if np.isnan(corr):
                progress_print("    Pearson 相关系数为 NaN，返回默认值 0.5", level="WARNING")
                return 0.5, 0.0
            
            # s_e = (corr + 1) / 2 映射到 [0, 1]
            edge_similarity = (corr + 1) / 2
            
            progress_print(f"    Pearson 相关系数: {corr:.4f} (p={p_value:.4e})")
            progress_print(f"    边相似度 s_e = (corr + 1) / 2 = {edge_similarity:.4f}")
            
            return edge_similarity, corr
            
        except Exception as e:
            logger.warning(f"Pearson 相关系数计算失败: {e}")
            return 0.5, 0.0
    
    # =========================================================================
    # 主计算函数
    # =========================================================================
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        计算 VEGA 迁移性分数（优化版本）
        
        严格遵循 VEGA 框架:
        1. 构建文本图 (Textual Graph)
        2. 构建视觉图 (Visual Graph) - PCA + Shrinkage
        3. 计算节点相似度 ($s_n$) - 自适应温度缩放
        4. 计算边相似度 ($s_e$) - Bhattacharyya 系数
        5. 融合: vega_score = node_weight * s_n + edge_weight * s_e
        
        Args:
            features: 图像特征 [N, D]
            text_embeddings: 文本嵌入 [K, D]
            pseudo_labels: 伪标签 [N] (可选，会自动计算)
            return_details: 是否返回详细信息
            
        Returns:
            VEGA 分数，或包含详细信息的字典
        """
        total_start = time.time()
        
        progress_print("=" * 60)
        progress_print("开始计算 VEGA 分数 (优化版本)")
        progress_print(f"  PCA 维度: {self.pca_dim}")
        progress_print(f"  Shrinkage alpha: {self.shrinkage_alpha}")
        progress_print(f"  Node weight: {self.node_weight}")
        progress_print(f"  Edge weight: {self.edge_weight}")
        progress_print("  节点相似度: 自适应温度缩放 (实例级标准化)")
        progress_print("  边相似度: Bhattacharyya 系数 (相似度度量)")
        progress_print("  融合: s = node_weight * s_n + edge_weight * s_e")
        progress_print("=" * 60)
        
        # 转换为 Tensor
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        n_samples, n_features = visual_features.shape
        n_classes = text_embeddings.shape[0]
        
        progress_print(f"数据维度: 样本数={n_samples}, 特征维度={n_features}, 类别数={n_classes}")
        
        # 获取伪标签
        if pseudo_labels is None:
            # 自动计算伪标签
            visual_normalized = self._normalize_features(visual_features)
            text_normalized = self._normalize_features(text_embeddings)
            cosine_similarity = visual_normalized @ text_normalized.T
            pseudo_labels = cosine_similarity.argmax(dim=1)
        else:
            pseudo_labels = self._to_tensor(pseudo_labels).long()
        
        # =====================================
        # 步骤 1: 构建文本图
        # =====================================
        textual_nodes, textual_edges = self.build_textual_graph(text_embeddings)
        
        # =====================================
        # 步骤 2: 构建视觉图 (PCA + Shrinkage + Bhattacharyya 系数)
        # =====================================
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_features, pseudo_labels, n_classes
        )
        
        # =====================================
        # 步骤 3: 计算节点相似度 (自适应温度缩放)
        # =====================================
        node_similarity = self.compute_node_similarity(
            visual_features, text_embeddings, pseudo_labels
        )
        
        # =====================================
        # 步骤 4: 计算边相似度 (Bhattacharyya 系数)
        # =====================================
        edge_similarity = 0.0
        pearson_corr = 0.0
        
        if visual_edges is not None and len(class_counts) >= 2:
            edge_similarity, pearson_corr = self.compute_edge_similarity(textual_edges, visual_edges)
        
        # =====================================
        # 步骤 5: 自然融合
        # =====================================
        vega_score = self.node_weight * node_similarity + self.edge_weight * edge_similarity
        
        total_time = time.time() - total_start
        
        # 输出结果
        progress_print("-" * 60)
        progress_print(f"VEGA 最终分数 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f} (weight={self.node_weight})")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f} (weight={self.edge_weight})")
        progress_print(f"  s = {self.node_weight:.1f} * {node_similarity:.4f} + {self.edge_weight:.1f} * {edge_similarity:.4f}")
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
                'adaptive_temperature': True,
                'optimization_version': 'VEGAOptimizedScorer'
            }
        
        return vega_score


def compute_vega_score_optimized(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    node_weight: float = 1.0,
    edge_weight: float = 1.0
) -> float:
    """
    计算 VEGA 分数的便捷函数（优化版本）
    
    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        pseudo_labels: 伪标签 [N]
        pca_dim: PCA 降维维度 (默认 64)
        shrinkage_alpha: Shrinkage 正则化参数 (默认 0.1)
        node_weight: 节点相似度权重 (默认 1.0)
        edge_weight: 边相似度权重 (默认 1.0)
        
    Returns:
        VEGA 分数
    """
    vega = VEGAOptimizedScorer(
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        node_weight=node_weight,
        edge_weight=edge_weight
    )
    return vega.compute_score(features, text_embeddings, pseudo_labels)


def compute_tau_at_k(
    predicted_scores: Union[np.ndarray, List[float]],
    ground_truth_scores: Union[np.ndarray, List[float]],
    k: int = 5,
    return_details: bool = False
) -> Union[float, Tuple[float, float, int]]:
    """
    计算 Tau@K (默认 Tau@5)。

    定义:
    - 先按 ground-truth 分数选出 Top-K 模型；
    - 再在这些模型上计算 predicted 与 ground-truth 的 Kendall tau。

    Args:
        predicted_scores: 预测分数，长度为 N
        ground_truth_scores: 真实分数，长度为 N
        k: Top-K 截断，默认 5
        return_details: 是否返回 (tau_k, p_value, effective_k)

    Returns:
        tau_k，或 (tau_k, p_value, effective_k)
    """
    pred = np.asarray(predicted_scores, dtype=np.float64)
    gt = np.asarray(ground_truth_scores, dtype=np.float64)

    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    n = pred.shape[0]
    if n < 2:
        if return_details:
            return float("nan"), float("nan"), min(max(int(k), 1), n)
        return float("nan")

    effective_k = min(max(int(k), 1), n)
    if effective_k < 2:
        if return_details:
            return float("nan"), float("nan"), effective_k
        return float("nan")

    # 在 ground-truth Top-K 上评估排序一致性
    topk_idx = np.argsort(gt)[-effective_k:]
    tau_k, p_value = kendalltau(pred[topk_idx], gt[topk_idx])

    if return_details:
        return float(tau_k), float(p_value), effective_k
    return float(tau_k)


# 别名
VEGAOptimized = VEGAOptimizedScorer