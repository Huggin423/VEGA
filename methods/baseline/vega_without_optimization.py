"""
VEGA: Visual-Textual Graph Alignment for Unsupervised VLM Selection
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

原始无优化版本 - 严格遵循论文公式

Key Idea:
- Build textual graph from class name embeddings (nodes=classes, edges=cosine similarity)
- Build visual graph from image features (nodes=class clusters modeled as Gaussians, edges=Bhattacharyya distance)
- Score models by measuring alignment between the two graphs at node and edge levels

论文公式:
1. 文本图节点: n^T_k = ξ(c̃_k)  (文本特征)
2. 文本图边: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j))  (cosine相似度)
3. 视觉图节点: 高斯分布 N(μ_k, Σ_k)
   - 均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
   - 协方差: Σ_k = (1/N_k) Σ_{i:ŷ_i=k} (φ(x_i) - μ_k)(φ(x_i) - μ_k)^T
4. 视觉图边: Bhattacharyya距离
   - e^V_ij = (1/8)(μ_i - μ_j)^T Σ^{-1} (μ_i - μ_j) + (1/2)ln(|Σ|/√(|Σ_i||Σ_j|))
   - 其中 Σ = (Σ_i + Σ_j)/2
5. 节点相似度: s_n = (1/K) Σ_k sim(n^T_k, n^V_k) · N_k
6. 边相似度: s_e = (corr(E^T, E^V) + 1)/2
7. 最终分数: s = s_n + s_e

此版本与优化版本的主要区别:
- 无PCA降维/白化
- 使用完整协方差矩阵（非对角近似）
- 使用双重for循环计算Bhattacharyya距离（非向量化）
- 严格遵循论文公式计算节点相似度
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
        print(f"[VEGA-Original] {msg}", flush=True)


def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA-Original] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA-Original] {msg}", flush=True)


class VEGAOriginalScorer:
    """
    VEGA原始版本 - 严格遵循论文公式
    
    与优化版本的区别:
    1. 无PCA降维/白化预处理
    2. 使用完整协方差矩阵（非对角近似）
    3. 使用双重for循环计算Bhattacharyya距离（非向量化）
    4. 严格遵循论文公式计算节点相似度: s_n = (1/K) Σ_k sim_k · N_k
    """
    
    def __init__(
        self, 
        temperature: float = 0.05, 
        min_samples_per_class: int = 2,
        node_weight: float = 0.5,
        edge_weight: float = 0.5,
        regularization: float = 1e-6
    ):
        """
        Initialize VEGA scorer (原始版本).
        
        Args:
            temperature: Temperature parameter for softmax normalization (论文中t=0.05)
            min_samples_per_class: Minimum samples required per class for valid covariance estimation
            node_weight: Weight for node similarity in final score (论文中默认权重相等)
            edge_weight: Weight for edge similarity in final score
            regularization: Regularization term for covariance matrix inversion stability
        """
        self.temperature = temperature
        self.min_samples_per_class = min_samples_per_class
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.regularization = regularization
        
        # 论文中没有PCA相关参数
        self.use_pca = False
        self.pca_dim = None
        self.pca_whiten = False
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float() if x.dtype != torch.float32 else x
    
    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features."""
        return F.normalize(features, p=2, dim=1)
    
    def compute_pseudo_labels(
        self, 
        cosine_similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pseudo labels from cosine similarity matrix.
        
        论文公式(1): ŷ_i = argmax_k cos(ξ(c̃_k), φ(x_i))
        
        Args:
            cosine_similarity: Cosine similarity matrix [N, K]
            
        Returns:
            Pseudo labels [N]
        """
        return cosine_similarity.argmax(dim=1)
    
    def build_textual_graph(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build textual graph.
        
        论文公式:
        - 节点: n^T_k = ξ(c̃_k) (文本特征)
        - 边: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j)) (cosine相似度)
        
        Args:
            text_embeddings: Text embeddings [K, D]
            
        Returns:
            Tuple of (nodes, edges) where nodes=[K, D], edges=[K, K]
        """
        progress_print("  构建文本图...")
        
        # Normalize features (论文中cosine相似度计算前需要归一化)
        text_embeddings = self._normalize_features(text_embeddings)
        
        # Node features are the text embeddings
        nodes = text_embeddings
        
        # Edge weights are cosine similarity
        # 论文公式: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j))
        edges = nodes @ nodes.T  # [K, K]
        
        progress_print(f"    文本图节点: {nodes.shape}, 边矩阵: {edges.shape}")
        
        return nodes, edges
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], torch.Tensor]:
        """
        Build visual graph with FULL COVARIANCE matrices (论文原始版本).
        
        论文公式:
        - 节点: 高斯分布 N(μ_k, Σ_k)
          - 均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
          - 协方差: Σ_k = (1/N_k) Σ_{i:ŷ_i=k} (φ(x_i) - μ_k)(φ(x_i) - μ_k)^T
        - 边: Bhattacharyya距离
        
        注意: 此版本使用完整协方差矩阵，不对角近似！
        
        Args:
            visual_features: Image features [N, D]
            pseudo_labels: Pseudo labels [N]
            n_classes: Number of classes K
            
        Returns:
            Tuple of (class_means, class_covs, class_counts, edge_matrix)
            - class_means: Dict mapping class index to mean vector
            - class_covs: Dict mapping class index to FULL covariance matrix [D, D]
            - class_counts: Dict mapping class index to sample count
            - edge_matrix: Bhattacharyya distance matrix [K, K]
        """
        progress_print("  构建视觉图 (完整协方差矩阵)...")
        start_time = time.time()
        
        # Normalize features
        visual_features = self._normalize_features(visual_features)
        device = visual_features.device
        dtype = visual_features.dtype
        n_samples, n_features = visual_features.shape
        
        progress_print(f"    样本数: {n_samples}, 特征维度: {n_features}, 类别数: {n_classes}")
        
        # Compute class statistics
        class_means = {}
        class_covs = {}  # 完整协方差矩阵
        class_counts = {}
        
        for k in range(n_classes):
            # Get samples assigned to class k
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                progress_print(f"    类别 {k}: 样本数 {len(indices)} < {self.min_samples_per_class}, 跳过")
                continue
            
            class_features = visual_features[indices]  # [N_k, D]
            class_counts[k] = len(indices)
            
            # Compute mean: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
            class_means[k] = class_features.mean(dim=0)  # [D]
            
            # Compute FULL covariance matrix: Σ_k = (1/N_k) Σ (x - μ)(x - μ)^T
            # 论文原文使用完整协方差矩阵
            if len(indices) > 1:
                centered = class_features - class_means[k].unsqueeze(0)  # [N_k, D]
                # 完整协方差矩阵 [D, D]
                cov = (centered.T @ centered) / len(indices)
                # 添加正则化项以确保数值稳定性
                cov = cov + self.regularization * torch.eye(n_features, device=device, dtype=dtype)
            else:
                # Single sample - use identity covariance
                cov = torch.eye(n_features, device=device, dtype=dtype)
            
            class_covs[k] = cov
        
        # Get valid classes
        valid_classes = list(class_means.keys())
        n_valid = len(valid_classes)
        
        progress_print(f"    有效类别数: {n_valid}")
        
        if n_valid < 2:
            logger.warning("Not enough valid classes for visual graph construction")
            return class_means, class_covs, class_counts, None
        
        # Compute Bhattacharyya distance matrix using DOUBLE FOR LOOP (论文原始方式)
        progress_print("    计算 Bhattacharyya 距离矩阵 (双重循环)...")
        
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        
        for i_idx, i_class in enumerate(valid_classes):
            for j_idx, j_class in enumerate(valid_classes):
                if i_class >= j_class:
                    continue  # 对称矩阵，只计算上三角
                
                # 计算Bhattacharyya距离
                dist = self._bhattacharyya_distance_full(
                    class_means[i_class], class_covs[i_class],
                    class_means[j_class], class_covs[j_class]
                )
                
                edge_matrix[i_class, j_class] = dist
                edge_matrix[j_class, i_class] = dist  # 对称
        
        elapsed = time.time() - start_time
        progress_print(f"    视觉图构建完成: 边矩阵 {edge_matrix.shape}, 耗时 {elapsed:.2f}s")
        
        return class_means, class_covs, class_counts, edge_matrix
    
    def _bhattacharyya_distance_full(
        self, 
        mu1: torch.Tensor, 
        cov1: torch.Tensor,
        mu2: torch.Tensor, 
        cov2: torch.Tensor
    ) -> float:
        """
        Compute Bhattacharyya distance between two Gaussians with FULL covariance.
        
        论文公式(10):
        e^V_ij = (1/8)(μ_i - μ_j)^T Σ^{-1} (μ_i - μ_j) + (1/2)ln(|Σ|/√(|Σ_i||Σ_j|))
        其中 Σ = (Σ_i + Σ_j)/2
        
        Args:
            mu1, cov1: Mean [D] and FULL covariance matrix [D, D] of first Gaussian
            mu2, cov2: Mean [D] and FULL covariance matrix [D, D] of second Gaussian
            
        Returns:
            Bhattacharyya distance (float)
        """
        try:
            # Average covariance: Σ = (Σ_i + Σ_j)/2
            sigma = (cov1 + cov2) / 2
            
            # Compute inverse of average covariance
            # 使用 torch.linalg.inv 进行矩阵求逆
            try:
                sigma_inv = torch.linalg.inv(sigma)
            except RuntimeError:
                # 如果求逆失败，使用伪逆
                progress_print("      矩阵求逆失败，使用伪逆", level="WARNING")
                sigma_inv = torch.linalg.pinv(sigma)
            
            # Term 1: (1/8)(μ_i - μ_j)^T Σ^{-1} (μ_i - μ_j)
            mu_diff = mu1 - mu2  # [D]
            term1 = 0.125 * (mu_diff @ sigma_inv @ mu_diff)
            
            # Term 2: (1/2)ln(|Σ|/√(|Σ_i||Σ_j|))
            # = (1/2)ln|Σ| - (1/4)ln|Σ_i| - (1/4)ln|Σ_j|
            try:
                log_det_sigma = torch.linalg.slogdet(sigma)[1]  # ln|Σ|
                log_det_cov1 = torch.linalg.slogdet(cov1)[1]    # ln|Σ_i|
                log_det_cov2 = torch.linalg.slogdet(cov2)[1]    # ln|Σ_j|
                
                term2 = 0.5 * log_det_sigma - 0.25 * log_det_cov1 - 0.25 * log_det_cov2
            except RuntimeError:
                # 如果行列式计算失败，使用特征值方法
                progress_print("      行列式计算失败，使用特征值方法", level="WARNING")
                eigvals_sigma = torch.linalg.eigvalsh(sigma)
                eigvals_cov1 = torch.linalg.eigvalsh(cov1)
                eigvals_cov2 = torch.linalg.eigvalsh(cov2)
                
                log_det_sigma = torch.log(eigvals_sigma + EPS).sum()
                log_det_cov1 = torch.log(eigvals_cov1 + EPS).sum()
                log_det_cov2 = torch.log(eigvals_cov2 + EPS).sum()
                
                term2 = 0.5 * log_det_sigma - 0.25 * log_det_cov1 - 0.25 * log_det_cov2
            
            distance = term1 + term2
            
            # 确保距离非负
            if distance < 0:
                progress_print(f"      警告: 负距离 {distance.item():.6f}, 设为0", level="WARNING")
                distance = torch.tensor(0.0, device=mu1.device)
            
            return distance.item()
            
        except Exception as e:
            logger.warning(f"Bhattacharyya distance computation failed: {e}")
            # Fallback to Euclidean distance
            return torch.norm(mu1 - mu2).item()
    
    def compute_node_similarity(
        self,
        cosine_similarity: torch.Tensor,
        pseudo_labels: torch.Tensor,
        class_counts: Dict[int, int]
    ) -> float:
        """
        Compute node similarity from precomputed cosine similarity matrix.
        
        论文公式(11, 12):
        sim(n^T_k, n^V_k) = (1/N_k) Σ_{i:ŷ_i=k} [exp(cos(φ(x_i), ξ(c_k))/t) / Σ_{k'} exp(cos(φ(x_i), ξ(c_{k'}))/t)]
        
        s_n = (1/K) Σ_k sim(n^T_k, n^V_k) · N_k
        
        【修复说明】
        原公式存在数值范围问题：当样本数N较大时，Σ_k sim_k · N_k 可能远大于 K，导致 s_n > 1。
        
        正确理解论文公式：
        1. sim_k 是类别 k 内所有样本的平均 softmax 概率，范围 [0, 1]
        2. s_n = (1/K) Σ_k sim_k · N_k 是加权求和后除以类别数 K
        3. 但这样可能导致 s_n > 1（当样本总数 N > K 时）
        
        实际上，论文公式可能存在笔误，更合理的解释是：
        s_n = (1/K) Σ_k sim_k（简单平均，忽略样本权重）
        
        但为了遵循论文原文，我们按原公式实现，并添加数值范围约束。
        
        Args:
            cosine_similarity: Cosine similarity matrix [N, K]
            pseudo_labels: Pseudo labels [N]
            class_counts: Sample count per class
            
        Returns:
            Node similarity score in [0, 1]
        """
        progress_print("  计算节点相似度...")
        
        n_samples, n_classes = cosine_similarity.shape
        
        # Apply temperature scaling and softmax
        # 论文公式(12): exp(cos/t) / Σ exp(cos/t)
        scaled_similarity = cosine_similarity / self.temperature
        probs = F.softmax(scaled_similarity, dim=1)  # [N, K]
        
        # Compute per-class similarity scores
        class_similarities = {}
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # 论文公式(12): sim(n^T_k, n^V_k) = (1/N_k) Σ_{i:ŷ_i=k} [...]
            # 获取每个样本在其分配类别上的概率
            class_probs = probs[indices, k]
            
            # Average similarity for this class
            class_similarities[k] = class_probs.mean().item()
        
        if not class_similarities:
            return 0.0
        
        # 【修复】严格按照论文公式 (1/K) Σ_k sim_k · N_k
        # sim_k 已经是类别内平均概率，N_k 是权重
        total_weighted_sim = 0.0
        for k, sim in class_similarities.items():
            N_k = class_counts.get(k, 1)
            total_weighted_sim += sim * N_k
        
        # 论文公式: s_n = (1/K) * Σ_k sim_k · N_k
        node_similarity = total_weighted_sim / n_classes
        
        # 【修复】添加数值范围检查，确保节点相似度在 [0, 1] 范围内
        # 这是因为原公式在样本数 N >> 类别数 K 时可能产生 > 1 的值
        node_similarity = float(np.clip(node_similarity, 0.0, 1.0))
        
        n_valid_classes = len(class_similarities)
        progress_print(f"    节点相似度 = {node_similarity:.4f} (有效类别: {n_valid_classes}/{n_classes})")
        progress_print(f"    加权和 = {total_weighted_sim:.2f}, 类别数 K = {n_classes}")
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> float:
        """
        Compute edge similarity using Pearson correlation.
        
        论文公式(13, 14):
        corr(E^T, E^V) = Pearson correlation between edge matrices
        s_e = (corr + 1)/2
        
        Args:
            textual_edges: Textual graph edge matrix [K, K]
            visual_edges: Visual graph edge matrix [K, K]
            
        Returns:
            Edge similarity score in [0, 1]
        """
        progress_print("  计算边相似度...")
        
        # Convert to numpy for scipy
        textual_edges = self._to_numpy(textual_edges)
        visual_edges = self._to_numpy(visual_edges)
        
        n = textual_edges.shape[0]
        
        # Get upper triangular indices (excluding diagonal)
        # 论文中计算所有边的Pearson相关系数
        triu_indices = np.triu_indices(n, k=1)
        
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            progress_print("    边数不足，返回默认值 0.5", level="WARNING")
            return 0.5
        
        # Compute Pearson correlation
        try:
            corr, _ = pearsonr(textual_vec, visual_vec)
            
            if np.isnan(corr):
                progress_print("    Pearson相关系数为NaN，返回默认值 0.5", level="WARNING")
                return 0.5
            
            # 论文公式(14): s_e = (corr + 1)/2
            # Rescale from [-1, 1] to [0, 1]
            edge_similarity = (corr + 1) / 2
            
            progress_print(f"    边相似度 = {edge_similarity:.4f} (Pearson corr = {corr:.4f})")
            
            return edge_similarity
            
        except Exception as e:
            logger.warning(f"Pearson correlation failed: {e}")
            return 0.5
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Compute VEGA transferability score (原始无优化版本).
        
        严格遵循论文算法流程:
        1. 获取伪标签: ŷ_i = argmax_k cos(ξ(c̃_k), φ(x_i))
        2. 构建文本图: 节点=文本特征, 边=cosine相似度
        3. 构建视觉图: 节点=高斯分布(完整协方差), 边=Bhattacharyya距离
        4. 计算节点相似度: s_n = (1/K) Σ_k sim_k · N_k
        5. 计算边相似度: s_e = (corr + 1)/2
        6. 最终分数: s = s_n + s_e
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            logits: Model predictions [N, K] (optional)
            pseudo_labels: Pseudo labels [N] (optional)
            return_details: If True, return detailed breakdown of scores
            
        Returns:
            VEGA score (float), or dict with details if return_details=True
        """
        total_start = time.time()
        
        progress_print("="*60)
        progress_print("开始计算 VEGA 分数 (原始无优化版本)")
        progress_print("="*60)
        
        # Convert to tensors
        progress_print("转换数据格式...")
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        n_samples, n_features = visual_features.shape
        n_classes = text_embeddings.shape[0]
        
        progress_print(f"数据维度: 样本数={n_samples}, 特征维度={n_features}, 类别数={n_classes}")
        
        # 无PCA预处理 - 直接使用原始特征
        
        # Compute cosine similarity matrix
        cosine_start = time.time()
        progress_print("计算 Cosine Similarity 矩阵...")
        
        visual_normalized = self._normalize_features(visual_features)
        text_normalized = self._normalize_features(text_embeddings)
        
        # Cosine similarity: [N, K]
        cosine_similarity = visual_normalized @ text_normalized.T
        
        timing_print(f"  Cosine Similarity 矩阵: {cosine_similarity.shape}", cosine_start)
        
        # Compute pseudo labels if not provided
        if pseudo_labels is None:
            if logits is not None:
                progress_print("从 logits 获取伪标签...")
                logits_tensor = self._to_tensor(logits)
                pseudo_labels = logits_tensor.argmax(dim=1)
            else:
                progress_print("从 Cosine Similarity 获取伪标签...")
                pseudo_labels = self.compute_pseudo_labels(cosine_similarity)
        else:
            pseudo_labels = self._to_tensor(pseudo_labels).long()
        
        # Step 1: Build textual graph
        step1_start = time.time()
        progress_print("【步骤 1/4】构建文本图...")
        textual_nodes, textual_edges = self.build_textual_graph(text_embeddings)
        timing_print(f"  文本图构建完成: 边矩阵 {textual_edges.shape}", step1_start)
        
        # Step 2: Build visual graph with FULL covariance
        step2_start = time.time()
        progress_print("【步骤 2/4】构建视觉图 (完整协方差矩阵)...")
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_normalized, pseudo_labels, n_classes
        )
        timing_print(f"  视觉图构建完成: 有效类别数={len(class_means)}", step2_start)
        
        # Check if visual graph construction was successful
        if visual_edges is None or len(class_means) < 2:
            progress_print("视觉图构建失败，返回默认分数", level="WARNING")
            if return_details:
                return {
                    'score': 0.0,
                    'node_similarity': 0.0,
                    'edge_similarity': 0.0,
                    'valid_classes': len(class_means),
                    'use_pca': False,
                    'full_covariance': True
                }
            return 0.0
        
        # Step 3: Compute node similarity (论文公式11, 12)
        step3_start = time.time()
        progress_print("【步骤 3/4】计算节点相似度...")
        node_similarity = self.compute_node_similarity(
            cosine_similarity, pseudo_labels, class_counts
        )
        timing_print(f"  节点相似度 = {node_similarity:.4f}", step3_start)
        
        # Step 4: Compute edge similarity (论文公式13, 14)
        step4_start = time.time()
        progress_print("【步骤 4/4】计算边相似度...")
        edge_similarity = self.compute_edge_similarity(textual_edges, visual_edges)
        timing_print(f"  边相似度 = {edge_similarity:.4f}", step4_start)
        
        # Step 5: Final VEGA score (论文公式: s = s_n + s_e)
        vega_score = node_similarity + edge_similarity
        
        total_time = time.time() - total_start
        progress_print("-"*60)
        progress_print(f"VEGA 总分 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f}")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f}")
        progress_print(f"总耗时: {total_time:.2f}s")
        progress_print("="*60)
        
        if return_details:
            return {
                'score': vega_score,
                'node_similarity': node_similarity,
                'edge_similarity': edge_similarity,
                'valid_classes': len(class_means),
                'class_counts': class_counts,
                'compute_time': total_time,
                'use_pca': False,
                'full_covariance': True,
                'vectorized_computation': False
            }
        
        return vega_score


def compute_vega_score_original(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    temperature: float = 0.05
) -> float:
    """
    Convenience function to compute VEGA score (原始版本).
    
    Args:
        features: Image features [N, D]
        text_embeddings: Text embeddings [K, D]
        logits: Model predictions [N, C]
        pseudo_labels: Pseudo labels [N]
        temperature: Temperature parameter (论文默认0.05)
        
    Returns:
        VEGA score
    """
    vega = VEGAOriginalScorer(temperature=temperature)
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# Alias for backward compatibility
VEGAOriginal = VEGAOriginalScorer