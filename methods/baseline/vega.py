"""
VEGA: Visual-Textual Graph Alignment for Unsupervised VLM Selection
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

Key Idea:
- Build textual graph from class name embeddings (nodes=classes, edges=cosine similarity)
- Build visual graph from image features (nodes=class clusters modeled as Gaussians, edges=Bhattacharyya distance)
- Score models by measuring alignment between the two graphs at node and edge levels

更新日志:
- 2026-03-06: 添加详细进度日志，优化 Bhattacharyya 距离计算，添加缓存支持
- 2026-03-06: 性能优化版本
  - 添加 PCA 降维/白化预处理（去相关，满足对角近似假设）
  - 使用对角协方差近似（Diagonal Covariance）替代全协方差矩阵
  - 向量化 edge_matrix 计算（移除双重 for 循环）
  - 复用 Cosine Similarity 矩阵
  - 添加数值稳定性保护（epsilon）
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Dict, List, Tuple
from scipy.stats import pearsonr
import logging
import sys
import os
import json
import hashlib
import time
from pathlib import Path

# 配置日志输出到控制台
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
    """打印进度信息，确保立即输出"""
    if VERBOSE_LOGGING or level == "WARNING":
        print(f"[VEGA] {msg}", flush=True)

def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA] {msg}", flush=True)


class VEGAScorer:
    """
    VEGA-based model selection scorer (优化版本).
    
    Implements the paper's algorithm with the following optimizations:
    1. PCA whitening preprocessing for decorrelation
    2. Diagonal covariance approximation (avoids matrix inversion)
    3. Vectorized Bhattacharyya distance computation
    4. Reused cosine similarity matrix
    5. Numerical stability with epsilon
    
    Usage:
        vega = VEGAScorer(temperature=0.05, use_pca=True, pca_dim=256)
        score = vega.compute_score(visual_features, text_embeddings, logits)
    """
    
    def __init__(
        self, 
        temperature: float = 0.05, 
        min_samples_per_class: int = 1,
        use_pca: bool = True,
        pca_dim: int = 512,
        pca_whiten: bool = False,
        node_weight: float = 0.5,
        edge_weight: float = 0.5
    ):
        """
        Initialize VEGA scorer.
        
        Args:
            temperature: Temperature parameter for softmax normalization in node similarity
            min_samples_per_class: Minimum samples required per class for valid computation
            use_pca: Whether to use PCA dimensionality reduction
            pca_dim: Target dimension after PCA (ignored if use_pca=False)
            pca_whiten: Whether to whiten features (disabled by default to preserve semantic structure)
            node_weight: Weight for node similarity in final score (default: 0.5)
            edge_weight: Weight for edge similarity in final score (default: 0.5)
        """
        self.temperature = temperature
        self.min_samples_per_class = min_samples_per_class
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.pca_whiten = pca_whiten
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        
        # PCA components (will be fitted during compute_score)
        self.pca_mean = None
        self.pca_components = None
        self.pca_explained_variance = None
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float() if x.dtype != torch.float32 else x
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features."""
        return F.normalize(features, p=2, dim=1)
    
    def _fit_pca(self, features: torch.Tensor) -> torch.Tensor:
        """
        Fit PCA on features and transform them.
        
        PCA whitening achieves:
        1. Dimensionality reduction (faster computation)
        2. Decorrelation (makes diagonal covariance assumption valid)
        3. Variance normalization (improves numerical stability)
        
        Args:
            features: Input features [N, D]
            
        Returns:
            Transformed features [N, pca_dim]
        """
        n_samples, n_features = features.shape
        
        # Determine target dimension
        target_dim = min(self.pca_dim, n_features, n_samples)
        
        if target_dim >= n_features:
            # No need for PCA
            progress_print(f"  PCA: 特征维度 {n_features} <= 目标维度 {target_dim}，跳过 PCA")
            return features
        
        progress_print(f"  PCA: 降维 {n_features} -> {target_dim} (白化={self.pca_whiten})")
        
        # Center features
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        
        # Compute covariance matrix
        # Use SVD for numerical stability: centered = U @ S @ V^T
        # Covariance ≈ V @ (S^2 / (n-1)) @ V^T
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            
            # Take top components
            V = Vh[:target_dim, :].T  # [D, target_dim]
            S = S[:target_dim]  # [target_dim]
            
            # Transform: centered @ V = U @ diag(S)
            transformed = centered @ V  # [N, target_dim]
            
            if self.pca_whiten:
                # Whiten: divide by sqrt(variance) = sqrt(S^2 / (n-1)) = S / sqrt(n-1)
                # This makes the covariance matrix approximately identity
                scale = S / np.sqrt(n_samples - 1 + EPS)
                transformed = transformed / (scale.unsqueeze(0) + EPS)
            
            # Store PCA parameters
            self.pca_mean = mean
            self.pca_components = V
            self.pca_explained_variance = S ** 2 / (n_samples - 1 + EPS)
            
            return transformed
            
        except Exception as e:
            logger.warning(f"PCA computation failed: {e}, using original features")
            return features
    
    def _transform_pca(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform features using fitted PCA.
        
        Args:
            features: Input features [N, D]
            
        Returns:
            Transformed features [N, pca_dim]
        """
        if self.pca_components is None:
            return features
        
        centered = features - self.pca_mean
        
        if self.pca_whiten:
            # Whiten transformation
            transformed = centered @ self.pca_components
            scale = torch.sqrt(self.pca_explained_variance + EPS)
            transformed = transformed / (scale.unsqueeze(0) + EPS)
        else:
            # Standard PCA transformation
            transformed = centered @ self.pca_components
        
        return transformed
    
    def compute_pseudo_labels(
        self, 
        cosine_similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pseudo labels from cosine similarity matrix.
        
        Args:
            cosine_similarity: Cosine similarity matrix [N, K]
            
        Returns:
            Pseudo labels [N]
        """
        return cosine_similarity.argmax(dim=1)
    
    def build_textual_graph(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build textual graph.
        
        Nodes: text embeddings of each class [K, D]
        Edges: cosine similarity between class embeddings [K, K]
        
        Args:
            text_embeddings: Text embeddings [K, D]
            
        Returns:
            Tuple of (nodes, edges) where nodes=[K, D], edges=[K, K]
        """
        # Normalize features
        text_embeddings = self._normalize_features(text_embeddings)
        
        # Node features are the text embeddings
        nodes = text_embeddings
        
        # Edge weights are cosine similarity
        edges = nodes @ nodes.T  # [K, K]
        
        return nodes, edges
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], torch.Tensor]:
        """
        Build visual graph with optimized diagonal covariance approximation.
        
        Nodes: Each class is modeled as a Gaussian distribution N(mu_k, sigma_k^2 * I)
               where sigma_k^2 is the per-dimension variance (diagonal covariance)
        Edges: Simplified Bhattacharyya distance between class Gaussians
        
        Args:
            visual_features: Image features [N, D]
            pseudo_labels: Pseudo labels [N]
            n_classes: Number of classes K
            
        Returns:
            Tuple of (class_means, class_vars, class_counts, edge_matrix)
            - class_means: Dict mapping class index to mean vector
            - class_vars: Dict mapping class index to variance vector (diagonal)
            - class_counts: Dict mapping class index to sample count
            - edge_matrix: Bhattacharyya distance matrix [K, K]
        """
        # Normalize features
        visual_features = self._normalize_features(visual_features)
        device = visual_features.device
        dtype = visual_features.dtype
        
        # Compute class statistics
        class_means = {}
        class_vars = {}  # Diagonal covariance: variance per dimension
        class_counts = {}
        
        for k in range(n_classes):
            # Get samples assigned to class k
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                continue
            
            class_features = visual_features[indices]
            class_counts[k] = len(indices)
            
            # Compute mean
            class_means[k] = class_features.mean(dim=0)
            
            # Compute variance (diagonal of covariance matrix)
            # Use biased estimator (divide by n) for stability with small samples
            if len(indices) > 1:
                var = class_features.var(dim=0, unbiased=False)  # [D]
            else:
                # Single sample - use small variance
                var = torch.ones(visual_features.shape[1], device=device, dtype=dtype) * 0.1
            
            # Add epsilon for numerical stability
            class_vars[k] = var + EPS
        
        # Get valid classes
        valid_classes = list(class_means.keys())
        n_valid = len(valid_classes)
        
        if n_valid < 2:
            logger.warning("Not enough valid classes for visual graph construction")
            return class_means, class_vars, class_counts, None
        
        # Stack means and variances for vectorized computation
        # Create index mapping: class_id -> matrix index
        class_to_idx = {k: i for i, k in enumerate(valid_classes)}
        n_valid = len(valid_classes)
        
        # Stack all means and vars into matrices
        means_matrix = torch.stack([class_means[k] for k in valid_classes], dim=0)  # [n_valid, D]
        vars_matrix = torch.stack([class_vars[k] for k in valid_classes], dim=0)  # [n_valid, D]
        
        # Vectorized Bhattacharyya distance computation
        # For diagonal covariance:
        # D_B = (1/8) * sum_d (mu1_d - mu2_d)^2 / sigma_d 
        #       + (1/2) * sum_d log(sigma_d) - (1/4) * sum_d (log(sigma1_d) + log(sigma2_d))
        #     = (1/8) * sum_d (mu1_d - mu2_d)^2 / sigma_d 
        #       + (1/4) * sum_d log(sigma_d) - (1/4) * sum_d (log(sigma1_d) + log(sigma2_d))
        
        # Compute pairwise mean differences: [n_valid, n_valid, D]
        # mu_diff[i, j, :] = means_matrix[i] - means_matrix[j]
        mu_diff = means_matrix.unsqueeze(1) - means_matrix.unsqueeze(0)  # [n_valid, n_valid, D]
        
        # Compute average variance: sigma_avg[i, j, :] = (vars_matrix[i] + vars_matrix[j]) / 2
        sigma_avg = (vars_matrix.unsqueeze(1) + vars_matrix.unsqueeze(0)) / 2  # [n_valid, n_valid, D]
        
        # Term 1: (1/8) * sum_d (mu1_d - mu2_d)^2 / sigma_avg_d
        # Avoid division by zero with epsilon
        term1 = (mu_diff ** 2) / (sigma_avg + EPS)  # [n_valid, n_valid, D]
        term1_sum = term1.sum(dim=-1) * 0.125  # [n_valid, n_valid]
        
        # Term 2: (1/4) * sum_d log(sigma_avg_d)
        # Use log1p for numerical stability: log(sigma_avg) = log1p(sigma_avg - 1)
        log_sigma_avg = torch.log(sigma_avg + EPS)  # [n_valid, n_valid, D]
        term2 = log_sigma_avg.sum(dim=-1) * 0.25  # [n_valid, n_valid]
        
        # Term 3: -(1/4) * sum_d (log(sigma1_d) + log(sigma2_d))
        log_vars = torch.log(vars_matrix + EPS)  # [n_valid, D]
        # log_vars_sum[i, j] = log_vars[i].sum() + log_vars[j].sum()
        log_vars_sum = log_vars.unsqueeze(1) + log_vars.unsqueeze(0)  # [n_valid, n_valid, D]
        term3 = log_vars_sum.sum(dim=-1) * 0.25  # [n_valid, n_valid]
        
        # Total Bhattacharyya distance
        bh_distance = term1_sum + term2 - term3  # [n_valid, n_valid]
        
        # Create full edge matrix [K, K]
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        
        # Fill in the computed distances
        for i_idx, i_class in enumerate(valid_classes):
            for j_idx, j_class in enumerate(valid_classes):
                if i_class != j_class:
                    edge_matrix[i_class, j_class] = bh_distance[i_idx, j_idx]
        
        # Note: We return class_vars instead of class_covs for compatibility
        # The name 'class_covs' is kept for backward compatibility but contains variance vectors
        return class_means, class_vars, class_counts, edge_matrix
    
    def _bhattacharyya_distance(
        self, 
        mu1: torch.Tensor, 
        var1: torch.Tensor,
        mu2: torch.Tensor, 
        var2: torch.Tensor
    ) -> float:
        """
        Compute Bhattacharyya distance between two diagonal Gaussians.
        
        Simplified formula for diagonal covariance:
        D_B = (1/8) * sum_d (mu1_d - mu2_d)^2 / sigma_d 
              + (1/4) * sum_d log(sigma_d) - (1/4) * sum_d (log(var1_d) + log(var2_d))
        where sigma_d = (var1_d + var2_d) / 2
        
        Args:
            mu1, var1: Mean and variance (diagonal) of first Gaussian
            mu2, var2: Mean and variance (diagonal) of second Gaussian
            
        Returns:
            Bhattacharyya distance (float)
        """
        try:
            # Average variance
            sigma_avg = (var1 + var2) / 2
            
            # Term 1: Mahalanobis-like term
            term1 = ((mu1 - mu2) ** 2) / (sigma_avg + EPS)
            term1_sum = term1.sum().item() * 0.125
            
            # Term 2: log determinant term (simplified for diagonal)
            log_sigma = torch.log(sigma_avg + EPS).sum().item()
            log_var1 = torch.log(var1 + EPS).sum().item()
            log_var2 = torch.log(var2 + EPS).sum().item()
            term2 = 0.25 * log_sigma - 0.25 * (log_var1 + log_var2)
            
            return term1_sum + term2
            
        except Exception as e:
            logger.warning(f"Bhattacharyya distance computation failed: {e}")
            return torch.norm(mu1 - mu2).item()
    
    def compute_node_similarity(
        self,
        cosine_similarity: torch.Tensor,
        pseudo_labels: torch.Tensor,
        class_counts: Dict[int, int]
    ) -> float:
        """
        Compute node similarity from precomputed cosine similarity matrix.
        
        Node similarity measures the average distance from visual features 
        within a cluster to the corresponding textual feature.
        
        sim_k = (1/N_k) * sum_{v in V_k} [exp(cos(v, t_k)/t) / sum_{k'} exp(cos(v, t_k')/t)]
        s_n = (1/K) * sum_k sim_k * N_k
        
        Args:
            cosine_similarity: Cosine similarity matrix [N, K]
            pseudo_labels: Pseudo labels [N]
            class_counts: Sample count per class
            
        Returns:
            Node similarity score in [0, 1]
        """
        n_samples, n_classes = cosine_similarity.shape
        
        # Apply temperature scaling and softmax
        scaled_similarity = cosine_similarity / self.temperature
        probs = F.softmax(scaled_similarity, dim=1)  # [N, K]
        
        # Compute per-class similarity scores
        class_similarities = {}
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # Get the probability assigned to the correct class for each sample
            class_probs = probs[indices, k]
            
            # Average similarity for this class
            class_similarities[k] = class_probs.mean().item()
        
        if not class_similarities:
            return 0.0
        
        # Compute weighted average
        total_weighted_sim = 0.0
        total_weight = 0
        
        for k, sim in class_similarities.items():
            weight = class_counts.get(k, 1)
            total_weighted_sim += sim * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        node_similarity = total_weighted_sim / total_weight
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> float:
        """
        Compute edge similarity using Pearson correlation.
        
        Edge similarity measures the correlation between the edge structures
        of textual and visual graphs.
        
        s_e = (PearsonCorr(vec(E^T), vec(E^V)) + 1) / 2
        
        Args:
            textual_edges: Textual graph edge matrix [K, K]
            visual_edges: Visual graph edge matrix [K, K]
            
        Returns:
            Edge similarity score in [0, 1]
        """
        # Convert to numpy for scipy
        if isinstance(textual_edges, torch.Tensor):
            textual_edges = textual_edges.cpu().numpy()
        if isinstance(visual_edges, torch.Tensor):
            visual_edges = visual_edges.cpu().numpy()
        
        # Flatten the matrices (excluding diagonal)
        n = textual_edges.shape[0]
        
        # Get upper triangular indices (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            return 0.5  # Neutral value if not enough edges
        
        # Compute Pearson correlation
        try:
            corr, _ = pearsonr(textual_vec, visual_vec)
            
            # Handle NaN
            if np.isnan(corr):
                return 0.5
            
            # Rescale from [-1, 1] to [0, 1]
            edge_similarity = (corr + 1) / 2
            
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
        Compute VEGA transferability score (优化版本).
        
        Optimizations:
        1. PCA dimensionality reduction and whitening
        2. Diagonal covariance approximation (faster Bhattacharyya distance)
        3. Vectorized edge matrix computation
        4. Reused cosine similarity matrix
        5. Numerical stability with epsilon
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            logits: Model predictions [N, K] (optional, used for pseudo labels if provided)
            pseudo_labels: Pseudo labels [N] (optional, derived from features if not provided)
            return_details: If True, return detailed breakdown of scores
            
        Returns:
            VEGA score (float), or dict with details if return_details=True
        """
        total_start = time.time()
        
        # Convert to tensors
        progress_print("转换数据格式...")
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        n_samples, n_features = visual_features.shape
        n_classes = text_embeddings.shape[0]
        
        progress_print(f"数据维度: 样本数={n_samples}, 特征维度={n_features}, 类别数={n_classes}")
        
        # Apply PCA if enabled
        if self.use_pca and n_features > self.pca_dim:
            pca_start = time.time()
            progress_print("【预处理】PCA 降维/白化...")
            
            # Fit PCA on visual features
            visual_features = self._fit_pca(visual_features)
            
            # Transform text embeddings using the same PCA
            if self.pca_components is not None:
                text_embeddings = self._transform_pca(text_embeddings)
            
            timing_print(f"  PCA 完成: 新维度={visual_features.shape[1]}", pca_start)
        
        # Compute cosine similarity matrix (reused for pseudo labels and node similarity)
        cosine_start = time.time()
        progress_print("【复用计算】计算 Cosine Similarity 矩阵...")
        
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
        
        # Step 2: Build visual graph (with diagonal covariance)
        step2_start = time.time()
        progress_print("【步骤 2/4】构建视觉图（对角协方差近似）...")
        class_means, class_vars, class_counts, visual_edges = self.build_visual_graph(
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
                    'pca_dim': visual_features.shape[1] if self.use_pca else n_features
                }
            return 0.0
        
        # Step 3: Compute node similarity (reusing cosine similarity)
        step3_start = time.time()
        progress_print("【步骤 3/4】计算节点相似度...")
        node_similarity = self.compute_node_similarity(
            cosine_similarity, pseudo_labels, class_counts
        )
        timing_print(f"  节点相似度 = {node_similarity:.4f}", step3_start)
        
        # Step 4: Compute edge similarity
        step4_start = time.time()
        progress_print("【步骤 4/4】计算边相似度...")
        edge_similarity = self.compute_edge_similarity(textual_edges, visual_edges)
        timing_print(f"  边相似度 = {edge_similarity:.4f}", step4_start)
        
        # Step 5: Final VEGA score
        vega_score = node_similarity + edge_similarity
        
        total_time = time.time() - total_start
        progress_print(f"VEGA 总分 = {vega_score:.4f} (总耗时: {total_time:.2f}s)")
        
        if return_details:
            return {
                'score': vega_score,
                'node_similarity': node_similarity,
                'edge_similarity': edge_similarity,
                'valid_classes': len(class_means),
                'class_counts': class_counts,
                'compute_time': total_time,
                'pca_dim': visual_features.shape[1] if self.use_pca else n_features,
                'diagonal_covariance': True
            }
        
        return vega_score


class VEGAPlus(VEGAScorer):
    """
    Extended VEGA with confidence-weighted scoring.
    
    This is an experimental extension that combines VEGA score
    with prediction confidence for potentially better ranking.
    """
    
    def __init__(
        self, 
        temperature: float = 0.05, 
        confidence_weight: float = 0.3,
        use_pca: bool = True,
        pca_dim: int = 256,
        pca_whiten: bool = True
    ):
        """
        Initialize VEGA+ scorer.
        
        Args:
            temperature: Temperature for node similarity
            confidence_weight: Weight for confidence term (0-1)
            use_pca: Whether to use PCA dimensionality reduction
            pca_dim: Target dimension after PCA
            pca_whiten: Whether to whiten features
        """
        super().__init__(
            temperature=temperature, 
            use_pca=use_pca, 
            pca_dim=pca_dim, 
            pca_whiten=pca_whiten
        )
        self.confidence_weight = confidence_weight
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Compute VEGA+ score with confidence weighting.
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            logits: Model predictions [N, K]
            pseudo_labels: Pseudo labels [N]
            return_details: If True, return detailed breakdown
            
        Returns:
            VEGA+ score
        """
        # Get base VEGA score
        base_result = super().compute_score(
            features, text_embeddings, logits, pseudo_labels, return_details=True
        )
        
        base_score = base_result['score']
        
        if logits is None:
            if return_details:
                base_result['confidence'] = None
                return base_result
            return base_score
        
        # Compute confidence
        logits_tensor = self._to_tensor(logits)
        probs = F.softmax(logits_tensor, dim=1)
        max_probs, _ = probs.max(dim=1)
        confidence = max_probs.mean().item()
        
        # Combine scores
        final_score = (1 - self.confidence_weight) * base_score + \
                      self.confidence_weight * confidence
        
        if return_details:
            base_result['confidence'] = confidence
            base_result['final_score'] = final_score
            return base_result
        
        return final_score


def compute_vega_score(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    temperature: float = 0.05,
    use_pca: bool = True,
    pca_dim: int = 256
) -> float:
    """
    Convenience function to compute VEGA score.
    
    Args:
        features: Image features [N, D]
        text_embeddings: Text embeddings [K, D]
        logits: Model predictions [N, C]
        pseudo_labels: Pseudo labels [N]
        temperature: Temperature parameter
        use_pca: Whether to use PCA
        pca_dim: Target dimension after PCA
        
    Returns:
        VEGA score
    """
    vega = VEGAScorer(temperature=temperature, use_pca=use_pca, pca_dim=pca_dim)
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# Backward compatibility - keep old class name as alias
VEGA = VEGAScorer