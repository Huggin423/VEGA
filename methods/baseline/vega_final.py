"""
VEGA Final Version: Visual-Textual Graph Alignment with PCA Reduction and Weighted Fusion
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

最终版本 - 解决高维特征的"维度诅咒"问题

核心改进 (相对于 VEGARobustScorer):
1. PCA 降维 - 将高维特征 (D=512/768/1024) 降至低维潜在空间 (pca_dim=64)
   - 解决 O(D³) 计算瓶颈
   - 恢复 Bhattacharyya 距离的几何意义
   - 使用 torch.pca_lowrank 快速计算

2. 加权融合 - 解决 s_n 和 s_e 尺度不匹配问题
   - s_n ~ 0.01 (1/K), s_e ~ 0.5 (随机相关系数)
   - 使用加权融合: score = w_n * s_n + w_e * s_e
   - 默认权重 w_n=1.0, w_e=0.0 (仅使用节点相似度)

3. 保留 Shrinkage 正则化和向量化计算
   - 在 PCA 降维后的特征上应用完整协方差 + Shrinkage
   - 批量矩阵运算，无 for 循环

论文公式参考:
1. 文本图: 节点=文本特征, 边=cosine相似度 (不变)
2. 视觉图: 
   - PCA 降维: φ'(x) = PCA(φ(x)) → [N, pca_dim]
   - 节点: 高斯分布 N(μ_k, Σ_k) 在低维空间
   - 边: Bhattacharyya 距离在低维空间
3. 节点相似度: s_n = Σ_k sim_k · N_k / N_total (不变)
4. 边相似度: s_e = (corr + 1)/2 (在低维空间计算)
5. 最终分数: s = w_n * s_n + w_e * s_e

更新日志:
- 2026-03-11: 创建最终版本，解决维度诅咒问题
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
        print(f"[VEGA-Final] {msg}", flush=True)


def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA-Final] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA-Final] {msg}", flush=True)


class VEGAFinalScorer:
    """
    VEGA 最终版本评分器
    
    核心改进 (相对于 VEGARobustScorer):
    
    1. **PCA 降维 (解决维度诅咒)**
       - 问题: 高维特征 D=512/768/1024 导致:
         - O(D³) 计算瓶颈 (矩阵求逆、行列式)
         - Bhattacharyya 距离失去几何意义 (所有点等距)
         - Edge Similarity 为纯噪声
       - 解决: 在计算视觉图前，将特征降至 pca_dim=64
       - 使用 torch.pca_lowrank 快速计算
    
    2. **加权融合 (解决尺度不匹配)**
       - 问题: s_n ~ 0.01 (1/K), s_e ~ 0.5 (随机)
       - 解决: score = w_n * s_n + w_e * s_e
       - 默认: w_n=1.0, w_e=0.0 (仅使用节点相似度)
    
    3. **保留 Shrinkage 和向量化计算**
       - 在 PCA 降维后的特征上应用
       - 保证协方差矩阵正定、可逆
    
    使用方法:
        vega = VEGAFinalScorer(
            temperature=0.05,
            pca_dim=64,
            shrinkage_alpha=0.1,
            node_weight=1.0,
            edge_weight=0.0
        )
        score = vega.compute_score(visual_features, text_embeddings, logits)
    """
    
    def __init__(
        self, 
        temperature: float = 0.05, 
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        node_weight: float = 1.0,
        edge_weight: float = 0.0
    ):
        """
        初始化 VEGA 最终版本评分器
        
        Args:
            temperature: softmax 归一化的温度参数 (论文默认 t=0.05)
            min_samples_per_class: 每个类别最少样本数，用于有效的方差估计
            pca_dim: PCA 降维后的维度 (默认 64)
            shrinkage_alpha: Shrinkage 正则化参数 α (默认 0.1)
            node_weight: 节点相似度权重 w_n (默认 1.0)
            edge_weight: 边相似度权重 w_e (默认 0.0)
        """
        self.temperature = temperature
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.node_weight = node_weight
        self.edge_weight = edge_weight
    
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
        
        【核心改进 1: PCA 降维解决维度诅咒】
        
        高维特征 (D=512/768/1024) 存在以下问题:
        1. 计算瓶颈: O(D³) 矩阵求逆和行列式计算
        2. 几何失真: 在高维空间中，所有点对之间的距离趋于相等
        3. 噪声放大: Edge Similarity 变成纯噪声
        
        PCA 降维后 (pca_dim=64):
        1. 计算速度: O(64³) << O(1024³)
        2. 几何意义: 恢复距离的区分度
        3. 信噪比: Edge Similarity 更有意义
        
        使用 torch.pca_lowrank 进行快速 SVD-based PCA:
        - 输入: [N, D] 原始特征
        - 输出: [N, pca_dim] 降维后特征
        - 复杂度: O(N * D * min(N, D)) 而非 O(D³)
        
        Args:
            features: 输入特征 [N, D]
            pca_dim: 目标维度
            
        Returns:
            (reduced_features, components, explained_variance) 元组
            - reduced_features: 降维后特征 [N, pca_dim]
            - components: 主成分 [pca_dim, D]
            - explained_variance: 解释方差 [pca_dim]
        """
        n_samples, n_features = features.shape
        
        # 如果原始维度已经小于等于目标维度，无需降维
        if n_features <= pca_dim:
            progress_print(f"    原始维度 {n_features} <= 目标维度 {pca_dim}，跳过 PCA")
            return features, None, None
        
        progress_print(f"    应用 PCA: {n_features} -> {pca_dim}")
        
        # 中心化
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        
        # 使用 torch.pca_lowrank 进行快速 PCA
        # 返回: U [N, pca_dim], S [pca_dim], V [D, pca_dim]
        # 使得 centered ≈ U @ diag(S) @ V.T
        U, S, V = torch.pca_lowrank(centered, q=pca_dim, center=False)
        
        # 降维后的特征: [N, pca_dim]
        # reduced = U * S (即投影到主成分空间)
        reduced_features = U * S  # [N, pca_dim]
        
        # 计算解释方差比例
        total_var = (centered ** 2).sum()
        explained_var = (S ** 2).sum()
        explained_ratio = explained_var / total_var
        
        progress_print(f"    PCA 解释方差比例: {explained_ratio:.2%}")
        
        return reduced_features, V, S
    
    def compute_pseudo_labels(
        self, 
        cosine_similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        从 cosine similarity 矩阵计算伪标签
        
        论文公式(1): ŷ_i = argmax_k cos(ξ(c̃_k), φ(x_i))
        
        Args:
            cosine_similarity: Cosine similarity 矩阵 [N, K]
            
        Returns:
            伪标签 [N]
        """
        return cosine_similarity.argmax(dim=1)
    
    def build_textual_graph(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建文本图
        
        论文公式:
        - 节点: n^T_k = ξ(c̃_k) (文本特征)
        - 边: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j)) (cosine相似度)
        
        Args:
            text_embeddings: 文本嵌入 [K, D]
            
        Returns:
            (nodes, edges) 元组，其中 nodes=[K, D], edges=[K, K]
        """
        progress_print("  构建文本图...")
        
        # L2 归一化
        text_embeddings = self._normalize_features(text_embeddings)
        
        # 节点特征即为文本嵌入
        nodes = text_embeddings
        
        # 边权重为 cosine similarity
        edges = nodes @ nodes.T  # [K, K]
        
        progress_print(f"    文本图节点: {nodes.shape}, 边矩阵: {edges.shape}")
        
        return nodes, edges
    
    def _compute_shrunk_covariance(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Shrinkage 正则化后的完整协方差矩阵
        
        公式:
        Σ_shrunk = (1 - α) · Σ + α · T
        T = trace(Σ)/D · I
        
        Args:
            features: 类别特征 [N_k, pca_dim]
            
        Returns:
            shrunk_covariance: [pca_dim, pca_dim]
        """
        n_samples, n_features = features.shape
        device = features.device
        dtype = features.dtype
        
        if n_samples < 2:
            return torch.eye(n_features, device=device, dtype=dtype)
        
        # 计算样本协方差矩阵
        cov = torch.cov(features.T, correction=0)
        
        # Shrinkage 正则化
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
        构建视觉图（在 PCA 降维后的特征上）
        
        流程:
        1. 对输入特征进行 PCA 降维
        2. 在低维空间计算均值和协方差
        3. 应用 Shrinkage 正则化
        4. 向量化计算 Bhattacharyya 距离
        
        Args:
            visual_features: 图像特征 [N, D] (原始高维)
            pseudo_labels: 伪标签 [N]
            n_classes: 类别数 K
            
        Returns:
            (class_means, class_covs, class_counts, edge_matrix) 元组
        """
        progress_print("  构建视觉图 (PCA 降维 + 完整协方差 + Shrinkage)...")
        start_time = time.time()
        
        device = visual_features.device
        dtype = visual_features.dtype
        n_samples, n_features = visual_features.shape
        
        progress_print(f"    原始维度: {n_features}, 样本数: {n_samples}, 类别数: {n_classes}")
        
        # ============================================================
        # 【核心改进 1】应用 PCA 降维
        # ============================================================
        pca_start = time.time()
        reduced_features, pca_components, pca_singular_values = self._apply_pca(
            visual_features, self.pca_dim
        )
        pca_time = time.time() - pca_start
        
        progress_print(f"    PCA 降维完成: {visual_features.shape} -> {reduced_features.shape}, 耗时 {pca_time:.2f}s")
        
        # L2 归一化降维后的特征
        reduced_features = self._normalize_features(reduced_features)
        
        # ============================================================
        # 步骤 2: 计算每个类别的统计量
        # ============================================================
        class_means_list = []
        class_covs_list = []
        class_counts = {}
        valid_classes = []
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                continue
            
            class_features = reduced_features[indices]  # [N_k, pca_dim]
            class_counts[k] = len(indices)
            valid_classes.append(k)
            
            # 计算均值
            class_mean = class_features.mean(dim=0)
            class_means_list.append(class_mean)
            
            # 计算 Shrinkage 正则化后的协方差
            class_cov = self._compute_shrunk_covariance(class_features)
            class_covs_list.append(class_cov)
        
        n_valid = len(valid_classes)
        progress_print(f"    有效类别数: {n_valid}")
        
        if n_valid < 2:
            logger.warning("有效类别数不足，无法构建视觉图")
            return None, None, class_counts, None
        
        # 堆叠成张量
        class_means = torch.stack(class_means_list, dim=0)
        class_covs = torch.stack(class_covs_list, dim=0)
        
        progress_print(f"    均值矩阵: {class_means.shape}, 协方差张量: {class_covs.shape}")
        
        # ============================================================
        # 步骤 3: 向量化计算 Bhattacharyya 距离矩阵
        # ============================================================
        progress_print("    计算 Bhattacharyya 距离矩阵 (向量化)...")
        
        edge_matrix = self._compute_bhattacharyya_distance_vectorized(
            class_means, class_covs, valid_classes, n_classes, device, dtype
        )
        
        elapsed = time.time() - start_time
        progress_print(f"    视觉图构建完成: 边矩阵 {edge_matrix.shape}, 总耗时 {elapsed:.2f}s")
        
        return class_means, class_covs, class_counts, edge_matrix
    
    def _compute_bhattacharyya_distance_vectorized(
        self,
        class_means: torch.Tensor,
        class_covs: torch.Tensor,
        valid_classes: List[int],
        n_classes: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        向量化计算 Bhattacharyya 距离矩阵 (在低维空间)
        
        由于已经降维到 pca_dim=64，计算复杂度从 O(D³) 降低到 O(64³)
        
        Args:
            class_means: 类别均值矩阵 [K_valid, pca_dim]
            class_covs: 类别协方差张量 [K_valid, pca_dim, pca_dim]
            valid_classes: 有效类别列表
            n_classes: 总类别数 K
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            Bhattacharyya 距离矩阵 [K, K]
        """
        n_valid = len(valid_classes)
        n_features = class_means.shape[1]  # pca_dim
        
        progress_print(f"      在 {n_features} 维空间计算距离...")
        
        # 准备广播所需的张量
        means_i = class_means.unsqueeze(1).unsqueeze(-1)  # [K, 1, pca_dim, 1]
        means_j = class_means.unsqueeze(0).unsqueeze(-1)  # [1, K, pca_dim, 1]
        covs_i = class_covs.unsqueeze(1)  # [K, 1, pca_dim, pca_dim]
        covs_j = class_covs.unsqueeze(0)  # [1, K, pca_dim, pca_dim]
        
        # 计算 Σ_avg = (Σ_1 + Σ_2) / 2
        sigma_avg = (covs_i + covs_j) / 2  # [K, K, pca_dim, pca_dim]
        
        # Term1: (1/8) * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2)
        mu_diff = means_i - means_j  # [K, K, pca_dim, 1]
        sigma_avg_inv = torch.linalg.inv(sigma_avg)  # [K, K, pca_dim, pca_dim]
        quad_form = sigma_avg_inv @ mu_diff  # [K, K, pca_dim, 1]
        term1 = 0.125 * (mu_diff.transpose(-2, -1) @ quad_form).squeeze(-1).squeeze(-1)
        
        # Term2: (1/2)ln|Σ_avg| - (1/4)ln|Σ_1| - (1/4)ln|Σ_2|
        sign_avg, logdet_avg = torch.linalg.slogdet(sigma_avg)
        sign_i, logdet_i = torch.linalg.slogdet(class_covs)
        sign_j, logdet_j = torch.linalg.slogdet(class_covs)
        
        term2 = 0.5 * logdet_avg - 0.25 * logdet_i.unsqueeze(1) - 0.25 * logdet_j.unsqueeze(0)
        
        # 合并
        bh_distance = term1 + term2
        
        # 确保对角线为 0
        bh_distance = bh_distance * (1 - torch.eye(n_valid, device=device, dtype=dtype))
        
        # 确保非负
        bh_distance = torch.clamp(bh_distance, min=0.0)
        
        # 填充到完整矩阵
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        for i_idx, i_class in enumerate(valid_classes):
            for j_idx, j_class in enumerate(valid_classes):
                if i_class != j_class:
                    edge_matrix[i_class, j_class] = bh_distance[i_idx, j_idx]
        
        return edge_matrix
    
    def compute_node_similarity(
        self,
        cosine_similarity: torch.Tensor,
        pseudo_labels: torch.Tensor,
        class_counts: Dict[int, int]
    ) -> float:
        """
        计算节点相似度
        
        s_n = Σ_k sim(n^T_k, n^V_k) · N_k / N_total
        
        Args:
            cosine_similarity: Cosine similarity 矩阵 [N, K]
            pseudo_labels: 伪标签 [N]
            class_counts: 每个类别的样本数 {class_idx: count}
            
        Returns:
            节点相似度分数 [0, 1]
        """
        progress_print("  计算节点相似度...")
        
        n_samples, n_classes = cosine_similarity.shape
        
        # 应用温度缩放和 softmax
        scaled_similarity = cosine_similarity / self.temperature
        probs = F.softmax(scaled_similarity, dim=1)
        
        # 计算每个类别的相似度分数
        class_similarities = {}
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            class_probs = probs[indices, k]
            class_similarities[k] = class_probs.mean().item()
        
        if not class_similarities:
            return 0.0
        
        # 计算加权和，除以总样本数
        total_weighted_sim = 0.0
        total_samples = 0
        
        for k, sim in class_similarities.items():
            N_k = class_counts.get(k, 1)
            total_weighted_sim += sim * N_k
            total_samples += N_k
        
        if total_samples > 0:
            node_similarity = total_weighted_sim / total_samples
        else:
            node_similarity = 0.0
        
        n_valid_classes = len(class_similarities)
        progress_print(f"    节点相似度 = {node_similarity:.4f} (有效类别: {n_valid_classes}/{n_classes})")
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算边相似度（使用 Pearson 相关系数）
        
        Args:
            textual_edges: 文本图边矩阵 [K, K]
            visual_edges: 视觉图边矩阵 [K, K]
            
        Returns:
            (edge_similarity, pearson_correlation) 元组
        """
        progress_print("  计算边相似度...")
        
        textual_edges = self._to_numpy(textual_edges)
        visual_edges = self._to_numpy(visual_edges)
        
        n = textual_edges.shape[0]
        
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
        计算 VEGA 迁移性分数（最终版本）
        
        算法流程:
        1. 获取伪标签
        2. 构建文本图
        3. PCA 降维 + 构建视觉图
        4. 计算节点相似度 s_n
        5. 计算边相似度 s_e
        6. 加权融合: s = w_n * s_n + w_e * s_e
        
        Args:
            features: 图像特征 [N, D]
            text_embeddings: 文本嵌入 [K, D]
            logits: 模型预测 [N, K]
            pseudo_labels: 伪标签 [N]
            return_details: 是否返回详细信息
            
        Returns:
            VEGA 分数，或包含详细信息的字典
        """
        total_start = time.time()
        
        progress_print("=" * 60)
        progress_print("开始计算 VEGA 分数 (最终版本)")
        progress_print(f"  PCA 维度: {self.pca_dim}")
        progress_print(f"  节点权重: {self.node_weight}")
        progress_print(f"  边权重: {self.edge_weight}")
        progress_print("=" * 60)
        
        # 转换为 Tensor
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        n_samples, n_features = visual_features.shape
        n_classes = text_embeddings.shape[0]
        
        progress_print(f"数据维度: 样本数={n_samples}, 特征维度={n_features}, 类别数={n_classes}")
        
        # 计算 Cosine Similarity 矩阵
        visual_normalized = self._normalize_features(visual_features)
        text_normalized = self._normalize_features(text_embeddings)
        cosine_similarity = visual_normalized @ text_normalized.T
        
        # 获取伪标签
        if pseudo_labels is None:
            if logits is not None:
                logits_tensor = self._to_tensor(logits)
                pseudo_labels = logits_tensor.argmax(dim=1)
            else:
                pseudo_labels = self.compute_pseudo_labels(cosine_similarity)
        else:
            pseudo_labels = self._to_tensor(pseudo_labels).long()
        
        # 步骤 1: 构建文本图
        textual_nodes, textual_edges = self.build_textual_graph(text_embeddings)
        
        # 步骤 2: 构建视觉图 (PCA + 完整协方差 + Shrinkage)
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_normalized, pseudo_labels, n_classes
        )
        
        # 检查视觉图是否构建成功
        if visual_edges is None or len(class_counts) < 2:
            progress_print("视觉图构建失败，返回默认分数", level="WARNING")
            if return_details:
                return {
                    'score': 0.0,
                    'node_similarity': 0.0,
                    'edge_similarity': 0.0,
                    'pearson_correlation': 0.0,
                    'valid_classes': len(class_counts),
                    'pca_dim': self.pca_dim,
                    'original_dim': n_features,
                    'node_weight': self.node_weight,
                    'edge_weight': self.edge_weight
                }
            return 0.0
        
        # 步骤 3: 计算节点相似度
        node_similarity = self.compute_node_similarity(
            cosine_similarity, pseudo_labels, class_counts
        )
        
        # 步骤 4: 计算边相似度
        edge_similarity, pearson_corr = self.compute_edge_similarity(textual_edges, visual_edges)
        
        # 步骤 5: 加权融合
        # 【核心改进 2】score = w_n * s_n + w_e * s_e
        vega_score = self.node_weight * node_similarity + self.edge_weight * edge_similarity
        
        total_time = time.time() - total_start
        
        progress_print("-" * 60)
        progress_print(f"VEGA 最终分数 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f} (权重 = {self.node_weight})")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f} (权重 = {self.edge_weight})")
        progress_print(f"  加权分数 = {self.node_weight:.1f} * {node_similarity:.4f} + {self.edge_weight:.1f} * {edge_similarity:.4f} = {vega_score:.4f}")
        progress_print(f"  Pearson corr = {pearson_corr:.4f}")
        progress_print(f"总耗时: {total_time:.2f}s")
        progress_print("=" * 60)
        
        if return_details:
            return {
                'score': vega_score,
                'node_similarity': node_similarity,
                'edge_similarity': edge_similarity,
                'pearson_correlation': pearson_corr,
                'valid_classes': len(class_counts),
                'class_counts': class_counts,
                'compute_time': total_time,
                'pca_dim': self.pca_dim,
                'original_dim': n_features,
                'node_weight': self.node_weight,
                'edge_weight': self.edge_weight,
                'full_covariance': True,
                'shrinkage_alpha': self.shrinkage_alpha,
                'vectorized_computation': True
            }
        
        return vega_score


def compute_vega_score_final(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    temperature: float = 0.05,
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    node_weight: float = 1.0,
    edge_weight: float = 0.0
) -> float:
    """
    计算 VEGA 分数的便捷函数（最终版本）
    
    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        logits: 模型预测 [N, K]
        pseudo_labels: 伪标签 [N]
        temperature: 温度参数 (论文默认 0.05)
        pca_dim: PCA 降维维度 (默认 64)
        shrinkage_alpha: Shrinkage 正则化参数 (默认 0.1)
        node_weight: 节点相似度权重 (默认 1.0)
        edge_weight: 边相似度权重 (默认 0.0)
        
    Returns:
        VEGA 分数
    """
    vega = VEGAFinalScorer(
        temperature=temperature,
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        node_weight=node_weight,
        edge_weight=edge_weight
    )
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# 别名
VEGAFinal = VEGAFinalScorer