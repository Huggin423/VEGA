"""
VEGA Perfect Version: Visual-Textual Graph Alignment with Corrected Softmax Temperature
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

完美版本 - 修复 Softmax 温度问题

核心修复 (相对于 VEGAUltimateScorer):

1. 【修复 s_n Softmax 温度】
   - **诊断发现**: 原论文建议的温度 t=0.05 对 VLM 物理上不正确
   - **原因**: 标准 CLIP 模型使用 logit_scale ≈ 100.0，对应温度 t=0.01
   - **问题**: 温度 t=0.05 太"热"，会将分布压平，导致 s_n ≈ 1/K = 0.01
   - **修复**: 使用正确的温度 t=0.01
   - 计算: probs = softmax(cosine_similarity / 0.01, dim=1)
   - s_n = mean(probs[range(N), pseudo_labels])

2. 【保留 s_e 修复】
   - exp(-bh_distance) 修复非常成功
   - Pearson 相关系数正确显示正值（好模型）和 0（坏模型）
   - 保持不变！

3. 【保留 PCA 和 Shrinkage】
   - 运行完美且快速
   - 保持不变！

更新日志:
- 2026-03-11: 创建完美版本，修复 Softmax 温度问题
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
        print(f"[VEGA-Perfect] {msg}", flush=True)


def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA-Perfect] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA-Perfect] {msg}", flush=True)


class VEGAPerfectScorer:
    """
    VEGA 完美版本评分器
    
    核心修复:
    
    1. **修复 s_n Softmax 温度**
       - 问题: 原论文 t=0.05 太"热"，导致 s_n ≈ 1/K
       - 原因: CLIP 的 logit_scale ≈ 100 对应 t=0.01
       - 修复: 使用正确温度 t=0.01
       - s_n = mean(softmax(cos/t) at pseudo_labels)
    
    2. **保留 s_e 修复 (exp(-bh_distance))**
       - Bhattacharyya 系数作为相似度度量
       - Pearson 相关正确显示正值（好模型）
    
    3. **保留 PCA 和 Shrinkage**
       - 解决维度诅咒，保证协方差正定
    
    使用方法:
        vega = VEGAPerfectScorer(pca_dim=64, shrinkage_alpha=0.1, temperature=0.01)
        score = vega.compute_score(visual_features, text_embeddings, logits)
    """
    
    def __init__(
        self, 
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        temperature: float = 0.01  # 【关键修复】默认温度从 0.05 改为 0.01
    ):
        """
        初始化 VEGA 完美版本评分器
        
        Args:
            min_samples_per_class: 每个类别最少样本数
            pca_dim: PCA 降维后的维度 (默认 64)
            shrinkage_alpha: Shrinkage 正则化参数 α (默认 0.1)
            temperature: Softmax 温度参数 (默认 0.01，匹配 CLIP logit_scale)
        """
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.temperature = temperature  # 修复后的温度
        
        progress_print(f"初始化 VEGAPerfectScorer:")
        progress_print(f"  PCA 维度: {pca_dim}")
        progress_print(f"  Shrinkage alpha: {shrinkage_alpha}")
        progress_print(f"  Softmax 温度: {temperature} (CLIP logit_scale 对应值)")
    
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
        
        保留自 VEGAUltimateScorer - 运行完美
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
    
    def compute_pseudo_labels(
        self, 
        cosine_similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        从 cosine similarity 矩阵计算伪标签
        
        论文公式(1): ŷ_i = argmax_k cos(ξ(c̃_k), φ(x_i))
        """
        return cosine_similarity.argmax(dim=1)
    
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
        
        保留自 VEGAUltimateScorer - 运行完美
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
        
        【保留自 VEGAUltimateScorer - 非常成功】
        
        数学推导:
        D_B = 1/8 * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2) + 1/2 * ln(|Σ_avg| / sqrt(|Σ_1||Σ_2|))
        BC = exp(-D_B) ∈ [0, 1]
        
        BC 高 = 两个分布相似 = 好
        与 Cosine 相似度一致！
        
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
    
    def compute_node_similarity(
        self,
        cosine_similarity: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> float:
        """
        计算节点相似度 (Softmax 版本，修复温度)
        
        【关键修复: Softmax 温度】
        
        原论文问题:
        - 使用固定温度 t=0.05
        - 这对 VLM 物理上不正确
        - CLIP 的 logit_scale ≈ 100，对应温度 t=0.01
        - 温度 0.05 太"热"，会将分布压平
        - 导致 s_n ≈ 1/K = 0.01
        
        修复:
        - 使用正确温度 t=0.01 (默认值)
        - probs = softmax(cosine_similarity / temperature, dim=1)
        - s_n = mean(probs[i, pseudo_label[i]])
        - 这测量的是: 平均而言，模型对其预测类别的置信度
        
        Args:
            cosine_similarity: Cosine similarity 矩阵 [N, K]
            pseudo_labels: 伪标签 [N]
            
        Returns:
            节点相似度分数 [范围 (0, 1)]
        """
        progress_print("  计算节点相似度 (Softmax 版本，温度=0.01)...")
        
        n_samples = cosine_similarity.shape[0]
        
        # 【关键修复】使用正确的温度 0.01
        # probs = softmax(cos / t, dim=1)
        probs = F.softmax(cosine_similarity / self.temperature, dim=1)
        
        # s_n = mean(probs[i, pseudo_label[i]])
        # 这是模型对伪标签的平均置信度
        node_similarity = probs[torch.arange(n_samples), pseudo_labels].mean().item()
        
        progress_print(f"    Softmax 温度: {self.temperature}")
        progress_print(f"    节点相似度 = {node_similarity:.4f} (平均最大 Softmax 概率)")
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算边相似度
        
        【保留自 VEGAUltimateScorer - 非常成功】
        
        两个矩阵都是相似度:
        - textual_edges: Cosine 相似度 [K, K]
        - visual_edges: Bhattacharyya 系数 [K, K]
        
        Pearson 相关系数为正（好模型有正相关）
        """
        progress_print("  计算边相似度 (相似度 vs 相似度)...")
        
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
        计算 VEGA 迁移性分数（完美版本）
        
        算法流程:
        1. 计算 Cosine 相似度矩阵
        2. 获取伪标签
        3. 构建文本图 (Cosine 相似度)
        4. 构建视觉图 (PCA + Bhattacharyya 系数)
        5. 计算 s_n = mean(softmax(cos/t) at pseudo_labels)  # 修复温度
        6. 计算 s_e = (Pearson + 1) / 2
        7. 返回 s = s_n + s_e
        
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
        progress_print("开始计算 VEGA 分数 (完美版本)")
        progress_print(f"  PCA 维度: {self.pca_dim}")
        progress_print(f"  Shrinkage alpha: {self.shrinkage_alpha}")
        progress_print(f"  Softmax 温度: {self.temperature} (修复: 0.05 -> 0.01)")
        progress_print("  s_n = mean(softmax(cos/t) at pseudo_labels)")
        progress_print("  s_e = (Pearson + 1) / 2 (exp(-bh_distance) 修复)")
        progress_print("  s = s_n + s_e")
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
        
        # 步骤 2: 构建视觉图
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
                    'temperature': self.temperature
                }
            return 0.0
        
        # 步骤 3: 计算节点相似度 (Softmax 版本，修复温度)
        node_similarity = self.compute_node_similarity(cosine_similarity, pseudo_labels)
        
        # 步骤 4: 计算边相似度 (保留 exp(-bh_distance) 修复)
        edge_similarity, pearson_corr = self.compute_edge_similarity(textual_edges, visual_edges)
        
        # 步骤 5: 融合分数
        vega_score = node_similarity + edge_similarity
        
        total_time = time.time() - total_start
        
        progress_print("-" * 60)
        progress_print(f"VEGA 最终分数 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f}")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f}")
        progress_print(f"  s = s_n + s_e")
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
                'full_covariance': True,
                'shrinkage_alpha': self.shrinkage_alpha,
                'temperature': self.temperature,
                'bhattacharyya_coefficient': True,
                'softmax_temperature_fixed': True
            }
        
        return vega_score


def compute_vega_score_perfect(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    temperature: float = 0.01
) -> float:
    """
    计算 VEGA 分数的便捷函数（完美版本）
    
    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        logits: 模型预测 [N, K]
        pseudo_labels: 伪标签 [N]
        pca_dim: PCA 降维维度 (默认 64)
        shrinkage_alpha: Shrinkage 正则化参数 (默认 0.1)
        temperature: Softmax 温度 (默认 0.01，CLIP 对应值)
        
    Returns:
        VEGA 分数
    """
    vega = VEGAPerfectScorer(
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        temperature=temperature
    )
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# 别名
VEGAPerfect = VEGAPerfectScorer