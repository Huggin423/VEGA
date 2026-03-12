"""
VEGA v3 Version: Visual-Textual Graph Alignment with Paper-Faithful Implementation
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

v3版本 - 严格遵循论文公式，修复 v2 版本的问题

核心修正:

1. 【节点相似度 ($s_n$) - 固定温度缩放】
   - 问题: v2 的实例级 z-score 标准化抹去了绝对对齐幅度
   - 修正: 使用论文中的固定温度 t=0.05
   - 公式 (Eq.12):
     cosine_sim = normalize(visual) @ normalize(text).T   # [N, K]
     probs = softmax(cosine_sim / t, dim=1)               # [N, K]
     s_n = probs[arange(N), pseudo_labels].mean() * (N / K)

2. 【边相似度 ($s_e$) - 负 Bhattacharyya 距离】
   - 问题: v2 使用 exp(-D_B) 引入非线性变换，扭曲 Pearson 相关性
   - 修正: 直接使用负距离 visual_edge = -D_B
   - 优势: 保持距离矩阵的线性结构，与 cosine 相似度方向一致

3. 【保留 v2 的优化】
   - PCA 降维 (pca_dim=64) 解决 O(D³) 维度诅咒
   - Ledoit-Wolf Shrinkage 正则化防止奇异矩阵
   - 向量化 batched 计算

更新日志:
- 2026-03-12: 创建 v3 版本，修复节点相似度和边相似度的计算
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
        print(f"[VEGA-v3] {msg}", flush=True)


class VEGAv3Scorer:
    """
    VEGA v3 版本评分器

    严格遵循论文公式，修复 v2 版本的问题

    框架流程:
    1. 构建文本图 (Textual Graph)
       - 节点: 文本嵌入 (L2 归一化)
       - 边: Cosine 相似度矩阵 [K, K]

    2. 构建视觉图 (Visual Graph)
       - PCA 降维解决维度诅咒
       - Shrinkage 正则化防止奇异矩阵
       - 节点: 类别均值
       - 边: 负 Bhattacharyya 距离矩阵

    3. 计算节点相似度 ($s_n$)
       - 固定温度缩放 (t=0.05)
       - Softmax 获取概率分布
       - s_n = mean(softmax[伪标签]) * (N/K)

    4. 计算边相似度 ($s_e$)
       - Pearson 相关性 (文本边 vs 视觉边)
       - 两者方向一致 (越大越相似)
       - s_e = (corr + 1) / 2

    5. 融合
       - vega_score = node_weight * s_n + edge_weight * s_e

    使用方法:
        vega = VEGAv3Scorer(pca_dim=64, shrinkage_alpha=0.1, temperature=0.05)
        score = vega.compute_score(visual_features, text_embeddings)
    """

    def __init__(
        self,
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        temperature: float = 0.05,
        node_weight: float = 1.0,
        edge_weight: float = 1.0
    ):
        """
        初始化 VEGA v3 版本评分器

        Args:
            min_samples_per_class: 每个类别最少样本数
            pca_dim: PCA 降维后的维度 (默认 64)
            shrinkage_alpha: Shrinkage 正则化参数 α (默认 0.1)
            temperature: Softmax 温度参数 t (默认 0.05)
            node_weight: 节点相似度权重 (默认 1.0)
            edge_weight: 边相似度权重 (默认 1.0)
        """
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.temperature = temperature
        self.node_weight = node_weight
        self.edge_weight = edge_weight

        progress_print(f"初始化 VEGAv3Scorer:")
        progress_print(f"  PCA 维度: {pca_dim}")
        progress_print(f"  Shrinkage alpha: {shrinkage_alpha}")
        progress_print(f"  Temperature: {temperature}")
        progress_print(f"  Node weight: {node_weight}")
        progress_print(f"  Edge weight: {edge_weight}")
        progress_print(f"  节点相似度: 固定温度缩放 (t={temperature})")
        progress_print(f"  边相似度: 负 Bhattacharyya 距离")

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
    # 鲁棒视觉图 (Dimensionality & Stability)
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
        4. 计算负 Bhattacharyya 距离矩阵

        Args:
            visual_features: 视觉特征 [N, D]
            pseudo_labels: 伪标签 [N]
            n_classes: 类别数 K

        Returns:
            (class_means, class_covs, class_counts, edge_matrix) 元组
            edge_matrix: 负 Bhattacharyya 距离矩阵 [K, K]
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

        # 步骤 4: 计算负 Bhattacharyya 距离矩阵
        progress_print("    计算 Bhattacharyya 距离矩阵 (向量化)...")

        edge_matrix = self._compute_negative_bhattacharyya_distance_vectorized(
            class_means, class_covs, valid_classes, n_classes, device, dtype
        )

        elapsed = time.time() - start_time
        progress_print(f"    视觉图构建完成: 边矩阵 {edge_matrix.shape}, 总耗时 {elapsed:.2f}s")

        return class_means, class_covs, class_counts, edge_matrix

    def _compute_negative_bhattacharyya_distance_vectorized(
        self,
        class_means: torch.Tensor,
        class_covs: torch.Tensor,
        valid_classes: List[int],
        n_classes: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        向量化计算负 Bhattacharyya 距离矩阵

        【v3 修正】

        问题 (v2):
        - 使用 exp(-D_B) 将距离转换为系数
        - 这引入非线性变换，扭曲 Pearson 相关性
        - 大距离被压缩到接近 0，降低区分度

        修正 (v3):
        - 直接使用负距离: visual_edge = -D_B
        - 保持距离矩阵的线性结构
        - 与 cosine 相似度方向一致 (越大越相似)
        - Pearson 相关性直接度量拓扑对齐

        数学推导:
        D_B = 1/8 * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2) + 1/2 * ln(|Σ_avg| / sqrt(|Σ_1||Σ_2|))
        visual_edge = -D_B (越大越相似)

        Args:
            class_means: [K_valid, pca_dim]
            class_covs: [K_valid, pca_dim, pca_dim]
            valid_classes: 有效类别列表
            n_classes: 总类别数
            device: 设备
            dtype: 数据类型

        Returns:
            负 Bhattacharyya 距离矩阵 [K, K] (对角线=0.0)
        """
        n_valid = len(valid_classes)
        n_features = class_means.shape[1]

        progress_print(f"      在 {n_features} 维空间计算 Bhattacharyya 距离...")

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

        # 【v3 关键修正】使用负距离而非 exp(-D_B)
        # 这样保持线性结构，Pearson 相关性直接度量对齐
        neg_bh_distance = -bh_distance  # [K_valid, K_valid]

        progress_print(f"      负 Bhattacharyya 距离范围: [{neg_bh_distance.min().item():.4f}, {neg_bh_distance.max().item():.4f}]")

        # 填充到完整矩阵
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        for i_idx, i_class in enumerate(valid_classes):
            for j_idx, j_class in enumerate(valid_classes):
                edge_matrix[i_class, j_class] = neg_bh_distance[i_idx, j_idx]

        return edge_matrix

    # =========================================================================
    # 节点相似度计算 (固定温度缩放)
    # =========================================================================

    def compute_node_similarity(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> float:
        """
        计算节点相似度 (固定温度缩放版本)

        【v3 修正】

        问题 (v2):
        - 实例级 z-score 标准化抹去了绝对对齐幅度
        - 强模型和弱模型变得难以区分
        - 与视觉图使用 PCA 特征不一致

        修正 (v3):
        - 使用固定温度 t=0.05 (论文默认值)
        - 直接应用于原始 cosine 相似度
        - 严格遵循论文 Eq.(12)

        计算:
        1. 计算 cosine_similarity = normalize(visual) @ normalize(text).T
        2. Softmax: probs = softmax(cosine_sim / t, dim=1)
        3. s_n = probs[arange(N), pseudo_labels].mean() * (N / K)

        注意 N/K 因子:
        - 论文 Eq.(11): s_n = (1/K) * sum_i softmax_i
        - 等价于: s_n = mean_i(softmax_i) * (N/K)
        - 不是简单的样本均值!

        Args:
            visual_features: 视觉特征 [N, D]
            text_embeddings: 文本嵌入 [K, D]
            pseudo_labels: 伪标签 [N]

        Returns:
            节点相似度分数
        """
        progress_print("  计算节点相似度 (固定温度缩放)...")

        n_samples = visual_features.shape[0]
        n_classes = text_embeddings.shape[0]

        # 步骤 1: 计算 Cosine 相似度
        visual_normalized = self._normalize_features(visual_features)
        text_normalized = self._normalize_features(text_embeddings)
        cosine_similarity = visual_normalized @ text_normalized.T  # [N, K]

        progress_print(f"    Cosine 相似度范围: [{cosine_similarity.min().item():.4f}, {cosine_similarity.max().item():.4f}]")

        # 步骤 2: Softmax (固定温度)
        # 论文 Eq.(12): probs = softmax(cosine_sim / t, dim=1)
        probs = F.softmax(cosine_similarity / self.temperature, dim=1)  # [N, K]

        progress_print(f"    Softmax 概率范围: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

        # 步骤 3: 计算 s_n
        # 论文 Eq.(11): s_n = (1/K) * sum_i softmax_i
        # 等价于: s_n = mean_i(softmax_i) * (N/K)
        softmax_at_pseudo_labels = probs[torch.arange(n_samples), pseudo_labels]
        s_n = softmax_at_pseudo_labels.mean().item() * (n_samples / n_classes)

        progress_print(f"    固定温度 (t={self.temperature}) Node Similarity = {s_n:.4f}")
        progress_print(f"    (mean softmax = {softmax_at_pseudo_labels.mean().item():.4f}, N/K = {n_samples}/{n_classes} = {n_samples/n_classes:.2f})")

        return s_n

    # =========================================================================
    # 边相似度计算
    # =========================================================================

    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算边相似度

        【v3 修正】

        现在:
        - textual_edges: Cosine 相似度 [K, K] (越大越相似)
        - visual_edges: 负 Bhattacharyya 距离 [K, K] (越大越相似)

        两者方向一致，Pearson 相关性直接度量拓扑对齐!

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
        计算 VEGA 迁移性分数（v3 版本）

        严格遵循 VEGA 论文:
        1. 构建文本图 (Textual Graph)
        2. 构建视觉图 (Visual Graph) - PCA + Shrinkage
        3. 计算节点相似度 ($s_n$) - 固定温度缩放
        4. 计算边相似度 ($s_e$) - 负 Bhattacharyya 距离
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
        progress_print("开始计算 VEGA 分数 (v3 版本)")
        progress_print(f"  PCA 维度: {self.pca_dim}")
        progress_print(f"  Shrinkage alpha: {self.shrinkage_alpha}")
        progress_print(f"  Temperature: {self.temperature}")
        progress_print(f"  Node weight: {self.node_weight}")
        progress_print(f"  Edge weight: {self.edge_weight}")
        progress_print("  节点相似度: 固定温度缩放 (t={self.temperature})")
        progress_print("  边相似度: 负 Bhattacharyya 距离")
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
        # 步骤 2: 构建视觉图 (PCA + Shrinkage + 负 Bhattacharyya 距离)
        # =====================================
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_features, pseudo_labels, n_classes
        )

        # =====================================
        # 步骤 3: 计算节点相似度 (固定温度缩放)
        # =====================================
        node_similarity = self.compute_node_similarity(
            visual_features, text_embeddings, pseudo_labels
        )

        # =====================================
        # 步骤 4: 计算边相似度 (负 Bhattacharyya 距离)
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
                'temperature': self.temperature,
                'node_weight': self.node_weight,
                'edge_weight': self.edge_weight,
                'negative_bhattacharyya_distance': True,
                'fixed_temperature': True,
                'optimization_version': 'VEGAv3Scorer'
            }

        return vega_score


def compute_vega_v3_score(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    temperature: float = 0.05,
    node_weight: float = 1.0,
    edge_weight: float = 1.0
) -> float:
    """
    计算 VEGA v3 分数的便捷函数

    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        pseudo_labels: 伪标签 [N]
        pca_dim: PCA 降维维度 (默认 64)
        shrinkage_alpha: Shrinkage 正则化参数 (默认 0.1)
        temperature: Softmax 温度参数 (默认 0.05)
        node_weight: 节点相似度权重 (默认 1.0)
        edge_weight: 边相似度权重 (默认 1.0)

    Returns:
        VEGA 分数
    """
    vega = VEGAv3Scorer(
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        temperature=temperature,
        node_weight=node_weight,
        edge_weight=edge_weight
    )
    return vega.compute_score(features, text_embeddings, pseudo_labels)


# 别名
VEGAv3 = VEGAv3Scorer