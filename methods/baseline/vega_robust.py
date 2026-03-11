"""
VEGA Robust Version: Visual-Textual Graph Alignment with Full Covariance and Shrinkage Regularization
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

鲁棒版本 - 使用完整协方差矩阵 + 收缩正则化，解决高维特征下的奇异性问题

核心改进:
1. 完整协方差矩阵 + Shrinkage 正则化 - 保证协方差矩阵严格正定，可逆
2. 向量化 Bhattacharyya 距离计算 - 使用 PyTorch 批量矩阵运算，无 for 循环
3. 正确的 Node Similarity - 使用 N_total 归一化，自然落在 [0,1]
4. Edge Similarity - 上三角元素 Pearson 相关系数，NaN 回退到 0.5

论文公式参考:
1. 文本图节点: n^T_k = ξ(c̃_k)  (文本特征)
2. 文本图边: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j))  (cosine相似度)
3. 视觉图节点: 高斯分布 N(μ_k, Σ_k)
   - 均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
   - 完整协方差: Σ_k = cov(φ(x_i)) + Shrinkage Regularization
   - Shrinkage: Σ_shrunk = (1-α)Σ + α·T, T = trace(Σ)/D · I
4. 视觉图边: Bhattacharyya 距离 (完整协方差版本)
   - D_B = (1/8)(μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2) + (1/2)ln|Σ_avg| - (1/4)ln|Σ_1| - (1/4)ln|Σ_2|
   - 其中 Σ_avg = (Σ_1 + Σ_2)/2
5. 节点相似度: s_n = Σ_k sim(n^T_k, n^V_k) · N_k / N_total
6. 边相似度: s_e = (corr(E^T, E^V) + 1)/2
7. 最终分数: s = s_n + s_e

更新日志:
- 2026-03-11: 创建鲁棒版本，实现完整协方差 + Shrinkage 正则化
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
        print(f"[VEGA-Robust] {msg}", flush=True)


def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA-Robust] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA-Robust] {msg}", flush=True)


class VEGARobustScorer:
    """
    VEGA 鲁棒版本评分器
    
    核心改进 (相对于对角协方差版本):
    
    1. **完整协方差矩阵 + Shrinkage 正则化**
       - 问题: VLM 特征维度 D=512/768，样本数 N_k << D，样本协方差矩阵必然 rank-deficient
       - 解决: 使用 Shrinkage 正则化保证协方差矩阵严格正定
       - 公式: Σ_shrunk = (1-α)Σ + α·T
         - α = 0.1 (收缩超参数)
         - T = trace(Σ)/D · I (目标对角矩阵)
       - 当 N_k = 1 时，回退到单位矩阵
    
    2. **向量化 Bhattacharyya 距离计算 (关键!)**
       - 使用 PyTorch 批量矩阵运算，消除双重 for 循环
       - 堆叠均值 [K, D] 和协方差 [K, D, D]
       - 利用广播机制计算 K×K 距离矩阵
       - 使用 torch.linalg.inv() 批量求逆
       - 使用 torch.linalg.slogdet() 计算 log-determinant，避免溢出
    
    3. **正确的 Node Similarity**
       - s_n = Σ_k sim_k · N_k / N_total
       - 自然落在 [0, 1] 范围，无需 clip
    
    4. **Edge Similarity**
       - 上三角元素 Pearson 相关系数
       - NaN 时回退到 0.5
    
    使用方法:
        vega = VEGARobustScorer(temperature=0.05, shrinkage_alpha=0.1)
        score = vega.compute_score(visual_features, text_embeddings, logits)
    """
    
    def __init__(
        self, 
        temperature: float = 0.05, 
        min_samples_per_class: int = 2,
        shrinkage_alpha: float = 0.1
    ):
        """
        初始化 VEGA 鲁棒版本评分器
        
        Args:
            temperature: softmax 归一化的温度参数 (论文默认 t=0.05)
            min_samples_per_class: 每个类别最少样本数，用于有效的方差估计
            shrinkage_alpha: Shrinkage 正则化参数 α (默认 0.1)
                           Σ_shrunk = (1-α)Σ + α·T
        """
        self.temperature = temperature
        self.min_samples_per_class = min_samples_per_class
        self.shrinkage_alpha = shrinkage_alpha
    
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
        # 论文公式: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j))
        edges = nodes @ nodes.T  # [K, K]
        
        progress_print(f"    文本图节点: {nodes.shape}, 边矩阵: {edges.shape}")
        
        return nodes, edges
    
    def _compute_shrunk_covariance(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Shrinkage 正则化后的完整协方差矩阵
        
        【核心改进 1: Shrinkage Regularization】
        
        当样本数 N_k < 特征维度 D 时，样本协方差矩阵是 rank-deficient 的，
        不可逆且行列式为 0。Shrinkage 正则化通过向对角目标矩阵收缩来保证正定性。
        
        公式:
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  Σ_shrunk = (1 - α) · Σ + α · T                                             │
        │                                                                              │
        │  其中:                                                                        │
        │  - Σ: 样本协方差矩阵 [D, D]                                                   │
        │  - α: 收缩参数 (默认 0.1)                                                     │
        │  - T = trace(Σ)/D · I: 目标对角矩阵                                          │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        理论保证:
        - T 是正定的 (对角元素 > 0)
        - Σ_shrunk 是 Σ 和 T 的凸组合
        - 当 α > 0 时，Σ_shrunk 严格正定，可逆
        
        Args:
            features: 类别特征 [N_k, D]
            
        Returns:
            shrunk_covariance: Shrinkage 正则化后的协方差矩阵 [D, D]
        """
        n_samples, n_features = features.shape
        device = features.device
        dtype = features.dtype
        
        if n_samples < 2:
            # 单样本: 回退到单位矩阵
            return torch.eye(n_features, device=device, dtype=dtype)
        
        # 计算样本协方差矩阵
        # 使用有偏估计 (unbiased=False) 更稳定
        # cov 返回 [D, D]
        cov = torch.cov(features.T, correction=0)  # correction=0 表示有偏估计
        
        # Shrinkage 正则化
        alpha = self.shrinkage_alpha
        
        # 目标矩阵 T = trace(Σ)/D · I
        trace_cov = torch.trace(cov)
        target_diag = trace_cov / n_features
        T = target_diag * torch.eye(n_features, device=device, dtype=dtype)
        
        # Σ_shrunk = (1-α)Σ + αT
        shrunk_cov = (1 - alpha) * cov + alpha * T
        
        return shrunk_cov
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], torch.Tensor]:
        """
        构建视觉图（使用完整协方差 + Shrinkage 正则化）
        
        【核心改进 1: 完整协方差矩阵】
        相对于对角协方差近似，完整协方差保留了特征维度之间的相关性，
        更准确地建模高维 VLM 特征的分布。
        
        论文公式:
        - 节点: 高斯分布 N(μ_k, Σ_k)
          - 均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
          - 协方差: Σ_k = Shrinkage(cov(φ(x_i)))
        - 边: Bhattacharyya 距离 (完整协方差版本)
        
        Args:
            visual_features: 图像特征 [N, D]
            pseudo_labels: 伪标签 [N]
            n_classes: 类别数 K
            
        Returns:
            (class_means, class_covs, class_counts, edge_matrix) 元组
            - class_means: 类别均值矩阵 [K_valid, D]
            - class_covs: 类别协方差张量 [K_valid, D, D]
            - class_counts: 类别样本数字典 {class_idx: count}
            - edge_matrix: Bhattacharyya 距离矩阵 [K, K]
        """
        progress_print("  构建视觉图 (完整协方差 + Shrinkage 正则化)...")
        start_time = time.time()
        
        # L2 归一化特征
        visual_features = self._normalize_features(visual_features)
        device = visual_features.device
        dtype = visual_features.dtype
        n_samples, n_features = visual_features.shape
        
        progress_print(f"    样本数: {n_samples}, 特征维度: {n_features}, 类别数: {n_classes}")
        
        # ============================================================
        # 步骤 1: 计算每个类别的统计量（均值和完整协方差）
        # ============================================================
        class_means_list = []
        class_covs_list = []
        class_counts = {}
        valid_classes = []
        
        for k in range(n_classes):
            # 获取分配到类别 k 的样本
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                progress_print(f"    类别 {k}: 样本数 {len(indices)} < {self.min_samples_per_class}, 跳过")
                continue
            
            class_features = visual_features[indices]  # [N_k, D]
            class_counts[k] = len(indices)
            valid_classes.append(k)
            
            # 计算均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
            class_mean = class_features.mean(dim=0)  # [D]
            class_means_list.append(class_mean)
            
            # 【核心改进 1】计算 Shrinkage 正则化后的完整协方差
            class_cov = self._compute_shrunk_covariance(class_features)  # [D, D]
            class_covs_list.append(class_cov)
        
        n_valid = len(valid_classes)
        
        progress_print(f"    有效类别数: {n_valid}")
        
        if n_valid < 2:
            logger.warning("有效类别数不足，无法构建视觉图")
            return None, None, class_counts, None
        
        # 堆叠成张量
        # class_means: [K_valid, D]
        # class_covs: [K_valid, D, D]
        class_means = torch.stack(class_means_list, dim=0)
        class_covs = torch.stack(class_covs_list, dim=0)
        
        progress_print(f"    均值矩阵: {class_means.shape}, 协方差张量: {class_covs.shape}")
        
        # ============================================================
        # 步骤 2: 向量化计算 Bhattacharyya 距离矩阵
        # 【核心改进 2: 完全向量化，无 for 循环】
        # ============================================================
        progress_print("    计算 Bhattacharyya 距离矩阵 (向量化)...")
        
        edge_matrix = self._compute_bhattacharyya_distance_vectorized(
            class_means, class_covs, valid_classes, n_classes, device, dtype
        )
        
        elapsed = time.time() - start_time
        progress_print(f"    视觉图构建完成: 边矩阵 {edge_matrix.shape}, 耗时 {elapsed:.2f}s")
        
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
        向量化计算 Bhattacharyya 距离矩阵 (完整协方差版本)
        
        【核心改进 2: 完全向量化的 Bhattacharyya 距离】
        
        Bhattacharyya 距离用于衡量两个高斯分布之间的相似度。
        对于两个高斯分布 N(μ_1, Σ_1) 和 N(μ_2, Σ_2):
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  D_B = Term1 + Term2                                                         │
        │                                                                              │
        │  Term1 (马氏距离项):                                                          │
        │    = (1/8) · (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2)                           │
        │                                                                              │
        │  Term2 (Log-Determinant 项):                                                 │
        │    = (1/2) · ln|Σ_avg| - (1/4) · ln|Σ_1| - (1/4) · ln|Σ_2|                  │
        │                                                                              │
        │  其中 Σ_avg = (Σ_1 + Σ_2) / 2                                               │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        向量化实现关键:
        1. 堆叠所有类别的均值 [K, D] 和协方差 [K, D, D]
        2. 利用广播机制计算所有 K×K 组合
        3. 使用 torch.linalg.inv() 批量求逆
        4. 使用 torch.linalg.slogdet() 计算 log-determinant，避免数值溢出
        
        张量形状说明:
        - means: [K, D]
        - covs: [K, D, D]
        - mu_diff: [K, K, D, 1] - 均值差异
        - sigma_avg: [K, K, D, D] - 平均协方差
        - sigma_avg_inv: [K, K, D, D] - 逆矩阵
        - term1: [K, K] - 马氏距离项
        - term2: [K, K] - Log-det 项
        
        Args:
            class_means: 类别均值矩阵 [K_valid, D]
            class_covs: 类别协方差张量 [K_valid, D, D]
            valid_classes: 有效类别列表
            n_classes: 总类别数 K
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            Bhattacharyya 距离矩阵 [K, K]
        """
        n_valid = len(valid_classes)
        n_features = class_means.shape[1]
        
        # ============================================================
        # 步骤 1: 准备广播所需的张量
        # ============================================================
        
        # means: [K, D] -> 扩展为 [K, 1, D, 1] 和 [1, K, D, 1]
        # 用于计算所有类别对的均值差异
        means_i = class_means.unsqueeze(1).unsqueeze(-1)  # [K, 1, D, 1]
        means_j = class_means.unsqueeze(0).unsqueeze(-1)  # [1, K, D, 1]
        
        # covs: [K, D, D] -> 扩展为 [K, 1, D, D] 和 [1, K, D, D]
        # 用于计算所有类别对的平均协方差
        covs_i = class_covs.unsqueeze(1)  # [K, 1, D, D]
        covs_j = class_covs.unsqueeze(0)  # [1, K, D, D]
        
        # ============================================================
        # 步骤 2: 计算 Σ_avg = (Σ_1 + Σ_2) / 2
        # ============================================================
        # sigma_avg: [K, K, D, D]
        sigma_avg = (covs_i + covs_j) / 2  # [K, K, D, D]
        
        # ============================================================
        # 步骤 3: 计算 Term1 = (1/8) * (μ_1 - μ_2)^T Σ_avg^{-1} (μ_1 - μ_2)
        # ============================================================
        progress_print("      计算 Term1 (马氏距离项)...")
        
        # mu_diff: [K, K, D, 1]
        mu_diff = means_i - means_j  # [K, K, D, 1]
        
        # 批量矩阵求逆: sigma_avg_inv: [K, K, D, D]
        # 使用 torch.linalg.inv 进行批量求逆
        sigma_avg_inv = torch.linalg.inv(sigma_avg)  # [K, K, D, D]
        
        # 计算 Σ_avg^{-1} @ (μ_1 - μ_2)
        # [K, K, D, D] @ [K, K, D, 1] -> [K, K, D, 1]
        quad_form = sigma_avg_inv @ mu_diff  # [K, K, D, 1]
        
        # 计算 (μ_1 - μ_2)^T @ (Σ_avg^{-1} @ (μ_1 - μ_2))
        # [K, K, 1, D] @ [K, K, D, 1] -> [K, K, 1, 1] -> squeeze -> [K, K]
        term1 = 0.125 * (mu_diff.transpose(-2, -1) @ quad_form).squeeze(-1).squeeze(-1)  # [K, K]
        
        # ============================================================
        # 步骤 4: 计算 Term2 = (1/2)ln|Σ_avg| - (1/4)ln|Σ_1| - (1/4)ln|Σ_2|
        # 使用 slogdet 避免数值溢出
        # ============================================================
        progress_print("      计算 Term2 (Log-Determinant 项)...")
        
        # slogdet 返回 (sign, logabsdet)
        # sign: 行列式的符号 (+1 或 -1)
        # logabsdet: log(|det|)
        # 对于正定矩阵，sign 应该是 +1
        
        # log|Σ_avg|: [K, K]
        sign_avg, logdet_avg = torch.linalg.slogdet(sigma_avg)  # sign: [K, K], logdet: [K, K]
        # 对于正定矩阵，sign 应该接近 1，这里我们假设 Shrinkage 保证了正定性
        
        # log|Σ_i| 和 log|Σ_j|: [K]
        sign_i, logdet_i = torch.linalg.slogdet(class_covs)  # sign: [K], logdet: [K]
        sign_j, logdet_j = torch.linalg.slogdet(class_covs)  # sign: [K], logdet: [K]
        
        # 广播 logdet 以计算 Term2
        # logdet_avg: [K, K]
        # logdet_i.unsqueeze(1): [K, 1] -> 广播到 [K, K]
        # logdet_j.unsqueeze(0): [1, K] -> 广播到 [K, K]
        
        # Term2 = 0.5 * log|Σ_avg| - 0.25 * log|Σ_i| - 0.25 * log|Σ_j|
        term2 = 0.5 * logdet_avg - 0.25 * logdet_i.unsqueeze(1) - 0.25 * logdet_j.unsqueeze(0)  # [K, K]
        
        # ============================================================
        # 步骤 5: 合并 Term1 和 Term2，得到 Bhattacharyya 距离
        # ============================================================
        bh_distance = term1 + term2  # [K, K]
        
        # 确保对角线为 0（同一类别的距离为 0）
        bh_distance = bh_distance * (1 - torch.eye(n_valid, device=device, dtype=dtype))
        
        # 确保距离非负（理论上应为非负，但数值误差可能导致小的负值）
        bh_distance = torch.clamp(bh_distance, min=0.0)
        
        # ============================================================
        # 步骤 6: 将距离填充到完整的 [K, K] 矩阵中
        # ============================================================
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        
        # 使用向量化索引填充
        valid_indices = torch.tensor(valid_classes, device=device)
        row_idx = valid_indices.unsqueeze(1).expand(n_valid, n_valid)
        col_idx = valid_indices.unsqueeze(0).expand(n_valid, n_valid)
        
        # 排除对角线
        mask = row_idx != col_idx
        edge_matrix[row_idx[mask], col_idx[mask]] = bh_distance[~torch.eye(n_valid, device=device, dtype=torch.bool)]
        
        # 上面的一行代码有问题，重新用正确的方式填充
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
        
        【约束 3: 正确的 Node Similarity】
        
        s_n = Σ_k sim(n^T_k, n^V_k) · N_k / N_total
        
        其中 sim(n^T_k, n^V_k) 是类别 k 的视觉特征与文本特征的平均匹配概率。
        
        这个公式是所有样本在其分配类别上 softmax 概率的加权平均，
        自然落在 [0, 1] 区间，无需 np.clip。
        
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
        probs = F.softmax(scaled_similarity, dim=1)  # [N, K]
        
        # 计算每个类别的相似度分数
        class_similarities = {}
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # 获取每个样本在其分配类别上的概率
            class_probs = probs[indices, k]
            
            # 该类别的平均相似度
            class_similarities[k] = class_probs.mean().item()
        
        if not class_similarities:
            return 0.0
        
        # ============================================================
        # 【约束 3】计算加权和，除以总样本数 N_total
        # ============================================================
        total_weighted_sim = 0.0
        total_samples = 0  # N_total
        
        for k, sim in class_similarities.items():
            N_k = class_counts.get(k, 1)
            total_weighted_sim += sim * N_k
            total_samples += N_k
        
        # s_n = Σ_k sim_k · N_k / N_total
        if total_samples > 0:
            node_similarity = total_weighted_sim / total_samples
        else:
            node_similarity = 0.0
        
        # 无需 clip，自然落在 [0, 1]
        
        n_valid_classes = len(class_similarities)
        progress_print(f"    节点相似度 = {node_similarity:.4f} (有效类别: {n_valid_classes}/{n_classes})")
        progress_print(f"    加权和 = {total_weighted_sim:.2f}, 总样本数 N_total = {total_samples}")
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算边相似度（使用 Pearson 相关系数）
        
        【约束 4: Edge Similarity】
        
        1. 提取文本边矩阵和视觉边矩阵的上三角元素
        2. 计算 Pearson 相关系数
        3. 如果 NaN，回退到 0.5
        4. s_e = (corr + 1) / 2
        
        Args:
            textual_edges: 文本图边矩阵 [K, K]
            visual_edges: 视觉图边矩阵 [K, K]
            
        Returns:
            (edge_similarity, pearson_correlation) 元组
        """
        progress_print("  计算边相似度...")
        
        # 转换为 numpy 数组
        textual_edges = self._to_numpy(textual_edges)
        visual_edges = self._to_numpy(visual_edges)
        
        n = textual_edges.shape[0]
        
        # 提取上三角元素（排除对角线）
        triu_indices = np.triu_indices(n, k=1)
        
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            progress_print("    边数不足，返回默认值 0.5", level="WARNING")
            return 0.5, 0.0
        
        # 计算 Pearson 相关系数
        try:
            corr, _ = pearsonr(textual_vec, visual_vec)
            
            # 【约束 4】处理 NaN 情况
            if np.isnan(corr):
                progress_print("    Pearson 相关系数为 NaN，返回默认值 0.5", level="WARNING")
                return 0.5, 0.0
            
            # s_e = (corr + 1) / 2
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
        计算 VEGA 迁移性分数（鲁棒版本）
        
        算法流程:
        1. 获取伪标签: ŷ_i = argmax_k cos(ξ(c̃_k), φ(x_i))
        2. 构建文本图: 节点=文本特征, 边=cosine相似度
        3. 构建视觉图: 节点=高斯分布(完整协方差+Shrinkage), 边=Bhattacharyya距离
        4. 计算节点相似度: s_n = Σ_k sim_k · N_k / N_total
        5. 计算边相似度: s_e = (corr + 1)/2
        6. 最终分数: s = s_n + s_e
        
        Args:
            features: 图像特征 [N, D]
            text_embeddings: 文本嵌入 [K, D]
            logits: 模型预测 [N, K] (可选，用于获取伪标签)
            pseudo_labels: 伪标签 [N] (可选，若未提供则从特征计算)
            return_details: 是否返回详细信息
            
        Returns:
            VEGA 分数 (float)，或包含详细信息的字典
            详细信息包含:
            - score: 最终分数
            - node_similarity: 节点相似度
            - edge_similarity: 边相似度
            - pearson_correlation: Pearson 相关系数
        """
        total_start = time.time()
        
        progress_print("=" * 60)
        progress_print("开始计算 VEGA 分数 (鲁棒版本)")
        progress_print("=" * 60)
        
        # 转换为 Tensor
        progress_print("转换数据格式...")
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        n_samples, n_features = visual_features.shape
        n_classes = text_embeddings.shape[0]
        
        progress_print(f"数据维度: 样本数={n_samples}, 特征维度={n_features}, 类别数={n_classes}")
        
        # 计算 Cosine Similarity 矩阵
        cosine_start = time.time()
        progress_print("计算 Cosine Similarity 矩阵...")
        
        visual_normalized = self._normalize_features(visual_features)
        text_normalized = self._normalize_features(text_embeddings)
        
        # Cosine similarity: [N, K]
        cosine_similarity = visual_normalized @ text_normalized.T
        
        timing_print(f"  Cosine Similarity 矩阵: {cosine_similarity.shape}", cosine_start)
        
        # 如果未提供伪标签，则计算伪标签
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
        
        # 步骤 1: 构建文本图
        step1_start = time.time()
        progress_print("【步骤 1/4】构建文本图...")
        textual_nodes, textual_edges = self.build_textual_graph(text_embeddings)
        timing_print(f"  文本图构建完成: 边矩阵 {textual_edges.shape}", step1_start)
        
        # 步骤 2: 构建视觉图（完整协方差 + Shrinkage）
        step2_start = time.time()
        progress_print("【步骤 2/4】构建视觉图 (完整协方差 + Shrinkage)...")
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_normalized, pseudo_labels, n_classes
        )
        timing_print(f"  视觉图构建完成: 有效类别数={len(class_counts)}", step2_start)
        
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
                    'full_covariance': True,
                    'shrinkage_alpha': self.shrinkage_alpha,
                    'vectorized_computation': True
                }
            return 0.0
        
        # 步骤 3: 计算节点相似度
        step3_start = time.time()
        progress_print("【步骤 3/4】计算节点相似度...")
        node_similarity = self.compute_node_similarity(
            cosine_similarity, pseudo_labels, class_counts
        )
        timing_print(f"  节点相似度 = {node_similarity:.4f}", step3_start)
        
        # 步骤 4: 计算边相似度
        step4_start = time.time()
        progress_print("【步骤 4/4】计算边相似度...")
        edge_similarity, pearson_corr = self.compute_edge_similarity(textual_edges, visual_edges)
        timing_print(f"  边相似度 = {edge_similarity:.4f}", step4_start)
        
        # 步骤 5: 最终 VEGA 分数 s = s_n + s_e
        vega_score = node_similarity + edge_similarity
        
        total_time = time.time() - total_start
        progress_print("-" * 60)
        progress_print(f"VEGA 总分 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f}")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f}")
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
                'full_covariance': True,
                'shrinkage_alpha': self.shrinkage_alpha,
                'vectorized_computation': True
            }
        
        return vega_score


def compute_vega_score_robust(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    temperature: float = 0.05,
    shrinkage_alpha: float = 0.1
) -> float:
    """
    计算 VEGA 分数的便捷函数（鲁棒版本）
    
    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        logits: 模型预测 [N, K]
        pseudo_labels: 伪标签 [N]
        temperature: 温度参数 (论文默认 0.05)
        shrinkage_alpha: Shrinkage 正则化参数 (默认 0.1)
        
    Returns:
        VEGA 分数
    """
    vega = VEGARobustScorer(temperature=temperature, shrinkage_alpha=shrinkage_alpha)
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# 别名，用于向后兼容
VEGARobust = VEGARobustScorer