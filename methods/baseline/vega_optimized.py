"""
VEGA 优化版本: Visual-Textual Graph Alignment for Unsupervised VLM Selection
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

优化版本 - 修复数值稳定性问题，提升计算性能

核心优化点:
1. 修复 Node Similarity 的范围溢出 Bug - 除以总样本数 N_total 而非类别数 K
2. 采用对角协方差近似 (Diagonal Covariance Approximation) - 避免高维协方差矩阵不可逆问题
3. 向量化 Bhattacharyya 距离计算 - 利用 PyTorch 广播机制消除双重 for 循环
4. 保留 Edge Similarity 正确逻辑 - 使用上三角元素计算 Pearson 相关系数

论文公式参考:
1. 文本图节点: n^T_k = ξ(c̃_k)  (文本特征)
2. 文本图边: e^T_ij = cos(ξ(c̃_i), ξ(c̃_j))  (cosine相似度)
3. 视觉图节点: 高斯分布 N(μ_k, Σ_k) → 简化为对角协方差 N(μ_k, diag(σ²_k))
   - 均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
   - 对角方差: σ²_k = diag(cov(φ(x_i))) + regularization
4. 视觉图边: Bhattacharyya距离 (对角协方差简化版)
   - e^V_ij = (1/8) Σ_d (μ_i,d - μ_j,d)² / σ_avg,d + (1/2) Σ_d log(σ_avg,d) - (1/4) Σ_d (log(σ²_i,d) + log(σ²_j,d))
   - 其中 σ_avg,d = (σ²_i,d + σ²_j,d) / 2
5. 节点相似度: s_n = Σ_k sim(n^T_k, n^V_k) · N_k / N_total  【关键修复】
6. 边相似度: s_e = (corr(E^T, E^V) + 1)/2
7. 最终分数: s = s_n + s_e

更新日志:
- 2026-03-11: 创建优化版本，实现四项关键优化
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
        print(f"[VEGA-Optimized] {msg}", flush=True)


def timing_print(msg: str, start_time: float = None):
    """打印带时间的进度信息"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"[VEGA-Optimized] {msg} (耗时: {elapsed:.2f}s)", flush=True)
    else:
        print(f"[VEGA-Optimized] {msg}", flush=True)


class VEGAOptimizedScorer:
    """
    VEGA 优化版本评分器
    
    核心优化:
    1. **修复 Node Similarity 范围溢出 Bug**
       - 论文公式 s_n = (1/K) Σ_k sim_k · N_k 存在笔误
       - 当样本数 N >> 类别数 K 时，分数会远超 1
       - 修正为: s_n = Σ_k sim_k · N_k / N_total (除以总样本数)
       - 移除 np.clip 截断，让分数自然落在 [0,1]
    
    2. **对角协方差近似 (Diagonal Covariance Approximation)**
       - VLM 特征维度高达 512/768，小样本下全协方差矩阵必然不可逆
       - 将协方差矩阵简化为对角矩阵（只保留每个维度的方差）
       - 添加正则化项 self.regularization 防止除零
    
    3. **向量化 Bhattacharyya 距离计算**
       - 利用 PyTorch 广播机制，将 K×K 的距离计算向量化
       - 对角协方差下：矩阵求逆变为逐元素倒数，行列式变为 log_sum
       - 消除双重 for 循环，大幅提升计算效率
    
    4. **保留 Edge Similarity 正确逻辑**
       - 使用 np.triu_indices(n, k=1) 提取上三角元素
       - 计算 Pearson 相关系数，NaN 时返回 0.5
    
    使用方法:
        vega = VEGAOptimizedScorer(temperature=0.05, regularization=1e-6)
        score = vega.compute_score(visual_features, text_embeddings, logits)
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
        初始化 VEGA 优化版本评分器
        
        Args:
            temperature: softmax 归一化的温度参数 (论文默认 t=0.05)
            min_samples_per_class: 每个类别最少样本数，用于有效的方差估计
            node_weight: 节点相似度在最终分数中的权重 (默认 0.5)
            edge_weight: 边相似度在最终分数中的权重 (默认 0.5)
            regularization: 对角方差的正则化项，防止除零 (默认 1e-6)
        """
        self.temperature = temperature
        self.min_samples_per_class = min_samples_per_class
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.regularization = regularization
    
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
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], torch.Tensor]:
        """
        构建视觉图（使用对角协方差近似）
        
        【优化 2: 对角协方差近似】
        VLM 特征维度高达 512/768，小样本下全协方差矩阵必然不可逆，且求行列式易产生 NaN。
        我们将每个类的协方差矩阵简化为对角矩阵（即只计算每个特征维度的方差）。
        
        论文公式 (简化版):
        - 节点: 高斯分布 N(μ_k, diag(σ²_k))
          - 均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
          - 对角方差: σ²_k = Var(φ(x_i)) + regularization
        - 边: 简化的 Bhattacharyya 距离
        
        Args:
            visual_features: 图像特征 [N, D]
            pseudo_labels: 伪标签 [N]
            n_classes: 类别数 K
            
        Returns:
            (class_means, class_vars, class_counts, edge_matrix) 元组
            - class_means: 类别均值字典 {class_idx: mean_vector}
            - class_vars: 类别方差字典 {class_idx: variance_vector} (对角协方差)
            - class_counts: 类别样本数字典 {class_idx: count}
            - edge_matrix: Bhattacharyya 距离矩阵 [K, K]
        """
        progress_print("  构建视觉图 (对角协方差近似)...")
        start_time = time.time()
        
        # L2 归一化特征
        visual_features = self._normalize_features(visual_features)
        device = visual_features.device
        dtype = visual_features.dtype
        n_samples, n_features = visual_features.shape
        
        progress_print(f"    样本数: {n_samples}, 特征维度: {n_features}, 类别数: {n_classes}")
        
        # ============================================================
        # 步骤 1: 计算每个类别的统计量（均值和对角方差）
        # ============================================================
        class_means = {}
        class_vars = {}  # 对角方差（1D Tensor）
        class_counts = {}
        
        for k in range(n_classes):
            # 获取分配到类别 k 的样本
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                progress_print(f"    类别 {k}: 样本数 {len(indices)} < {self.min_samples_per_class}, 跳过")
                continue
            
            class_features = visual_features[indices]  # [N_k, D]
            class_counts[k] = len(indices)
            
            # 计算均值: μ_k = (1/N_k) Σ_{i:ŷ_i=k} φ(x_i)
            class_means[k] = class_features.mean(dim=0)  # [D]
            
            # 【优化 2: 对角协方差】
            # 只计算每个特征维度的方差，而非完整的协方差矩阵
            # 方差公式: σ²_k,d = (1/N_k) Σ_{i:ŷ_i=k} (φ(x_i,d) - μ_k,d)²
            # 使用有偏估计 (unbiased=False) 提高小样本稳定性
            if len(indices) > 1:
                var = class_features.var(dim=0, unbiased=False)  # [D]
            else:
                # 单样本情况：使用较小的默认方差
                var = torch.ones(n_features, device=device, dtype=dtype) * 0.1
            
            # 【关键】添加正则化项防止除零
            # 正则化确保方差严格为正，避免 log(0) 和除以零
            class_vars[k] = var + self.regularization
        
        # 获取有效类别
        valid_classes = list(class_means.keys())
        n_valid = len(valid_classes)
        
        progress_print(f"    有效类别数: {n_valid}")
        
        if n_valid < 2:
            logger.warning("有效类别数不足，无法构建视觉图")
            return class_means, class_vars, class_counts, None
        
        # ============================================================
        # 步骤 2: 向量化计算 Bhattacharyya 距离矩阵
        # 【优化 3: 向量化计算，消除双重 for 循环】
        # ============================================================
        progress_print("    计算 Bhattacharyya 距离矩阵 (向量化)...")
        
        edge_matrix = self._compute_bhattacharyya_distance_vectorized(
            class_means, class_vars, valid_classes, n_classes, device, dtype
        )
        
        elapsed = time.time() - start_time
        progress_print(f"    视觉图构建完成: 边矩阵 {edge_matrix.shape}, 耗时 {elapsed:.2f}s")
        
        return class_means, class_vars, class_counts, edge_matrix
    
    def _compute_bhattacharyya_distance_vectorized(
        self,
        class_means: Dict[int, torch.Tensor],
        class_vars: Dict[int, torch.Tensor],
        valid_classes: List[int],
        n_classes: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        向量化计算 Bhattacharyya 距离矩阵
        
        【优化 3: 向量化计算详解】
        
        Bhattacharyya 距离用于衡量两个高斯分布之间的相似度。
        对于两个高斯分布 N(μ_1, Σ_1) 和 N(μ_2, Σ_2)，原始公式为:
        
        D_B = (1/8)(μ_1 - μ_2)^T Σ^{-1} (μ_1 - μ_2) + (1/2)ln(|Σ|/√(|Σ_1||Σ_2|))
        
        其中 Σ = (Σ_1 + Σ_2)/2
        
        当协方差矩阵为对角矩阵时（Σ = diag(σ²)），公式可大幅简化:
        1. 矩阵求逆 Σ^{-1} 变为逐元素倒数 1/σ²
        2. 行列式 |Σ| 变为对角元素乘积的绝对值，即 Π σ²，log 后变为 Σ log(σ²)
        
        简化后的公式:
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │  D_B = Term1 + Term2                                                         │
        │                                                                              │
        │  Term1 (马氏距离项):                                                          │
        │    = (1/8) * Σ_d (μ_1,d - μ_2,d)² / σ_avg,d                                 │
        │    其中 σ_avg,d = (σ²_1,d + σ²_2,d) / 2                                      │
        │                                                                              │
        │  Term2 (行列式项):                                                            │
        │    = (1/2) * Σ_d log(σ_avg,d) - (1/4) * Σ_d log(σ²_1,d) - (1/4) * Σ_d log(σ²_2,d) │
        │    = (1/4) * Σ_d log(σ_avg,d) - (1/4) * Σ_d (log(σ²_1,d) + log(σ²_2,d))      │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        向量化实现思路:
        1. 将所有类别的均值和方差堆叠成矩阵 [K_valid, D]
        2. 利用 PyTorch 广播机制，一次性计算所有类别对的距离
        3. 广播维度: [K_valid, 1, D] op [1, K_valid, D] → [K_valid, K_valid, D]
        
        Args:
            class_means: 类别均值字典 {class_idx: mean_vector [D]}
            class_vars: 类别方差字典 {class_idx: variance_vector [D]}
            valid_classes: 有效类别列表
            n_classes: 总类别数 K
            device: 计算设备
            dtype: 数据类型
            
        Returns:
            Bhattacharyya 距离矩阵 [K, K]
        """
        n_valid = len(valid_classes)
        
        # ============================================================
        # 步骤 1: 将均值和方差堆叠成矩阵
        # ============================================================
        # means_matrix: [n_valid, D] - 每行是一个类别的均值向量
        means_matrix = torch.stack([class_means[k] for k in valid_classes], dim=0)
        
        # vars_matrix: [n_valid, D] - 每行是一个类别的方差向量（对角协方差的对角元素）
        vars_matrix = torch.stack([class_vars[k] for k in valid_classes], dim=0)
        
        # ============================================================
        # 步骤 2: 利用广播计算所有类别对之间的距离分量
        # ============================================================
        
        # ----------------------------------------------------------------
        # 计算 Term1: (1/8) * Σ_d (μ_i,d - μ_j,d)² / σ_avg,d
        # ----------------------------------------------------------------
        
        # 均值差异矩阵: mu_diff[i, j, d] = means_matrix[i, d] - means_matrix[j, d]
        # 广播: [n_valid, 1, D] - [1, n_valid, D] → [n_valid, n_valid, D]
        mu_diff = means_matrix.unsqueeze(1) - means_matrix.unsqueeze(0)  # [n_valid, n_valid, D]
        
        # 平均方差矩阵: sigma_avg[i, j, d] = (vars_matrix[i, d] + vars_matrix[j, d]) / 2
        # 广播: [n_valid, 1, D] + [1, n_valid, D] → [n_valid, n_valid, D]
        sigma_avg = (vars_matrix.unsqueeze(1) + vars_matrix.unsqueeze(0)) / 2  # [n_valid, n_valid, D]
        
        # Term1 的被除部分: (μ_i - μ_j)² / σ_avg
        # 添加 EPS 防止除零（虽然已有正则化，但这里再加一层保护）
        term1_per_dim = (mu_diff ** 2) / (sigma_avg + EPS)  # [n_valid, n_valid, D]
        
        # Term1: 对特征维度求和，乘以 1/8
        term1 = term1_per_dim.sum(dim=-1) * 0.125  # [n_valid, n_valid]
        
        # ----------------------------------------------------------------
        # 计算 Term2: (1/4) * Σ_d log(σ_avg,d) - (1/4) * Σ_d (log(σ²_i,d) + log(σ²_j,d))
        # ----------------------------------------------------------------
        
        # log(σ_avg): 对应行列式 |Σ| = Π σ_avg,d 的 log，即 Σ log(σ_avg,d)
        log_sigma_avg = torch.log(sigma_avg + EPS)  # [n_valid, n_valid, D]
        
        # Term2 第一部分: (1/4) * Σ_d log(σ_avg,d)
        # 注意: 原公式中是 (1/2) * log(|Σ|)，其中 |Σ| = Π σ_avg,d
        # 所以 (1/2) * Σ log(σ_avg,d) = (1/2) * Σ log(σ_avg,d)
        # 但实际上应该是 (1/2) * Σ log(σ_avg,d)，因为 log(|Σ|/√(|Σ_1||Σ_2|)) = log|Σ| - 1/2*log|Σ_1| - 1/2*log|Σ_2|
        # = Σ log(σ_avg) - 1/2 * Σ log(σ²_1) - 1/2 * Σ log(σ²_2)
        # 化简后: = Σ log(σ_avg) - 1/4 * Σ (log(σ²_1) + log(σ²_2))，其中 σ²_avg = (σ²_1 + σ²_2)/2
        # 更精确的推导:
        # log(|Σ|/√(|Σ_1||Σ_2|)) = log|Σ| - 0.5*log|Σ_1| - 0.5*log|Σ_2|
        # 对于对角矩阵: log|Σ| = Σ log(σ²_avg) = Σ log((σ²_1 + σ²_2)/2)
        # = Σ log(σ²_1 + σ²_2) - Σ log(2)
        # log|Σ_1| = Σ log(σ²_1), log|Σ_2| = Σ log(σ²_2)
        # 所以行列式项 = 0.5 * (Σ log((σ²_1 + σ²_2)/2) - 0.5*Σ log(σ²_1) - 0.5*Σ log(σ²_2))
        # = 0.5 * Σ log((σ²_1 + σ²_2)/2) - 0.25 * Σ log(σ²_1) - 0.25 * Σ log(σ²_2)
        
        # Term2: 0.5 * sum(log(sigma_avg)) - 0.25 * sum(log(var_i)) - 0.25 * sum(log(var_j))
        log_vars = torch.log(vars_matrix + EPS)  # [n_valid, D]
        
        # Term2 完整计算
        term2 = 0.5 * log_sigma_avg.sum(dim=-1)  # 0.5 * Σ log(σ_avg) [n_valid, n_valid]
        
        # 需要 -0.25 * (Σ log(var_i) + Σ log(var_j))
        # log_vars_sum[i, j] = Σ log(var_i) + Σ log(var_j)
        # 首先计算每个类别的 log 方差和
        log_vars_sum_per_class = log_vars.sum(dim=-1)  # [n_valid]
        # 然后广播求和: log_vars_sum[i, j] = log_vars_sum_per_class[i] + log_vars_sum_per_class[j]
        log_vars_sum = log_vars_sum_per_class.unsqueeze(1) + log_vars_sum_per_class.unsqueeze(0)  # [n_valid, n_valid]
        
        term2 = term2 - 0.25 * log_vars_sum  # [n_valid, n_valid]
        
        # ============================================================
        # 步骤 3: 合并 Term1 和 Term2，得到最终的 Bhattacharyya 距离
        # ============================================================
        bh_distance = term1 + term2  # [n_valid, n_valid]
        
        # 确保对角线为 0（同一类别的距离为 0）
        bh_distance = bh_distance * (1 - torch.eye(n_valid, device=device, dtype=dtype))
        
        # 确保距离非负（理论上应为非负，但数值误差可能导致小的负值）
        bh_distance = torch.clamp(bh_distance, min=0.0)
        
        # ============================================================
        # 步骤 4: 将距离填充到完整的 [K, K] 矩阵中
        # ============================================================
        edge_matrix = torch.zeros(n_classes, n_classes, device=device, dtype=dtype)
        
        # 建立 valid_classes 索引到原始类别索引的映射
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
        
        【优化 1: 修复范围溢出 Bug】
        
        论文公式(11, 12):
        sim(n^T_k, n^V_k) = (1/N_k) Σ_{i:ŷ_i=k} [exp(cos(φ(x_i), ξ(c_k))/t) / Σ_{k'} exp(cos(φ(x_i), ξ(c_{k'}))/t)]
        
        原论文公式: s_n = (1/K) Σ_k sim(n^T_k, n^V_k) · N_k
        
        【问题】原公式存在笔误：当样本数 N 较大时，Σ_k sim_k · N_k 可能远大于 K，导致 s_n > 1。
        例如：K=10 类别，每类 100 样本，即使 sim_k=0.5（平均概率），
        s_n = (1/10) * 0.5 * 100 * 10 = 50，远超 1！
        
        【修复】正确的归一化方式应该是除以总样本数 N_total，而非类别数 K:
        s_n = Σ_k sim_k · N_k / N_total
        
        这样 s_n 就是所有样本在其分配类别上 softmax 概率的加权平均，自然落在 [0, 1] 区间。
        
        【验证】假设每个样本在其分配类别上的概率为 p，则：
        s_n = Σ_k (p * N_k) * N_k / N_total = p * Σ_k N_k² / N_total
        当各类别样本数相等时：s_n = p * K * (N/K)² / N = p * N / K
        这仍然可能超过 1... 让我重新理解。
        
        【重新理解论文公式】
        sim(n^T_k, n^V_k) 实际上是在衡量类别 k 的视觉特征与文本特征的匹配程度。
        对于每个样本 i，它被分配到类别 ŷ_i，我们计算该样本在类别 ŷ_i 上的 softmax 概率。
        然后对每个类别内的所有样本取平均，得到该类别的 sim_k。
        
        s_n = (1/K) Σ_k sim_k · N_k 的问题在于：
        - sim_k 是一个 [0,1] 的值（softmax 概率的平均）
        - N_k 是样本数，可能很大
        - 除以 K 而非 N_total 导致权重分配不合理
        
        正确的公式应该是：
        s_n = Σ_k sim_k · N_k / Σ_k N_k = Σ_k sim_k · N_k / N_total
        
        这样 s_n 表示所有样本在其分配类别上 softmax 概率的加权平均。
        
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
        # 论文公式(12): exp(cos/t) / Σ exp(cos/t)
        scaled_similarity = cosine_similarity / self.temperature
        probs = F.softmax(scaled_similarity, dim=1)  # [N, K]
        
        # 计算每个类别的相似度分数
        class_similarities = {}
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # 论文公式(12): sim(n^T_k, n^V_k) = (1/N_k) Σ_{i:ŷ_i=k} [...]
            # 获取每个样本在其分配类别上的概率
            class_probs = probs[indices, k]
            
            # 该类别的平均相似度
            class_similarities[k] = class_probs.mean().item()
        
        if not class_similarities:
            return 0.0
        
        # ============================================================
        # 【关键修复】计算加权和，然后除以总样本数而非类别数
        # ============================================================
        total_weighted_sim = 0.0
        total_samples = 0  # N_total = Σ_k N_k (只统计有效类别)
        
        for k, sim in class_similarities.items():
            N_k = class_counts.get(k, 1)
            total_weighted_sim += sim * N_k
            total_samples += N_k
        
        # 【修复】除以总样本数 N_total，让分数自然落在 [0, 1]
        # 原 Bug: node_similarity = total_weighted_sim / n_classes
        # 修复后: node_similarity = total_weighted_sim / total_samples
        if total_samples > 0:
            node_similarity = total_weighted_sim / total_samples
        else:
            node_similarity = 0.0
        
        # 【移除 np.clip】现在分数自然落在 [0, 1] 区间，无需截断
        # 原代码: node_similarity = float(np.clip(node_similarity, 0.0, 1.0))
        # 修复后: 直接使用计算结果
        
        n_valid_classes = len(class_similarities)
        progress_print(f"    节点相似度 = {node_similarity:.4f} (有效类别: {n_valid_classes}/{n_classes})")
        progress_print(f"    加权和 = {total_weighted_sim:.2f}, 总样本数 N_total = {total_samples}")
        progress_print(f"    【修复】除以 N_total={total_samples} 而非 K={n_classes}")
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> float:
        """
        计算边相似度（使用 Pearson 相关系数）
        
        【优化 4: 保留正确逻辑】
        使用 np.triu_indices(n, k=1) 提取上三角元素计算 Pearson 相关系数。
        如果相关系数计算结果为 NaN，捕获异常并返回 0.5。
        
        论文公式(13, 14):
        corr(E^T, E^V) = E^T 和 E^V 边矩阵之间的 Pearson 相关系数
        s_e = (corr + 1)/2  将 [-1, 1] 映射到 [0, 1]
        
        Args:
            textual_edges: 文本图边矩阵 [K, K]
            visual_edges: 视觉图边矩阵 [K, K]
            
        Returns:
            边相似度分数 [0, 1]
        """
        progress_print("  计算边相似度...")
        
        # 转换为 numpy 数组
        textual_edges = self._to_numpy(textual_edges)
        visual_edges = self._to_numpy(visual_edges)
        
        n = textual_edges.shape[0]
        
        # 提取上三角元素（排除对角线）
        # 论文中计算所有边的 Pearson 相关系数
        triu_indices = np.triu_indices(n, k=1)
        
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            progress_print("    边数不足，返回默认值 0.5", level="WARNING")
            return 0.5
        
        # 计算 Pearson 相关系数
        try:
            corr, _ = pearsonr(textual_vec, visual_vec)
            
            # 【优化 4】处理 NaN 情况
            if np.isnan(corr):
                progress_print("    Pearson 相关系数为 NaN，返回默认值 0.5", level="WARNING")
                return 0.5
            
            # 论文公式(14): s_e = (corr + 1)/2
            # 将 [-1, 1] 映射到 [0, 1]
            edge_similarity = (corr + 1) / 2
            
            progress_print(f"    边相似度 = {edge_similarity:.4f} (Pearson corr = {corr:.4f})")
            
            return edge_similarity
            
        except Exception as e:
            logger.warning(f"Pearson 相关系数计算失败: {e}")
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
        计算 VEGA 迁移性分数（优化版本）
        
        算法流程:
        1. 获取伪标签: ŷ_i = argmax_k cos(ξ(c̃_k), φ(x_i))
        2. 构建文本图: 节点=文本特征, 边=cosine相似度
        3. 构建视觉图: 节点=高斯分布(对角协方差), 边=Bhattacharyya距离
        4. 计算节点相似度: s_n = Σ_k sim_k · N_k / N_total  【修复后】
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
        """
        total_start = time.time()
        
        progress_print("=" * 60)
        progress_print("开始计算 VEGA 分数 (优化版本)")
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
        
        # 步骤 2: 构建视觉图（对角协方差近似）
        step2_start = time.time()
        progress_print("【步骤 2/4】构建视觉图 (对角协方差近似)...")
        class_means, class_vars, class_counts, visual_edges = self.build_visual_graph(
            visual_normalized, pseudo_labels, n_classes
        )
        timing_print(f"  视觉图构建完成: 有效类别数={len(class_means)}", step2_start)
        
        # 检查视觉图是否构建成功
        if visual_edges is None or len(class_means) < 2:
            progress_print("视觉图构建失败，返回默认分数", level="WARNING")
            if return_details:
                return {
                    'score': 0.0,
                    'node_similarity': 0.0,
                    'edge_similarity': 0.0,
                    'valid_classes': len(class_means),
                    'diagonal_covariance': True,
                    'vectorized_computation': True,
                    'node_similarity_fix': True
                }
            return 0.0
        
        # 步骤 3: 计算节点相似度 (论文公式11, 12，已修复)
        step3_start = time.time()
        progress_print("【步骤 3/4】计算节点相似度...")
        node_similarity = self.compute_node_similarity(
            cosine_similarity, pseudo_labels, class_counts
        )
        timing_print(f"  节点相似度 = {node_similarity:.4f}", step3_start)
        
        # 步骤 4: 计算边相似度 (论文公式13, 14)
        step4_start = time.time()
        progress_print("【步骤 4/4】计算边相似度...")
        edge_similarity = self.compute_edge_similarity(textual_edges, visual_edges)
        timing_print(f"  边相似度 = {edge_similarity:.4f}", step4_start)
        
        # 步骤 5: 最终 VEGA 分数 (论文公式: s = s_n + s_e)
        vega_score = node_similarity + edge_similarity
        
        total_time = time.time() - total_start
        progress_print("-" * 60)
        progress_print(f"VEGA 总分 = {vega_score:.4f}")
        progress_print(f"  节点相似度 s_n = {node_similarity:.4f}")
        progress_print(f"  边相似度 s_e = {edge_similarity:.4f}")
        progress_print(f"总耗时: {total_time:.2f}s")
        progress_print("=" * 60)
        
        if return_details:
            return {
                'score': vega_score,
                'node_similarity': node_similarity,
                'edge_similarity': edge_similarity,
                'valid_classes': len(class_means),
                'class_counts': class_counts,
                'compute_time': total_time,
                'diagonal_covariance': True,
                'vectorized_computation': True,
                'node_similarity_fix': True
            }
        
        return vega_score


def compute_vega_score_optimized(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    temperature: float = 0.05,
    regularization: float = 1e-6
) -> float:
    """
    计算 VEGA 分数的便捷函数（优化版本）
    
    Args:
        features: 图像特征 [N, D]
        text_embeddings: 文本嵌入 [K, D]
        logits: 模型预测 [N, K]
        pseudo_labels: 伪标签 [N]
        temperature: 温度参数 (论文默认 0.05)
        regularization: 对角方差正则化项 (默认 1e-6)
        
    Returns:
        VEGA 分数
    """
    vega = VEGAOptimizedScorer(temperature=temperature, regularization=regularization)
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# 别名，用于向后兼容
VEGAOptimized = VEGAOptimizedScorer