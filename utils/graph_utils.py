import torch
import numpy as np


def compute_cosine_similarity_matrix(x, y=None):
    """计算余弦相似度矩阵"""
    if y is None:
        y = x
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)
    return torch.mm(x_norm, y_norm.t())


def bhattacharyya_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    计算两个高斯分布之间的 Bhattacharyya 距离 (VEGA Eq. 10)
    """
    mean_diff = (mu1 - mu2).unsqueeze(1)
    avg_cov = (cov1 + cov2) / 2

    # 为了数值稳定性，添加微小扰动
    dims = avg_cov.shape[0]
    avg_cov += torch.eye(dims, device=avg_cov.device) * eps

    # 第一项: 均值差异项
    # 使用 solve 而不是 inverse 提高稳定性
    term1 = (1 / 8) * torch.mm(torch.mm(mean_diff.t(), torch.inverse(avg_cov)), mean_diff)

    # 第二项: 协方差项
    sign1, logdet1 = torch.slogdet(cov1 + torch.eye(dims, device=cov1.device) * eps)
    sign2, logdet2 = torch.slogdet(cov2 + torch.eye(dims, device=cov2.device) * eps)
    sign_avg, logdet_avg = torch.slogdet(avg_cov)

    term2 = (1 / 2) * (logdet_avg - 0.5 * (logdet1 + logdet2))

    return (term1 + term2).item()


def pearson_correlation(matrix1, matrix2):
    """计算两个矩阵之间的 Pearson 相关系数 (VEGA Eq. 13)"""
    x = matrix1.flatten()
    y = matrix2.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return cost.item()