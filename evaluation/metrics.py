"""
Evaluation metrics for VLM model selection experiments.
Includes ranking metrics and correlation measures.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Union, Optional


def compute_rank_correlation(
    predicted_scores: Union[List[float], np.ndarray],
    ground_truth_scores: Union[List[float], np.ndarray],
    method: str = 'spearman'
) -> float:
    """
    Compute rank correlation between predicted and ground truth rankings.
    
    Args:
        predicted_scores: Predicted transferability scores
        ground_truth_scores: Ground truth (actual) performance scores
        method: 'spearman' or 'kendall' correlation
        
    Returns:
        Correlation coefficient
    """
    predicted_scores = np.array(predicted_scores)
    ground_truth_scores = np.array(ground_truth_scores)
    
    if method == 'spearman':
        corr, _ = stats.spearmanr(predicted_scores, ground_truth_scores)
    elif method == 'kendall':
        corr, _ = stats.kendalltau(predicted_scores, ground_truth_scores)
    elif method == 'pearson':
        corr, _ = stats.pearsonr(predicted_scores, ground_truth_scores)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return corr if not np.isnan(corr) else 0.0


def compute_top_k_accuracy(
    predicted_ranking: Union[List[str], np.ndarray],
    optimal_ranking: Union[List[str], np.ndarray],
    k: int = 5
) -> float:
    """
    Compute top-k accuracy for model selection.
    
    Args:
        predicted_ranking: Model names sorted by predicted scores (descending)
        optimal_ranking: Model names sorted by ground truth performance (descending)
        k: Number of top models to consider
        
    Returns:
        Top-k accuracy (fraction of overlap)
    """
    if isinstance(predicted_ranking, np.ndarray):
        predicted_ranking = predicted_ranking.tolist()
    if isinstance(optimal_ranking, np.ndarray):
        optimal_ranking = optimal_ranking.tolist()
    
    predicted_top_k = set(predicted_ranking[:k])
    optimal_top_k = set(optimal_ranking[:k])
    
    overlap = len(predicted_top_k & optimal_top_k)
    
    return overlap / k


def compute_weighted_tau(
    predicted_ranking: Union[List[str], np.ndarray],
    optimal_ranking: Union[List[str], np.ndarray],
    k: int = None
) -> float:
    """
    Compute weighted Kendall's tau (gives more weight to top ranks).
    
    Args:
        predicted_ranking: Model names sorted by predicted scores (descending)
        optimal_ranking: Model names sorted by ground truth performance (descending)
        k: Number of top ranks to weight (None = all)
        
    Returns:
        Weighted tau score
    """
    if isinstance(predicted_ranking, np.ndarray):
        predicted_ranking = predicted_ranking.tolist()
    if isinstance(optimal_ranking, np.ndarray):
        optimal_ranking = optimal_ranking.tolist()
    
    n = len(predicted_ranking)
    if k is None:
        k = n
    
    # Create rank dictionaries
    pred_ranks = {model: i for i, model in enumerate(predicted_ranking)}
    opt_ranks = {model: i for i, model in enumerate(optimal_ranking)}
    
    # Compute weighted tau
    concordant = 0
    discordant = 0
    total_weight = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            model_i = predicted_ranking[i]
            model_j = predicted_ranking[j]
            
            pred_diff = pred_ranks[model_i] - pred_ranks[model_j]
            opt_diff = opt_ranks[model_i] - opt_ranks[model_j]
            
            # Weight based on rank position
            weight = 1.0 / (1 + min(i, j))
            
            if (pred_diff > 0 and opt_diff > 0) or (pred_diff < 0 and opt_diff < 0):
                concordant += weight
            else:
                discordant += weight
            
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return (concordant - discordant) / total_weight


def compute_mean_reciprocal_rank(
    predicted_ranking: Union[List[str], np.ndarray],
    optimal_best: str
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for finding the best model.
    
    Args:
        predicted_ranking: Model names sorted by predicted scores (descending)
        optimal_best: Name of the actually best model
        
    Returns:
        Reciprocal rank (1/rank of best model)
    """
    if isinstance(predicted_ranking, np.ndarray):
        predicted_ranking = predicted_ranking.tolist()
    
    for i, model in enumerate(predicted_ranking):
        if model == optimal_best:
            return 1.0 / (i + 1)
    
    return 0.0


def compute_full_metrics(
    predicted_scores: Dict[str, float],
    ground_truth_scores: Dict[str, float],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predicted_scores: Dict mapping model name to predicted score
        ground_truth_scores: Dict mapping model name to actual performance
        k_values: List of k values for top-k accuracy
        
    Returns:
        Dictionary containing all metrics
    """
    # Get common models
    common_models = list(set(predicted_scores.keys()) & set(ground_truth_scores.keys()))
    
    if not common_models:
        return {}
    
    # Extract scores for common models
    pred_scores = [predicted_scores[m] for m in common_models]
    gt_scores = [ground_truth_scores[m] for m in common_models]
    
    # Create rankings
    sorted_by_pred = sorted(common_models, key=lambda x: predicted_scores[x], reverse=True)
    sorted_by_gt = sorted(common_models, key=lambda x: ground_truth_scores[x], reverse=True)
    
    metrics = {}
    
    # Correlation metrics
    metrics['spearman'] = compute_rank_correlation(pred_scores, gt_scores, 'spearman')
    metrics['kendall'] = compute_rank_correlation(pred_scores, gt_scores, 'kendall')
    metrics['pearson'] = compute_rank_correlation(pred_scores, gt_scores, 'pearson')
    
    # Top-k accuracy
    for k in k_values:
        if k <= len(common_models):
            metrics[f'top_{k}_acc'] = compute_top_k_accuracy(sorted_by_pred, sorted_by_gt, k)
    
    # Weighted tau
    metrics['weighted_tau'] = compute_weighted_tau(sorted_by_pred, sorted_by_gt)
    
    # MRR
    if sorted_by_gt:
        metrics['mrr'] = compute_mean_reciprocal_rank(sorted_by_pred, sorted_by_gt[0])
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric name -> value
        title: Title for the output
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    for name, value in metrics.items():
        print(f"{name:25s}: {value:.4f}")
    
    print(f"{'='*50}\n")