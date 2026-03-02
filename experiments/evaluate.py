"""
Evaluation utilities for model selection experiments.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from evaluation.metrics import (
    compute_rank_correlation,
    compute_top_k_accuracy,
    compute_full_metrics,
    print_metrics
)


def evaluate_model_selection(
    predicted_scores: Dict[str, float],
    ground_truth_scores: Dict[str, float],
    method_name: str = "Method",
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model selection performance.
    
    Args:
        predicted_scores: Dict mapping model name to predicted score
        ground_truth_scores: Dict mapping model name to actual performance
        method_name: Name of the method for display
        k_values: List of k values for top-k accuracy
        verbose: Whether to print results
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = compute_full_metrics(predicted_scores, ground_truth_scores, k_values)
    
    if verbose:
        print_metrics(metrics, f"{method_name} Evaluation Results")
    
    return metrics


def compare_methods(
    results: Dict[str, Dict[str, float]],
    output_path: str = None
) -> str:
    """
    Compare results from multiple methods.
    
    Args:
        results: Dict mapping method name to metrics dict
        output_path: Optional path to save comparison table
        
    Returns:
        Formatted comparison table string
    """
    # Get all metric names
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    
    # Build comparison table
    lines = []
    lines.append("\n" + "="*80)
    lines.append("Method Comparison".center(80))
    lines.append("="*80)
    
    # Header
    header = f"{'Metric':<20}"
    for method in results.keys():
        header += f"{method:>15}"
    lines.append(header)
    lines.append("-"*80)
    
    # Metrics rows
    for metric in all_metrics:
        row = f"{metric:<20}"
        for method in results.keys():
            value = results[method].get(metric, float('nan'))
            row += f"{value:>15.4f}"
        lines.append(row)
    
    lines.append("="*80 + "\n")
    
    table = "\n".join(lines)
    print(table)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
    
    return table


def save_results(
    results: Dict,
    output_dir: str,
    filename: str = None
) -> str:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Optional filename (auto-generated if None)
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")
    return output_path


def load_results(results_path: str) -> Dict:
    """
    Load experiment results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Results dictionary
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def aggregate_results(
    results_list: List[Dict[str, Dict[str, float]]],
    method_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple runs/datasets.
    
    Args:
        results_list: List of results dictionaries
        method_names: List of method names (if not in results)
        
    Returns:
        Aggregated results with mean and std
    """
    if method_names is None:
        method_names = list(results_list[0].keys())
    
    aggregated = {}
    
    for method in method_names:
        # Collect all metrics for this method
        method_results = [r[method] for r in results_list if method in r]
        
        if not method_results:
            continue
        
        # Get all metric names
        metric_names = set()
        for r in method_results:
            metric_names.update(r.keys())
        
        aggregated[method] = {}
        for metric in metric_names:
            values = [r.get(metric, float('nan')) for r in method_results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregated[method][f"{metric}_mean"] = np.mean(values)
                aggregated[method][f"{metric}_std"] = np.std(values)
    
    return aggregated


if __name__ == "__main__":
    # Example usage
    print("Evaluation utilities")
    print("Import and use evaluate_model_selection(), compare_methods(), etc.")