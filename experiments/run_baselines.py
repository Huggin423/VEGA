"""
Run baseline experiments for VLM model selection.
Includes LogME and VEGA baseline methods.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.baseline.logme import LogME
from methods.baseline.vega import VEGAScorer, compute_vega_score
from evaluation.metrics import compute_full_metrics, print_metrics


def run_logme_experiment(
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    ground_truth_acc: Dict[str, float],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run LogME experiment on multiple models.
    
    Args:
        features_dict: Dict mapping model name to features [N, D]
        labels: Ground truth labels [N]
        ground_truth_acc: Dict mapping model name to accuracy
        verbose: Whether to print progress
        
    Returns:
        Evaluation metrics
    """
    logme_scores = {}
    
    model_names = list(features_dict.keys())
    iterator = tqdm(model_names, desc="LogME") if verbose else model_names
    
    for model_name in iterator:
        features = features_dict[model_name]
        
        try:
            logme = LogME()
            score = logme.fit(features, labels)
            logme_scores[model_name] = score
        except Exception as e:
            print(f"Error computing LogME for {model_name}: {e}")
            logme_scores[model_name] = float('-inf')
    
    # Compute evaluation metrics
    metrics = compute_full_metrics(logme_scores, ground_truth_acc)
    
    if verbose:
        print_metrics(metrics, "LogME Results")
    
    return metrics


def run_vega_experiment(
    features_dict: Dict[str, np.ndarray],
    text_embeddings: np.ndarray,
    logits_dict: Dict[str, np.ndarray],
    ground_truth_acc: Dict[str, float],
    k_neighbors: int = 10,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run VEGA experiment on multiple models.
    
    Args:
        features_dict: Dict mapping model name to features [N, D]
        text_embeddings: Text embeddings [C, D]
        logits_dict: Dict mapping model name to logits [N, C]
        ground_truth_acc: Dict mapping model name to accuracy
        k_neighbors: Number of neighbors for graph construction
        verbose: Whether to print progress
        
    Returns:
        Evaluation metrics
    """
    vega_scores = {}
    
    model_names = list(features_dict.keys())
    iterator = tqdm(model_names, desc="VEGA") if verbose else model_names
    
    vega = VEGAScorer(k_neighbors=k_neighbors)
    
    for model_name in iterator:
        features = features_dict[model_name]
        logits = logits_dict.get(model_name)
        
        try:
            score = vega.compute_score(features, text_embeddings, logits)
            vega_scores[model_name] = score
        except Exception as e:
            print(f"Error computing VEGA for {model_name}: {e}")
            vega_scores[model_name] = 0.0
    
    # Compute evaluation metrics
    metrics = compute_full_metrics(vega_scores, ground_truth_acc)
    
    if verbose:
        print_metrics(metrics, "VEGA Results")
    
    return metrics


def run_all_baselines(
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    text_embeddings: np.ndarray,
    logits_dict: Dict[str, np.ndarray],
    ground_truth_acc: Dict[str, float],
    output_dir: str = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run all baseline experiments.
    
    Args:
        features_dict: Dict mapping model name to features [N, D]
        labels: Ground truth labels [N]
        text_embeddings: Text embeddings [C, D]
        logits_dict: Dict mapping model name to logits [N, C]
        ground_truth_acc: Dict mapping model name to accuracy
        output_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping method name to metrics
    """
    results = {}
    
    # Run LogME
    print("\n" + "="*60)
    print("Running LogME baseline...")
    print("="*60)
    results['LogME'] = run_logme_experiment(features_dict, labels, ground_truth_acc, verbose)
    
    # Run VEGA
    print("\n" + "="*60)
    print("Running VEGA baseline...")
    print("="*60)
    results['VEGA'] = run_vega_experiment(features_dict, text_embeddings, logits_dict, ground_truth_acc, verbose=verbose)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Baseline experiment runner")
    print("Import and use run_logme_experiment() or run_vega_experiment()")