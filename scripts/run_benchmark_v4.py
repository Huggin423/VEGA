#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VEGA v4 Benchmark Script
========================
Benchmark script for VEGAv4Scorer with confidence-weighted visual graph nodes
and contrastive-aware edge similarity.

Key Features:
- Confidence-weighted visual graph nodes using prediction confidence to weight samples
- Contrastive-aware edge similarity using contrastive-weighted Pearson correlation

Author: VEGA Team
Date: 2024
"""

import os
import sys
import time
import pickle
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from methods.baseline.vega_v4 import VEGAv4Scorer
from methods.baseline.logme import LogME
from utils.data_processor import DatasetManager

# =============================================================================
# Configuration
# =============================================================================

CACHE_DIR = project_root / "cache" / "benchmark_v4"
ENABLE_CACHE = True

# Ensure cache directory exists
if ENABLE_CACHE:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Progress Display Tools
# =============================================================================

class ProgressBar:
    """Simple progress bar for tracking dataset processing."""
    
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n=1):
        self.current += n
        elapsed = time.time() - self.start_time
        if self.total > 0:
            percent = 100 * self.current / self.total
            bar_len = 40
            filled = int(bar_len * self.current / self.total)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r{self.desc}: [{bar}] {percent:5.1f}% ({self.current}/{self.total}) | {elapsed:.1f}s", end='', flush=True)
    
    def close(self):
        print()


def print_header(title, char="=", width=70):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_section(title, char="-", width=60):
    """Print a formatted section header."""
    print()
    print(f"\n{char * width}")
    print(f" {title}")
    print(char * width)


def print_info(label, value, indent=2):
    """Print formatted info line."""
    print(f"{' ' * indent}{label}: {value}")


def print_metric(label, value, indent=2, format_str=".4f"):
    """Print formatted metric."""
    if isinstance(value, (int, float)):
        print(f"{' ' * indent}{label}: {value:{format_str}}")
    else:
        print(f"{' ' * indent}{label}: {value}")


# =============================================================================
# Cache System
# =============================================================================

def get_cache_key(dataset_name, model_name, pca_dim, temperature, conf_threshold, 
                  tau_contrast, node_weight, edge_weight):
    """Generate a unique cache key based on all parameters."""
    key_str = f"{dataset_name}_{model_name}_pca{pca_dim}_temp{temperature}_conf{conf_threshold}_tau{tau_contrast}_nw{node_weight}_ew{edge_weight}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cache(cache_key):
    """Load cached results if available."""
    if not ENABLE_CACHE:
        return None
    
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
    return None


def save_cache(cache_key, data):
    """Save results to cache."""
    if not ENABLE_CACHE:
        return
    
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_logits_data(dataset_name, model_name, data_dir="data"):
    """Load logits and labels for a dataset."""
    data_path = project_root / data_dir / dataset_name / model_name
    
    # Try different file patterns
    patterns = [
        ("logits.npy", "labels.npy"),
        ("predictions.npy", "labels.npy"),
        (f"{model_name}_logits.npy", f"{model_name}_labels.npy"),
    ]
    
    for logits_file, labels_file in patterns:
        logits_path = data_path / logits_file
        labels_path = data_path / labels_file
        
        if logits_path.exists() and labels_path.exists():
            logits = np.load(logits_path)
            labels = np.load(labels_path)
            return logits, labels
    
    # Try DatasetManager as fallback
    try:
        manager = DatasetManager(dataset_name, model_name)
        data = manager.load_logits_data()
        if data is not None:
            return data['logits'], data['labels']
    except Exception:
        pass
    
    return None, None


def load_image_features(dataset_name, model_name, data_dir="data"):
    """Load image features for a dataset."""
    data_path = project_root / data_dir / dataset_name / model_name
    
    patterns = [
        "image_features.npy",
        "img_features.npy",
        "features.npy",
        f"{model_name}_image_features.npy",
    ]
    
    for pattern in patterns:
        feat_path = data_path / pattern
        if feat_path.exists():
            return np.load(feat_path)
    
    return None


def load_text_features(dataset_name, model_name, data_dir="data"):
    """Load text features for a dataset."""
    data_path = project_root / data_dir / dataset_name / model_name
    
    patterns = [
        "text_features.npy",
        "txt_features.npy",
        "text_embeddings.npy",
        f"{model_name}_text_features.npy",
    ]
    
    for pattern in patterns:
        feat_path = data_path / pattern
        if feat_path.exists():
            return np.load(feat_path)
    
    return None


# =============================================================================
# VEGA v4 Score Computation
# =============================================================================

def compute_vega_score_v4(image_features, text_features, labels, logits=None,
                          pca_dim=64, temperature=0.05, conf_threshold=0.5,
                          tau_contrast=0.1, node_weight=1.0, edge_weight=1.0,
                          verbose=True):
    """
    Compute VEGA v4 score with confidence-weighted nodes and contrastive-aware edges.
    
    Args:
        image_features: Image feature matrix (N, D)
        text_features: Text feature matrix (N, D) or class prototypes
        labels: Ground truth labels (N,)
        logits: Optional prediction logits for confidence weighting
        pca_dim: Target dimension after PCA
        temperature: Temperature for probability computation
        conf_threshold: Confidence threshold for filtering high-quality samples
        tau_contrast: Contrastive temperature for edge similarity
        node_weight: Weight for node-level score
        edge_weight: Weight for edge-level score
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing scores and detailed information
    """
    if verbose:
        print_section("VEGA v4 Score Computation")
        print_info("Configuration", "")
        print_info("PCA dimension", pca_dim, indent=4)
        print_info("Temperature", temperature, indent=4)
        print_info("Confidence threshold", conf_threshold, indent=4)
        print_info("Contrastive tau", tau_contrast, indent=4)
        print_info("Node weight", node_weight, indent=4)
        print_info("Edge weight", edge_weight, indent=4)
    
    # Initialize VEGAv4Scorer
    scorer = VEGAv4Scorer(
        pca_dim=pca_dim,
        temperature=temperature,
        conf_threshold=conf_threshold,
        tau_contrast=tau_contrast,
        node_weight=node_weight,
        edge_weight=edge_weight
    )
    
    # Compute score
    start_time = time.time()
    result = scorer.score(
        image_features=image_features,
        text_features=text_features,
        labels=labels,
        logits=logits
    )
    elapsed_time = time.time() - start_time
    
    if verbose:
        print_info("Computation time", f"{elapsed_time:.3f}s")
        print_info("VEGA v4 Score", f"{result['score']:.4f}")
        if 'node_score' in result:
            print_info("Node Score (confidence-weighted)", f"{result['node_score']:.4f}")
        if 'edge_score' in result:
            print_info("Edge Score (contrastive-aware)", f"{result['edge_score']:.4f}")
        if 'graph_info' in result:
            info = result['graph_info']
            print_info("Graph Statistics", "")
            if 'num_nodes' in info:
                print_info("Number of nodes", info['num_nodes'], indent=4)
            if 'num_edges' in info:
                print_info("Number of edges", info['num_edges'], indent=4)
            if 'avg_confidence' in info:
                print_info("Average confidence", f"{info['avg_confidence']:.4f}", indent=4)
    
    return result


# =============================================================================
# LogME Score Computation
# =============================================================================

def compute_logme_score(features, labels, verbose=True):
    """Compute LogME score as baseline comparison."""
    if verbose:
        print_section("LogME Score Computation")
    
    logme = LogME()
    
    start_time = time.time()
    score = logme.fit(features, labels)
    elapsed_time = time.time() - start_time
    
    if verbose:
        print_info("Computation time", f"{elapsed_time:.3f}s")
        print_info("LogME Score", f"{score:.4f}")
    
    return score


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(predicted_scores, ground_truth_ranks, verbose=True):
    """
    Compute evaluation metrics for transferability prediction.
    
    Args:
        predicted_scores: Predicted transferability scores
        ground_truth_ranks: Ground truth ranks (lower is better)
    
    Returns:
        Dictionary containing various metrics
    """
    from scipy.stats import spearmanr, kendalltau
    
    metrics = {}
    
    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(predicted_scores, ground_truth_ranks)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_p_value'] = spearman_p
    
    # Kendall's tau
    kendall_tau, kendall_p = kendalltau(predicted_scores, ground_truth_ranks)
    metrics['kendall_tau'] = kendall_tau
    metrics['kendall_p_value'] = kendall_p
    
    # Pearson correlation
    from scipy.stats import pearsonr
    pearson_corr, pearson_p = pearsonr(predicted_scores, ground_truth_ranks)
    metrics['pearson_correlation'] = pearson_corr
    metrics['pearson_p_value'] = pearson_p
    
    if verbose:
        print_section("Evaluation Metrics")
        print_metric("Spearman Correlation", metrics['spearman_correlation'])
        print_metric("Kendall's Tau", metrics['kendall_tau'])
        print_metric("Pearson Correlation", metrics['pearson_correlation'])
    
    return metrics


# =============================================================================
# Main Benchmark Function
# =============================================================================

def run_single_dataset_benchmark(dataset_name, model_name, 
                                 pca_dim=64, temperature=0.05,
                                 conf_threshold=0.5, tau_contrast=0.1,
                                 node_weight=1.0, edge_weight=1.0,
                                 use_cache=True, verbose=True):
    """
    Run VEGA v4 benchmark on a single dataset-model pair.
    
    Returns:
        Dictionary containing all scores and metrics
    """
    # Generate cache key
    cache_key = get_cache_key(
        dataset_name, model_name, pca_dim, temperature,
        conf_threshold, tau_contrast, node_weight, edge_weight
    )
    
    # Try to load from cache
    if use_cache:
        cached_result = load_cache(cache_key)
        if cached_result is not None:
            if verbose:
                print(f"Loaded cached results for {dataset_name}/{model_name}")
            return cached_result
    
    if verbose:
        print_header(f"Benchmark: {dataset_name} - {model_name}")
        print_info("Dataset", dataset_name)
        print_info("Model", model_name)
    
    # Load data
    if verbose:
        print_section("Loading Data")
    
    logits, labels = load_logits_data(dataset_name, model_name)
    image_features = load_image_features(dataset_name, model_name)
    text_features = load_text_features(dataset_name, model_name)
    
    if image_features is None:
        if verbose:
            print(f"Warning: Could not load image features for {dataset_name}/{model_name}")
        return None
    
    if labels is None:
        if verbose:
            print(f"Warning: Could not load labels for {dataset_name}/{model_name}")
        return None
    
    if verbose:
        print_info("Image features shape", image_features.shape)
        if text_features is not None:
            print_info("Text features shape", text_features.shape)
        if logits is not None:
            print_info("Logits shape", logits.shape)
        print_info("Number of samples", len(labels))
        print_info("Number of classes", len(np.unique(labels)))
    
    # Compute VEGA v4 score
    vega_result = compute_vega_score_v4(
        image_features=image_features,
        text_features=text_features,
        labels=labels,
        logits=logits,
        pca_dim=pca_dim,
        temperature=temperature,
        conf_threshold=conf_threshold,
        tau_contrast=tau_contrast,
        node_weight=node_weight,
        edge_weight=edge_weight,
        verbose=verbose
    )
    
    # Compute LogME score for comparison
    logme_score = compute_logme_score(image_features, labels, verbose=verbose)
    
    # Compile results
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'vega_score': vega_result['score'],
        'vega_node_score': vega_result.get('node_score'),
        'vega_edge_score': vega_result.get('edge_score'),
        'logme_score': logme_score,
        'graph_info': vega_result.get('graph_info', {}),
        'config': {
            'pca_dim': pca_dim,
            'temperature': temperature,
            'conf_threshold': conf_threshold,
            'tau_contrast': tau_contrast,
            'node_weight': node_weight,
            'edge_weight': edge_weight
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to cache
    if use_cache:
        save_cache(cache_key, result)
    
    return result


def run_multi_dataset_benchmark(datasets, models=None,
                                pca_dim=64, temperature=0.05,
                                conf_threshold=0.5, tau_contrast=0.1,
                                node_weight=1.0, edge_weight=1.0,
                                use_cache=True, verbose=True):
    """
    Run VEGA v4 benchmark across multiple datasets and models.
    
    Args:
        datasets: List of dataset names or list of (dataset, model) tuples
        models: List of model names (if datasets are just dataset names)
        ... other parameters same as single dataset
    
    Returns:
        Dictionary containing all results and summary statistics
    """
    print_header("VEGA v4 Benchmark Suite", "=")
    print_info("Configuration", "")
    print_info("PCA dimension", pca_dim, indent=4)
    print_info("Temperature", temperature, indent=4)
    print_info("Confidence threshold", conf_threshold, indent=4)
    print_info("Contrastive tau", tau_contrast, indent=4)
    print_info("Node weight", node_weight, indent=4)
    print_info("Edge weight", edge_weight, indent=4)
    print_info("Cache enabled", use_cache, indent=4)
    
    all_results = []
    
    # Determine benchmark pairs
    if models is not None and isinstance(datasets[0], str):
        # datasets are dataset names, models are specified separately
        benchmark_pairs = [(d, m) for d in datasets for m in models]
    elif isinstance(datasets[0], (list, tuple)):
        # datasets are (dataset, model) tuples
        benchmark_pairs = datasets
    else:
        # Try to auto-detect models for each dataset
        benchmark_pairs = []
        for dataset in datasets:
            dataset_path = project_root / "data" / dataset
            if dataset_path.exists():
                for model_dir in dataset_path.iterdir():
                    if model_dir.is_dir():
                        benchmark_pairs.append((dataset, model_dir.name))
    
    total = len(benchmark_pairs)
    progress = ProgressBar(total, "Running Benchmarks")
    
    for dataset_name, model_name in benchmark_pairs:
        try:
            result = run_single_dataset_benchmark(
                dataset_name=dataset_name,
                model_name=model_name,
                pca_dim=pca_dim,
                temperature=temperature,
                conf_threshold=conf_threshold,
                tau_contrast=tau_contrast,
                node_weight=node_weight,
                edge_weight=edge_weight,
                use_cache=use_cache,
                verbose=verbose
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\nError processing {dataset_name}/{model_name}: {e}")
        
        progress.update()
    
    progress.close()
    
    # Compute summary statistics
    if all_results:
        vega_scores = [r['vega_score'] for r in all_results if r['vega_score'] is not None]
        logme_scores = [r['logme_score'] for r in all_results if r['logme_score'] is not None]
        
        summary = {
            'total_datasets': len(all_results),
            'vega_mean': np.mean(vega_scores) if vega_scores else None,
            'vega_std': np.std(vega_scores) if vega_scores else None,
            'vega_min': np.min(vega_scores) if vega_scores else None,
            'vega_max': np.max(vega_scores) if vega_scores else None,
            'logme_mean': np.mean(logme_scores) if logme_scores else None,
            'logme_std': np.std(logme_scores) if logme_scores else None,
        }
    else:
        summary = {
            'total_datasets': 0,
            'error': 'No successful results'
        }
    
    return {
        'results': all_results,
        'summary': summary,
        'config': {
            'pca_dim': pca_dim,
            'temperature': temperature,
            'conf_threshold': conf_threshold,
            'tau_contrast': tau_contrast,
            'node_weight': node_weight,
            'edge_weight': edge_weight
        },
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# Print Final Results
# =============================================================================

def print_final_results(benchmark_results):
    """Print formatted final results summary."""
    print_header("Benchmark Results Summary", "=")
    
    results = benchmark_results['results']
    summary = benchmark_results['summary']
    config = benchmark_results['config']
    
    # Print configuration
    print_section("Configuration")
    print_info("PCA dimension", config['pca_dim'])
    print_info("Temperature", config['temperature'])
    print_info("Confidence threshold", config['conf_threshold'])
    print_info("Contrastive tau", config['tau_contrast'])
    print_info("Node weight", config['node_weight'])
    print_info("Edge weight", config['edge_weight'])
    
    # Print summary statistics
    print_section("Summary Statistics")
    print_info("Total datasets processed", summary['total_datasets'])
    if summary.get('vega_mean') is not None:
        print_metric("VEGA v4 Mean Score", summary['vega_mean'])
        print_metric("VEGA v4 Std Dev", summary['vega_std'])
        print_metric("VEGA v4 Min Score", summary['vega_min'])
        print_metric("VEGA v4 Max Score", summary['vega_max'])
    if summary.get('logme_mean') is not None:
        print_metric("LogME Mean Score", summary['logme_mean'])
        print_metric("LogME Std Dev", summary['logme_std'])
    
    # Print individual results
    print_section("Individual Results")
    print(f"\n{'Dataset':<25} {'Model':<20} {'VEGA v4':>12} {'Node':>10} {'Edge':>10} {'LogME':>10}")
    print("-" * 90)
    
    for r in results:
        dataset = r['dataset'][:24] if len(r['dataset']) > 24 else r['dataset']
        model = r['model'][:19] if len(r['model']) > 19 else r['model']
        vega = f"{r['vega_score']:.4f}" if r['vega_score'] is not None else "N/A"
        node = f"{r['vega_node_score']:.4f}" if r.get('vega_node_score') is not None else "N/A"
        edge = f"{r['vega_edge_score']:.4f}" if r.get('vega_edge_score') is not None else "N/A"
        logme = f"{r['logme_score']:.4f}" if r['logme_score'] is not None else "N/A"
        print(f"{dataset:<25} {model:<20} {vega:>12} {node:>10} {edge:>10} {logme:>10}")
    
    print("\n" + "=" * 90)


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main entry point for VEGA v4 benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VEGA v4 Benchmark: Confidence-Weighted Visual Graph with Contrastive-Aware Edges"
    )
    parser.add_argument(
        '--dataset', '-d', type=str, default=None,
        help='Dataset name (default: run all available datasets)'
    )
    parser.add_argument(
        '--model', '-m', type=str, default=None,
        help='Model name (default: all available models)'
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='List of datasets to benchmark'
    )
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='List of models to benchmark'
    )
    parser.add_argument(
        '--pca-dim', type=int, default=64,
        help='PCA dimension (default: 64)'
    )
    parser.add_argument(
        '--temperature', '-t', type=float, default=0.05,
        help='Temperature for probability computation (default: 0.05)'
    )
    parser.add_argument(
        '--conf-threshold', '-c', type=float, default=0.5,
        help='Confidence threshold for sample filtering (default: 0.5)'
    )
    parser.add_argument(
        '--tau-contrast', type=float, default=0.1,
        help='Contrastive temperature for edge similarity (default: 0.1)'
    )
    parser.add_argument(
        '--node-weight', type=float, default=1.0,
        help='Weight for node-level score (default: 1.0)'
    )
    parser.add_argument(
        '--edge-weight', type=float, default=1.0,
        help='Weight for edge-level score (default: 1.0)'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching of results'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    use_cache = not args.no_cache
    
    # Run benchmark
    if args.dataset is not None and args.model is not None:
        # Single dataset-model pair
        result = run_single_dataset_benchmark(
            dataset_name=args.dataset,
            model_name=args.model,
            pca_dim=args.pca_dim,
            temperature=args.temperature,
            conf_threshold=args.conf_threshold,
            tau_contrast=args.tau_contrast,
            node_weight=args.node_weight,
            edge_weight=args.edge_weight,
            use_cache=use_cache,
            verbose=verbose
        )
        results = {'results': [result] if result else [], 'summary': {}, 'config': {}}
    else:
        # Multiple datasets
        datasets = args.datasets if args.datasets else []
        models = args.models if args.models else None
        
        # If no datasets specified, discover available ones
        if not datasets:
            data_dir = project_root / "data"
            if data_dir.exists():
                datasets = [d.name for d in data_dir.iterdir() if d.is_dir()]
        
        results = run_multi_dataset_benchmark(
            datasets=datasets,
            models=models,
            pca_dim=args.pca_dim,
            temperature=args.temperature,
            conf_threshold=args.conf_threshold,
            tau_contrast=args.tau_contrast,
            node_weight=args.node_weight,
            edge_weight=args.edge_weight,
            use_cache=use_cache,
            verbose=verbose
        )
    
    # Print results
    if verbose and results['results']:
        print_final_results(results)
    
    # Save to output file if specified
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()