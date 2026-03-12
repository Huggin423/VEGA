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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VEGA v4 Benchmark Script
Benchmark script for VEGAv4Scorer with confidence-weighted visual graph nodes
and contrastive-aware edge similarity.

Key Features:
- Confidence-weighted visual graph nodes using prediction confidence to weight samples
- Contrastive-aware edge similarity using contrastive-weighted Pearson correlation

Data Structure (SWAB):
- Logits: ptm_stats/logits/{model}__{dataset}.pth
- Image features: ptm_stats/stats_on_hist_task/img_feat/{model}.pkl
- Text features: ptm_stats/class_text_feat/{model}.pkl
- Class names: data/datasets/classnames/{dataset}.txt
- Class-level accuracy: ptm_stats/stats_on_hist_task/class_level_acc/{model}.pkl

Author: VEGA Team
Date: 2024
"""

import os
import sys
import time
import pickle
import hashlib
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from methods.baseline.vega_v4 import VEGAv4Scorer
from methods.baseline.logme import LogME
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

# Default data directory - can be configured for SWAB project structure
# On server: /root/mxy/SWAB
# Local: use project_root / "data" or configure via --data-dir
DEFAULT_DATA_DIR = project_root / "data"

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
# Data Loading Functions (SWAB Structure)
# =============================================================================

def load_logits_data(dataset_name, model_name, data_dir=None):
    """
    Load logits and labels for a dataset using SWAB structure.
    
    SWAB path: ptm_stats/logits/{model}__{dataset}.pth
    
    Returns:
        logits: numpy array (N, num_classes)
        labels: numpy array (N,)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
    # SWAB structure: ptm_stats/logits/{model}__{dataset}.pth
    logits_path = data_dir / "ptm_stats" / "logits" / f"{model_name}__{dataset_name}.pth"
    
    if logits_path.exists():
        try:
            data = torch.load(logits_path, map_location='cpu')
            if isinstance(data, dict):
                logits = data.get('logits', data.get('predictions', None))
                labels = data.get('labels', data.get('targets', None))
                if logits is not None and labels is not None:
                    if isinstance(logits, torch.Tensor):
                        logits = logits.numpy()
                    if isinstance(labels, torch.Tensor):
                        labels = labels.numpy()
                    return logits, labels
            elif isinstance(data, (list, tuple)):
                logits, labels = data[0], data[1]
                if isinstance(logits, torch.Tensor):
                    logits = logits.numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                return logits, labels
            elif isinstance(data, torch.Tensor):
                # Only logits, need to find labels separately
                logits = data.numpy()
                labels = load_labels_from_logits_file(logits_path, dataset_name, model_name, data_dir)
                return logits, labels
        except Exception as e:
            print(f"Warning: Failed to load logits from {logits_path}: {e}")
    
    # Fallback: try alternative paths
    alt_paths = [
        data_dir / dataset_name / model_name / "logits.npy",
        data_dir / dataset_name / model_name / "predictions.npy",
        data_dir / "logits" / f"{dataset_name}_{model_name}.npy",
    ]
    
    for path in alt_paths:
        if path.exists():
            try:
                logits = np.load(path)
                labels_path = path.parent / "labels.npy"
                if labels_path.exists():
                    labels = np.load(labels_path)
                    return logits, labels
            except Exception:
                continue
    
    return None, None


def load_labels_from_logits_file(logits_path, dataset_name, model_name, data_dir):
    """Try to load labels from alternative sources."""
    data_dir = Path(data_dir)
    
    # Try class-level accuracy file which may contain labels
    acc_path = data_dir / "ptm_stats" / "stats_on_hist_task" / "class_level_acc" / f"{model_name}.pkl"
    if acc_path.exists():
        try:
            with open(acc_path, 'rb') as f:
                acc_data = pickle.load(f)
                if isinstance(acc_data, dict):
                    # Look for labels in the accuracy data
                    for key in ['labels', 'targets', 'gt', 'ground_truth']:
                        if key in acc_data:
                            labels = acc_data[key]
                            if isinstance(labels, torch.Tensor):
                                labels = labels.numpy()
                            return labels
        except Exception:
            pass
    
    return None


def load_image_features(model_name, data_dir=None):
    """
    Load image features using SWAB structure.
    
    SWAB path: ptm_stats/stats_on_hist_task/img_feat/{model}.pkl
    
    Returns:
        features: dict with class statistics or numpy array
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
    # SWAB structure: ptm_stats/stats_on_hist_task/img_feat/{model}.pkl
    img_feat_path = data_dir / "ptm_stats" / "stats_on_hist_task" / "img_feat" / f"{model_name}.pkl"
    
    if img_feat_path.exists():
        try:
            with open(img_feat_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                # Could be class-wise statistics: {class_id: {'mean': ..., 'cov': ...}}
                # Or could be direct feature storage
                return data
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, torch.Tensor):
                return data.numpy()
        except Exception as e:
            print(f"Warning: Failed to load image features from {img_feat_path}: {e}")
    
    # Fallback paths
    alt_paths = [
        data_dir / "features" / f"{model_name}_img.npy",
        data_dir / "image_features" / f"{model_name}.npy",
    ]
    
    for path in alt_paths:
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                continue
    
    return None


def load_text_features(model_name, data_dir=None):
    """
    Load text features (class embeddings) using SWAB structure.
    
    SWAB path: ptm_stats/class_text_feat/{model}.pkl
    
    Returns:
        features: numpy array (num_classes, feature_dim) or dict
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
    # SWAB structure: ptm_stats/class_text_feat/{model}.pkl
    text_feat_path = data_dir / "ptm_stats" / "class_text_feat" / f"{model_name}.pkl"
    
    if text_feat_path.exists():
        try:
            with open(text_feat_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                # Could be {class_name: embedding} format
                # Convert to array if possible
                if 'embeddings' in data:
                    emb = data['embeddings']
                    if isinstance(emb, torch.Tensor):
                        return emb.numpy()
                    return emb
                # Try to stack class embeddings
                try:
                    embeddings = []
                    for key, val in data.items():
                        if isinstance(val, (np.ndarray, torch.Tensor)):
                            if isinstance(val, torch.Tensor):
                                val = val.numpy()
                            embeddings.append(val)
                    if embeddings:
                        return np.stack(embeddings)
                except Exception:
                    pass
                return data
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, torch.Tensor):
                return data.numpy()
        except Exception as e:
            print(f"Warning: Failed to load text features from {text_feat_path}: {e}")
    
    # Fallback paths
    alt_paths = [
        data_dir / "features" / f"{model_name}_text.npy",
        data_dir / "text_features" / f"{model_name}.npy",
    ]
    
    for path in alt_paths:
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                continue
    
    return None


def load_class_names(dataset_name, data_dir=None):
    """
    Load class names for a dataset.
    
    SWAB path: data/datasets/classnames/{dataset}.txt
    
    Returns:
        class_names: list of class name strings
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
    # SWAB structure: data/datasets/classnames/{dataset}.txt
    classnames_path = data_dir / "data" / "datasets" / "classnames" / f"{dataset_name}.txt"
    
    if classnames_path.exists():
        try:
            with open(classnames_path, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
            return class_names
        except Exception as e:
            print(f"Warning: Failed to load class names from {classnames_path}: {e}")
    
    # Alternative path patterns
    alt_paths = [
        data_dir / "classnames" / f"{dataset_name}.txt",
        data_dir / "class_names" / f"{dataset_name}.txt",
        data_dir / dataset_name / "class_names.txt",
    ]
    
    for path in alt_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                return class_names
            except Exception:
                continue
    
    return None


# =============================================================================
# VEGA v4 Score Computation
# =============================================================================

def compute_vega_score_v4(image_features, text_features, class_names=None,
                          pca_dim=64, temperature=0.05, conf_threshold=0.5,
                          tau_contrast=0.1, node_weight=1.0, edge_weight=1.0,
                          verbose=True):
    """
    Compute VEGA v4 score with confidence-weighted nodes and contrastive-aware edges.
    
    VEGAv4Scorer.compute_score signature:
        visual_features: Image feature statistics (dict or array)
        text_embeddings: Text embeddings for each class
        class_names: List of class names
    
    Args:
        image_features: Image feature statistics dict or array
            - If dict: {class_id: {'mean': array, 'cov': array, ...}}
            - If array: (N, D) raw features (will need labels to compute stats)
        text_features: Text embeddings (num_classes, D) or dict
        class_names: List of class name strings
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
    
    # Compute score using VEGAv4Scorer.compute_score
    start_time = time.time()
    try:
        score = scorer.compute_score(
            visual_features=image_features,
            text_embeddings=text_features,
            class_names=class_names
        )
        elapsed_time = time.time() - start_time
        
        # Build result dictionary
        result = {
            'score': score,
            'node_score': getattr(scorer, 'last_node_score', None),
            'edge_score': getattr(scorer, 'last_edge_score', None),
            'graph_info': getattr(scorer, 'graph_info', {})
        }
        
        if verbose:
            print_info("Computation time", f"{elapsed_time:.3f}s")
            print_info("VEGA v4 Score", f"{score:.4f}")
            if result['node_score'] is not None:
                print_info("Node Score (confidence-weighted)", f"{result['node_score']:.4f}")
            if result['edge_score'] is not None:
                print_info("Edge Score (contrastive-aware)", f"{result['edge_score']:.4f}")
            if result['graph_info']:
                info = result['graph_info']
                print_info("Graph Statistics", "")
                if 'num_nodes' in info:
                    print_info("Number of nodes", info['num_nodes'], indent=4)
                if 'num_edges' in info:
                    print_info("Number of edges", info['num_edges'], indent=4)
                if 'avg_confidence' in info:
                    print_info("Average confidence", f"{info['avg_confidence']:.4f}", indent=4)
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error computing VEGA v4 score: {e}")
        import traceback
        traceback.print_exc()
        return {
            'score': None,
            'node_score': None,
            'edge_score': None,
            'graph_info': {},
            'error': str(e)
        }


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
                                 data_dir=None,
                                 use_cache=True, verbose=True):
    """
    Run VEGA v4 benchmark on a single dataset-model pair.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR10', 'ImageNet')
        model_name: Name of the model (e.g., 'ViT-B-16', 'RN50')
        pca_dim: Target dimension after PCA
        temperature: Temperature for probability computation
        conf_threshold: Confidence threshold for filtering high-quality samples
        tau_contrast: Contrastive temperature for edge similarity
        node_weight: Weight for node-level score
        edge_weight: Weight for edge-level score
        data_dir: Base data directory (default: DEFAULT_DATA_DIR)
        use_cache: Whether to use caching
        verbose: Whether to print progress information
    
    Returns:
        Dictionary containing all scores and metrics
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
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
        print_info("Data directory", str(data_dir))
    
    # Load data using SWAB structure
    if verbose:
        print_section("Loading Data")
    
    # Load image features (class-wise statistics)
    image_features = load_image_features(model_name, data_dir)
    if image_features is None:
        if verbose:
            print(f"Warning: Could not load image features for {model_name}")
        return None
    
    # Load text features (class embeddings)
    text_features = load_text_features(model_name, data_dir)
    if text_features is None:
        if verbose:
            print(f"Warning: Could not load text features for {model_name}")
        return None
    
    # Load class names
    class_names = load_class_names(dataset_name, data_dir)
    if class_names is None:
        if verbose:
            print(f"Warning: Could not load class names for {dataset_name}")
        # Try to infer number of classes from text features
        if isinstance(text_features, dict):
            class_names = list(text_features.keys()) if text_features else None
        elif isinstance(text_features, np.ndarray):
            class_names = [f"class_{i}" for i in range(text_features.shape[0])]
    
    # Load logits and labels (optional, for additional info)
    logits, labels = load_logits_data(dataset_name, model_name, data_dir)
    
    if verbose:
        if isinstance(image_features, dict):
            print_info("Image features type", "dict (class-wise statistics)")
            print_info("Number of classes", len(image_features))
            # Print sample structure
            if image_features:
                first_key = list(image_features.keys())[0]
                first_val = image_features[first_key]
                if isinstance(first_val, dict):
                    print_info("Feature structure", f"keys: {list(first_val.keys())}")
        else:
            print_info("Image features shape", image_features.shape)
        
        if isinstance(text_features, dict):
            print_info("Text features type", "dict (class embeddings)")
            print_info("Number of classes", len(text_features))
        else:
            print_info("Text features shape", text_features.shape)
        
        if class_names is not None:
            print_info("Number of class names", len(class_names))
        
        if logits is not None:
            print_info("Logits shape", logits.shape)
        if labels is not None:
            print_info("Labels shape", labels.shape)
            print_info("Number of unique labels", len(np.unique(labels)))
    
    # Compute VEGA v4 score
    vega_result = compute_vega_score_v4(
        image_features=image_features,
        text_features=text_features,
        class_names=class_names,
        pca_dim=pca_dim,
        temperature=temperature,
        conf_threshold=conf_threshold,
        tau_contrast=tau_contrast,
        node_weight=node_weight,
        edge_weight=edge_weight,
        verbose=verbose
    )
    
    # Compute LogME score for comparison (if we have raw features)
    logme_score = None
    if isinstance(image_features, np.ndarray) and labels is not None:
        logme_score = compute_logme_score(image_features, labels, verbose=verbose)
    elif verbose:
        print_info("LogME Score", "Skipped (class-wise statistics format, need raw features)")
    
    # Compile results
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'vega_score': vega_result['score'],
        'vega_node_score': vega_result.get('node_score'),
        'vega_edge_score': vega_result.get('edge_score'),
        'logme_score': logme_score,
        'graph_info': vega_result.get('graph_info', {}),
        'error': vega_result.get('error'),
        'config': {
            'pca_dim': pca_dim,
            'temperature': temperature,
            'conf_threshold': conf_threshold,
            'tau_contrast': tau_contrast,
            'node_weight': node_weight,
            'edge_weight': edge_weight,
            'data_dir': str(data_dir)
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
                                data_dir=None,
                                use_cache=True, verbose=True):
    """
    Run VEGA v4 benchmark across multiple datasets and models.
    
    Args:
        datasets: List of dataset names or list of (dataset, model) tuples
        models: List of model names (if datasets are just dataset names)
        pca_dim: Target dimension after PCA
        temperature: Temperature for probability computation
        conf_threshold: Confidence threshold for filtering high-quality samples
        tau_contrast: Contrastive temperature for edge similarity
        node_weight: Weight for node-level score
        edge_weight: Weight for edge-level score
        data_dir: Base data directory (default: DEFAULT_DATA_DIR or /root/mxy/SWAB)
        use_cache: Whether to use caching
        verbose: Whether to print progress information
    
    Returns:
        Dictionary containing all results and summary statistics
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
    print_header("VEGA v4 Benchmark Suite", "=")
    print_info("Configuration", "")
    print_info("Data directory", str(data_dir), indent=4)
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
        # Try to auto-detect available model-dataset pairs from logits directory
        benchmark_pairs = []
        logits_dir = data_dir / "ptm_stats" / "logits"
        if logits_dir.exists():
            for logits_file in logits_dir.glob("*.pth"):
                # Parse filename: {model}__{dataset}.pth
                name = logits_file.stem
                if "__" in name:
                    parts = name.split("__")
                    if len(parts) >= 2:
                        model_name = parts[0]
                        dataset_name = "__".join(parts[1:])  # Handle dataset names with __
                        benchmark_pairs.append((dataset_name, model_name))
        
        if not benchmark_pairs:
            # Fallback: try dataset directories
            for dataset in datasets:
                dataset_path = data_dir / dataset
                if dataset_path.exists():
                    for model_dir in dataset_path.iterdir():
                        if model_dir.is_dir():
                            benchmark_pairs.append((dataset, model_dir.name))
    
    if not benchmark_pairs:
        print("Warning: No benchmark pairs found!")
        return {
            'results': [],
            'summary': {'total_datasets': 0, 'error': 'No benchmark pairs found'},
            'config': {},
            'timestamp': datetime.now().isoformat()
        }
    
    total = len(benchmark_pairs)
    print_info("Total benchmark pairs found", total)
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
                data_dir=data_dir,
                use_cache=use_cache,
                verbose=verbose
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\nError processing {dataset_name}/{model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        progress.update()
    
    progress.close()
    
    # Compute summary statistics
    if all_results:
        vega_scores = [r['vega_score'] for r in all_results if r.get('vega_score') is not None]
        logme_scores = [r['logme_score'] for r in all_results if r.get('logme_score') is not None]
        errors = [r for r in all_results if r.get('error')]
        
        summary = {
            'total_datasets': len(all_results),
            'successful': len(vega_scores),
            'errors': len(errors),
            'vega_mean': float(np.mean(vega_scores)) if vega_scores else None,
            'vega_std': float(np.std(vega_scores)) if vega_scores else None,
            'vega_min': float(np.min(vega_scores)) if vega_scores else None,
            'vega_max': float(np.max(vega_scores)) if vega_scores else None,
            'logme_mean': float(np.mean(logme_scores)) if logme_scores else None,
            'logme_std': float(np.std(logme_scores)) if logme_scores else None,
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
            'edge_weight': edge_weight,
            'data_dir': str(data_dir)
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
        description="VEGA v4 Benchmark: Confidence-Weighted Visual Graph with Contrastive-Aware Edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on single dataset-model pair
  python scripts/run_benchmark_v4.py -d CIFAR10 -m ViT-B-16
  
  # Run with SWAB data directory
  python scripts/run_benchmark_v4.py -d CIFAR10 -m ViT-B-16 --data-dir /root/mxy/SWAB
  
  # Run on multiple datasets
  python scripts/run_benchmark_v4.py --datasets CIFAR10 CIFAR100 --models ViT-B-16 RN50
  
  # Run with custom parameters
  python scripts/run_benchmark_v4.py -d CIFAR10 -m ViT-B-16 --conf-threshold 0.7 --tau-contrast 0.15
  
  # Save results to JSON
  python scripts/run_benchmark_v4.py -d CIFAR10 -m ViT-B-16 -o results.json
"""
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
        '--data-dir', type=str, default=None,
        help='Base data directory (default: project_root/data or /root/mxy/SWAB on server)'
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
    parser.add_argument(
        '--list-pairs', action='store_true',
        help='List available dataset-model pairs and exit'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    use_cache = not args.no_cache
    
    # Determine data directory
    data_dir = args.data_dir
    if data_dir is None:
        # Try SWAB default path on server
        swab_path = Path("/root/mxy/SWAB")
        if swab_path.exists():
            data_dir = swab_path
        else:
            data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    
    # List available pairs if requested
    if args.list_pairs:
        print_header("Available Dataset-Model Pairs")
        logits_dir = data_dir / "ptm_stats" / "logits"
        if logits_dir.exists():
            pairs = []
            for logits_file in sorted(logits_dir.glob("*.pth")):
                name = logits_file.stem
                if "__" in name:
                    parts = name.split("__")
                    if len(parts) >= 2:
                        model_name = parts[0]
                        dataset_name = "__".join(parts[1:])
                        pairs.append((dataset_name, model_name))
            
            if pairs:
                print(f"\nFound {len(pairs)} dataset-model pairs in {logits_dir}:")
                print(f"\n{'Dataset':<30} {'Model':<20}")
                print("-" * 50)
                for dataset, model in pairs:
                    print(f"{dataset:<30} {model:<20}")
            else:
                print(f"No dataset-model pairs found in {logits_dir}")
        else:
            print(f"Logits directory not found: {logits_dir}")
            print("\nSearching in data directories...")
            # Fallback to searching data directory
            for search_dir in [data_dir, DEFAULT_DATA_DIR]:
                if search_dir.exists():
                    print(f"\nContents of {search_dir}:")
                    for item in sorted(search_dir.iterdir()):
                        print(f"  {item.name}")
        return None
    
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
            data_dir=data_dir,
            use_cache=use_cache,
            verbose=verbose
        )
        results = {'results': [result] if result else [], 'summary': {}, 'config': {}}
    else:
        # Multiple datasets
        datasets = args.datasets if args.datasets else []
        models = args.models if args.models else None
        
        # If no datasets specified, discover available ones from logits directory
        if not datasets:
            logits_dir = data_dir / "ptm_stats" / "logits"
            if logits_dir.exists():
                datasets = []
                for logits_file in logits_dir.glob("*.pth"):
                    name = logits_file.stem
                    if "__" in name:
                        parts = name.split("__")
                        if len(parts) >= 2:
                            datasets.append("__".join(parts[1:]))
                datasets = list(set(datasets))
        
        results = run_multi_dataset_benchmark(
            datasets=datasets,
            models=models,
            pca_dim=args.pca_dim,
            temperature=args.temperature,
            conf_threshold=args.conf_threshold,
            tau_contrast=args.tau_contrast,
            node_weight=args.node_weight,
            edge_weight=args.edge_weight,
            data_dir=data_dir,
            use_cache=use_cache,
            verbose=verbose
        )
    
    # Print results
    if verbose and results.get('results'):
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
