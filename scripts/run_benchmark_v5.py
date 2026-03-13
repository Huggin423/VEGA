#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEGA V5 Benchmark - Confidence-Enhanced Node Similarity
========================================================
Based on VEGA V2 architecture with V4's confidence-aware node features:
  - Confidence threshold filtering
  - Confidence-weighted statistics
  - Effective sample size (Kish formula)

Note: Contrast-weighted edge similarity (tau_contrast) is NOT included.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from methods.baseline.vega_v5 import VEGAv5Scorer


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # PCA dimension
    "pca_dim": 64,
    
    # Shrinkage for covariance estimation
    "shrinkage_alpha": 0.1,
    
    # Graph weights
    "node_weight": 1.0,
    "edge_weight": 1.0,
    
    # V5 specific: Confidence threshold
    "conf_threshold": 0.0,  # 0.0 means no filtering
    
    # V5 specific: Use confidence weighting
    "use_confidence_weighting": True,
    
    # Minimum samples per class
    "min_samples_per_class": 2,
}


# ============================================================================
# VEGA V5 Score Computation
# ============================================================================

def compute_vega_v5_score(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    model_name: str = "",
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
    conf_threshold: float = 0.0,
    use_confidence_weighting: bool = True,
    min_samples_per_class: int = 2,
    return_stats: bool = False,
    device: str = "cuda",
) -> tuple:
    """
    Compute VEGA V5 score with confidence-aware node similarity.
    
    Args:
        img_features: Image features of shape (N, D)
        text_features: Text features of shape (C, D) where C is number of classes
        model_name: Model name for logging
        pca_dim: Target dimension for PCA reduction
        shrinkage_alpha: Shrinkage intensity for covariance estimation
        node_weight: Weight for node similarity component
        edge_weight: Weight for edge similarity component
        conf_threshold: Minimum confidence to include sample in graph construction
        use_confidence_weighting: Whether to use confidence-weighted statistics
        min_samples_per_class: Minimum samples required per class
        return_stats: Whether to return confidence statistics
        device: Device to run computation on
        
    Returns:
        Tuple of (vega_score, node_score, edge_score, [confidence_stats])
    """
    # Move to device and ensure float32
    img_features = img_features.to(device).float()
    text_features = text_features.to(device).float()
    
    # Normalize features
    img_normalized = F.normalize(img_features, dim=1)
    text_normalized = F.normalize(text_features, dim=1)
    
    # Compute similarity matrix (N, C)
    sim_matrix = img_normalized @ text_normalized.T
    
    # Get probability distribution over classes
    probs = F.softmax(sim_matrix, dim=1)
    
    # Get pseudo labels (argmax)
    pseudo_labels = probs.argmax(dim=1)
    
    # Get confidence weights (probability of assigned class)
    n_samples = img_features.shape[0]
    conf_weights = probs[torch.arange(n_samples, device=device), pseudo_labels]
    
    # Number of classes
    n_classes = text_features.shape[0]
    
    # Initialize VEGA V5 scorer
    scorer = VEGAv5Scorer(
        min_samples_per_class=min_samples_per_class,
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        node_weight=node_weight,
        edge_weight=edge_weight,
        conf_threshold=conf_threshold,
        use_confidence_weighting=use_confidence_weighting,
    )
    
    # Compute VEGA score
    vega_score, node_score, edge_score = scorer.compute_vega_score(
        visual_features=img_features,
        pseudo_labels=pseudo_labels,
        conf_weights=conf_weights,
        n_classes=n_classes,
    )
    
    if return_stats:
        confidence_stats = scorer.get_confidence_stats()
        return vega_score, node_score, edge_score, confidence_stats
    
    return vega_score, node_score, edge_score


def run_vega_v5_on_features(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    class_names: list = None,
    config: dict = None,
    model_name: str = "",
    dataset_name: str = "",
    verbose: bool = True,
) -> dict:
    """
    Run VEGA V5 on pre-extracted features.
    
    Args:
        img_features: Image features of shape (N, D)
        text_features: Text features of shape (C, D)
        class_names: List of class names
        config: Configuration dictionary
        model_name: Model name for logging
        dataset_name: Dataset name for logging
        verbose: Whether to print progress
        
    Returns:
        Dict containing scores and statistics
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VEGA V5 Benchmark")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Image features: {img_features.shape}")
        print(f"Text features: {text_features.shape}")
        print(f"\nConfiguration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
    
    # Run VEGA V5
    start_time = time.time()
    
    vega_score, node_score, edge_score, confidence_stats = compute_vega_v5_score(
        img_features=img_features,
        text_features=text_features,
        model_name=model_name,
        return_stats=True,
        **config
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract statistics
    class_n_eff = confidence_stats['class_n_eff']
    
    if verbose:
        print(f"\nResults:")
        print(f"  VEGA score shape: {vega_score.shape}")
        print(f"  Node score shape: {node_score.shape}")
        print(f"  Edge score shape: {edge_score.shape}")
        print(f"  Score range: [{vega_score.min().item():.4f}, {vega_score.max().item():.4f}]")
        print(f"  Node score mean: {node_score.mean().item():.4f}")
        print(f"  Edge score mean: {edge_score.mean().item():.4f}")
        print(f"  Avg effective sample size: {class_n_eff.mean().item():.2f}")
        print(f"\n  Elapsed time: {elapsed_time:.2f}s")
    
    return {
        "vega_score": vega_score,
        "node_score": node_score,
        "edge_score": edge_score,
        "confidence_stats": confidence_stats,
        "config": config,
        "elapsed_time": elapsed_time,
    }


# ============================================================================
# Feature Loading Utilities
# ============================================================================

def load_features_from_npy(feature_path: str) -> torch.Tensor:
    """Load features from numpy file."""
    features = np.load(feature_path)
    return torch.from_numpy(features)


def load_features_from_pt(feature_path: str) -> torch.Tensor:
    """Load features from PyTorch file."""
    features = torch.load(feature_path)
    return features


def load_features(feature_path: str) -> tuple:
    """
    Load features from file.
    
    Args:
        feature_path: Path to feature file (.npy or .pt)
        
    Returns:
        Tuple of (img_features, text_features, class_names)
    """
    path = Path(feature_path)
    
    if path.suffix == ".npy":
        features = load_features_from_npy(feature_path)
        return features, None, None
    elif path.suffix == ".pt":
        data = torch.load(feature_path)
        if isinstance(data, dict):
            img_features = data.get("img_features", data.get("image_features"))
            text_features = data.get("text_features", data.get("class_features"))
            class_names = data.get("class_names", data.get("classes"))
            return img_features, text_features, class_names
        return data, None, None
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VEGA V5 Benchmark - Confidence-Enhanced Node Similarity"
    )
    
    parser.add_argument(
        "--feature_path", "-f",
        type=str,
        required=True,
        help="Path to feature file (.npy or .pt)"
    )
    
    parser.add_argument(
        "--img_features",
        type=str,
        default=None,
        help="Path to image features (if separate from feature_path)"
    )
    
    parser.add_argument(
        "--text_features",
        type=str,
        default=None,
        help="Path to text features (if separate from feature_path)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results/vega_v5",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        default="",
        help="Model name for logging"
    )
    
    parser.add_argument(
        "--dataset_name", "-d",
        type=str,
        default="",
        help="Dataset name for logging"
    )
    
    # VEGA V5 specific arguments
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=DEFAULT_CONFIG["pca_dim"],
        help="PCA dimension"
    )
    
    parser.add_argument(
        "--shrinkage_alpha",
        type=float,
        default=DEFAULT_CONFIG["shrinkage_alpha"],
        help="Shrinkage alpha for covariance estimation"
    )
    
    parser.add_argument(
        "--node_weight",
        type=float,
        default=DEFAULT_CONFIG["node_weight"],
        help="Weight for node similarity"
    )
    
    parser.add_argument(
        "--edge_weight",
        type=float,
        default=DEFAULT_CONFIG["edge_weight"],
        help="Weight for edge similarity"
    )
    
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CONFIG["conf_threshold"],
        help="Confidence threshold for sample filtering (0.0 = no filtering)"
    )
    
    parser.add_argument(
        "--no_confidence_weighting",
        action="store_true",
        help="Disable confidence weighting for statistics"
    )
    
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=DEFAULT_CONFIG["min_samples_per_class"],
        help="Minimum samples per class"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results to file"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build configuration
    config = {
        "pca_dim": args.pca_dim,
        "shrinkage_alpha": args.shrinkage_alpha,
        "node_weight": args.node_weight,
        "edge_weight": args.edge_weight,
        "conf_threshold": args.conf_threshold,
        "use_confidence_weighting": not args.no_confidence_weighting,
        "min_samples_per_class": args.min_samples_per_class,
    }
    
    # Load features
    print(f"Loading features from {args.feature_path}...")
    
    if args.img_features and args.text_features:
        # Separate files for image and text features
        img_features = load_features_from_npy(args.img_features) if args.img_features.endswith(".npy") else load_features_from_pt(args.img_features)
        text_features = load_features_from_npy(args.text_features) if args.text_features.endswith(".npy") else load_features_from_pt(args.text_features)
        class_names = None
    else:
        # Single file
        img_features, text_features, class_names = load_features(args.feature_path)
    
    print(f"Image features shape: {img_features.shape}")
    if text_features is not None:
        print(f"Text features shape: {text_features.shape}")
    
    # Run VEGA V5
    results = run_vega_v5_on_features(
        img_features=img_features,
        text_features=text_features,
        class_names=class_names,
        config=config,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        verbose=not args.quiet,
    )
    
    # Save results if requested
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"vega_v5_{timestamp}.pt"
        
        # Prepare results for saving
        save_dict = {
            "vega_score": results["vega_score"].cpu(),
            "node_score": results["node_score"].cpu(),
            "edge_score": results["edge_score"].cpu(),
            "confidence_stats": {
                "class_n_eff": results["confidence_stats"]["class_n_eff"].cpu(),
            },
            "config": results["config"],
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "timestamp": timestamp,
        }
        
        torch.save(save_dict, output_file)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()