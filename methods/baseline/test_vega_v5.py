"""
Test script for VEGA V5 Scorer
"""

import torch
import numpy as np
from vega_v5 import VEGAv5Scorer, compute_vega_v5_score


def test_basic_functionality():
    """Test basic VEGA V5 functionality."""
    print("=" * 60)
    print("Testing VEGA V5 Scorer")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 200
    n_features = 128
    n_classes = 5
    
    # Generate random features
    visual_features = torch.randn(n_samples, n_features)
    
    # Generate pseudo labels (balanced classes)
    pseudo_labels = torch.randint(0, n_classes, (n_samples,))
    
    # Generate confidence weights (simulating model confidence)
    conf_weights = torch.rand(n_samples)  # Random confidence between 0 and 1
    
    print(f"\nInput shapes:")
    print(f"  Visual features: {visual_features.shape}")
    print(f"  Pseudo labels: {pseudo_labels.shape}")
    print(f"  Confidence weights: {conf_weights.shape}")
    print(f"  Number of classes: {n_classes}")
    
    # Test 1: Basic scorer without confidence threshold
    print("\n" + "-" * 40)
    print("Test 1: Basic scorer (no confidence threshold)")
    print("-" * 40)
    
    scorer = VEGAv5Scorer(
        min_samples_per_class=2,
        pca_dim=32,
        shrinkage_alpha=0.1,
        conf_threshold=0.0,
        use_confidence_weighting=True
    )
    
    total_score, S_node, S_edge = scorer.compute_vega_score(
        visual_features, pseudo_labels, conf_weights, n_classes
    )
    
    print(f"Output shapes:")
    print(f"  Total score: {total_score.shape}")
    print(f"  Node similarity: {S_node.shape}")
    print(f"  Edge similarity: {S_edge.shape}")
    print(f"\nNode similarity matrix (first 3x3):")
    print(S_node[:3, :3].numpy())
    
    # Test 2: With confidence threshold
    print("\n" + "-" * 40)
    print("Test 2: With confidence threshold (0.5)")
    print("-" * 40)
    
    scorer_threshold = VEGAv5Scorer(
        min_samples_per_class=2,
        pca_dim=32,
        shrinkage_alpha=0.1,
        conf_threshold=0.5,
        use_confidence_weighting=True
    )
    
    total_score_2, S_node_2, S_edge_2 = scorer_threshold.compute_vega_score(
        visual_features, pseudo_labels, conf_weights, n_classes
    )
    
    print(f"Node similarity matrix (first 3x3):")
    print(S_node_2[:3, :3].numpy())
    
    # Get confidence stats
    stats = scorer_threshold.get_confidence_stats()
    print(f"\nEffective sample sizes per class:")
    print(stats['class_n_eff'].numpy())
    
    # Test 3: Without confidence weighting
    print("\n" + "-" * 40)
    print("Test 3: Without confidence weighting")
    print("-" * 40)
    
    scorer_no_weight = VEGAv5Scorer(
        min_samples_per_class=2,
        pca_dim=32,
        shrinkage_alpha=0.1,
        conf_threshold=0.0,
        use_confidence_weighting=False
    )
    
    total_score_3, S_node_3, S_edge_3 = scorer_no_weight.compute_vega_score(
        visual_features, pseudo_labels, conf_weights, n_classes
    )
    
    print(f"Node similarity matrix (first 3x3):")
    print(S_node_3[:3, :3].numpy())
    
    # Test 4: Convenience function
    print("\n" + "-" * 40)
    print("Test 4: Convenience function")
    print("-" * 40)
    
    score = compute_vega_v5_score(
        visual_features, pseudo_labels, conf_weights, n_classes,
        conf_threshold=0.3,
        pca_dim=32
    )
    print(f"Output shape: {score.shape}")
    print(f"Score range: [{score.min().item():.4f}, {score.max().item():.4f}]")
    
    # Test 5: Compare with different thresholds
    print("\n" + "-" * 40)
    print("Test 5: Threshold sensitivity analysis")
    print("-" * 40)
    
    thresholds = [0.0, 0.3, 0.5, 0.7]
    for thresh in thresholds:
        scorer = VEGAv5Scorer(conf_threshold=thresh, pca_dim=32)
        _, S_node, _ = scorer.compute_vega_score(
            visual_features, pseudo_labels, conf_weights, n_classes
        )
        stats = scorer.get_confidence_stats()
        n_eff_avg = stats['class_n_eff'].mean().item()
        print(f"  Threshold {thresh:.1f}: avg_n_eff = {n_eff_avg:.2f}, "
              f"score_mean = {S_node.mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_functionality()