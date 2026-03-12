#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for VEGARobustScorer
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from methods.baseline.vega_robust import VEGARobustScorer

def test_vega_robust():
    """Test VEGARobustScorer with synthetic data"""
    
    print("=" * 60)
    print("Testing VEGARobustScorer")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test parameters
    N = 100  # samples
    D = 64   # feature dimension (smaller for quick test)
    K = 10   # classes
    
    # Generate random features
    img_features = np.random.randn(N, D).astype(np.float32)
    text_features = np.random.randn(K, D).astype(np.float32)
    logits = np.random.randn(N, K).astype(np.float32)
    
    print(f"\nInput shapes:")
    print(f"  Image features: {img_features.shape}")
    print(f"  Text features: {text_features.shape}")
    print(f"  Logits: {logits.shape}")
    
    # Create scorer
    vega = VEGARobustScorer(
        temperature=0.05,
        shrinkage_alpha=0.1
    )
    
    # Compute score
    result = vega.compute_score(
        img_features, 
        text_features, 
        logits=logits, 
        return_details=True
    )
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"  Score: {result['score']:.4f}")
    print(f"  Node similarity: {result['node_similarity']:.4f}")
    print(f"  Edge similarity: {result['edge_similarity']:.4f}")
    print(f"  Pearson correlation: {result['pearson_correlation']:.4f}")
    print(f"  Valid classes: {result['valid_classes']}")
    print(f"  Full covariance: {result['full_covariance']}")
    print(f"  Shrinkage alpha: {result['shrinkage_alpha']}")
    
    # Verify results
    assert 0.0 <= result['node_similarity'] <= 1.0, "Node similarity out of range"
    assert 0.0 <= result['edge_similarity'] <= 1.0, "Edge similarity out of range"
    assert result['score'] == result['node_similarity'] + result['edge_similarity'], "Score mismatch"
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    
    return True


def test_high_dimensional():
    """Test with high-dimensional features (512-D, like CLIP)"""
    
    print("\n" + "=" * 60)
    print("Testing with high-dimensional features (512-D)")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    N = 50   # samples (small, N << D)
    D = 512  # CLIP feature dimension
    K = 10   # classes
    
    img_features = np.random.randn(N, D).astype(np.float32)
    text_features = np.random.randn(K, D).astype(np.float32)
    logits = np.random.randn(N, K).astype(np.float32)
    
    print(f"\nInput shapes (N={N} << D={D}):")
    print(f"  Image features: {img_features.shape}")
    print(f"  Text features: {text_features.shape}")
    
    vega = VEGARobustScorer(temperature=0.05, shrinkage_alpha=0.1)
    
    try:
        result = vega.compute_score(img_features, text_features, logits=logits, return_details=True)
        
        print(f"\nResults:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Node similarity: {result['node_similarity']:.4f}")
        print(f"  Edge similarity: {result['edge_similarity']:.4f}")
        print(f"  Pearson correlation: {result['pearson_correlation']:.4f}")
        
        print("\nHigh-dimensional test PASSED!")
        return True
    except Exception as e:
        print(f"\nHigh-dimensional test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = True
    
    try:
        success = test_vega_robust() and success
        success = test_high_dimensional() and success
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED!")
        print("=" * 60)
        sys.exit(1)