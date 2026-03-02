"""
Test script for VEGA implementation.
Tests the core algorithms with synthetic data.

Run this script on the server with proper environment setup:
    python methods/test_vega.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.baseline.vega import VEGAScorer, VEGAPlus, compute_vega_score


def test_vega_basic():
    """Test VEGA with simple synthetic data."""
    print("=" * 60)
    print("Test 1: Basic VEGA Score Computation")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    n_samples = 100
    n_classes = 5
    feature_dim = 64
    
    # Generate random features (simulating image features)
    visual_features = torch.randn(n_samples, feature_dim)
    
    # Generate text features (class prototypes)
    text_features = torch.randn(n_classes, feature_dim)
    
    # Generate logits (random predictions)
    logits = torch.randn(n_samples, n_classes)
    
    # Compute VEGA score
    scorer = VEGAScorer(temperature=0.05)
    result = scorer.compute_score(visual_features, text_features, logits, return_details=True)
    
    print(f"Visual features shape: {visual_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"\nVEGA Score: {result['score']:.4f}")
    print(f"  - Node Similarity (s_n): {result['node_similarity']:.4f}")
    print(f"  - Edge Similarity (s_e): {result['edge_similarity']:.4f}")
    print(f"  - Valid Classes: {result['valid_classes']}")
    
    # Verify score is in expected range
    assert 0 <= result['node_similarity'] <= 1, "Node similarity should be in [0, 1]"
    assert 0 <= result['edge_similarity'] <= 1, "Edge similarity should be in [0, 1]"
    assert 0 <= result['score'] <= 2, "Total score should be in [0, 2]"
    
    print("\n✅ Test 1 PASSED!")
    return True


def test_vega_well_aligned():
    """Test VEGA with well-aligned visual and textual features."""
    print("\n" + "=" * 60)
    print("Test 2: Well-aligned Features (Expected Higher Score)")
    print("=" * 60)
    
    np.random.seed(123)
    torch.manual_seed(123)
    
    n_samples_per_class = 20
    n_classes = 4
    feature_dim = 32
    
    # Create well-clustered visual features
    visual_features_list = []
    labels_list = []
    
    for k in range(n_classes):
        # Each class has features centered around a prototype
        center = torch.randn(feature_dim) * 2
        class_features = center + torch.randn(n_samples_per_class, feature_dim) * 0.3
        visual_features_list.append(class_features)
        labels_list.extend([k] * n_samples_per_class)
    
    visual_features = torch.cat(visual_features_list, dim=0)
    pseudo_labels = torch.tensor(labels_list)
    
    # Text features are similar to class centers
    text_features = torch.randn(n_classes, feature_dim)
    for k in range(n_classes):
        # Text features are close to their corresponding visual clusters
        class_mean = visual_features[k * n_samples_per_class : (k + 1) * n_samples_per_class].mean(dim=0)
        text_features[k] = class_mean + torch.randn(feature_dim) * 0.1
    
    # Create logits that match pseudo-labels
    logits = torch.zeros(len(visual_features), n_classes)
    for i, label in enumerate(pseudo_labels):
        logits[i, label] = 2.0  # High confidence for correct class
        logits[i] += torch.randn(n_classes) * 0.3
    
    scorer = VEGAScorer(temperature=0.05)
    result = scorer.compute_score(visual_features, text_features, logits, pseudo_labels, return_details=True)
    
    print(f"VEGA Score: {result['score']:.4f}")
    print(f"  - Node Similarity: {result['node_similarity']:.4f}")
    print(f"  - Edge Similarity: {result['edge_similarity']:.4f}")
    print(f"  - Valid Classes: {result['valid_classes']}")
    
    print("\n✅ Test 2 PASSED!")
    return True


def test_vega_poorly_aligned():
    """Test VEGA with poorly-aligned features."""
    print("\n" + "=" * 60)
    print("Test 3: Poorly-aligned Features (Expected Lower Score)")
    print("=" * 60)
    
    np.random.seed(456)
    torch.manual_seed(456)
    
    n_samples = 80
    n_classes = 4
    feature_dim = 32
    
    # Random visual features (no clear clustering)
    visual_features = torch.randn(n_samples, feature_dim)
    
    # Random text features (unrelated to visual features)
    text_features = torch.randn(n_classes, feature_dim) * 3
    
    # Random logits
    logits = torch.randn(n_samples, n_classes)
    
    scorer = VEGAScorer(temperature=0.05)
    result = scorer.compute_score(visual_features, text_features, logits, return_details=True)
    
    print(f"VEGA Score: {result['score']:.4f}")
    print(f"  - Node Similarity: {result['node_similarity']:.4f}")
    print(f"  - Edge Similarity: {result['edge_similarity']:.4f}")
    
    print("\n✅ Test 3 PASSED!")
    return True


def test_bhattacharyya_distance():
    """Test Bhattacharyya distance computation."""
    print("\n" + "=" * 60)
    print("Test 4: Bhattacharyya Distance")
    print("=" * 60)
    
    scorer = VEGAScorer()
    
    # Test 1: Identical distributions -> distance should be ~0
    mu = torch.randn(16)
    cov = torch.eye(16)
    cov = cov + torch.randn(16, 16) * 0.1
    cov = cov @ cov.T  # Make positive definite
    cov = cov + torch.eye(16) * 0.1  # Add regularization
    
    dist_identical = scorer._bhattacharyya_distance(mu, cov, mu, cov)
    print(f"Identical distributions distance: {dist_identical:.6f} (should be ~0)")
    
    # Test 2: Different distributions
    mu1 = torch.zeros(16)
    mu2 = torch.ones(16) * 2
    cov1 = torch.eye(16)
    cov2 = torch.eye(16) * 2
    
    dist_diff = scorer._bhattacharyya_distance(mu1, cov1, mu2, cov2)
    print(f"Different distributions distance: {dist_diff:.4f} (should be > 0)")
    
    assert dist_identical < dist_diff, "Identical distributions should have smaller distance"
    
    print("\n✅ Test 4 PASSED!")
    return True


def test_numpy_input():
    """Test VEGA with numpy arrays as input."""
    print("\n" + "=" * 60)
    print("Test 5: NumPy Input Compatibility")
    print("=" * 60)
    
    np.random.seed(789)
    
    n_samples = 50
    n_classes = 3
    feature_dim = 16
    
    # Use numpy arrays
    visual_features = np.random.randn(n_samples, feature_dim).astype(np.float32)
    text_features = np.random.randn(n_classes, feature_dim).astype(np.float32)
    logits = np.random.randn(n_samples, n_classes).astype(np.float32)
    
    score = compute_vega_score(visual_features, text_features, logits)
    
    print(f"NumPy input VEGA Score: {score:.4f}")
    print("\n✅ Test 5 PASSED!")
    return True


def test_vega_plus():
    """Test VEGA+ with confidence weighting."""
    print("\n" + "=" * 60)
    print("Test 6: VEGA+ with Confidence Weighting")
    print("=" * 60)
    
    np.random.seed(999)
    torch.manual_seed(999)
    
    n_samples = 60
    n_classes = 4
    feature_dim = 32
    
    visual_features = torch.randn(n_samples, feature_dim)
    text_features = torch.randn(n_classes, feature_dim)
    logits = torch.randn(n_samples, n_classes)
    
    scorer = VEGAPlus(temperature=0.05, confidence_weight=0.3)
    result = scorer.compute_score(visual_features, text_features, logits, return_details=True)
    
    print(f"VEGA+ Score: {result.get('final_score', result['score']):.4f}")
    print(f"  - Base Score: {result['score']:.4f}")
    print(f"  - Confidence: {result.get('confidence', 'N/A')}")
    
    print("\n✅ Test 6 PASSED!")
    return True


def test_graph_construction():
    """Test textual and visual graph construction."""
    print("\n" + "=" * 60)
    print("Test 7: Graph Construction")
    print("=" * 60)
    
    torch.manual_seed(111)
    
    n_classes = 5
    feature_dim = 16
    
    # Test textual graph
    text_features = torch.randn(n_classes, feature_dim)
    scorer = VEGAScorer()
    
    nodes, edges = scorer.build_textual_graph(text_features)
    
    print(f"Textual Graph:")
    print(f"  - Nodes shape: {nodes.shape}")
    print(f"  - Edges shape: {edges.shape}")
    print(f"  - Edge range: [{edges.min():.4f}, {edges.max():.4f}]")
    
    assert nodes.shape == (n_classes, feature_dim), "Node shape mismatch"
    assert edges.shape == (n_classes, n_classes), "Edge shape mismatch"
    
    # Test visual graph
    n_samples = 50
    visual_features = torch.randn(n_samples, feature_dim)
    pseudo_labels = torch.randint(0, n_classes, (n_samples,))
    
    class_means, class_covs, class_counts, edge_matrix = scorer.build_visual_graph(
        visual_features, pseudo_labels, n_classes
    )
    
    print(f"\nVisual Graph:")
    print(f"  - Valid classes: {list(class_means.keys())}")
    print(f"  - Class counts: {class_counts}")
    
    if edge_matrix is not None:
        print(f"  - Edge matrix shape: {edge_matrix.shape}")
    
    print("\n✅ Test 7 PASSED!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VEGA Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_vega_basic,
        test_vega_well_aligned,
        test_vega_poorly_aligned,
        test_bhattacharyya_distance,
        test_numpy_input,
        test_vega_plus,
        test_graph_construction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)