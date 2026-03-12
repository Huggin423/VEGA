"""
Test script for VEGA v4 implementation.
"""

import numpy as np
import torch
from methods.baseline.vega_v4 import VEGAv4Scorer, compute_vega_v4_score


def test_basic_functionality():
    """Test basic functionality of VEGAv4Scorer."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    feat_dim = 512
    
    # Generate visual features (clustered by class)
    visual_features = np.zeros((n_samples, feat_dim))
    for i in range(n_samples):
        class_idx = i // (n_samples // n_classes)
        center = np.random.randn(feat_dim)
        visual_features[i] = center + 0.1 * np.random.randn(feat_dim)
    
    # Generate text embeddings
    text_embeddings = np.random.randn(n_classes, feat_dim)
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Test with default parameters
    scorer = VEGAv4Scorer()
    score = scorer.compute_score(visual_features, text_embeddings, class_names)
    
    print(f"Score: {score:.4f}")
    print(f"Score is in valid range [0, 2]: {0 <= score <= 2}")
    
    assert isinstance(score, float), "Score should be a float"
    assert 0 <= score <= 2, f"Score should be in [0, 2], got {score}"
    
    print("✓ Test 1 passed!\n")


def test_return_details():
    """Test return_details parameter."""
    print("=" * 60)
    print("Test 2: Return Details")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 50
    n_classes = 3
    feat_dim = 256
    
    visual_features = np.random.randn(n_samples, feat_dim)
    text_embeddings = np.random.randn(n_classes, feat_dim)
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    details = compute_vega_v4_score(
        visual_features, text_embeddings, class_names,
        return_details=True
    )
    
    print("Keys in details:")
    for key in details.keys():
        print(f"  - {key}")
    
    # Check expected keys
    expected_keys = [
        'score', 'node_similarity', 'edge_similarity',
        'pseudo_labels', 'confidence_weights', 'class_effective_sizes',
        'adj_text', 'adj_visual', 'probs'
    ]
    for key in expected_keys:
        assert key in details, f"Missing key: {key}"
    
    # Check shapes
    assert details['pseudo_labels'].shape == (n_samples,), \
        f"Wrong pseudo_labels shape: {details['pseudo_labels'].shape}"
    assert details['confidence_weights'].shape == (n_samples,), \
        f"Wrong confidence_weights shape: {details['confidence_weights'].shape}"
    assert details['adj_text'].shape == (n_classes, n_classes), \
        f"Wrong adj_text shape: {details['adj_text'].shape}"
    assert details['adj_visual'].shape == (n_classes, n_classes), \
        f"Wrong adj_visual shape: {details['adj_visual'].shape}"
    
    print(f"Node similarity: {details['node_similarity']:.4f}")
    print(f"Edge similarity: {details['edge_similarity']:.4f}")
    print(f"Effective sample sizes: {details['class_effective_sizes']}")
    print("✓ Test 2 passed!\n")


def test_confidence_threshold():
    """Test different confidence threshold values."""
    print("=" * 60)
    print("Test 3: Confidence Threshold")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 100
    n_classes = 4
    feat_dim = 256
    
    visual_features = np.random.randn(n_samples, feat_dim)
    text_embeddings = np.random.randn(n_classes, feat_dim)
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Test with different thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    scores = []
    
    for thresh in thresholds:
        score = compute_vega_v4_score(
            visual_features, text_embeddings, class_names,
            conf_threshold=thresh
        )
        scores.append(score)
        print(f"  threshold={thresh:.1f}: score={score:.4f}")
    
    # All scores should be valid
    assert all(0 <= s <= 2 for s in scores), "All scores should be in [0, 2]"
    
    print("✓ Test 3 passed!\n")


def test_tau_contrast():
    """Test different tau_contrast values."""
    print("=" * 60)
    print("Test 4: Tau Contrast")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 80
    n_classes = 5
    feat_dim = 256
    
    visual_features = np.random.randn(n_samples, feat_dim)
    text_embeddings = np.random.randn(n_classes, feat_dim)
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Test with different tau values
    taus = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for tau in taus:
        score = compute_vega_v4_score(
            visual_features, text_embeddings, class_names,
            tau_contrast=tau
        )
        print(f"  tau_contrast={tau:.2f}: score={score:.4f}")
    
    print("✓ Test 4 passed!\n")


def test_torch_tensors():
    """Test with PyTorch tensors as input."""
    print("=" * 60)
    print("Test 5: PyTorch Tensor Input")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 60
    n_classes = 3
    feat_dim = 128
    
    visual_features = torch.randn(n_samples, feat_dim)
    text_embeddings = torch.randn(n_classes, feat_dim)
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    score = compute_vega_v4_score(visual_features, text_embeddings, class_names)
    
    print(f"Score with torch tensors: {score:.4f}")
    assert isinstance(score, float), "Score should be a float"
    
    print("✓ Test 5 passed!\n")


def test_weighted_pearsonr():
    """Test weighted Pearson correlation."""
    print("=" * 60)
    print("Test 6: Weighted Pearson Correlation")
    print("=" * 60)
    
    scorer = VEGAv4Scorer()
    
    # Test case 1: uniform weights should equal standard pearsonr
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    w_uniform = np.ones(5)
    
    corr_weighted = scorer._weighted_pearsonr(x, y, w_uniform)
    
    from scipy.stats import pearsonr
    corr_standard, _ = pearsonr(x, y)
    
    print(f"Weighted corr (uniform): {corr_weighted:.4f}")
    print(f"Standard corr: {corr_standard:.4f}")
    print(f"Difference: {abs(corr_weighted - corr_standard):.6f}")
    
    assert abs(corr_weighted - corr_standard) < 1e-6, \
        "Uniform weights should match standard Pearson"
    
    # Test case 2: non-uniform weights
    w_custom = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # Emphasize last point
    corr_custom = scorer._weighted_pearsonr(x, y, w_custom)
    print(f"Weighted corr (custom): {corr_custom:.4f}")
    
    print("✓ Test 6 passed!\n")


def test_node_similarity():
    """Test node similarity computation."""
    print("=" * 60)
    print("Test 7: Node Similarity")
    print("=" * 60)
    
    scorer = VEGAv4Scorer()
    
    # Create test probabilities
    np.random.seed(42)
    n_samples = 50
    n_classes = 4
    
    probs = np.random.rand(n_samples, n_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
    pseudo_labels = probs.argmax(axis=1)
    
    s_n, conf_weights = scorer.compute_node_similarity(probs, pseudo_labels, return_weights=True)
    
    print(f"Node similarity: {s_n:.4f}")
    print(f"Confidence weights shape: {conf_weights.shape}")
    print(f"Confidence weights range: [{conf_weights.min():.4f}, {conf_weights.max():.4f}]")
    
    assert 0 <= s_n <= 1, f"Node similarity should be in [0, 1], got {s_n}"
    assert len(conf_weights) == n_samples, "Confidence weights length mismatch"
    
    print("✓ Test 7 passed!\n")


def test_edge_similarity():
    """Test edge similarity computation."""
    print("=" * 60)
    print("Test 8: Edge Similarity")
    print("=" * 60)
    
    scorer = VEGAv4Scorer(tau_contrast=0.1)
    
    # Create test adjacency matrices
    np.random.seed(42)
    n_classes = 5
    
    # Create correlated adjacency matrices
    adj_text = np.abs(np.random.randn(n_classes, n_classes))
    adj_text = (adj_text + adj_text.T) / 2
    np.fill_diagonal(adj_text, 0)
    adj_text = adj_text / (adj_text.sum(axis=1, keepdims=True) + 1e-8)
    
    # Add noise to create visual adjacency
    noise = 0.1 * np.random.randn(n_classes, n_classes)
    adj_visual = np.clip(adj_text + noise, 0, 1)
    adj_visual = (adj_visual + adj_visual.T) / 2
    np.fill_diagonal(adj_visual, 0)
    adj_visual = adj_visual / (adj_visual.sum(axis=1, keepdims=True) + 1e-8)
    
    s_e = scorer.compute_edge_similarity(adj_text, adj_visual)
    
    print(f"Edge similarity: {s_e:.4f}")
    assert 0 <= s_e <= 1, f"Edge similarity should be in [0, 1], got {s_e}"
    
    print("✓ Test 8 passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VEGA v4 Test Suite")
    print("=" * 60 + "\n")
    
    test_basic_functionality()
    test_return_details()
    test_confidence_threshold()
    test_tau_contrast()
    test_torch_tensors()
    test_weighted_pearsonr()
    test_node_similarity()
    test_edge_similarity()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()