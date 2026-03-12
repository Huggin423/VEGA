# VEGA Experiment Progress Log

## Overview

This document tracks the development and experimentation progress for the VEGA (Visual-Graph Enhanced Alignment) scoring methods.

---

## 2024-01 - VEGA v4 Development

### Objective
Implement VEGA v4 based on VEGA v2 (VEGAOptimizedScorer) with two targeted enhancements:

1. **Enhancement 1: Confidence-Weighted Visual Graph Nodes**
   - Use prediction confidence to weight samples when computing class-wise statistics
   - Apply confidence threshold filtering for visual graph construction
   - Compute confidence-weighted mean and covariance
   - Track effective sample size using Kish formula

2. **Enhancement 2: Contrastive-Aware Edge Similarity**
   - Implement weighted Pearson correlation for edge similarity
   - Use softmax over textual edge similarities as contrastive weights
   - Temperature parameter `tau_contrast` controls weighting sharpness

### Implementation Details

#### New Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conf_threshold` | float | 0.5 | Minimum confidence for samples in visual graph |
| `tau_contrast` | float | 0.1 | Temperature for contrastive weighting |

#### Key Methods Modified from v2

1. **`build_visual_graph()`**
   - Added `probs` and `conf_weights` parameters
   - Implements confidence-weighted mean: `μ_k = Σ conf_i * x_i / Σ conf_i`
   - Implements confidence-weighted covariance with shrinkage
   - Returns effective sample sizes per class

2. **`compute_node_similarity()`**
   - Added `return_weights` option
   - Computes confidence-weighted average: `s_n = Σ conf_i * p_i / Σ conf_i`

3. **`compute_edge_similarity()`**
   - Replaced standard Pearson with weighted Pearson correlation
   - Uses contrastive weights: `w_ij = exp(e_ij^T / tau) / Σ exp(...)`

4. **`_weighted_pearsonr()`** (new)
   - Helper function for weighted correlation computation
   - Properly handles numerical stability with EPS

### Files Created
- `methods/baseline/vega_v4.py` - Main implementation
- `test_vega_v4.py` - Test suite

### Test Results
All 8 tests passed:
1. ✓ Basic functionality
2. ✓ Return details
3. ✓ Confidence threshold variations
4. ✓ Tau contrast variations
5. ✓ PyTorch tensor input
6. ✓ Weighted Pearson correlation
7. ✓ Node similarity computation
8. ✓ Edge similarity computation

### Next Steps
- [ ] Run on benchmark datasets
- [ ] Compare v4 vs v2 performance
- [ ] Hyperparameter tuning (conf_threshold, tau_contrast)
- [ ] Document optimal configurations

---

## Version History

| Version | Date | Status | Description |
|---------|------|--------|-------------|
| v1 | - | Complete | Initial VEGA implementation |
| v2 | - | Complete | Optimized scorer with PCA + Ledoit-Wolf shrinkage |
| v4 | 2024-01 | Complete | Confidence-weighted nodes + Contrastive-aware edges |

---

## Notes

### Design Decisions

1. **Confidence Weighting Rationale**
   - Low-confidence predictions are often from misclassified samples
   - Including them equally in class statistics corrupts the visual graph
   - Weighting by confidence reduces noise from uncertain predictions

2. **Contrastive Weighting Rationale**
   - Hard negatives (semantically similar classes) carry more signal
   - Softmax weighting over textual edges emphasizes these pairs
   - Temperature controls trade-off between uniform and peaked weighting

3. **Kish Effective Sample Size**
   - When using weighted statistics, effective N decreases
   - Formula: `N_eff = (Σ w)² / Σ w²`
   - Useful for monitoring stability of class statistics

### Implementation Notes

- All computations use numpy for stability
- PyTorch tensors are automatically converted
- Numerical stability ensured with EPS constant
- Fallback mechanisms when samples per class < threshold