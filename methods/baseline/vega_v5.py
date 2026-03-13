"""
VEGA V5: Confidence-Enhanced Node Similarity Scorer
=====================================================
Based on VEGA V2 (PyTorch backend, adaptive temperature scaling)
Enhanced with V4's confidence-aware node features:
  - Confidence threshold filtering
  - Confidence-weighted statistics
  - Effective sample size (Kish formula)

Note: Contrast-weighted edge similarity (tau_contrast) is NOT included.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Constants
EPS = 1e-8


class VEGAv5Scorer:
    """
    VEGA V5 Scorer: Confidence-Enhanced Node Similarity
    
    Combines V2's PyTorch efficiency with V4's confidence-aware node processing.
    
    Key Features:
    1. PyTorch backend (from V2)
    2. Adaptive temperature scaling for node similarity (from V2)
    3. Confidence threshold filtering (from V4)
    4. Confidence-weighted statistics (from V4)
    5. Effective sample size calculation (from V4)
    
    Excluded from V4:
    - Contrast-weighted edge similarity (tau_contrast)
    """
    
    def __init__(
        self,
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        conf_threshold: float = 0.0,  # Confidence threshold for node filtering
        use_confidence_weighting: bool = True,  # Use confidence-weighted statistics
    ):
        """
        Initialize VEGA V5 Scorer.
        
        Args:
            min_samples_per_class: Minimum samples required per class
            pca_dim: Target dimension for PCA reduction
            shrinkage_alpha: Shrinkage intensity for covariance estimation
            node_weight: Weight for node similarity component
            edge_weight: Weight for edge similarity component
            conf_threshold: Minimum confidence to include sample in graph construction
            use_confidence_weighting: Whether to use confidence-weighted statistics
        """
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.conf_threshold = conf_threshold
        self.use_confidence_weighting = use_confidence_weighting
        
        # Cache
        self._class_means: Optional[torch.Tensor] = None
        self._class_covs: Optional[torch.Tensor] = None
        self._class_n_eff: Optional[torch.Tensor] = None  # Effective sample sizes
        self._n_classes: int = 0
        self._pca_components: Optional[torch.Tensor] = None
    
    def _ensure_tensor(self, x) -> torch.Tensor:
        """Ensure input is a tensor on the correct device."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.float()
    
    def _compute_pca(
        self, 
        X: torch.Tensor, 
        n_components: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PCA using PyTorch's low-rank approximation.
        
        Args:
            X: Input tensor of shape (N, D)
            n_components: Number of principal components
            
        Returns:
            transformed: PCA-transformed data of shape (N, n_components)
            components: Principal components of shape (n_components, D)
        """
        # Center the data
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        
        # Use torch.pca_lowrank for efficient PCA
        U, S, V = torch.pca_lowrank(X_centered, q=n_components)
        
        # Transform data
        transformed = X_centered @ V
        
        return transformed, V
    
    def _compute_shrunk_covariance(
        self,
        X: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Ledoit-Wolf shrunk covariance estimate.
        
        Args:
            X: Input tensor of shape (N, D)
            weights: Optional weights of shape (N,)
            
        Returns:
            Shrunk covariance matrix of shape (D, D)
        """
        n, d = X.shape
        
        if weights is None:
            # Standard covariance
            mean = X.mean(dim=0, keepdim=True)
            X_centered = X - mean
            cov = (X_centered.T @ X_centered) / (n - 1 + EPS)
        else:
            # Weighted covariance (confidence-weighted)
            w = weights / (weights.sum() + EPS)
            
            # Weighted mean
            mean = (w.unsqueeze(1) * X).sum(dim=0, keepdim=True)
            X_centered = X - mean
            
            # Weighted covariance
            cov = (X_centered.T * w.unsqueeze(0)) @ X_centered
            
            # Apply shrinkage
            n_eff = self._compute_effective_sample_size(weights)
            cov = cov * n_eff / (n_eff - 1 + EPS)
        
        # Ledoit-Wolf shrinkage towards identity
        identity = torch.eye(d, device=X.device, dtype=X.dtype)
        shrunk_cov = (1 - self.shrinkage_alpha) * cov + self.shrinkage_alpha * identity
        
        return shrunk_cov
    
    def _compute_effective_sample_size(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute effective sample size using Kish formula.
        
        n_eff = (sum(w))^2 / sum(w^2)
        
        Args:
            weights: Weights tensor of shape (N,)
            
        Returns:
            Effective sample size (scalar tensor)
        """
        sum_w = weights.sum()
        sum_w_sq = (weights ** 2).sum()
        n_eff = (sum_w ** 2) / (sum_w_sq + EPS)
        return n_eff
    
    def _compute_bhattacharyya_coefficient(
        self,
        mean1: torch.Tensor,
        cov1: torch.Tensor,
        mean2: torch.Tensor,
        cov2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Bhattacharyya coefficient between two Gaussian distributions.
        
        BC = exp(-0.5 * d^2)
        where d^2 is the Bhattacharyya distance.
        
        Args:
            mean1, cov1: Mean and covariance of first distribution
            mean2, cov2: Mean and covariance of second distribution
            
        Returns:
            Bhattacharyya coefficient (scalar)
        """
        # Difference in means
        diff = mean1 - mean2
        
        # Average covariance
        avg_cov = (cov1 + cov2) / 2
        
        # Compute inverse of average covariance
        try:
            avg_cov_inv = torch.linalg.inv(avg_cov)
        except:
            # Fallback: add small diagonal for numerical stability
            avg_cov = avg_cov + EPS * torch.eye(avg_cov.shape[0], device=avg_cov.device)
            avg_cov_inv = torch.linalg.inv(avg_cov)
        
        # Mahalanobis distance term
        maha_sq = diff @ avg_cov_inv @ diff
        
        # Determinant term
        det_avg = torch.linalg.det(avg_cov)
        det1 = torch.linalg.det(cov1)
        det2 = torch.linalg.det(cov2)
        
        # Log determinant term (for numerical stability)
        log_det_term = 0.5 * torch.log(det_avg + EPS) - 0.25 * (torch.log(det1 + EPS) + torch.log(det2 + EPS))
        
        # Bhattacharyya distance
        b_dist = 0.125 * maha_sq + log_det_term
        
        # Bhattacharyya coefficient
        bc = torch.exp(-b_dist)
        
        return bc
    
    def build_visual_graph(
        self,
        visual_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        conf_weights: torch.Tensor,
        n_classes: int
    ) -> None:
        """
        Build visual graph with confidence-aware statistics.
        
        Args:
            visual_features: Visual features of shape (N, D)
            pseudo_labels: Pseudo labels of shape (N,)
            conf_weights: Confidence weights of shape (N,)
            n_classes: Number of classes
        """
        X = self._ensure_tensor(visual_features)
        labels = self._ensure_tensor(pseudo_labels).long()
        conf = self._ensure_tensor(conf_weights)
        
        n, d = X.shape
        self._n_classes = n_classes
        
        # Apply PCA if needed
        if d > self.pca_dim:
            X, self._pca_components = self._compute_pca(X, self.pca_dim)
            d = self.pca_dim
        
        # Initialize storage
        class_means = torch.zeros(n_classes, d, device=X.device, dtype=X.dtype)
        class_covs = torch.zeros(n_classes, d, d, device=X.device, dtype=X.dtype)
        class_n_eff = torch.zeros(n_classes, device=X.device, dtype=X.dtype)
        
        for k in range(n_classes):
            mask_k = (labels == k)
            n_k = mask_k.sum().item()
            
            if n_k < self.min_samples_per_class:
                # Not enough samples - use identity covariance
                class_covs[k] = torch.eye(d, device=X.device, dtype=X.dtype)
                continue
            
            X_k = X[mask_k]
            conf_k = conf[mask_k]
            
            # Confidence threshold filtering (from V4)
            if self.conf_threshold > 0:
                high_conf_mask = conf_k >= self.conf_threshold
                n_high_conf = high_conf_mask.sum().item()
                
                if n_high_conf < self.min_samples_per_class:
                    # Fallback: use all samples if not enough high-confidence ones
                    high_conf_mask = torch.ones_like(conf_k, dtype=torch.bool)
            
            if self.use_confidence_weighting and conf_k.sum() > EPS:
                # Confidence-weighted statistics (from V4)
                X_filtered = X_k if self.conf_threshold == 0 else X_k[high_conf_mask] if high_conf_mask is not None else X_k
                conf_filtered = conf_k if self.conf_threshold == 0 else conf_k[high_conf_mask] if high_conf_mask is not None else conf_k
                
                # Normalize weights
                w = conf_filtered / (conf_filtered.sum() + EPS)
                
                # Weighted mean
                class_means[k] = (w.unsqueeze(1) * X_filtered).sum(dim=0)
                
                # Weighted covariance with shrinkage
                class_covs[k] = self._compute_shrunk_covariance(X_filtered, conf_filtered)
                
                # Effective sample size (Kish formula)
                class_n_eff[k] = self._compute_effective_sample_size(conf_filtered)
            else:
                # Standard statistics (from V2)
                class_means[k] = X_k.mean(dim=0)
                class_covs[k] = self._compute_shrunk_covariance(X_k)
                class_n_eff[k] = float(n_k)
        
        # Store results
        self._class_means = class_means
        self._class_covs = class_covs
        self._class_n_eff = class_n_eff
    
    def compute_node_similarity(
        self,
        class_features: Dict[int, torch.Tensor],
        class_probs: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute node similarity matrix using Bhattacharyya coefficients
        with adaptive temperature scaling (V2 style).
        
        Note: Confidence-weighted average similarity is optional via class_probs.
        
        Args:
            class_features: Dict mapping class index to features
            class_probs: Optional dict mapping class index to probabilities (for weighted averaging)
            
        Returns:
            Node similarity matrix of shape (n_classes, n_classes)
        """
        if self._class_means is None:
            raise ValueError("Must call build_visual_graph first")
        
        n = self._n_classes
        device = self._class_means.device
        dtype = self._class_means.dtype
        
        # Compute pairwise Bhattacharyya coefficients
        S_node = torch.zeros(n, n, device=device, dtype=dtype)
        
        for i in range(n):
            for j in range(i, n):
                if self._class_covs[i].sum() == 0 or self._class_covs[j].sum() == 0:
                    bc = torch.tensor(0.0, device=device, dtype=dtype)
                else:
                    bc = self._compute_bhattacharyya_coefficient(
                        self._class_means[i], self._class_covs[i],
                        self._class_means[j], self._class_covs[j]
                    )
                S_node[i, j] = bc
                S_node[j, i] = bc
        
        # Adaptive temperature scaling (V2 style)
        # This normalizes the similarity scores to be more interpretable
        S_node_flat = S_node.view(-1)
        mean_s = S_node_flat.mean()
        std_s = S_node_flat.std() + EPS
        
        # Instance-level normalization (adaptive temperature)
        S_node_scaled = (S_node - mean_s) / std_s
        
        # Convert to similarity via sigmoid
        S_node_scaled = torch.sigmoid(S_node_scaled)
        
        # Set diagonal to 1 (self-similarity)
        S_node_scaled.fill_diagonal_(1.0)
        
        return S_node_scaled
    
    def compute_edge_similarity(
        self,
        edge_indices: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute edge similarity matrix using Pearson correlation.
        
        Note: This does NOT include contrast weighting from V4.
        
        Args:
            edge_indices: Edge indices of shape (2, E)
            edge_features: Optional edge features
            
        Returns:
            Edge similarity matrix
        """
        if self._class_means is None:
            raise ValueError("Must call build_visual_graph first")
        
        n = self._n_classes
        
        # Compute edge similarity based on class relationships
        # Simple cosine similarity between class means
        means_normalized = F.normalize(self._class_means, dim=1)
        S_edge = means_normalized @ means_normalized.t()
        
        return S_edge
    
    def compute_vega_score(
        self,
        visual_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        conf_weights: torch.Tensor,
        n_classes: int,
        class_features: Optional[Dict[int, torch.Tensor]] = None,
        class_probs: Optional[Dict[int, torch.Tensor]] = None,
        edge_indices: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VEGA score combining node and edge similarities.
        
        Args:
            visual_features: Visual features of shape (N, D)
            pseudo_labels: Pseudo labels of shape (N,)
            conf_weights: Confidence weights of shape (N,)
            n_classes: Number of classes
            class_features: Optional dict for node similarity computation
            class_probs: Optional dict for weighted node similarity
            edge_indices: Optional edge indices
            edge_features: Optional edge features
            
        Returns:
            Tuple of (total_score, node_score, edge_score)
        """
        # Build visual graph with confidence-aware statistics
        self.build_visual_graph(visual_features, pseudo_labels, conf_weights, n_classes)
        
        # Compute similarities
        S_node = self.compute_node_similarity(class_features or {}, class_probs)
        S_edge = self.compute_edge_similarity(edge_indices or torch.zeros(2, 0), edge_features)
        
        # Combine with weights
        total_score = self.node_weight * S_node + self.edge_weight * S_edge
        
        return total_score, S_node, S_edge
    
    def get_confidence_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get confidence-related statistics for analysis.
        
        Returns:
            Dict containing:
                - class_means: Computed class means
                - class_covs: Computed class covariances
                - class_n_eff: Effective sample sizes per class
        """
        return {
            'class_means': self._class_means,
            'class_covs': self._class_covs,
            'class_n_eff': self._class_n_eff
        }
    
    def forward(
        self,
        visual_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        conf_weights: torch.Tensor,
        n_classes: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass computing VEGA score.
        
        Args:
            visual_features: Visual features of shape (N, D)
            pseudo_labels: Pseudo labels of shape (N,)
            conf_weights: Confidence weights of shape (N,)
            n_classes: Number of classes
            
        Returns:
            Combined similarity score matrix
        """
        total_score, _, _ = self.compute_vega_score(
            visual_features, pseudo_labels, conf_weights, n_classes, **kwargs
        )
        return total_score


# Convenience function for backward compatibility
def compute_vega_v5_score(
    visual_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    conf_weights: torch.Tensor,
    n_classes: int,
    conf_threshold: float = 0.0,
    use_confidence_weighting: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function to compute VEGA V5 score.
    
    Args:
        visual_features: Visual features of shape (N, D)
        pseudo_labels: Pseudo labels of shape (N,)
        conf_weights: Confidence weights of shape (N,)
        n_classes: Number of classes
        conf_threshold: Minimum confidence to include sample
        use_confidence_weighting: Whether to use confidence-weighted statistics
        **kwargs: Additional arguments passed to VEGAv5Scorer
        
    Returns:
        Combined similarity score matrix
    """
    scorer = VEGAv5Scorer(
        conf_threshold=conf_threshold,
        use_confidence_weighting=use_confidence_weighting,
        **kwargs
    )
    return scorer.forward(visual_features, pseudo_labels, conf_weights, n_classes)