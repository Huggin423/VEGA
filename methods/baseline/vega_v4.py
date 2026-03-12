"""
VEGA v4: Enhanced Visual-Graph Alignment Scorer

Based on VEGA v2 (VEGAOptimizedScorer) with two targeted enhancements:
1. Confidence-Weighted Visual Graph Nodes: Uses prediction confidence to weight
   samples when computing class-wise statistics for visual graph construction.
2. Contrastive-Aware Edge Similarity: Applies contrastive weighting to Pearson
   correlation when computing edge similarity scores.

Author: VEGA Team
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler


# Small epsilon for numerical stability
EPS = 1e-8


class VEGAv4Scorer:
    """
    VEGA v4 Scorer with confidence-weighted visual graph nodes and
    contrastive-aware edge similarity.
    
    Parameters
    ----------
    min_samples_per_class : int, default=2
        Minimum samples required per class to compute statistics.
    pca_dim : int, default=64
        Target dimension for PCA reduction. Set to None to skip PCA.
    shrinkage_alpha : float, default=0.1
        Regularization strength for covariance shrinkage.
    node_weight : float, default=1.0
        Weight for node similarity component.
    edge_weight : float, default=1.0
        Weight for edge similarity component.
    conf_threshold : float, default=0.5
        Minimum confidence threshold for samples to be included in
        visual graph node statistics.
    tau_contrast : float, default=0.1
        Temperature for contrastive weighting in edge similarity.
    device : str or torch.device, optional
        Device for tensor computations.
    """
    
    def __init__(
        self,
        min_samples_per_class: int = 2,
        pca_dim: int = 64,
        shrinkage_alpha: float = 0.1,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        conf_threshold: float = 0.5,
        tau_contrast: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.min_samples_per_class = min_samples_per_class
        self.pca_dim = pca_dim
        self.shrinkage_alpha = shrinkage_alpha
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.conf_threshold = conf_threshold
        self.tau_contrast = tau_contrast
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Fitted components
        self.pca_: Optional[PCA] = None
        self.scaler_: Optional[StandardScaler] = None
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor on device."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        return x.float().to(self.device)
    
    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    def _normalize_features(
        self, 
        features: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """L2-normalize features."""
        features = self._to_numpy(features)
        return normalize(features, norm='l2', axis=1)
    
    def _apply_pca(
        self,
        features: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        if self.pca_dim is None or features.shape[1] <= self.pca_dim:
            return features
        
        if fit:
            self.pca_ = PCA(n_components=self.pca_dim, random_state=42)
            return self.pca_.fit_transform(features)
        else:
            if self.pca_ is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            return self.pca_.transform(features)
    
    def _compute_shrunk_covariance(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute covariance with Ledoit-Wolf shrinkage.
        
        If weights are provided, computes weighted covariance.
        """
        if weights is not None:
            # Weighted covariance
            weights = weights / (weights.sum() + EPS)
            X_centered = X - np.average(X, axis=0, weights=weights)
            weighted_cov = (X_centered.T * weights) @ X_centered
            # Apply Ledoit-Wolf shrinkage
            lw = LedoitWolf().fit(X)
            shrunk_cov = (
                (1 - self.shrinkage_alpha) * weighted_cov + 
                self.shrinkage_alpha * lw.covariance_
            )
            return shrunk_cov
        else:
            # Standard Ledoit-Wolf covariance
            lw = LedoitWolf().fit(X)
            return lw.covariance_
    
    def build_textual_graph(
        self,
        text_embeddings: Union[np.ndarray, torch.Tensor],
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Build textual graph from class embeddings.
        
        Parameters
        ----------
        text_embeddings : array-like of shape (num_classes, embed_dim)
            Text embeddings for each class.
        class_names : list of str
            Class names corresponding to embeddings.
        
        Returns
        -------
        adj_text : np.ndarray of shape (num_classes, num_classes)
            Normalized adjacency matrix for textual graph.
        class_to_idx : dict
            Mapping from class name to index.
        """
        text_embeddings = self._normalize_features(text_embeddings)
        
        # Compute cosine similarity
        sim_matrix = text_embeddings @ text_embeddings.T
        
        # Normalize to adjacency (remove self-loops, normalize)
        np.fill_diagonal(sim_matrix, 0)
        
        # Row-normalize
        row_sums = sim_matrix.sum(axis=1, keepdims=True) + EPS
        adj_text = sim_matrix / row_sums
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        return adj_text, class_to_idx
    
    def build_visual_graph(
        self,
        visual_features: Union[np.ndarray, torch.Tensor],
        pseudo_labels: np.ndarray,
        probs: np.ndarray,
        conf_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Build visual graph from visual features with confidence weighting.
        
        Parameters
        ----------
        visual_features : array-like of shape (n_samples, feat_dim)
            Visual features for each sample.
        pseudo_labels : np.ndarray of shape (n_samples,)
            Pseudo-labels for each sample.
        probs : np.ndarray of shape (n_samples, num_classes)
            Prediction probabilities for each sample.
        conf_weights : np.ndarray, optional
            Pre-computed confidence weights. If None, computed from probs.
        
        Returns
        -------
        adj_visual : np.ndarray
            Normalized adjacency matrix.
        class_means : np.ndarray of shape (num_classes, pca_dim)
            Class-wise mean feature vectors.
        class_covs : list of np.ndarray
            Class-wise covariance matrices.
        class_effective_sizes : list of float
            Effective sample sizes (Kish formula) for each class.
        """
        # Normalize and reduce dimensions
        visual_features = self._normalize_features(visual_features)
        visual_features = self._apply_pca(visual_features, fit=True)
        
        num_classes = probs.shape[1]
        
        # Compute confidence weights if not provided
        if conf_weights is None:
            conf_weights = probs[np.arange(len(pseudo_labels)), pseudo_labels]
        
        # Initialize outputs
        class_means = np.zeros((num_classes, visual_features.shape[1]))
        class_covs = []
        class_effective_sizes = []
        
        for k in range(num_classes):
            # Get samples for class k
            mask_k = (pseudo_labels == k)
            n_samples = mask_k.sum()
            
            if n_samples < self.min_samples_per_class:
                # Fallback: use all samples with uniform weights
                class_means[k] = visual_features.mean(axis=0)
                class_covs.append(self._compute_shrunk_covariance(visual_features))
                class_effective_sizes.append(float(len(visual_features)))
                continue
            
            # Apply confidence threshold
            X_k = visual_features[mask_k]
            conf_k = conf_weights[mask_k]
            
            # Filter by confidence threshold
            high_conf_mask = conf_k >= self.conf_threshold
            
            if high_conf_mask.sum() < self.min_samples_per_class:
                # Fallback: use all samples for this class
                X_filtered = X_k
                conf_filtered = conf_k
            else:
                X_filtered = X_k[high_conf_mask]
                conf_filtered = conf_k[high_conf_mask]
            
            # Normalize weights
            w = conf_filtered / (conf_filtered.sum() + EPS)
            
            # Confidence-weighted mean
            class_means[k] = np.sum(w[:, np.newaxis] * X_filtered, axis=0)
            
            # Confidence-weighted covariance with shrinkage
            cov_k = self._compute_shrunk_covariance(X_filtered, weights=conf_filtered)
            class_covs.append(cov_k)
            
            # Effective sample size (Kish formula)
            sum_w = conf_filtered.sum()
            sum_w_sq = (conf_filtered ** 2).sum()
            n_eff = (sum_w ** 2) / (sum_w_sq + EPS)
            class_effective_sizes.append(n_eff)
        
        # Compute edge similarities using Bhattacharyya coefficient
        n_classes = num_classes
        adj_visual = np.zeros((n_classes, n_classes))
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                bc = self._compute_bhattacharyya_coefficient(
                    class_means[i], class_covs[i],
                    class_means[j], class_covs[j]
                )
                adj_visual[i, j] = bc
                adj_visual[j, i] = bc
        
        # Normalize adjacency
        np.fill_diagonal(adj_visual, 0)
        row_sums = adj_visual.sum(axis=1, keepdims=True) + EPS
        adj_visual = adj_visual / row_sums
        
        return adj_visual, class_means, class_covs, class_effective_sizes
    
    def _compute_bhattacharyya_coefficient(
        self,
        mu1: np.ndarray,
        cov1: np.ndarray,
        mu2: np.ndarray,
        cov2: np.ndarray
    ) -> float:
        """
        Compute Bhattacharyya coefficient between two Gaussian distributions.
        
        Uses diagonal approximation for numerical stability.
        """
        d = len(mu1)
        
        # Diagonal approximation
        var1 = np.diag(cov1) + EPS
        var2 = np.diag(cov2) + EPS
        
        # Average variance
        sigma = (var1 + var2) / 2
        
        # Mahalanobis distance term
        diff = mu1 - mu2
        maha_term = np.sum(diff ** 2 / sigma) / 4
        
        # Determinant term
        det_term = 0.5 * np.sum(np.log(var1 + var2) - np.log(4 * var1 * var2))
        
        # Bhattacharyya coefficient
        bc = np.exp(-maha_term - det_term)
        
        return float(np.clip(bc, 0, 1))
    
    def _compute_bhattacharyya_coefficient_vectorized(
        self,
        class_means: np.ndarray,
        class_covs: List[np.ndarray]
    ) -> np.ndarray:
        """Vectorized computation of all pairwise Bhattacharyya coefficients."""
        n_classes = len(class_means)
        bc_matrix = np.zeros((n_classes, n_classes))
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                bc = self._compute_bhattacharyya_coefficient(
                    class_means[i], class_covs[i],
                    class_means[j], class_covs[j]
                )
                bc_matrix[i, j] = bc
                bc_matrix[j, i] = bc
        
        return bc_matrix
    
    def compute_node_similarity(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        pseudo_labels: np.ndarray,
        return_weights: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute confidence-weighted node similarity.
        
        Parameters
        ----------
        probs : array-like of shape (n_samples, num_classes)
            Prediction probabilities.
        pseudo_labels : np.ndarray of shape (n_samples,)
            Pseudo-labels for each sample.
        return_weights : bool
            If True, also return confidence weights.
        
        Returns
        -------
        s_n : float
            Confidence-weighted node similarity score.
        conf_weights : np.ndarray, optional
            Confidence weights (if return_weights=True).
        """
        probs = self._to_numpy(probs)
        n_samples = len(pseudo_labels)
        
        # Get confidence for each sample's predicted class
        conf_weights = probs[np.arange(n_samples), pseudo_labels]
        
        # Confidence-weighted average of max probabilities
        total_weight = conf_weights.sum() + EPS
        s_n = (conf_weights * probs[np.arange(n_samples), pseudo_labels]).sum() / total_weight
        
        if return_weights:
            return s_n, conf_weights
        return s_n
    
    def _weighted_pearsonr(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray
    ) -> float:
        """
        Compute weighted Pearson correlation coefficient.
        
        Parameters
        ----------
        x, y : np.ndarray
            Input vectors.
        w : np.ndarray
            Weights (non-negative, will be normalized to sum to 1).
        
        Returns
        -------
        float
            Weighted Pearson correlation coefficient.
        """
        # Normalize weights
        w = w / (w.sum() + EPS)
        
        # Weighted means
        mu_x = np.sum(w * x)
        mu_y = np.sum(w * y)
        
        # Weighted covariance and standard deviations
        cov_xy = np.sum(w * (x - mu_x) * (y - mu_y))
        std_x = np.sqrt(np.sum(w * (x - mu_x) ** 2) + EPS)
        std_y = np.sqrt(np.sum(w * (y - mu_y) ** 2) + EPS)
        
        # Weighted correlation
        corr = cov_xy / (std_x * std_y + EPS)
        
        return float(np.clip(corr, -1, 1))
    
    def compute_edge_similarity(
        self,
        adj_text: np.ndarray,
        adj_visual: np.ndarray
    ) -> float:
        """
        Compute contrastive-aware edge similarity.
        
        Uses weighted Pearson correlation where weights are based on
        textual edge similarities (soft contrastive weighting).
        
        Parameters
        ----------
        adj_text : np.ndarray of shape (num_classes, num_classes)
            Textual adjacency matrix.
        adj_visual : np.ndarray of shape (num_classes, num_classes)
            Visual adjacency matrix.
        
        Returns
        -------
        s_e : float
            Edge similarity score in [0, 1].
        """
        n_classes = adj_text.shape[0]
        
        # Get upper triangle indices (excluding diagonal)
        triu_i, triu_j = np.triu_indices(n_classes, k=1)
        
        if len(triu_i) == 0:
            return 0.5
        
        # Extract edge similarities
        text_edges = adj_text[triu_i, triu_j]
        visual_edges = adj_visual[triu_i, triu_j]
        
        # Compute contrastive weights from textual edges
        # Using softmax with temperature
        weights_unnorm = np.exp(text_edges / self.tau_contrast)
        weights = weights_unnorm / (weights_unnorm.sum() + EPS)
        
        # Weighted Pearson correlation
        corr = self._weighted_pearsonr(text_edges, visual_edges, weights)
        
        # Map to [0, 1]
        s_e = (corr + 1) / 2
        
        return s_e
    
    def compute_score(
        self,
        visual_features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Compute VEGA v4 alignment score.
        
        Parameters
        ----------
        visual_features : array-like of shape (n_samples, feat_dim)
            Visual features for each sample.
        text_embeddings : array-like of shape (num_classes, embed_dim)
            Text embeddings for each class.
        class_names : list of str
            Class names.
        return_details : bool
            If True, return detailed breakdown.
        
        Returns
        -------
        score : float or dict
            Alignment score, or detailed dict if return_details=True.
        """
        # Convert to numpy
        visual_features = self._to_numpy(visual_features)
        text_embeddings = self._to_numpy(text_embeddings)
        
        n_samples = visual_features.shape[0]
        num_classes = text_embeddings.shape[0]
        
        # Step 1: Normalize visual features for similarity computation
        visual_features_norm = self._normalize_features(visual_features)
        text_embeddings_norm = self._normalize_features(text_embeddings)
        
        # Step 2: Compute similarity matrix and probabilities
        sim_matrix = visual_features_norm @ text_embeddings_norm.T
        
        # Softmax to get probabilities
        sim_max = sim_matrix.max(axis=1, keepdims=True)
        sim_exp = np.exp(sim_matrix - sim_max)
        probs = sim_exp / (sim_exp.sum(axis=1, keepdims=True) + EPS)
        
        # Step 3: Get pseudo-labels
        pseudo_labels = np.argmax(probs, axis=1)
        
        # Step 4: Compute node similarity with confidence weights
        s_n, conf_weights = self.compute_node_similarity(
            probs, pseudo_labels, return_weights=True
        )
        
        # Step 5: Build textual graph
        adj_text, class_to_idx = self.build_textual_graph(text_embeddings, class_names)
        
        # Step 6: Build visual graph with confidence weighting
        adj_visual, class_means, class_covs, class_eff_sizes = self.build_visual_graph(
            visual_features, pseudo_labels, probs, conf_weights
        )
        
        # Step 7: Compute edge similarity with contrastive weighting
        s_e = self.compute_edge_similarity(adj_text, adj_visual)
        
        # Step 8: Compute final score
        score = self.node_weight * s_n + self.edge_weight * s_e
        
        if return_details:
            return {
                'score': score,
                'node_similarity': s_n,
                'edge_similarity': s_e,
                'pseudo_labels': pseudo_labels,
                'confidence_weights': conf_weights,
                'class_effective_sizes': class_eff_sizes,
                'adj_text': adj_text,
                'adj_visual': adj_visual,
                'probs': probs,
            }
        
        return score


def compute_vega_v4_score(
    visual_features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
    min_samples_per_class: int = 2,
    pca_dim: int = 64,
    shrinkage_alpha: float = 0.1,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
    conf_threshold: float = 0.5,
    tau_contrast: float = 0.1,
    return_details: bool = False
) -> Union[float, Dict]:
    """
    Convenience function to compute VEGA v4 alignment score.
    
    Parameters
    ----------
    visual_features : array-like of shape (n_samples, feat_dim)
        Visual features for each sample.
    text_embeddings : array-like of shape (num_classes, embed_dim)
        Text embeddings for each class.
    class_names : list of str
        Class names.
    min_samples_per_class : int, default=2
        Minimum samples per class.
    pca_dim : int, default=64
        PCA dimension.
    shrinkage_alpha : float, default=0.1
        Shrinkage regularization.
    node_weight : float, default=1.0
        Weight for node similarity.
    edge_weight : float, default=1.0
        Weight for edge similarity.
    conf_threshold : float, default=0.5
        Confidence threshold for visual graph construction.
    tau_contrast : float, default=0.1
        Temperature for contrastive weighting.
    return_details : bool
        If True, return detailed breakdown.
    
    Returns
    -------
    score : float or dict
        Alignment score, or detailed dict if return_details=True.
    """
    scorer = VEGAv4Scorer(
        min_samples_per_class=min_samples_per_class,
        pca_dim=pca_dim,
        shrinkage_alpha=shrinkage_alpha,
        node_weight=node_weight,
        edge_weight=edge_weight,
        conf_threshold=conf_threshold,
        tau_contrast=tau_contrast,
    )
    
    return scorer.compute_score(
        visual_features=visual_features,
        text_embeddings=text_embeddings,
        class_names=class_names,
        return_details=return_details
    )