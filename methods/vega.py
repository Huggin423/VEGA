"""
VEGA: Visual-Text Semantic Graph for Model Selection
Reference: Based on VEGA paper methodology
Implements graph-based model selection for VLMs.

Key Idea:
- Build visual graph from image features
- Build semantic graph from text embeddings
- Score models by graph matching
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Dict, List, Tuple


class VEGAScorer:
    """
    VEGA-based model selection scorer.
    
    Computes transferability score based on semantic consistency
    between visual features and text embeddings.
    
    Usage:
        vega = VEGAScorer()
        score = vega.compute_score(features, text_embeddings, pseudo_labels)
    """
    
    def __init__(self, k_neighbors: int = 10, sigma: float = 1.0):
        """
        Initialize VEGA scorer.
        
        Args:
            k_neighbors: Number of neighbors for graph construction
            sigma: Bandwidth for Gaussian kernel
        """
        self.k_neighbors = k_neighbors
        self.sigma = sigma
    
    def build_knn_graph(self, features: torch.Tensor, k: int = None) -> torch.Tensor:
        """
        Build k-NN graph from features.
        
        Args:
            features: Feature matrix [N, D]
            k: Number of neighbors
            
        Returns:
            Adjacency matrix [N, N] with edge weights
        """
        if k is None:
            k = self.k_neighbors
            
        # Compute pairwise distances
        n = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = features @ features.T
        
        # Keep only k-nearest neighbors
        values, indices = torch.topk(similarity, k=k+1, dim=1)
        
        # Build sparse adjacency
        adj = torch.zeros_like(similarity)
        for i in range(n):
            adj[i, indices[i]] = values[i]
        
        # Make symmetric
        adj = (adj + adj.T) / 2
        
        # Apply Gaussian kernel
        adj = torch.exp((adj - 1) / self.sigma)
        
        # Remove self-loops
        adj = adj * (1 - torch.eye(n, device=adj.device))
        
        return adj
    
    def compute_laplacian(self, adj: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """
        Compute graph Laplacian matrix.
        
        Args:
            adj: Adjacency matrix [N, N]
            normalized: Whether to use normalized Laplacian
            
        Returns:
            Laplacian matrix [N, N]
        """
        # Degree matrix
        degree = adj.sum(dim=1)
        
        if normalized:
            # D^{-1/2}
            d_inv_sqrt = torch.pow(degree, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            D_inv_sqrt = torch.diag(d_inv_sqrt)
            
            # L = I - D^{-1/2} A D^{-1/2}
            L = torch.eye(adj.shape[0], device=adj.device) - D_inv_sqrt @ adj @ D_inv_sqrt
        else:
            # L = D - A
            D = torch.diag(degree)
            L = D - adj
        
        return L
    
    def compute_graph_matching_score(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> float:
        """
        Compute graph matching score between visual and semantic graphs.
        
        Args:
            visual_features: Image features [N, D]
            text_embeddings: Text embeddings per class [C, D]
            pseudo_labels: Predicted labels [N]
            
        Returns:
            Graph matching score
        """
        n_samples = visual_features.shape[0]
        n_classes = text_embeddings.shape[0]
        
        # Build visual graph
        visual_adj = self.build_knn_graph(visual_features)
        
        # Build semantic graph (class-level)
        # Nodes are classes, edges connect similar classes
        semantic_adj = self.build_knn_graph(text_embeddings)
        
        # Map visual samples to semantic nodes
        # Count samples per class
        class_counts = torch.zeros(n_classes, device=visual_features.device)
        for c in range(n_classes):
            class_counts[c] = (pseudo_labels == c).sum().float()
        
        # Compute semantic consistency
        # For each sample, check if its neighbors have the same pseudo-label
        consistency_scores = []
        
        for i in range(n_samples):
            neighbors = torch.where(visual_adj[i] > 0)[0]
            if len(neighbors) == 0:
                continue
            
            # Check label consistency with neighbors
            same_label = (pseudo_labels[neighbors] == pseudo_labels[i]).float()
            weights = visual_adj[i, neighbors]
            
            weighted_consistency = (same_label * weights).sum() / weights.sum()
            consistency_scores.append(weighted_consistency.item())
        
        if not consistency_scores:
            return 0.0
        
        return np.mean(consistency_scores)
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None
    ) -> float:
        """
        Compute VEGA transferability score.
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [C, D]
            logits: Model predictions [N, C] (optional)
            pseudo_labels: Pseudo labels [N] (optional, derived from logits if not provided)
            
        Returns:
            VEGA score
        """
        # Convert to tensors
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.from_numpy(text_embeddings).float()
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
            
        # Derive pseudo labels if not provided
        if pseudo_labels is None and logits is not None:
            pseudo_labels = logits.argmax(dim=1)
        elif isinstance(pseudo_labels, np.ndarray):
            pseudo_labels = torch.from_numpy(pseudo_labels).long()
        
        return self.compute_graph_matching_score(features, text_embeddings, pseudo_labels)


class VEGAPlus(VEGAScorer):
    """
    Extended VEGA with confidence-weighted scoring.
    
    Combines graph matching with prediction confidence.
    """
    
    def __init__(self, k_neighbors: int = 10, sigma: float = 1.0, 
                 confidence_weight: float = 0.5):
        """
        Initialize VEGA+ scorer.
        
        Args:
            k_neighbors: Number of neighbors for graph
            sigma: Gaussian kernel bandwidth
            confidence_weight: Weight for confidence term
        """
        super().__init__(k_neighbors, sigma)
        self.confidence_weight = confidence_weight
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None
    ) -> float:
        """
        Compute VEGA+ score with confidence weighting.
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [C, D]
            logits: Model predictions [N, C]
            pseudo_labels: Pseudo labels [N]
            
        Returns:
            VEGA+ score
        """
        # Base VEGA score
        base_score = super().compute_score(features, text_embeddings, logits, pseudo_labels)
        
        # Compute confidence
        if logits is None:
            return base_score
            
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        
        # Prediction confidence
        probs = F.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        confidence = max_probs.mean().item()
        
        # Combine scores
        final_score = (1 - self.confidence_weight) * base_score + \
                      self.confidence_weight * confidence
        
        return final_score


def compute_vega_score(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    k_neighbors: int = 10
) -> float:
    """
    Convenience function to compute VEGA score.
    
    Args:
        features: Image features [N, D]
        text_embeddings: Text embeddings [C, D]
        logits: Model predictions [N, C]
        pseudo_labels: Pseudo labels [N]
        k_neighbors: Number of neighbors
        
    Returns:
        VEGA score
    """
    vega = VEGAScorer(k_neighbors=k_neighbors)
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)