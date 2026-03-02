"""
VEGA: Visual-Textual Graph Alignment for Unsupervised VLM Selection
Reference: VEGA Paper - "Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks"

Key Idea:
- Build textual graph from class name embeddings (nodes=classes, edges=cosine similarity)
- Build visual graph from image features (nodes=class clusters modeled as Gaussians, edges=Bhattacharyya distance)
- Score models by measuring alignment between the two graphs at node and edge levels
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Dict, List, Tuple
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VEGAScorer:
    """
    VEGA-based model selection scorer.
    
    Implements the paper's algorithm:
    1. Extract features using VLM encoders
    2. Construct textual graph (K nodes, cosine similarity edges)
    3. Construct visual graph (K Gaussian clusters, Bhattacharyya distance edges)
    4. Compute node similarity (weighted average of normalized cosine similarity)
    5. Compute edge similarity (Pearson correlation of edge matrices)
    6. Final score = node_similarity + edge_similarity
    
    Usage:
        vega = VEGAScorer(temperature=0.05)
        score = vega.compute_score(visual_features, text_embeddings, logits)
    """
    
    def __init__(self, temperature: float = 0.05, min_samples_per_class: int = 1):
        """
        Initialize VEGA scorer.
        
        Args:
            temperature: Temperature parameter for softmax normalization in node similarity
            min_samples_per_class: Minimum samples required per class for valid computation
        """
        self.temperature = temperature
        self.min_samples_per_class = min_samples_per_class
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float() if x.dtype != torch.float32 else x
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features."""
        return F.normalize(features, p=2, dim=1)
    
    def compute_pseudo_labels(
        self, 
        visual_features: torch.Tensor, 
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pseudo labels by zero-shot classification.
        
        Args:
            visual_features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            
        Returns:
            Pseudo labels [N]
        """
        # Normalize features
        visual_features = self._normalize_features(visual_features)
        text_embeddings = self._normalize_features(text_embeddings)
        
        # Compute cosine similarity [N, K]
        similarity = visual_features @ text_embeddings.T
        
        # Get pseudo labels
        pseudo_labels = similarity.argmax(dim=1)
        
        return pseudo_labels
    
    def build_textual_graph(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build textual graph.
        
        Nodes: text embeddings of each class [K, D]
        Edges: cosine similarity between class embeddings [K, K]
        
        Args:
            text_embeddings: Text embeddings [K, D]
            
        Returns:
            Tuple of (nodes, edges) where nodes=[K, D], edges=[K, K]
        """
        # Normalize features
        text_embeddings = self._normalize_features(text_embeddings)
        
        # Node features are the text embeddings
        nodes = text_embeddings
        
        # Edge weights are cosine similarity
        edges = nodes @ nodes.T  # [K, K]
        
        return nodes, edges
    
    def build_visual_graph(
        self, 
        visual_features: torch.Tensor, 
        pseudo_labels: torch.Tensor,
        n_classes: int
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], torch.Tensor]:
        """
        Build visual graph.
        
        Nodes: Each class is modeled as a Gaussian distribution N(mu_k, Sigma_k)
        Edges: Bhattacharyya distance between class Gaussians
        
        Args:
            visual_features: Image features [N, D]
            pseudo_labels: Pseudo labels [N]
            n_classes: Number of classes K
            
        Returns:
            Tuple of (class_means, class_covs, class_counts, edge_matrix)
            - class_means: Dict mapping class index to mean vector
            - class_covs: Dict mapping class index to covariance matrix
            - class_counts: Dict mapping class index to sample count
            - edge_matrix: Bhattacharyya distance matrix [K, K]
        """
        # Normalize features
        visual_features = self._normalize_features(visual_features)
        
        # Compute class statistics
        class_means = {}
        class_covs = {}
        class_counts = {}
        
        for k in range(n_classes):
            # Get samples assigned to class k
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) < self.min_samples_per_class:
                logger.warning(f"Class {k} has only {len(indices)} samples, skipping...")
                continue
            
            class_features = visual_features[indices]
            class_counts[k] = len(indices)
            
            # Compute mean
            class_means[k] = class_features.mean(dim=0)
            
            # Compute covariance (regularized)
            if len(indices) > 1:
                centered = class_features - class_means[k].unsqueeze(0)
                cov = (centered.T @ centered) / (len(indices) - 1)
                # Add small regularization for numerical stability
                cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
            else:
                # Single sample - use identity
                cov = torch.eye(visual_features.shape[1], device=visual_features.device)
            
            class_covs[k] = cov
        
        # Compute Bhattacharyya distance between all pairs of classes
        valid_classes = list(class_means.keys())
        n_valid = len(valid_classes)
        
        if n_valid < 2:
            logger.warning("Not enough valid classes for visual graph construction")
            return class_means, class_covs, class_counts, None
        
        edge_matrix = torch.zeros(n_classes, n_classes, device=visual_features.device)
        
        for i in valid_classes:
            for j in valid_classes:
                if i == j:
                    continue
                edge_matrix[i, j] = self._bhattacharyya_distance(
                    class_means[i], class_covs[i],
                    class_means[j], class_covs[j]
                )
        
        return class_means, class_covs, class_counts, edge_matrix
    
    def _bhattacharyya_distance(
        self, 
        mu1: torch.Tensor, 
        sigma1: torch.Tensor,
        mu2: torch.Tensor, 
        sigma2: torch.Tensor
    ) -> float:
        """
        Compute Bhattacharyya distance between two Gaussian distributions.
        
        Bhattacharyya distance:
        D_B = (1/8) * (mu1 - mu2)^T * Sigma^{-1} * (mu1 - mu2) 
              + (1/2) * ln(|Sigma| / sqrt(|sigma1| * |sigma2|))
        where Sigma = (sigma1 + sigma2) / 2
        
        Args:
            mu1, sigma1: Mean and covariance of first Gaussian
            mu2, sigma2: Mean and covariance of second Gaussian
            
        Returns:
            Bhattacharyya distance (float)
        """
        try:
            # Compute average covariance
            sigma = (sigma1 + sigma2) / 2
            
            # Compute inverse of average covariance
            sigma_inv = torch.linalg.inv(sigma)
            
            # Compute Mahalanobis-like term
            diff = (mu1 - mu2).unsqueeze(1)  # [D, 1]
            term1 = 0.125 * (diff.T @ sigma_inv @ diff).item()
            
            # Compute determinant term
            det_sigma = torch.linalg.det(sigma)
            det_sigma1 = torch.linalg.det(sigma1)
            det_sigma2 = torch.linalg.det(sigma2)
            
            # Avoid log of negative or zero
            if det_sigma <= 0 or det_sigma1 <= 0 or det_sigma2 <= 0:
                # Fallback to simple distance
                return torch.norm(mu1 - mu2).item()
            
            term2 = 0.5 * torch.log(det_sigma / torch.sqrt(det_sigma1 * det_sigma2)).item()
            
            return term1 + term2
            
        except Exception as e:
            # Fallback to Euclidean distance on means
            logger.warning(f"Bhattacharyya distance computation failed: {e}, using Euclidean fallback")
            return torch.norm(mu1 - mu2).item()
    
    def compute_node_similarity(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        pseudo_labels: torch.Tensor,
        class_counts: Dict[int, int]
    ) -> float:
        """
        Compute node similarity.
        
        Node similarity measures the average distance from visual features 
        within a cluster to the corresponding textual feature.
        
        sim_k = (1/N_k) * sum_{v in V_k} [exp(cos(v, t_k)/t) / sum_{k'} exp(cos(v, t_k')/t)]
        s_n = (1/K) * sum_k sim_k * N_k
        
        Args:
            visual_features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            pseudo_labels: Pseudo labels [N]
            class_counts: Sample count per class
            
        Returns:
            Node similarity score in [0, 1]
        """
        # Normalize features
        visual_features = self._normalize_features(visual_features)
        text_embeddings = self._normalize_features(text_embeddings)
        
        n_classes = text_embeddings.shape[0]
        total_samples = visual_features.shape[0]
        
        # Compute cosine similarity between all images and all class texts [N, K]
        similarity = visual_features @ text_embeddings.T  # [N, K]
        
        # Apply temperature scaling and softmax
        scaled_similarity = similarity / self.temperature
        probs = F.softmax(scaled_similarity, dim=1)  # [N, K]
        
        # Compute per-class similarity scores
        class_similarities = {}
        
        for k in range(n_classes):
            mask = (pseudo_labels == k)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # Get the probability assigned to the correct class for each sample
            # This is the normalized similarity to the corresponding text embedding
            class_probs = probs[indices, k]
            
            # Average similarity for this class
            class_similarities[k] = class_probs.mean().item()
        
        if not class_similarities:
            return 0.0
        
        # Compute weighted average
        # s_n = (1/K) * sum_k sim_k * N_k
        # This is equivalent to weighted mean by class size
        total_weighted_sim = 0.0
        total_weight = 0
        
        for k, sim in class_similarities.items():
            weight = class_counts.get(k, 1)
            total_weighted_sim += sim * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        node_similarity = total_weighted_sim / total_weight
        
        return node_similarity
    
    def compute_edge_similarity(
        self,
        textual_edges: torch.Tensor,
        visual_edges: torch.Tensor
    ) -> float:
        """
        Compute edge similarity using Pearson correlation.
        
        Edge similarity measures the correlation between the edge structures
        of textual and visual graphs.
        
        s_e = (PearsonCorr(vec(E^T), vec(E^V)) + 1) / 2
        
        Args:
            textual_edges: Textual graph edge matrix [K, K]
            visual_edges: Visual graph edge matrix [K, K]
            
        Returns:
            Edge similarity score in [0, 1]
        """
        # Convert to numpy for scipy
        if isinstance(textual_edges, torch.Tensor):
            textual_edges = textual_edges.cpu().numpy()
        if isinstance(visual_edges, torch.Tensor):
            visual_edges = visual_edges.cpu().numpy()
        
        # Flatten the matrices (excluding diagonal)
        n = textual_edges.shape[0]
        
        # Get upper triangular indices (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        
        textual_vec = textual_edges[triu_indices]
        visual_vec = visual_edges[triu_indices]
        
        if len(textual_vec) < 2:
            return 0.5  # Neutral value if not enough edges
        
        # Compute Pearson correlation
        try:
            corr, _ = pearsonr(textual_vec, visual_vec)
            
            # Handle NaN
            if np.isnan(corr):
                return 0.5
            
            # Rescale from [-1, 1] to [0, 1]
            edge_similarity = (corr + 1) / 2
            
            return edge_similarity
            
        except Exception as e:
            logger.warning(f"Pearson correlation failed: {e}")
            return 0.5
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Compute VEGA transferability score.
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            logits: Model predictions [N, K] (optional, used for pseudo labels if provided)
            pseudo_labels: Pseudo labels [N] (optional, derived from features if not provided)
            return_details: If True, return detailed breakdown of scores
            
        Returns:
            VEGA score (float), or dict with details if return_details=True
        """
        # Convert to tensors
        visual_features = self._to_tensor(features)
        text_embeddings = self._to_tensor(text_embeddings)
        
        n_classes = text_embeddings.shape[0]
        
        # Compute pseudo labels if not provided
        if pseudo_labels is None:
            if logits is not None:
                logits_tensor = self._to_tensor(logits)
                pseudo_labels = logits_tensor.argmax(dim=1)
            else:
                pseudo_labels = self.compute_pseudo_labels(visual_features, text_embeddings)
        else:
            pseudo_labels = self._to_tensor(pseudo_labels).long()
        
        # Step 1: Build textual graph
        textual_nodes, textual_edges = self.build_textual_graph(text_embeddings)
        
        # Step 2: Build visual graph
        class_means, class_covs, class_counts, visual_edges = self.build_visual_graph(
            visual_features, pseudo_labels, n_classes
        )
        
        # Check if visual graph construction was successful
        if visual_edges is None or len(class_means) < 2:
            logger.warning("Visual graph construction failed, returning fallback score")
            if return_details:
                return {
                    'score': 0.0,
                    'node_similarity': 0.0,
                    'edge_similarity': 0.0,
                    'valid_classes': len(class_means)
                }
            return 0.0
        
        # Step 3: Compute node similarity
        node_similarity = self.compute_node_similarity(
            visual_features, text_embeddings, pseudo_labels, class_counts
        )
        
        # Step 4: Compute edge similarity
        edge_similarity = self.compute_edge_similarity(textual_edges, visual_edges)
        
        # Step 5: Final VEGA score
        vega_score = node_similarity + edge_similarity
        
        if return_details:
            return {
                'score': vega_score,
                'node_similarity': node_similarity,
                'edge_similarity': edge_similarity,
                'valid_classes': len(class_means),
                'class_counts': class_counts
            }
        
        return vega_score


class VEGAPlus(VEGAScorer):
    """
    Extended VEGA with confidence-weighted scoring.
    
    This is an experimental extension that combines VEGA score
    with prediction confidence for potentially better ranking.
    """
    
    def __init__(self, temperature: float = 0.05, confidence_weight: float = 0.3):
        """
        Initialize VEGA+ scorer.
        
        Args:
            temperature: Temperature for node similarity
            confidence_weight: Weight for confidence term (0-1)
        """
        super().__init__(temperature=temperature)
        self.confidence_weight = confidence_weight
    
    def compute_score(
        self,
        features: Union[np.ndarray, torch.Tensor],
        text_embeddings: Union[np.ndarray, torch.Tensor],
        logits: Union[np.ndarray, torch.Tensor] = None,
        pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Compute VEGA+ score with confidence weighting.
        
        Args:
            features: Image features [N, D]
            text_embeddings: Text embeddings [K, D]
            logits: Model predictions [N, K]
            pseudo_labels: Pseudo labels [N]
            return_details: If True, return detailed breakdown
            
        Returns:
            VEGA+ score
        """
        # Get base VEGA score
        base_result = super().compute_score(
            features, text_embeddings, logits, pseudo_labels, return_details=True
        )
        
        base_score = base_result['score']
        
        if logits is None:
            if return_details:
                base_result['confidence'] = None
                return base_result
            return base_score
        
        # Compute confidence
        logits_tensor = self._to_tensor(logits)
        probs = F.softmax(logits_tensor, dim=1)
        max_probs, _ = probs.max(dim=1)
        confidence = max_probs.mean().item()
        
        # Combine scores
        final_score = (1 - self.confidence_weight) * base_score + \
                      self.confidence_weight * confidence
        
        if return_details:
            base_result['confidence'] = confidence
            base_result['final_score'] = final_score
            return base_result
        
        return final_score


def compute_vega_score(
    features: Union[np.ndarray, torch.Tensor],
    text_embeddings: Union[np.ndarray, torch.Tensor],
    logits: Union[np.ndarray, torch.Tensor] = None,
    pseudo_labels: Union[np.ndarray, torch.Tensor] = None,
    temperature: float = 0.05
) -> float:
    """
    Convenience function to compute VEGA score.
    
    Args:
        features: Image features [N, D]
        text_embeddings: Text embeddings [K, D]
        logits: Model predictions [N, C]
        pseudo_labels: Pseudo labels [N]
        temperature: Temperature parameter
        
    Returns:
        VEGA score
    """
    vega = VEGAScorer(temperature=temperature)
    return vega.compute_score(features, text_embeddings, logits, pseudo_labels)


# Backward compatibility - keep old class name as alias
VEGA = VEGAScorer