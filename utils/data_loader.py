"""
Data loader for PTM (Pre-trained Model) statistics.
Handles loading features, logits, and calibration metrics from stored files.
"""

import os
import torch
import pickle
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PTMDataLoader:
    """
    Unified data loader for VLM model selection experiments.
    
    Handles loading:
    - Image features (from img_feat/)
    - Text classifiers (from text_classifier/)
    - Logits (from logits/)
    - Calibration metrics (from calibration_metrics/)
    - Class-level accuracy (from class_level_acc/)
    """
    
    def __init__(self, root_path: str):
        """
        Initialize the data loader.
        
        Args:
            root_path: Root directory of the project (containing ptm_stats/)
        """
        self.root_path = root_path
        
        # Path configurations
        self.stats_dir = os.path.join(root_path, 'ptm_stats', 'stats_on_hist_task')
        self.logits_dir = os.path.join(root_path, 'ptm_stats', 'logits')
        self.classnames_dir = os.path.join(root_path, 'data', 'datasets', 'classnames')
        
    def _load_pkl(self, path: str) -> Dict:
        """Load pickle file safely."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}
    
    def _load_classnames(self, dataset_name: str) -> Optional[List[str]]:
        """
        Load class names for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of class names or None if not found
        """
        txt_path = os.path.join(self.classnames_dir, f"{dataset_name}.txt")
        
        if not os.path.exists(txt_path):
            logger.warning(f"Classnames file not found: {txt_path}")
            return None
        
        with open(txt_path, 'r') as f:
            classnames = [line.strip() for line in f.readlines() if line.strip()]
        
        return classnames
    
    def _flatten_features(self, feat_dict: Dict, dataset_name: str) -> torch.Tensor:
        """
        Flatten class-wise feature dictionary to (N, D) tensor.
        
        Args:
            feat_dict: Dictionary mapping class names to features
            dataset_name: Name of the dataset (for class ordering)
            
        Returns:
            Flattened feature tensor of shape (N, D)
        """
        # Get correct class order
        ordered_classes = self._load_classnames(dataset_name)
        
        if ordered_classes is None:
            logger.info("Using dictionary insertion order for flattening features.")
            ordered_classes = list(feat_dict.keys())
        
        feature_list = []
        
        for cls_name in ordered_classes:
            if cls_name not in feat_dict:
                logger.warning(f"Class '{cls_name}' not found in feature dict keys. Skipping...")
                continue
            
            feats = feat_dict[cls_name]
            
            # Convert to tensor
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            elif isinstance(feats, list):
                feats = torch.tensor(feats)
            
            feature_list.append(feats)
        
        if not feature_list:
            raise ValueError(f"No features extracted for dataset {dataset_name}")
        
        return torch.cat(feature_list, dim=0)
    
    def load_image_features(self, model_name: str, dataset_name: str) -> torch.Tensor:
        """
        Load image features for a model on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Feature tensor of shape (N, D)
        """
        feat_path = os.path.join(self.stats_dir, 'img_feat', f"{model_name}.pkl")
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file missing: {feat_path}")
        
        full_pkl = self._load_pkl(feat_path)
        
        if dataset_name not in full_pkl:
            raise KeyError(f"Dataset '{dataset_name}' not found in pkl.")
        
        raw_content = full_pkl[dataset_name]
        
        if isinstance(raw_content, dict):
            return self._flatten_features(raw_content, dataset_name)
        else:
            # Already flattened
            if isinstance(raw_content, np.ndarray):
                return torch.from_numpy(raw_content)
            return raw_content
    
    def load_logits(self, model_name: str, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load logits and targets for a model on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (logits, targets) tensors
        """
        logits_filename = f"{model_name}__{dataset_name}.pth"
        logits_path = os.path.join(self.logits_dir, logits_filename)
        
        if not os.path.exists(logits_path):
            raise FileNotFoundError(f"Logits file missing: {logits_path}")
        
        payload = torch.load(logits_path, map_location='cpu')
        
        logits = None
        targets = None
        
        if isinstance(payload, dict):
            # Extract logits
            for k in ['logits', 'outputs', 'predictions']:
                if k in payload:
                    logits = payload[k]
                    break
            # Extract targets
            for k in ['targets', 'labels', 'ground_truth']:
                if k in payload:
                    targets = payload[k]
                    break
            
            # Fallback
            if logits is None or targets is None:
                keys = list(payload.keys())
                logits = payload[keys[0]]
                targets = payload[keys[1]]
                
        elif isinstance(payload, (tuple, list)):
            logits = payload[0]
            targets = payload[1]
        
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        
        return logits.float(), targets.long()
    
    def load_data(self, model_name: str, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load features, logits, and targets for a model on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (features, logits, targets)
        """
        features = self.load_image_features(model_name, dataset_name)
        logits, targets = self.load_logits(model_name, dataset_name)
        
        # Validate alignment
        if features.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Feature count ({features.shape[0]}) != Logits count ({logits.shape[0]})"
            )
        
        return features.float(), logits.float(), targets.long()
    
    def load_calibration_metrics(self, model_name: str, dataset_name: str) -> Dict:
        """
        Load calibration metrics (ECE, ACE) for a model on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with 'ece' and 'ace' values
        """
        calib_path = os.path.join(self.stats_dir, 'calibration_metrics', f"{model_name}.pkl")
        
        if not os.path.exists(calib_path):
            logger.warning(f"Calibration file not found: {calib_path}")
            return {}
        
        calib_data = self._load_pkl(calib_path)
        return calib_data.get(dataset_name, {})
    
    def load_class_accuracy(self, model_name: str, dataset_name: str) -> Dict[str, float]:
        """
        Load class-level accuracy for a model on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping class names to accuracy values
        """
        acc_path = os.path.join(self.stats_dir, 'class_level_acc', f"{model_name}.pkl")
        
        if not os.path.exists(acc_path):
            logger.warning(f"Accuracy file not found: {acc_path}")
            return {}
        
        acc_data = self._load_pkl(acc_path)
        return acc_data.get(dataset_name, {})
    
    def load_text_classifier(self, model_name: str, dataset_name: str) -> torch.Tensor:
        """
        Load text classifier weights for a model on a dataset.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Text classifier tensor of shape (num_classes, embed_dim)
        """
        text_path = os.path.join(self.stats_dir, 'text_classifier', f"{model_name}.pkl")
        
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text classifier file missing: {text_path}")
        
        text_data = self._load_pkl(text_path)
        
        if dataset_name not in text_data:
            raise KeyError(f"Dataset '{dataset_name}' not found in text classifier pkl.")
        
        weights = text_data[dataset_name]
        
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        
        return weights.float()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from img_feat directory."""
        feat_dir = os.path.join(self.stats_dir, 'img_feat')
        if not os.path.exists(feat_dir):
            return []
        
        return [f.replace('.pkl', '') for f in os.listdir(feat_dir) if f.endswith('.pkl')]
    
    def get_available_datasets(self, model_name: str) -> List[str]:
        """Get list of datasets available for a specific model."""
        feat_path = os.path.join(self.stats_dir, 'img_feat', f"{model_name}.pkl")
        
        if not os.path.exists(feat_path):
            return []
        
        data = self._load_pkl(feat_path)
        return list(data.keys())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help='Project root directory')
    args = parser.parse_args()
    
    loader = PTMDataLoader(args.root)
    
    # Test loading
    test_model = "RN50_openai"
    test_dataset = "cars"
    
    try:
        features, logits, targets = loader.load_data(test_model, test_dataset)
        print(f"\n✅ SUCCESS! Loaded {features.shape[0]} samples.")
        print(f"Features shape: {features.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Calculate accuracy
        acc = (logits.argmax(dim=1) == targets).float().mean()
        print(f"Zero-shot Accuracy: {acc:.4%}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")