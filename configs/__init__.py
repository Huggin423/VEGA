"""
Configuration module for VLM model selection experiments.
"""

from .dataset_config import DATASETS, get_dataset_list
from .model_config import MODELS, get_model_list

__all__ = ['DATASETS', 'MODELS', 'get_dataset_list', 'get_model_list']