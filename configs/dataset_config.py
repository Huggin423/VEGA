"""
Dataset configuration for VLM model selection experiments.
Contains dataset names, paths, and metadata.
"""

import os

# Dataset list used in experiments (from SWAB framework)
# 22 datasets total
DATASETS = [
    "cars",
    "cifar100",
    "clevr_closest_object_distance",
    "clevr_count_all",
    "country211",
    "diabetic_retinopathy",
    "dmlab",
    "dtd",
    "eurosat",
    "fer2013",
    "flowers",
    "gtsrb",
    "imagenet1k",
    "kitti_closest_vehicle_distance",
    "mnist",
    "pcam",
    "pets",
    "renderedsst2",
    "resisc45",
    "stl10",
    "sun397",
    "svhn",
    "voc2007",
]

# Datasets that failed in Oracle evaluation
FAILED_DATASETS = ["imagenet1k", "clevr_closest_object_distance"]

# Valid datasets for experiments
VALID_DATASETS = [d for d in DATASETS if d not in FAILED_DATASETS]


def get_dataset_list(exclude_failed=True):
    """
    Get list of datasets for experiments.
    
    Args:
        exclude_failed: Whether to exclude datasets that failed in Oracle evaluation
    
    Returns:
        List of dataset names
    """
    if exclude_failed:
        return VALID_DATASETS
    return DATASETS.copy()


def get_dataset_config_path(root_dir, dataset_name):
    """
    Get path to dataset classnames file.
    
    Args:
        root_dir: Project root directory
        dataset_name: Name of the dataset
    
    Returns:
        Path to classnames txt file
    """
    return os.path.join(root_dir, "data", "datasets", "classnames", f"{dataset_name}.txt")