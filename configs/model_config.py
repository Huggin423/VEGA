"""
Model configuration for VLM model selection experiments.
Contains model names, architectures, and metadata.
"""

import os
import yaml

# Model families for categorization
MODEL_FAMILIES = {
    "CLIP_ViT": [
        "ViT-B-32_openai",
        "ViT-B-16_openai",
        "ViT-L-14_openai",
        "ViT-L-14-336_openai",
    ],
    "CLIP_RN": [
        "RN50_openai",
        "RN101_openai",
        "RN50x4_openai",
        "RN50x16_openai",
        "RN50x64_openai",
    ],
    "CLIP_Laion": [
        "ViT-B-32_laion400m_e31",
        "ViT-B-32_laion400m_e32",
        "ViT-B-32_laion2b_e16",
        "ViT-B-32_laion2b_s34b_b79k",
        "ViT-B-16_laion400m_e32",
        "ViT-B-16-plus-240_laion400m_e32",
        "ViT-L-14_laion400m_e31",
        "ViT-L-14_laion400m_e32",
        "ViT-L-14_laion2b_s32b_b82k",
        "ViT-H-14_laion2b_s32b_b79k",
        "ViT-g-14_laion2b_s12b_b42k",
        "ViT-g-14_laion2b_s34b_b88k",
    ],
    "BLIP": [
        "BLIP_retrieval_base_coco",
        "BLIP_retrieval_base_f30k",
        "BLIP_retrieval_large_coco",
        "BLIP_retrieval_large_f30k",
    ],
    "BEIT3": [
        "BEIT3_retrieval_base_coco",
        "BEIT3_retrieval_base_f30k",
        "BEIT3_retrieval_large_coco",
        "BEIT3_retrieval_large_f30k",
    ],
    "CoCa": [
        "coca_ViT-B-32_laion2b_s13b_b90k",
        "coca_ViT-L-14_laion2b_s13b_b90k",
        "coca_ViT-B-32_mscoco_finetuned_laion2b_s13b_b90k",
        "coca_ViT-L-14_mscoco_finetuned_laion2b_s13b_b90k",
    ],
    "ConvNeXt": [
        "convnext_base_laion400m_s13b_b51k",
        "convnext_base_w_laion2b_s13b_b82k",
        "convnext_base_w_laion2b_s13b_b82k_augreg",
        "convnext_base_w_laion_aesthetic_s13b_b82k",
        "convnext_base_w_320_laion_aesthetic_s13b_b82k",
        "convnext_base_w_320_laion_aesthetic_s13b_b82k_augreg",
        "convnext_large_d_laion2b_s26b_b102k_augreg",
        "convnext_large_d_320_laion2b_s29b_b131k_ft",
        "convnext_large_d_320_laion2b_s29b_b131k_ft_soup",
    ],
}

# Flatten to get all models
MODELS = []
for family_models in MODEL_FAMILIES.values():
    MODELS.extend(family_models)


def get_model_list():
    """
    Get list of all available models.
    
    Returns:
        List of model names
    """
    return MODELS.copy()


def get_models_by_family(family_name):
    """
    Get models belonging to a specific family.
    
    Args:
        family_name: Name of the model family
    
    Returns:
        List of model names in that family
    """
    return MODEL_FAMILIES.get(family_name, []).copy()


def load_model_list_from_yaml(config_path):
    """
    Load model list from YAML configuration file.
    Supports formats:
    - ['RN50', 'openai']  -> automatically converts to 'RN50_openai'
    - 'ViT-B-32_openai'  -> keeps as is
    
    Args:
        config_path: Path to the YAML config file
    
    Returns:
        List of model names
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("models.yml must contain a list of models")

    model_list = []
    for item in data:
        if isinstance(item, list):
            # Handle ['RN50', 'openai'] format
            model_name = "_".join(item)
            model_list.append(model_name)
        elif isinstance(item, str):
            # Handle 'RN50_openai' format
            model_list.append(item)
        else:
            print(f"Warning: Skipping invalid model entry: {item}")

    return model_list


def get_model_family(model_name):
    """
    Get the family name for a given model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Family name or 'Unknown'
    """
    for family, models in MODEL_FAMILIES.items():
        if model_name in models:
            return family
    return "Unknown"