#!/usr/bin/env python
"""
提取 VEGA 所需的类别文本特征

VEGA 需要使用各个模型的 text encoder 对类别名称进行编码，
而不是使用 SWAB 的 caption_text_feat（这些是经过 LLM 扩写的）。

本脚本会：
1. 加载各个数据集的类别名称
2. 使用 open_clip / transformers 加载各个模型
3. 提取类别名称的文本嵌入
4. 保存到 ptm_stats/class_text_feat/ 目录

VEGA 的三个 Benchmark:
- Benchmark A: VLMs from CLIP Family (31个模型)
- Benchmark B: VLMs from Various Pre-training Algorithms (17个模型)
- Benchmark C: Combinations of VLM and Prompt Template (10模型×10模板)

运行环境：实验室服务器（需要 GPU 和模型文件）

更新：支持离线加载，优先使用本地模型文件
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# 设置离线模式环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# open_clip 用于加载 CLIP 类模型
try:
    import open_clip
except ImportError:
    print("请安装 open_clip: pip install open_clip_torch")
    sys.exit(1)


# ============================================================================
# 配置
# ============================================================================

# 数据目录
SWAB_ROOT = '/root/mxy/SWAB'
VEGA_ROOT = '/root/mxy/VEGA'

# 类别名文件目录
CLASSNAMES_DIR = os.path.join(SWAB_ROOT, 'data/datasets/classnames')

# 输出目录
OUTPUT_DIR = os.path.join(SWAB_ROOT, 'ptm_stats/class_text_feat')

# 默认 prompt template (CLIP 风格)
DEFAULT_TEMPLATE = "a photo of a {}."

# ============================================================================
# 本地模型路径配置
# ============================================================================

# HuggingFace 缓存目录（SWAB 中的缓存）
HF_CACHE_DIR = os.path.join(SWAB_ROOT, 'model/checkpoint')

# VEGA 本地模型目录
VEGA_MODELS_DIR = os.path.join(VEGA_ROOT, 'models')

# 额外模型目录
EXTRA_MODELS_DIR = '/root/mxy/models'


# ============================================================================
# VEGA 实验设置
# ============================================================================

# VEGA 论文使用的 10 个下游数据集
VEGA_DATASETS = [
    'cifar100', 'pets', 'flowers', 'svhn', 'gtsrb',
    'dtd', 'country211', 'sun397', 'mnist', 'fer2013',
]

# 完整数据集列表
ALL_DATASETS = [
    'cars', 'cifar100', 'flowers', 'pets', 'dtd',
    'eurosat', 'gtsrb', 'mnist', 'pcam', 'stl10',
    'sun397', 'svhn', 'voc2007', 'country211',
    'fer2013', 'renderedsst2', 'resisc45',
    'clevr_count_all', 'clevr_closest_object_distance',
    'diabetic_retinopathy', 'dmlab', 'kitti_closest_vehicle_distance'
]


# ============================================================================
# Benchmark A: VLMs from CLIP Family
# ============================================================================

# open_clip 模型配置，支持本地缓存路径
# 格式: (模型名称, open_clip模型名, pretrained标识, 本地缓存路径/None)
CLIP_FAMILY_MODELS = [
    # OpenAI CLIP (ResNet 系列) - 使用本地 .pt 文件
    ('RN50_openai', 'RN50', 'openai', None),  # 有本地 .pt 文件
    ('RN101_openai', 'RN101', 'openai', None),
    ('RN50x4_openai', 'RN50x4', 'openai', None),
    ('RN50x16_openai', 'RN50x16', 'openai', None),
    ('RN50x64_openai', 'RN50x64', 'openai', None),
    
    # OpenAI CLIP (ViT 系列)
    ('ViT-B-32_openai', 'ViT-B-32', 'openai', None),
    ('ViT-B-16_openai', 'ViT-B-16', 'openai', None),
    ('ViT-L-14_openai', 'ViT-L-14', 'openai', None),
    ('ViT-L-14-336_openai', 'ViT-L-14-336', 'openai', None),
    
    # LAION CLIP (ViT 系列) - 使用 HuggingFace 缓存目录
    ('ViT-B-32_laion400m_e31', 'ViT-B-32-quickgelu', 'laion400m_e31', None),
    ('ViT-B-32_laion400m_e32', 'ViT-B-32-quickgelu', 'laion400m_e32', None),
    ('ViT-B-32_laion2b_s34b_b79k', 'ViT-B-32', 'laion2b_s34b_b79k', 
     'models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K'),
    ('ViT-B-16_laion400m_e32', 'ViT-B-16', 'laion400m_e32', None),
    ('ViT-L-14_laion400m_e31', 'ViT-L-14', 'laion400m_e31', None),
    ('ViT-L-14_laion400m_e32', 'ViT-L-14', 'laion400m_e32', None),
    ('ViT-L-14_laion2b_s32b_b82k', 'ViT-L-14', 'laion2b_s32b_b82k',
     'models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K'),
    ('ViT-H-14_laion2b_s32b_b79k', 'ViT-H-14', 'laion2b_s32b_b79k',
     'models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K'),
    ('ViT-g-14_laion2b_s12b_b42k', 'ViT-g-14', 'laion2b_s12b_b42k',
     'models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K'),
    ('ViT-g-14_laion2b_s34b_b88k', 'ViT-g-14', 'laion2b_s34b_b88k',
     'models--laion--CLIP-ViT-g-14-laion2B-s34B-b88K'),
    
    # LAION CLIP (ConvNeXt 系列)
    ('convnext_base_laion400m_s13b_b51k', 'convnext_base', 'laion400m_s13b_b51k',
     'models--laion--CLIP-convnext_base-laion400M-s13B-b51K'),
    ('convnext_base_w_laion2b_s13b_b82k', 'convnext_base_w', 'laion2b_s13b_b82k',
     'models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K'),
    ('convnext_base_w_laion2b_s13b_b82k_augreg', 'convnext_base_w', 'laion2b_s13b_b82k_augreg',
     'models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K-augreg'),
    ('convnext_base_w_laion_aesthetic_s13b_b82k', 'convnext_base_w', 'laion_aesthetic_s13b_b82k',
     'models--laion--CLIP-convnext_base_w-laion_aesthetic-s13B-b82K'),
    ('convnext_base_w_320_laion_aesthetic_s13b_b82k', 'convnext_base_w_320', 'laion_aesthetic_s13b_b82k',
     'models--laion--CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K'),
    ('convnext_base_w_320_laion_aesthetic_s13b_b82k_augreg', 'convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg',
     'models--laion--CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg'),
    ('convnext_large_d_laion2b_s26b_b102k_augreg', 'convnext_large_d', 'laion2b_s26b_b102k_augreg',
     'models--laion--CLIP-convnext_large_d.laion2B-s26B-b102K-augreg'),
    ('convnext_large_d_320_laion2b_s29b_b131k_ft', 'convnext_large_d_320', 'laion2b_s29b_b131k_ft',
     'models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft'),
    ('convnext_large_d_320_laion2b_s29b_b131k_ft_soup', 'convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup',
     'models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup'),
    
    # CoCa 模型
    ('coca_ViT-B-32_laion2b_s13b_b90k', 'coca_ViT-B-32', 'laion2b_s13b_b90k',
     'models--laion--CoCa-ViT-B-32-laion2B-s13B-b90k'),
    ('coca_ViT-L-14_laion2b_s13b_b90k', 'coca_ViT-L-14', 'laion2b_s13b_b90k',
     'models--laion--CoCa-ViT-L-14-laion2B-s13B-b90k'),
    ('coca_ViT-B-32_mscoco_finetuned_laion2b_s13b_b90k', 'coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k',
     'models--laion--mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k'),
    ('coca_ViT-L-14_mscoco_finetuned_laion2b_s13b_b90k', 'coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k',
     'models--laion--mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k'),
    
    # 其他 ViT 变体
    ('ViT-B-16-plus-240_laion400m_e32', 'ViT-B-16-plus-240', 'laion400m_e32', None),
    ('ViT-B-32-quickgelu_laion400m_e32', 'ViT-B-32-quickgelu', 'laion400m_e32', None),
    ('ViT-B-32_laion2b_e16', 'ViT-B-32', 'laion2b_e16', None),
]


# ============================================================================
# Benchmark B: VLMs from Various Pre-training Algorithms (17个模型)
# ============================================================================

# 本地模型路径配置 - 按照实际目录结构
BENCHMARK_B_MODELS = {
    # VEGA/models 目录下的模型
    "AltCLIP": {
        "path": os.path.join(VEGA_MODELS_DIR, "BAAI-AltCLIP"),
        "hf_name": "BAAI/AltCLIP",
        "type": "transformers"
    },
    "StreetCLIP": {
        "path": os.path.join(VEGA_MODELS_DIR, "geolocal-street-clip"),
        "hf_name": "geolocal/StreetCLIP",
        "type": "transformers"
    },
    "SigLIP_base": {
        "path": os.path.join(VEGA_MODELS_DIR, "google-siglip-base-patch16-224"),
        "hf_name": "google/siglip-base-patch16-224",
        "type": "transformers"
    },
    "SigLIP_so400m": {
        "path": os.path.join(VEGA_MODELS_DIR, "google-siglip-so400m-patch14-384"),
        "hf_name": "google/siglip-so400m-patch14-384",
        "type": "transformers"
    },
    "BioCLIP": {
        "path": os.path.join(VEGA_MODELS_DIR, "imageomics-bioclip"),
        "hf_name": "imageomics/bioclip",
        "type": "open_clip"  # BioCLIP 使用 open_clip 格式
    },
    "ALIGN": {
        "path": os.path.join(VEGA_MODELS_DIR, "kakaobrain-align-base"),
        "hf_name": "kakaobrain/align-base",
        "type": "transformers"
    },
    "MetaCLIP": {
        "path": os.path.join(VEGA_MODELS_DIR, "metaclip-b32-fullcc2.5b"),
        "hf_name": "facebook/metaclip-b32-fullcc2.5b",
        "type": "open_clip"
    },
    "BiomedCLIP": {
        "path": os.path.join(VEGA_MODELS_DIR, "microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patcg16_224"),
        "hf_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "type": "open_clip"
    },
    "GroupViT": {
        "path": os.path.join(VEGA_MODELS_DIR, "nvidia-groupvit-gcc-yfcc"),
        "hf_name": "nvidia/groupvit-gcc-yfcc",
        "type": "transformers"
    },
    "QuiltNet": {
        "path": os.path.join(VEGA_MODELS_DIR, "wisdomik-QuiltNet-B-32"),
        "hf_name": "wisdomik/QuiltNet-B-32",
        "type": "open_clip"
    },
}

# 额外的 Benchmark B 模型（可能需要下载或有其他配置）
EXTRA_BENCHMARK_B_MODELS = {
    # 这些模型可能需要额外处理
    "CLIP_ViT-B-32": {
        "path": os.path.join(VEGA_MODELS_DIR, "openai-clip-vit-base-patch32"),
        "hf_name": "openai/clip-vit-base-patch32",
        "type": "transformers"
    },
    "CLIP_ViT-L-14": {
        "path": os.path.join(VEGA_MODELS_DIR, "openai-clip-vit-large-patch14"),
        "hf_name": "openai/clip-vit-large-patch14",
        "type": "transformers"
    },
}


# ============================================================================
# BLIP 和 BEIT3 模型
# ============================================================================

# BLIP 模型配置 - 使用本地 checkpoint
BLIP_MODELS = {
    'BLIP_retrieval_base_coco': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'model_base_retrieval_coco.pth'),
        'config': 'base',
    },
    'BLIP_retrieval_base_f30k': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'model_base_retrieval_flickr.pth'),
        'config': 'base',
    },
    'BLIP_retrieval_large_coco': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'model_large_retrieval_coco.pth'),
        'config': 'large',
    },
    'BLIP_retrieval_large_f30k': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'model_large_retrieval_flickr.pth'),
        'config': 'large',
    },
}

# BEIT3 模型配置 - 使用本地 checkpoint
BEIT3_MODELS = {
    'BEIT3_retrieval_base_coco': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'beit3_base_patch16_384_coco_retrieval.pth'),
        'config': 'base',
    },
    'BEIT3_retrieval_base_f30k': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'beit3_base_patch16_384_f30k_retrieval.pth'),
        'config': 'base',
    },
    'BEIT3_retrieval_large_coco': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'beit3_large_patch16_384_coco_retrieval.pth'),
        'config': 'large',
    },
    'BEIT3_retrieval_large_f30k': {
        'checkpoint': os.path.join(HF_CACHE_DIR, 'beit3_large_patch16_384_f30k_retrieval.pth'),
        'config': 'large',
    },
}


# ============================================================================
# 数据集类别名加载
# ============================================================================

def load_classnames(dataset_name: str) -> Optional[List[str]]:
    """加载数据集的类别名称"""
    txt_path = os.path.join(CLASSNAMES_DIR, f'{dataset_name}.txt')
    
    if not os.path.exists(txt_path):
        print(f"  [!] 类别名文件不存在: {txt_path}")
        return None
    
    with open(txt_path, 'r') as f:
        classnames = [line.strip() for line in f.readlines() if line.strip()]
    
    return classnames


def load_all_classnames(datasets: List[str] = None) -> Dict[str, List[str]]:
    """加载所有数据集的类别名称"""
    if datasets is None:
        datasets = ALL_DATASETS
    
    all_classnames = {}
    
    for dataset in datasets:
        classnames = load_classnames(dataset)
        if classnames:
            all_classnames[dataset] = classnames
            print(f"  {dataset}: {len(classnames)} 类别")
    
    return all_classnames


# ============================================================================
# 模型加载函数
# ============================================================================

def find_hf_cache_model(cache_dir: str, model_subdir: str) -> Optional[str]:
    """在 HuggingFace 缓存目录中查找模型"""
    full_path = os.path.join(cache_dir, model_subdir)
    if os.path.exists(full_path):
        # 查找 snapshots 目录
        snapshots_dir = os.path.join(full_path, 'snapshots')
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
    return None


def load_open_clip_model_cached(
    model_name: str, 
    pretrained: str, 
    cache_path: Optional[str] = None,
    device: str = 'cuda'
):
    """加载 open_clip 模型，支持本地缓存"""
    try:
        # 如果有本地缓存路径，尝试从缓存加载
        if cache_path:
            full_cache_path = find_hf_cache_model(HF_CACHE_DIR, cache_path)
            if full_cache_path:
                print(f"  从本地缓存加载: {full_cache_path}")
                # open_clip 可以从本地目录加载
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=full_cache_path,
                    device=device
                )
                tokenizer = open_clip.get_tokenizer(model_name)
                return model, tokenizer
        
        # 尝试直接加载（可能在系统缓存中）
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, tokenizer
        
    except Exception as e:
        print(f"  [!] 加载模型失败 {model_name} ({pretrained}): {e}")
        return None, None


def extract_text_features(
    model,
    tokenizer,
    classnames: List[str],
    template: str = DEFAULT_TEMPLATE,
    device: str = 'cuda'
) -> np.ndarray:
    """使用模型的 text encoder 提取类别文本特征"""
    model.eval()
    
    with torch.no_grad():
        prompts = [template.format(c) for c in classnames]
        tokens = tokenizer(prompts)
        tokens = tokens.to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()


def extract_features_for_openclip_model(
    model_config: Tuple,
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """为 open_clip 模型提取所有数据集的文本特征"""
    if len(model_config) == 4:
        ptm_name, model_name, pretrained, cache_path = model_config
    else:
        ptm_name, model_name, pretrained = model_config
        cache_path = None
    
    print(f"\n处理模型: {ptm_name}")
    print(f"  open_clip: {model_name}, pretrained: {pretrained}")
    
    model, tokenizer = load_open_clip_model_cached(model_name, pretrained, cache_path, device)
    if model is None:
        return {}
    
    results = {}
    
    for dataset, classnames in all_classnames.items():
        try:
            features = extract_text_features(
                model, tokenizer, classnames, DEFAULT_TEMPLATE, device
            )
            results[dataset] = features
            print(f"  ✓ {dataset}: {features.shape}")
        except Exception as e:
            print(f"  [!] {dataset} 失败: {e}")
    
    return results


def extract_features_for_hf_model_local(
    model_name: str,
    model_config: Dict,
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """使用本地 HuggingFace 模型提取文本特征"""
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    
    print(f"\n处理 HuggingFace 模型: {model_name}")
    
    model_path = model_config['path']
    model_type = model_config.get('type', 'transformers')
    
    print(f"  路径: {model_path}")
    print(f"  类型: {model_type}")
    
    try:
        # 检查本地路径是否存在
        if os.path.exists(model_path):
            print(f"  使用本地模型文件")
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        else:
            print(f"  [!] 本地路径不存在: {model_path}")
            return {}
        
        model.eval()
        results = {}
        
        with torch.no_grad():
            for dataset, classnames in all_classnames.items():
                try:
                    # 构造提示
                    if 'siglip' in model_name.lower():
                        prompts = [f"a photo of a {c}" for c in classnames]
                    else:
                        prompts = [f"a photo of a {c}" for c in classnames]
                    
                    inputs = processor(text=prompts, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 获取文本特征
                    if hasattr(model, 'get_text_features'):
                        text_features = model.get_text_features(**inputs)
                    elif hasattr(model, 'encode_text'):
                        text_features = model.encode_text(inputs['input_ids'])
                    else:
                        outputs = model(**inputs)
                        # 不同模型输出不同
                        if hasattr(outputs, 'text_embeds'):
                            text_features = outputs.text_embeds
                        elif hasattr(outputs, 'last_hidden_state'):
                            text_features = outputs.last_hidden_state[:, 0, :]
                        else:
                            print(f"  [!] 无法获取文本特征")
                            continue
                    
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    results[dataset] = text_features.cpu().numpy()
                    print(f"  ✓ {dataset}: {results[dataset].shape}")
                    
                except Exception as e:
                    print(f"  [!] {dataset} 失败: {e}")
        
        return results
        
    except Exception as e:
        print(f"  [!] 加载模型失败: {e}")
        return {}


def extract_features_for_open_clip_hf(
    model_name: str,
    model_config: Dict,
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """加载使用 open_clip 格式的 HuggingFace 模型 (如 BioCLIP, BiomedCLIP)"""
    import open_clip
    
    print(f"\n处理 open_clip 格式模型: {model_name}")
    
    model_path = model_config['path']
    print(f"  路径: {model_path}")
    
    try:
        # 查找模型文件
        if os.path.exists(model_path):
            # 尝试查找 open_clip_model.bin 或类似文件
            model_file = None
            for f in ['open_clip_pytorch_model.bin', 'pytorch_model.bin', 'model.safetensors']:
                candidate = os.path.join(model_path, f)
                if os.path.exists(candidate):
                    model_file = candidate
                    break
            
            if model_file:
                print(f"  找到模型文件: {model_file}")
                # 需要知道对应的架构
                # BioCLIP 使用 ViT-B-16
                # BiomedCLIP 使用 ViT-B-16
                if 'bioclip' in model_name.lower():
                    architecture = 'ViT-B-16'
                elif 'biomedclip' in model_name.lower():
                    architecture = 'ViT-B-16'
                elif 'metaclip' in model_name.lower():
                    architecture = 'ViT-B-32'
                elif 'quilt' in model_name.lower():
                    architecture = 'ViT-B-32'
                else:
                    architecture = 'ViT-B-16'  # 默认
                
                model, _, preprocess = open_clip.create_model_and_transforms(
                    architecture,
                    pretrained=model_file,
                    device=device
                )
                tokenizer = open_clip.get_tokenizer(architecture)
                
                results = {}
                for dataset, classnames in all_classnames.items():
                    try:
                        features = extract_text_features(
                            model, tokenizer, classnames, DEFAULT_TEMPLATE, device
                        )
                        results[dataset] = features
                        print(f"  ✓ {dataset}: {features.shape}")
                    except Exception as e:
                        print(f"  [!] {dataset} 失败: {e}")
                
                return results
            else:
                print(f"  [!] 未找到模型文件")
                return {}
        else:
            print(f"  [!] 路径不存在: {model_path}")
            return {}
            
    except Exception as e:
        print(f"  [!] 加载失败: {e}")
        return {}


def extract_features_for_blip_local(
    model_name: str,
    model_config: Dict,
    all_classnames: Dict[str, List[str]],
    swab_root: str,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """使用本地 BLIP 模型提取文本特征"""
    print(f"\n处理 BLIP 模型: {model_name}")
    
    checkpoint_path = model_config['checkpoint']
    
    if not os.path.exists(checkpoint_path):
        print(f"  [!] Checkpoint 不存在: {checkpoint_path}")
        return {}
    
    print(f"  Checkpoint: {checkpoint_path}")
    
    try:
        # 使用 SWAB 中的 BLIP 实现
        sys.path.insert(0, os.path.join(swab_root, 'model'))
        from blip_class import BlipRetrieval
        
        # 创建 BLIP 模型
        config_type = model_config['config']
        blip = BlipRetrieval(config_type, checkpoint_path, device)
        
        results = {}
        
        for dataset, classnames in all_classnames.items():
            try:
                # BLIP 使用特定的提示格式
                prompts = [f"a photo of a {c}" for c in classnames]
                
                text_features = blip.extract_text_features(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                results[dataset] = text_features.cpu().numpy()
                print(f"  ✓ {dataset}: {results[dataset].shape}")
                
            except Exception as e:
                print(f"  [!] {dataset} 失败: {e}")
        
        return results
        
    except Exception as e:
        print(f"  [!] 加载 BLIP 模型失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_features_for_beit3_local(
    model_name: str,
    model_config: Dict,
    all_classnames: Dict[str, List[str]],
    swab_root: str,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """使用本地 BEIT3 模型提取文本特征"""
    print(f"\n处理 BEIT3 模型: {model_name}")
    
    checkpoint_path = model_config['checkpoint']
    
    if not os.path.exists(checkpoint_path):
        print(f"  [!] Checkpoint 不存在: {checkpoint_path}")
        return {}
    
    print(f"  Checkpoint: {checkpoint_path}")
    
    try:
        # 使用 SWAB 中的 BEIT3 实现
        sys.path.insert(0, os.path.join(swab_root, 'model'))
        from beit_class import BEIT3Retrieval
        
        # 创建 BEIT3 模型
        config_type = model_config['config']
        beit3 = BEIT3Retrieval(config_type, checkpoint_path, device)
        
        results = {}
        
        for dataset, classnames in all_classnames.items():
            try:
                # BEIT3 使用特定的提示格式
                prompts = [f"a photo of a {c}" for c in classnames]
                
                text_features = beit3.extract_text_features(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                results[dataset] = text_features.cpu().numpy()
                print(f"  ✓ {dataset}: {results[dataset].shape}")
                
            except Exception as e:
                print(f"  [!] {dataset} 失败: {e}")
        
        return results
        
    except Exception as e:
        print(f"  [!] 加载 BEIT3 模型失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# 保存和加载
# ============================================================================

def save_text_features(model_name: str, features: Dict[str, np.ndarray]):
    """保存文本特征"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{model_name}.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"  保存到: {output_path}")


def load_existing_features(model_name: str) -> Dict[str, np.ndarray]:
    """加载已有的文本特征"""
    output_path = os.path.join(OUTPUT_DIR, f'{model_name}.pkl')
    
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            return pickle.load(f)
    return {}


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("提取 VEGA 所需的类别文本特征 (离线模式)")
    print("=" * 70)
    
    # 检查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"离线模式: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', 'not set')}")
    
    # 加载所有类别名
    print("\n加载数据集类别名:")
    all_classnames = load_all_classnames()
    print(f"共加载 {len(all_classnames)} 个数据集")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # =========================================================================
    # Benchmark A: CLIP Family
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark A: VLMs from CLIP Family")
    print("=" * 70)
    
    for model_config in CLIP_FAMILY_MODELS:
        ptm_name = model_config[0]
        
        existing = load_existing_features(ptm_name)
        if existing:
            print(f"\n[跳过] {ptm_name} 已存在 ({len(existing)} 数据集)")
            continue
        
        features = extract_features_for_openclip_model(model_config, all_classnames, device)
        
        if features:
            save_text_features(ptm_name, features)
    
    # =========================================================================
    # Benchmark B: Various Pre-training Algorithms
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark B: VLMs from Various Pre-training Algorithms")
    print("=" * 70)
    
    for model_name, model_config in BENCHMARK_B_MODELS.items():
        existing = load_existing_features(model_name)
        if existing:
            print(f"\n[跳过] {model_name} 已存在 ({len(existing)} 数据集)")
            continue
        
        model_type = model_config.get('type', 'transformers')
        
        if model_type == 'open_clip':
            features = extract_features_for_open_clip_hf(
                model_name, model_config, all_classnames, device
            )
        else:
            features = extract_features_for_hf_model_local(
                model_name, model_config, all_classnames, device
            )
        
        if features:
            save_text_features(model_name, features)
    
    # =========================================================================
    # BLIP 模型
    # =========================================================================
    print("\n" + "=" * 70)
    print("BLIP 模型")
    print("=" * 70)
    
    for model_name, model_config in BLIP_MODELS.items():
        existing = load_existing_features(model_name)
        if existing:
            print(f"\n[跳过] {model_name} 已存在 ({len(existing)} 数据集)")
            continue
        
        features = extract_features_for_blip_local(
            model_name, model_config, all_classnames, SWAB_ROOT, device
        )
        
        if features:
            save_text_features(model_name, features)
    
    # =========================================================================
    # BEIT3 模型
    # =========================================================================
    print("\n" + "=" * 70)
    print("BEIT3 模型")
    print("=" * 70)
    
    for model_name, model_config in BEIT3_MODELS.items():
        existing = load_existing_features(model_name)
        if existing:
            print(f"\n[跳过] {model_name} 已存在 ({len(existing)} 数据集)")
            continue
        
        features = extract_features_for_beit3_local(
            model_name, model_config, all_classnames, SWAB_ROOT, device
        )
        
        if features:
            save_text_features(model_name, features)
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    
    # 打印统计信息
    print(f"\n生成的文件:")
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.pkl'):
                fpath = os.path.join(OUTPUT_DIR, f)
                with open(fpath, 'rb') as fp:
                    data = pickle.load(fp)
                print(f"  {f}: {len(data)} 数据集")


def verify_features():
    """验证生成的特征"""
    print("\n" + "=" * 70)
    print("验证文本特征")
    print("=" * 70)
    
    if not os.path.exists(OUTPUT_DIR):
        print("输出目录不存在")
        return
    
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not f.endswith('.pkl'):
            continue
        
        fpath = os.path.join(OUTPUT_DIR, f)
        with open(fpath, 'rb') as fp:
            data = pickle.load(fp)
        
        print(f"\n{f}:")
        for dataset, features in sorted(data.items()):
            print(f"  {dataset}: shape={features.shape}, dtype={features.dtype}")


def print_model_status():
    """打印模型状态"""
    print("\n" + "=" * 70)
    print("模型状态检查")
    print("=" * 70)
    
    print("\nBenchmark A: CLIP Family 模型")
    print("-" * 40)
    
    for model_config in CLIP_FAMILY_MODELS:
        ptm_name = model_config[0]
        existing = load_existing_features(ptm_name)
        status = "✓ 已完成" if existing else "✗ 待处理"
        print(f"  {ptm_name}: {status}")
    
    print("\nBenchmark B: 各种预训练算法模型")
    print("-" * 40)
    
    for model_name, model_config in BENCHMARK_B_MODELS.items():
        existing = load_existing_features(model_name)
        status = "✓ 已完成" if existing else "✗ 待处理"
        path_exists = os.path.exists(model_config['path'])
        path_status = "存在" if path_exists else "不存在"
        print(f"  {model_name}: {status} (路径: {path_status})")
    
    print("\nBLIP 模型")
    print("-" * 40)
    
    for model_name, model_config in BLIP_MODELS.items():
        existing = load_existing_features(model_name)
        status = "✓ 已完成" if existing else "✗ 待处理"
        ckpt_exists = os.path.exists(model_config['checkpoint'])
        ckpt_status = "存在" if ckpt_exists else "不存在"
        print(f"  {model_name}: {status} (checkpoint: {ckpt_status})")
    
    print("\nBEIT3 模型")
    print("-" * 40)
    
    for model_name, model_config in BEIT3_MODELS.items():
        existing = load_existing_features(model_name)
        status = "✓ 已完成" if existing else "✗ 待处理"
        ckpt_exists = os.path.exists(model_config['checkpoint'])
        ckpt_status = "存在" if ckpt_exists else "不存在"
        print(f"  {model_name}: {status} (checkpoint: {ckpt_status})")


def count_completed_models():
    """统计已完成的模型数量"""
    total = 0
    completed = 0
    
    # Benchmark A
    for model_config in CLIP_FAMILY_MODELS:
        total += 1
        if load_existing_features(model_config[0]):
            completed += 1
    
    benchmark_a = (completed, total)
    
    # Benchmark B
    total = 0
    completed = 0
    for model_name in BENCHMARK_B_MODELS.keys():
        total += 1
        if load_existing_features(model_name):
            completed += 1
    
    benchmark_b = (completed, total)
    
    # BLIP
    total = 0
    completed = 0
    for model_name in BLIP_MODELS.keys():
        total += 1
        if load_existing_features(model_name):
            completed += 1
    
    blip = (completed, total)
    
    # BEIT3
    total = 0
    completed = 0
    for model_name in BEIT3_MODELS.keys():
        total += 1
        if load_existing_features(model_name):
            completed += 1
    
    beit3 = (completed, total)
    
    print("\n" + "=" * 70)
    print("模型完成统计")
    print("=" * 70)
    print(f"  Benchmark A (CLIP Family): {benchmark_a[0]}/{benchmark_a[1]}")
    print(f"  Benchmark B (各种预训练):  {benchmark_b[0]}/{benchmark_b[1]}")
    print(f"  BLIP 模型:                 {blip[0]}/{blip[1]}")
    print(f"  BEIT3 模型:                {beit3[0]}/{beit3[1]}")
    total_completed = benchmark_a[0] + benchmark_b[0] + blip[0] + beit3[0]
    total_models = benchmark_a[1] + benchmark_b[1] + blip[1] + beit3[1]
    print(f"  总计:                      {total_completed}/{total_models}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='仅验证已有特征')
    parser.add_argument('--status', action='store_true', help='打印模型状态')
    parser.add_argument('--count', action='store_true', help='统计完成数量')
    parser.add_argument('--swab_root', type=str, default=SWAB_ROOT, help='SWAB 根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--benchmark', type=str, choices=['A', 'B', 'ALL'], default='ALL',
                        help='运行哪个 benchmark')
    args = parser.parse_args()
    
    # 更新路径
    SWAB_ROOT = args.swab_root
    CLASSNAMES_DIR = os.path.join(SWAB_ROOT, 'data/datasets/classnames')
    OUTPUT_DIR = args.output_dir
    
    if args.status:
        print_model_status()
    elif args.count:
        count_completed_models()
    elif args.verify:
        verify_features()
    else:
        main()