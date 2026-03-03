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
# VEGA 实验设置
# ============================================================================

# VEGA 论文使用的 10 个下游数据集
# "We conduct performance prediction on ten common-used downstream datasets"
VEGA_DATASETS = [
    'cifar100',       # basic image recognition Cifar-100
    'pets',           # animal dataset Oxford Pets
    'flowers',        # plant dataset Oxford Flowers
    'svhn',           # street scene dataset SVHN
    'gtsrb',          # street scene dataset GTSRB
    'dtd',            # describable textures dataset DTD
    'country211',     # scene classification dataset Country211
    'sun397',         # scene classification dataset SUN397
    'mnist',          # digit dataset MNIST
    'fer2013',        # facial expression dataset Fer2013
]

# 完整数据集列表（包括 SWAB 中有的额外数据集）
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
# 基于 ptm_stats/logits 中已有的模型
# ============================================================================

# 从 SWAB 的 ptm_stats/logits 目录中提取的模型列表
# 这些模型已经跑过数据集，可以直接使用
CLIP_FAMILY_MODELS = [
    # OpenAI CLIP (ResNet 系列)
    ('RN50_openai', 'RN50', 'openai'),
    ('RN101_openai', 'RN101', 'openai'),
    ('RN50x4_openai', 'RN50x4', 'openai'),
    ('RN50x16_openai', 'RN50x16', 'openai'),
    ('RN50x64_openai', 'RN50x64', 'openai'),
    
    # OpenAI CLIP (ViT 系列)
    ('ViT-B-32_openai', 'ViT-B-32', 'openai'),
    ('ViT-B-16_openai', 'ViT-B-16', 'openai'),
    ('ViT-L-14_openai', 'ViT-L-14', 'openai'),
    ('ViT-L-14-336_openai', 'ViT-L-14-336', 'openai'),
    
    # LAION CLIP (ViT 系列)
    ('ViT-B-32_laion400m_e31', 'ViT-B-32-quickgelu', 'laion400m_e31'),
    ('ViT-B-32_laion400m_e32', 'ViT-B-32-quickgelu', 'laion400m_e32'),
    ('ViT-B-32_laion2b_s34b_b79k', 'ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-B-16_laion400m_e32', 'ViT-B-16', 'laion400m_e32'),
    ('ViT-L-14_laion400m_e31', 'ViT-L-14', 'laion400m_e31'),
    ('ViT-L-14_laion400m_e32', 'ViT-L-14', 'laion400m_e32'),
    ('ViT-L-14_laion2b_s32b_b82k', 'ViT-L-14', 'laion2b_s32b_b82k'),
    ('ViT-H-14_laion2b_s32b_b79k', 'ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-g-14_laion2b_s12b_b42k', 'ViT-g-14', 'laion2b_s12b_b42k'),
    ('ViT-g-14_laion2b_s34b_b88k', 'ViT-g-14', 'laion2b_s34b_b88k'),
    
    # LAION CLIP (ConvNeXt 系列)
    ('convnext_base_laion400m_s13b_b51k', 'convnext_base', 'laion400m_s13b_b51k'),
    ('convnext_base_w_laion2b_s13b_b82k', 'convnext_base_w', 'laion2b_s13b_b82k'),
    ('convnext_base_w_laion2b_s13b_b82k_augreg', 'convnext_base_w', 'laion2b_s13b_b82k_augreg'),
    ('convnext_base_w_laion_aesthetic_s13b_b82k', 'convnext_base_w', 'laion_aesthetic_s13b_b82k'),
    ('convnext_base_w_320_laion_aesthetic_s13b_b82k', 'convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
    ('convnext_base_w_320_laion_aesthetic_s13b_b82k_augreg', 'convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
    ('convnext_large_d_laion2b_s26b_b102k_augreg', 'convnext_large_d', 'laion2b_s26b_b102k_augreg'),
    ('convnext_large_d_320_laion2b_s29b_b131k_ft', 'convnext_large_d_320', 'laion2b_s29b_b131k_ft'),
    ('convnext_large_d_320_laion2b_s29b_b131k_ft_soup', 'convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),
    
    # CoCa 模型
    ('coca_ViT-B-32_laion2b_s13b_b90k', 'coca_ViT-B-32', 'laion2b_s13b_b90k'),
    ('coca_ViT-L-14_laion2b_s13b_b90k', 'coca_ViT-L-14', 'laion2b_s13b_b90k'),
    ('coca_ViT-B-32_mscoco_finetuned_laion2b_s13b_b90k', 'coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'),
    ('coca_ViT-L-14_mscoco_finetuned_laion2b_s13b_b90k', 'coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'),
    
    # 其他 ViT 变体
    ('ViT-B-16-plus-240_laion400m_e32', 'ViT-B-16-plus-240', 'laion400m_e32'),
    ('ViT-B-32-quickgelu_laion400m_e32', 'ViT-B-32-quickgelu', 'laion400m_e32'),
    ('ViT-B-32_laion2b_e16', 'ViT-B-32', 'laion2b_e16'),
]


# ============================================================================
# Benchmark B: VLMs from Various Pre-training Algorithms
# 使用用户提供的模型路径
# ============================================================================

# VEGA 论文中使用的各种预训练算法模型
# "We collect 17 models from Hugging Face from 10 commonly used VLM pre-training algorithms"
# 包括: ALIGN, AltCLIP, CLIP, GroupViT, SigLIP, StreetCLIP, MetaCLIP, BiomedCLIP, QuiltNet, BioCLIP

# 模型路径配置 (用户提供的)
ADDITIONAL_MODEL_PATHS = {
    # Benchmark B models from ~/models
    "AltCLIP": "/root/mxy/models/BAAI-AltCLIP",
    "StreetCLIP": "/root/mxy/models/geolocal-street-clip",
    "SigLIP_base": "/root/mxy/models/google-siglip-base-patch16-224",
    "SigLIP_so400m": "/root/mxy/models/google-siglip-so400m-patch14-384",
    "BioCLIP": "/root/mxy/models/imageomics-bioclip",
    "ALIGN": "/root/mxy/models/kakaobrain-align-base",
    "MetaCLIP": "/root/mxy/models/metaclip-b32-fullcc2.5b",
    "BiomedCLIP": "/root/mxy/models/microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "GroupViT": "/root/mxy/models/nvidia-groupvit-gcc-yfcc",
    "QuiltNet": "/root/mxy/models/wisdomik-QuiltNet-B-32",
}

# HuggingFace 模型名称映射
HUGGINGFACE_MODELS = {
    "AltCLIP": "BAAI/AltCLIP",
    "StreetCLIP": "geolocal/StreetCLIP",
    "SigLIP_base": "google/siglip-base-patch16-224",
    "SigLIP_so400m": "google/siglip-so400m-patch14-384",
    "BioCLIP": "imageomics/bioclip",
    "ALIGN": "kakaobrain/align-base",
    "MetaCLIP": "facebook/metaclip-b32-fullcc2.5b",
    "BiomedCLIP": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "GroupViT": "nvidia/groupvit-gcc-yfcc",
    "QuiltNet": "wisdomik/QuiltNet-B-32",
}


# ============================================================================
# BLIP 和 BEIT3 模型 (已在 SWAB 中有中间结果)
# ============================================================================

# 这些模型已经在 ptm_stats/logits 中有数据
# BLIP 和 BEIT3 需要特殊处理
BLIP_BEIT3_MODELS = [
    'BLIP_retrieval_base_coco',
    'BLIP_retrieval_base_f30k',
    'BLIP_retrieval_large_coco',
    'BLIP_retrieval_large_f30k',
    'BEIT3_retrieval_base_coco',
    'BEIT3_retrieval_base_f30k',
    'BEIT3_retrieval_large_coco',
    'BEIT3_retrieval_large_f30k',
]


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
# 模型加载和特征提取
# ============================================================================

def load_open_clip_model(model_name: str, pretrained: str, device: str = 'cuda'):
    """加载 open_clip 模型"""
    try:
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
    """
    使用模型的 text encoder 提取类别文本特征
    
    Args:
        model: open_clip 模型
        tokenizer: 对应的 tokenizer
        classnames: 类别名称列表
        template: prompt 模板
        device: 设备
        
    Returns:
        np.ndarray: [K, D] 类别文本嵌入矩阵
    """
    model.eval()
    
    with torch.no_grad():
        # 构造 prompt
        prompts = [template.format(c) for c in classnames]
        
        # Tokenize
        tokens = tokenizer(prompts)
        tokens = tokens.to(device)
        
        # 编码
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()


def extract_features_for_openclip_model(
    model_config: Tuple[str, str, str],
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    为 open_clip 模型提取所有数据集的文本特征
    """
    ptm_name, model_name, pretrained = model_config
    
    print(f"\n处理模型: {ptm_name}")
    print(f"  open_clip: {model_name}, pretrained: {pretrained}")
    
    # 加载模型
    model, tokenizer = load_open_clip_model(model_name, pretrained, device)
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


def extract_features_for_hf_model(
    model_name: str,
    model_path: str,
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    为 HuggingFace 模型提取文本特征
    
    支持: AltCLIP, SigLIP, BioCLIP, ALIGN, etc.
    """
    from transformers import AutoModel, AutoProcessor
    
    print(f"\n处理 HuggingFace 模型: {model_name}")
    print(f"  路径: {model_path}")
    
    try:
        # 尝试从本地路径加载
        if os.path.exists(model_path):
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(device)
        else:
            # 从 HuggingFace Hub 加载
            hf_name = HUGGINGFACE_MODELS.get(model_name, model_name)
            processor = AutoProcessor.from_pretrained(hf_name)
            model = AutoModel.from_pretrained(hf_name).to(device)
        
        model.eval()
        results = {}
        
        with torch.no_grad():
            for dataset, classnames in all_classnames.items():
                try:
                    prompts = [f"a photo of a {c}" for c in classnames]
                    
                    inputs = processor(text=prompts, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 不同模型的接口可能不同
                    if hasattr(model, 'get_text_features'):
                        text_features = model.get_text_features(**inputs)
                    elif hasattr(model, 'encode_text'):
                        text_features = model.encode_text(inputs['input_ids'])
                    else:
                        outputs = model(**inputs)
                        text_features = outputs.last_hidden_state[:, 0, :]
                    
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    results[dataset] = text_features.cpu().numpy()
                    print(f"  ✓ {dataset}: {results[dataset].shape}")
                    
                except Exception as e:
                    print(f"  [!] {dataset} 失败: {e}")
        
        return results
        
    except Exception as e:
        print(f"  [!] 加载模型失败: {e}")
        return {}


def extract_features_for_blip(
    model_name: str,
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """使用 BLIP 模型提取文本特征"""
    from transformers import BlipModel, BlipProcessor
    
    print(f"\n处理 BLIP 模型: {model_name}")
    
    # BLIP 模型的 HuggingFace 名称
    blip_hf_names = {
        'BLIP_retrieval_base_coco': 'Salesforce/blip-itm-base-coco',
        'BLIP_retrieval_base_f30k': 'Salesforce/blip-itm-base-flickr',
        'BLIP_retrieval_large_coco': 'Salesforce/blip-itm-large-coco',
        'BLIP_retrieval_large_f30k': 'Salesforce/blip-itm-large-flickr',
    }
    
    hf_name = blip_hf_names.get(model_name)
    if not hf_name:
        print(f"  [!] 未找到对应的 HuggingFace 模型")
        return {}
    
    try:
        processor = BlipProcessor.from_pretrained(hf_name)
        model = BlipModel.from_pretrained(hf_name).to(device)
        model.eval()
        
        results = {}
        
        with torch.no_grad():
            for dataset, classnames in all_classnames.items():
                try:
                    prompts = [f"a photo of a {c}" for c in classnames]
                    
                    inputs = processor(text=prompts, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    text_features = model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    results[dataset] = text_features.cpu().numpy()
                    print(f"  ✓ {dataset}: {results[dataset].shape}")
                    
                except Exception as e:
                    print(f"  [!] {dataset} 失败: {e}")
        
        return results
        
    except Exception as e:
        print(f"  [!] 加载 BLIP 模型失败: {e}")
        return {}


def extract_features_for_beit3(
    model_name: str,
    all_classnames: Dict[str, List[str]],
    swab_root: str,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    使用 BEIT3 模型提取文本特征
    
    BEIT3 使用 SWAB 中已有的模型文件
    需要参考 model/beit_class.py 的实现
    """
    print(f"\n处理 BEIT3 模型: {model_name}")
    
    # BEIT3 的实现较为复杂，这里标记为需要特殊处理
    # 可以参考 SWAB/model/beit_class.py
    print("  [!] BEIT3 需要特殊处理，请参考 SWAB/model/beit_class.py")
    print("  [!] 或者使用已有的 caption_text_feat 作为替代")
    
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
    print("提取 VEGA 所需的类别文本特征")
    print("=" * 70)
    
    # 检查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
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
        
        # 检查是否已存在
        existing = load_existing_features(ptm_name)
        if existing:
            print(f"\n[跳过] {ptm_name} 已存在 ({len(existing)} 数据集)")
            continue
        
        # 提取特征
        features = extract_features_for_openclip_model(model_config, all_classnames, device)
        
        if features:
            save_text_features(ptm_name, features)
    
    # =========================================================================
    # Benchmark B: Various Pre-training Algorithms
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark B: VLMs from Various Pre-training Algorithms")
    print("=" * 70)
    
    for model_name in ADDITIONAL_MODEL_PATHS.keys():
        # 检查是否已存在
        existing = load_existing_features(model_name)
        if existing:
            print(f"\n[跳过] {model_name} 已存在 ({len(existing)} 数据集)")
            continue
        
        model_path = ADDITIONAL_MODEL_PATHS[model_name]
        features = extract_features_for_hf_model(model_name, model_path, all_classnames, device)
        
        if features:
            save_text_features(model_name, features)
    
    # =========================================================================
    # BLIP 模型
    # =========================================================================
    print("\n" + "=" * 70)
    print("BLIP 模型")
    print("=" * 70)
    
    for model_name in BLIP_BEIT3_MODELS:
        if model_name.startswith('BLIP'):
            # 检查是否已存在
            existing = load_existing_features(model_name)
            if existing:
                print(f"\n[跳过] {model_name} 已存在 ({len(existing)} 数据集)")
                continue
            
            features = extract_features_for_blip(model_name, all_classnames, device)
            if features:
                save_text_features(model_name, features)
    
    # =========================================================================
    # BEIT3 模型
    # =========================================================================
    print("\n" + "=" * 70)
    print("BEIT3 模型")
    print("=" * 70)
    
    for model_name in BLIP_BEIT3_MODELS:
        if model_name.startswith('BEIT3'):
            existing = load_existing_features(model_name)
            if existing:
                print(f"\n[跳过] {model_name} 已存在 ({len(existing)} 数据集)")
                continue
            
            features = extract_features_for_beit3(model_name, all_classnames, SWAB_ROOT, device)
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


def print_vega_setup():
    """打印 VEGA 实验设置"""
    print("\n" + "=" * 70)
    print("VEGA 实验设置")
    print("=" * 70)
    
    print("\n下游数据集 (10个):")
    for ds in VEGA_DATASETS:
        print(f"  - {ds}")
    
    print("\nBenchmark A: VLMs from CLIP Family")
    print(f"  模型数: {len(CLIP_FAMILY_MODELS)}")
    print("  包含: OpenAI CLIP (ResNet + ViT), LAION CLIP (ViT + ConvNeXt), CoCa")
    
    print("\nBenchmark B: VLMs from Various Pre-training Algorithms")
    print(f"  模型数: {len(ADDITIONAL_MODEL_PATHS)}")
    print("  包含: AltCLIP, StreetCLIP, SigLIP, BioCLIP, ALIGN, MetaCLIP, BiomedCLIP, GroupViT, QuiltNet")
    
    print("\nBLIP 和 BEIT3 模型 (已在 SWAB 中有数据):")
    print(f"  模型数: {len(BLIP_BEIT3_MODELS)}")
    for m in BLIP_BEIT3_MODELS:
        print(f"  - {m}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='仅验证已有特征')
    parser.add_argument('--setup', action='store_true', help='打印 VEGA 实验设置')
    parser.add_argument('--swab_root', type=str, default=SWAB_ROOT, help='SWAB 根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--benchmark', type=str, choices=['A', 'B', 'ALL'], default='ALL',
                        help='运行哪个 benchmark (A=CLIP Family, B=Various Pre-training, ALL=全部)')
    args = parser.parse_args()
    
    # 更新路径
    SWAB_ROOT = args.swab_root
    CLASSNAMES_DIR = os.path.join(SWAB_ROOT, 'data/datasets/classnames')
    OUTPUT_DIR = args.output_dir
    
    if args.setup:
        print_vega_setup()
    elif args.verify:
        verify_features()
    else:
        main()