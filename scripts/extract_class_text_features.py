#!/usr/bin/env python
"""
提取 VEGA 所需的类别文本特征

VEGA 需要使用各个模型的 text encoder 对类别名称进行编码，
而不是使用 SWAB 的 caption_text_feat（这些是经过 LLM 扩写的）。

本脚本会：
1. 加载各个数据集的类别名称
2. 使用 open_clip 加载各个模型
3. 提取类别名称的文本嵌入
4. 保存到 ptm_stats/class_text_feat/ 目录

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

# 多模板（可以提升性能）
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of a {} in the wild.",
    "a photo of a {} in the scene.",
    "a photo of a {} on a white background.",
    "a photo of a {} on a black background.",
    "a photo of a {} with a person.",
    "a photo of a {} with a dog.",
    "a photo of a {} with a cat.",
    "a photo of a {} with a car.",
]

# 定义需要处理的模型（open_clip 格式）
# 格式: (model_name_in_ptm_stats, open_clip_model_name, open_clip_pretrained)
OPEN_CLIP_MODELS = [
    # OpenAI CLIP
    ('RN50_openai', 'RN50', 'openai'),
    ('RN101_openai', 'RN101', 'openai'),
    ('ViT-B-32_openai', 'ViT-B-32', 'openai'),
    ('ViT-B-16_openai', 'ViT-B-16', 'openai'),
    ('ViT-L-14_openai', 'ViT-L-14', 'openai'),
    ('ViT-L-14-336_openai', 'ViT-L-14-336', 'openai'),
    
    # LAION models
    ('ViT-B-32_laion2b_s34b_b79k', 'ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-B-16_laion400m_e32', 'ViT-B-16', 'laion400m_e32'),
    ('ViT-L-14_laion400m_e32', 'ViT-L-14', 'laion400m_e32'),
    ('ViT-H-14_laion2b_s32b_b79k', 'ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-g-14_laion2b_s12b_b42k', 'ViT-g-14', 'laion2b_s12b_b42k'),
    
    # ConvNeXt
    ('convnext_base_laion400m_s13b_b51k', 'convnext_base', 'laion400m_s13b_b51k'),
]

# 数据集列表
DATASETS = [
    'cars', 'cifar100', 'flowers', 'pets', 'dtd',
    'eurosat', 'gtsrb', 'mnist', 'pcam', 'stl10',
    'sun397', 'svhn', 'voc2007', 'country211',
    'fer2013', 'renderedsst2', 'resisc45',
    'clevr_count_all', 'clevr_closest_object_distance',
    'diabetic_retinopathy', 'dmlab', 'kitti_closest_vehicle_distance'
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


def load_all_classnames() -> Dict[str, List[str]]:
    """加载所有数据集的类别名称"""
    all_classnames = {}
    
    for dataset in DATASETS:
        classnames = load_classnames(dataset)
        if classnames:
            all_classnames[dataset] = classnames
            print(f"  {dataset}: {len(classnames)} 类别")
    
    return all_classnames


# ============================================================================
# 模型加载
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


# ============================================================================
# 文本特征提取
# ============================================================================

def extract_text_features(
    model,
    tokenizer,
    classnames: List[str],
    templates: List[str] = None,
    device: str = 'cuda'
) -> np.ndarray:
    """
    使用模型的 text encoder 提取类别文本特征
    
    Args:
        model: open_clip 模型
        tokenizer: 对应的 tokenizer
        classnames: 类别名称列表
        templates: prompt 模板列表（如果多个，会平均）
        device: 设备
        
    Returns:
        np.ndarray: [K, D] 类别文本嵌入矩阵
    """
    if templates is None:
        templates = [DEFAULT_TEMPLATE]
    
    model.eval()
    text_features_list = []
    
    with torch.no_grad():
        for template in templates:
            # 构造 prompt
            prompts = [template.format(c) for c in classnames]
            
            # Tokenize
            tokens = tokenizer(prompts)
            tokens = tokens.to(device)
            
            # 编码
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            text_features_list.append(text_features.cpu())
    
    # 如果有多个模板，取平均
    if len(text_features_list) > 1:
        text_features = torch.stack(text_features_list).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = text_features_list[0]
    
    return text_features.numpy()


def extract_features_for_model(
    model_config: Tuple[str, str, str],
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda',
    use_multi_templates: bool = False
) -> Dict[str, np.ndarray]:
    """
    为单个模型提取所有数据集的文本特征
    
    Args:
        model_config: (ptm_name, open_clip_name, pretrained)
        all_classnames: 所有数据集的类别名
        device: 设备
        use_multi_templates: 是否使用多模板
        
    Returns:
        Dict[dataset_name, text_features]
    """
    ptm_name, model_name, pretrained = model_config
    
    print(f"\n处理模型: {ptm_name}")
    print(f"  open_clip: {model_name}, pretrained: {pretrained}")
    
    # 加载模型
    model, tokenizer = load_open_clip_model(model_name, pretrained, device)
    if model is None:
        return {}
    
    # 选择模板
    templates = CLIP_TEMPLATES if use_multi_templates else [DEFAULT_TEMPLATE]
    
    results = {}
    
    for dataset, classnames in all_classnames.items():
        try:
            features = extract_text_features(
                model, tokenizer, classnames, templates, device
            )
            results[dataset] = features
            print(f"  ✓ {dataset}: {features.shape}")
        except Exception as e:
            print(f"  [!] {dataset} 失败: {e}")
    
    return results


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
# BLIP 和 BEIT3 模型处理
# ============================================================================

def extract_blip_features(
    all_classnames: Dict[str, List[str]],
    device: str = 'cuda'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    使用 BLIP 模型提取文本特征
    需要使用 transformers 库
    """
    try:
        from transformers import BlipModel, BlipProcessor
    except ImportError:
        print("[!] 需要安装 transformers: pip install transformers")
        return {}
    
    blip_models = [
        ('BLIP_retrieval_base_coco', 'Salesforce/blip-itm-base-coco'),
        ('BLIP_retrieval_large_coco', 'Salesforce/blip-itm-large-coco'),
    ]
    
    results = {}
    
    for ptm_name, hf_name in blip_models:
        print(f"\n处理 BLIP 模型: {ptm_name}")
        
        try:
            processor = BlipProcessor.from_pretrained(hf_name)
            model = BlipModel.from_pretrained(hf_name).to(device)
            model.eval()
            
            model_features = {}
            
            with torch.no_grad():
                for dataset, classnames in all_classnames.items():
                    prompts = [f"a photo of a {c}" for c in classnames]
                    
                    inputs = processor(text=prompts, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    text_features = model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    model_features[dataset] = text_features.cpu().numpy()
                    print(f"  ✓ {dataset}: {model_features[dataset].shape}")
            
            results[ptm_name] = model_features
            
        except Exception as e:
            print(f"  [!] 加载失败: {e}")
    
    return results


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
    
    # 处理 open_clip 模型
    print("\n" + "=" * 70)
    print("处理 OpenCLIP 模型")
    print("=" * 70)
    
    for model_config in OPEN_CLIP_MODELS:
        ptm_name = model_config[0]
        
        # 检查是否已存在
        existing = load_existing_features(ptm_name)
        if existing:
            print(f"\n[跳过] {ptm_name} 已存在")
            continue
        
        # 提取特征
        features = extract_features_for_model(
            model_config, all_classnames, device, use_multi_templates=False
        )
        
        if features:
            save_text_features(ptm_name, features)
    
    # 处理 BLIP 模型（可选）
    print("\n" + "=" * 70)
    print("处理 BLIP 模型 (需要 transformers)")
    print("=" * 70)
    
    blip_features = extract_blip_features(all_classnames, device)
    for ptm_name, features in blip_features.items():
        save_text_features(ptm_name, features)
    
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
        for dataset, features in data.items():
            print(f"  {dataset}: shape={features.shape}, dtype={features.dtype}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='仅验证已有特征')
    parser.add_argument('--swab_root', type=str, default=SWAB_ROOT, help='SWAB 根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    args = parser.parse_args()
    
    # 更新路径
    SWAB_ROOT = args.swab_root
    CLASSNAMES_DIR = os.path.join(SWAB_ROOT, 'data/datasets/classnames')
    OUTPUT_DIR = args.output_dir
    
    if args.verify:
        verify_features()
    else:
        main()