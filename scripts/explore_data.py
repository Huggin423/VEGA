#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据探索脚本：确认 ptm_stats 文件夹中的数据结构，以及 data 和 model 目录结构
运行环境：实验室服务器 /root/mxy/SWAB
"""

import os
import torch
import pickle
import numpy as np
from pathlib import Path


def explore_logits(data_dir, model_name='RN50_openai', dataset_name='cars'):
    """探索 logits 文件结构"""
    logits_path = os.path.join(data_dir, 'ptm_stats/logits', f'{model_name}__{dataset_name}.pth')
    
    print("=" * 60)
    print(f"探索 Logits 文件: {logits_path}")
    print("=" * 60)
    
    if not os.path.exists(logits_path):
        print(f"文件不存在: {logits_path}")
        return None
    
    logits = torch.load(logits_path)
    print(f"类型: {type(logits)}")
    
    if isinstance(logits, dict):
        print(f"Keys: {logits.keys()}")
        for k, v in logits.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, np.ndarray)):
                print(f"  {k}: len={len(v)}")
            else:
                print(f"  {k}: type={type(v)}")
    elif isinstance(logits, (list, tuple)):
        print(f"Length: {len(logits)}")
        for i, item in enumerate(logits):
            if hasattr(item, 'shape'):
                print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  [{i}]: type={type(item)}")
    elif hasattr(logits, 'shape'):
        print(f"Shape: {logits.shape}")
        print(f"Dtype: {logits.dtype}")
        print(f"Min: {logits.min():.4f}, Max: {logits.max():.4f}")
    
    return logits


def explore_img_feat(data_dir, model_name='RN50_openai'):
    """探索图像特征文件结构"""
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索图像特征文件: {feat_path}")
    print("=" * 60)
    
    if not os.path.exists(feat_path):
        print(f"文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        img_feat = pickle.load(f)
    
    print(f"类型: {type(img_feat)}")
    
    if isinstance(img_feat, dict):
        print(f"数据集数量: {len(img_feat)}")
        print(f"数据集列表: {list(img_feat.keys())}")
        
        # 检查第一个数据集的结构
        first_key = list(img_feat.keys())[0]
        first_val = img_feat[first_key]
        print(f"\n示例数据集 '{first_key}':")
        if hasattr(first_val, 'shape'):
            print(f"  Shape: {first_val.shape}")
            print(f"  Dtype: {first_val.dtype}")
        elif isinstance(first_val, dict):
            print(f"  Keys (前5个): {list(first_val.keys())[:5]}")
        elif isinstance(first_val, (list, np.ndarray)):
            print(f"  Length: {len(first_val)}")
    
    return img_feat


def explore_text_feat(data_dir, model_name='RN50_openai'):
    """探索文本特征文件结构"""
    # caption_text_feat 可能是类别名称的 embeddings
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索文本特征文件 (caption_text_feat): {feat_path}")
    print("=" * 60)
    
    if not os.path.exists(feat_path):
        print(f"文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        text_feat = pickle.load(f)
    
    print(f"类型: {type(text_feat)}")
    
    if isinstance(text_feat, dict):
        print(f"数据集数量: {len(text_feat)}")
        print(f"数据集列表: {list(text_feat.keys())}")
        
        # 检查第一个数据集的结构
        first_key = list(text_feat.keys())[0]
        first_val = text_feat[first_key]
        print(f"\n示例数据集 '{first_key}':")
        if hasattr(first_val, 'shape'):
            print(f"  Shape: {first_val.shape}")
            print(f"  Dtype: {first_val.dtype}")
            print(f"  特征维度: {first_val.shape[-1] if len(first_val.shape) > 1 else 'N/A'}")
        elif isinstance(first_val, dict):
            print(f"  Keys (前5个): {list(first_val.keys())[:5]}")
        elif isinstance(first_val, (list, np.ndarray)):
            print(f"  Length: {len(first_val)}")
    
    return text_feat


def explore_text_classifier(data_dir, model_name='RN50_openai'):
    """探索 text_classifier 文件结构 - VEGA 需要的类别文本嵌入"""
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/text_classifier', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索 text_classifier 文件: {feat_path}")
    print("=" * 60)
    
    if not os.path.exists(feat_path):
        print(f"文件不存在: {feat_path}")
        return None
    
    with open(feat_path, 'rb') as f:
        text_classifier = pickle.load(f)
    
    print(f"类型: {type(text_classifier)}")
    
    if isinstance(text_classifier, dict):
        print(f"数据集数量: {len(text_classifier)}")
        print(f"数据集列表: {list(text_classifier.keys())}")
        
        # 检查几个数据集的结构
        for dataset_name in ['cars', 'cifar100', 'pets']:
            if dataset_name in text_classifier:
                val = text_classifier[dataset_name]
                print(f"\n数据集 '{dataset_name}':")
                if hasattr(val, 'shape'):
                    print(f"  Shape: {val.shape}")
                    print(f"  Dtype: {val.dtype}")
                    print(f"  类别数量: {val.shape[0]}")
                    print(f"  特征维度: {val.shape[1]}")
                elif isinstance(val, dict):
                    print(f"  Keys (前5个): {list(val.keys())[:5]}")
                break
    
    return text_classifier


def explore_class_level_acc(data_dir, model_name='RN50_openai'):
    """探索类别级别准确率"""
    acc_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/class_level_acc', f'{model_name}.pkl')
    
    print("\n" + "=" * 60)
    print(f"探索类别级别准确率: {acc_path}")
    print("=" * 60)
    
    if not os.path.exists(acc_path):
        print(f"文件不存在: {acc_path}")
        return None
    
    with open(acc_path, 'rb') as f:
        acc = pickle.load(f)
    
    print(f"类型: {type(acc)}")
    
    if isinstance(acc, dict):
        print(f"数据集数量: {len(acc)}")
        
        # 检查 cars 数据集
        if 'cars' in acc:
            cars_acc = acc['cars']
            print(f"\nCars 数据集:")
            print(f"  类别数量: {len(cars_acc)}")
            # 计算平均准确率
            avg_acc = np.mean(list(cars_acc.values()))
            print(f"  平均准确率: {avg_acc:.4f}")
            print(f"  前3个类别: {list(cars_acc.items())[:3]}")
    
    return acc


def explore_classnames(data_dir):
    """
    探索 data/datasets/classnames 目录
    这是 VEGA 需要的类别名称来源
    """
    classnames_dir = os.path.join(data_dir, 'data/datasets/classnames')
    
    print("\n" + "=" * 60)
    print(f"探索类别名称目录: {classnames_dir}")
    print("=" * 60)
    
    if not os.path.exists(classnames_dir):
        print(f"目录不存在: {classnames_dir}")
        return None
    
    # 获取所有 .txt 文件
    txt_files = [f for f in os.listdir(classnames_dir) if f.endswith('.txt')]
    txt_files = sorted(txt_files)
    
    print(f"类别名称文件数量: {len(txt_files)}")
    print(f"\n所有数据集的类别名称文件:")
    for f in txt_files:
        print(f"  - {f}")
    
    # 检查几个数据集的类别名称
    sample_datasets = ['cars.txt', 'cifar100.txt', 'pets.txt']
    for dataset_file in sample_datasets:
        filepath = os.path.join(classnames_dir, dataset_file)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                classnames = [line.strip() for line in f.readlines() if line.strip()]
            print(f"\n{dataset_file}:")
            print(f"  类别数量: {len(classnames)}")
            print(f"  前5个类别: {classnames[:5]}")
    
    return txt_files


def explore_model_dir(data_dir):
    """
    探索 model 目录结构
    了解可用的 VLM 模型
    """
    model_dir = os.path.join(data_dir, 'model')
    
    print("\n" + "=" * 60)
    print(f"探索模型目录: {model_dir}")
    print("=" * 60)
    
    if not os.path.exists(model_dir):
        print(f"目录不存在: {model_dir}")
        return None
    
    # 列出目录内容
    items = os.listdir(model_dir)
    print(f"目录内容 ({len(items)} 项):")
    
    # 分类展示
    ckpt_files = []
    model_dirs = []
    other_files = []
    
    for item in items:
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            model_dirs.append(item)
        elif item.endswith('.pt') or item.endswith('.pth'):
            ckpt_files.append(item)
        else:
            other_files.append(item)
    
    print(f"\n检查点文件 ({len(ckpt_files)} 个):")
    for f in ckpt_files[:10]:
        filepath = os.path.join(model_dir, f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")
    if len(ckpt_files) > 10:
        print(f"  ... 还有 {len(ckpt_files) - 10} 个文件")
    
    print(f"\n模型目录 ({len(model_dirs)} 个):")
    for d in model_dirs:
        print(f"  - {d}/")
    
    print(f"\n其他文件 ({len(other_files)} 个):")
    for f in other_files:
        print(f"  - {f}")
    
    # 检查 models.yml 配置文件
    models_yml = os.path.join(model_dir, 'models.yml')
    if os.path.exists(models_yml):
        print(f"\n找到 models.yml 配置文件")
        with open(models_yml, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"文件大小: {len(content)} 字节")
        # 显示前几行
        lines = content.split('\n')[:20]
        print("前20行内容:")
        for line in lines:
            print(f"  {line}")
    
    return {
        'ckpt_files': ckpt_files,
        'model_dirs': model_dirs,
        'other_files': other_files
    }


def explore_templates(data_dir):
    """
    探索 prompt templates 目录
    VEGA 需要这些模板来生成文本嵌入
    """
    templates_dir = os.path.join(data_dir, 'LOVM/templates')
    
    print("\n" + "=" * 60)
    print(f"探索 Prompt Templates 目录: {templates_dir}")
    print("=" * 60)
    
    if not os.path.exists(templates_dir):
        print(f"目录不存在: {templates_dir}")
        return None
    
    txt_files = [f for f in os.listdir(templates_dir) if f.endswith('.txt')]
    txt_files = sorted(txt_files)
    
    print(f"模板文件数量: {len(txt_files)}")
    
    # 检查几个模板文件
    sample_files = ['cars.txt', 'cifar100.txt', 'pets.txt']
    for filename in sample_files:
        filepath = os.path.join(templates_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                templates = f.readlines()
            print(f"\n{filename}:")
            print(f"  模板数量: {len(templates)}")
            for t in templates[:3]:
                print(f"    {t.strip()}")
    
    return txt_files


def list_available_models(data_dir):
    """列出所有可用的模型（从 logits 文件推断）"""
    logits_dir = os.path.join(data_dir, 'ptm_stats/logits')
    
    print("\n" + "=" * 60)
    print("可用的模型列表（从 logits 推断）")
    print("=" * 60)
    
    if not os.path.exists(logits_dir):
        print(f"目录不存在: {logits_dir}")
        return [], []
    
    # 获取所有 .pth 文件
    files = [f for f in os.listdir(logits_dir) if f.endswith('.pth')]
    
    # 提取模型名称
    models = set()
    datasets = set()
    for f in files:
        # 格式: model__dataset.pth
        parts = f.replace('.pth', '').split('__')
        if len(parts) == 2:
            models.add(parts[0])
            datasets.add(parts[1])
    
    models = sorted(list(models))
    datasets = sorted(list(datasets))
    
    print(f"模型数量: {len(models)}")
    print(f"数据集数量: {len(datasets)}")
    print(f"\n前10个模型:")
    for m in models[:10]:
        print(f"  - {m}")
    print(f"\n所有数据集:")
    for d in datasets:
        print(f"  - {d}")
    
    return models, datasets


def check_vega_requirements(data_dir):
    """
    检查 VEGA 所需的数据是否完整
    """
    print("\n" + "=" * 60)
    print("VEGA 数据需求检查")
    print("=" * 60)
    
    # VEGA 需要:
    # 1. 图像特征 (img_feat)
    # 2. 类别文本嵌入 (需要从类别名称生成)
    # 3. Logits (用于生成伪标签)
    
    model_name = 'RN50_openai'
    
    # 检查图像特征
    img_feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', f'{model_name}.pkl')
    img_feat_ok = os.path.exists(img_feat_path)
    print(f"\n[{'✓' if img_feat_ok else '✗'}] 图像特征: {img_feat_path}")
    
    # 检查 logits
    logits_path = os.path.join(data_dir, 'ptm_stats/logits')
    logits_ok = os.path.exists(logits_path) and len(os.listdir(logits_path)) > 0
    print(f"[{'✓' if logits_ok else '✗'}] Logits 目录: {logits_path}")
    
    # 检查类别名称
    classnames_dir = os.path.join(data_dir, 'data/datasets/classnames')
    classnames_ok = os.path.exists(classnames_dir)
    print(f"[{'✓' if classnames_ok else '✗'}] 类别名称目录: {classnames_dir}")
    
    # 检查 text_classifier (当前可用的文本嵌入)
    text_classifier_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/text_classifier', f'{model_name}.pkl')
    text_classifier_ok = os.path.exists(text_classifier_path)
    print(f"[{'✓' if text_classifier_ok else '✗'}] Text Classifier: {text_classifier_path}")
    
    # 检查 caption_text_feat (可能不适用于 VEGA)
    caption_text_feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', f'{model_name}.pkl')
    caption_text_feat_ok = os.path.exists(caption_text_feat_path)
    print(f"[{'✓' if caption_text_feat_ok else '✗'}] Caption Text Feat: {caption_text_feat_path}")
    
    print("\n" + "-" * 60)
    print("结论:")
    if text_classifier_ok:
        print("  ✓ text_classifier 可用于 VEGA (包含类别文本嵌入)")
    else:
        print("  ✗ text_classifier 不可用，需要生成类别文本嵌入")
    
    if classnames_ok:
        print("  ✓ 类别名称可用，可以生成 VEGA 所需的文本嵌入")
    else:
        print("  ✗ 类别名称不可用，无法生成文本嵌入")
    
    return {
        'img_feat': img_feat_ok,
        'logits': logits_ok,
        'classnames': classnames_ok,
        'text_classifier': text_classifier_ok,
        'caption_text_feat': caption_text_feat_ok
    }


def main():
    # 数据目录（服务器上的 SWAB 项目路径）
    data_dir = '/root/mxy/SWAB'
    
    # 如果在本地测试，使用当前目录
    if not os.path.exists(data_dir):
        print(f"警告: 服务器路径不存在，使用当前目录")
        data_dir = '.'
    
    print("=" * 60)
    print("VEGA 数据探索")
    print("=" * 60)
    
    # 1. 列出可用模型
    models, datasets = list_available_models(data_dir)
    
    # 2. 探索 logits 结构
    logits = explore_logits(data_dir, 'RN50_openai', 'cars')
    
    # 3. 探索图像特征
    img_feat = explore_img_feat(data_dir, 'RN50_openai')
    
    # 4. 探索文本特征 (caption_text_feat)
    text_feat = explore_text_feat(data_dir, 'RN50_openai')
    
    # 5. 探索 text_classifier (VEGA 可能需要)
    text_classifier = explore_text_classifier(data_dir, 'RN50_openai')
    
    # 6. 探索类别级别准确率
    acc = explore_class_level_acc(data_dir, 'RN50_openai')
    
    # 7. 探索类别名称目录 (VEGA 需要)
    classnames = explore_classnames(data_dir)
    
    # 8. 探索模型目录
    model_info = explore_model_dir(data_dir)
    
    # 9. 探索 prompt templates
    templates = explore_templates(data_dir)
    
    # 10. 检查 VEGA 数据需求
    vega_check = check_vega_requirements(data_dir)
    
    print("\n" + "=" * 60)
    print("探索完成！")
    print("=" * 60)
    
    print("\n下一步建议:")
    if vega_check['text_classifier']:
        print("  1. text_classifier 可用，可以直接用于 VEGA")
        print("  2. 需要确认 text_classifier 的特征维度是否与 img_feat 对齐")
    else:
        print("  1. 需要创建脚本生成类别文本嵌入")
        print("  2. 参考 data/datasets/classnames/ 中的类别名称")
        print("  3. 使用 model/ 中的 VLM 文本编码器生成嵌入")
    
    print("\n生成文本嵌入的命令（在服务器运行）:")
    print("  python scripts/generate_vega_text_features.py")


if __name__ == '__main__':
    main()