import os
import torch
import pickle
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SWABDataLoader:
    def __init__(self, root_path):
        self.root_path = root_path

        # 路径配置
        self.feat_dir = os.path.join(root_path, 'ptm_stats', 'stats_on_hist_task', 'img_feat')
        self.logits_dir = os.path.join(root_path, 'ptm_stats', 'logits')
        self.classnames_dir = os.path.join(root_path, 'data', 'datasets', 'classnames')

    def _load_classnames(self, dataset_name):
        """
        加载数据集的类别名称列表，用于保证特征拼接的顺序正确
        """
        # 尝试寻找 classnames 文件
        txt_path = os.path.join(self.classnames_dir, f"{dataset_name}.txt")

        if not os.path.exists(txt_path):
            logger.warning(f"Classnames file not found: {txt_path}. Will fallback to alphabetical sorting.")
            return None

        with open(txt_path, 'r') as f:
            # 读取每一行，去除空白符
            classnames = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[DEBUG] Loaded {len(classnames)} classes from {txt_path}")
        return classnames

    def _flatten_features(self, feat_dict, dataset_name):
        """
        将按类别存储的特征字典展平为 (N, D) 的 Tensor
        """
        # 1. 获取正确的类别顺序
        ordered_classes = self._load_classnames(dataset_name)

        # 如果没找到文件，尝试使用字典按键排序（通常 ImageFolder 是按字母序）
        # 或者直接信任字典的插入顺序（Python 3.7+ 字典有序）
        if ordered_classes is None:
            # 这是一个兜底策略，优先尝试字典原本的顺序（因为你的Log显示原本顺序似乎是对的）
            logger.info("Using dictionary insertion order for flattening features.")
            ordered_classes = list(feat_dict.keys())

        feature_list = []
        total_samples = 0

        # 2. 按顺序提取特征
        for cls_name in ordered_classes:
            if cls_name not in feat_dict:
                # 这种情况下可能是 classnames.txt 和 pkl 的 key 不完全匹配（大小写或空格）
                # 这是一个潜在的风险点
                logger.warning(f"Class '{cls_name}' not found in feature dict keys. Skipping...")
                continue

            feats = feat_dict[cls_name]

            # 统一转为 Tensor
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            elif isinstance(feats, list):
                feats = torch.tensor(feats)

            feature_list.append(feats)
            total_samples += feats.shape[0]

        if not feature_list:
            raise ValueError(f"No features extracted for dataset {dataset_name}")

        # 3. 拼接
        all_features = torch.cat(feature_list, dim=0)
        print(f"[DEBUG] Flattened features: {len(feature_list)} classes, Total shape: {all_features.shape}")

        return all_features

    def load_data(self, model_name, dataset_name):
        print(f"\n--- [DEBUG] Loading {model_name} on {dataset_name} ---")

        # ----------------------------------------------------
        # 1. 加载 Features (.pkl)
        # ----------------------------------------------------
        feat_path = os.path.join(self.feat_dir, f"{model_name}.pkl")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file missing: {feat_path}")

        with open(feat_path, 'rb') as f:
            full_pkl = pickle.load(f)

        if dataset_name not in full_pkl:
            raise KeyError(f"Dataset '{dataset_name}' not found in pkl.")

        raw_content = full_pkl[dataset_name]

        # 处理字典结构的特征
        if isinstance(raw_content, dict):
            # 检查是否包含特殊key 'features' (有些pkl结构不同)
            if 'features' in raw_content and not isinstance(raw_content['features'], dict):
                features = torch.tensor(raw_content['features']) if not isinstance(raw_content['features'],
                                                                                   torch.Tensor) else raw_content[
                    'features']
            else:
                # 此时 raw_content 是 {'class_a': feat, 'class_b': feat}
                print(f"[DEBUG] Feature data is Class-wise Dictionary. Flattening...")
                features = self._flatten_features(raw_content, dataset_name)
        else:
            # 已经是数组/Tensor
            features = torch.tensor(raw_content) if not isinstance(raw_content, torch.Tensor) else raw_content

        # ----------------------------------------------------
        # 2. 加载 Logits (.pth)
        # ----------------------------------------------------
        logits_filename = f"{model_name}__{dataset_name}.pth"
        logits_path = os.path.join(self.logits_dir, logits_filename)

        if not os.path.exists(logits_path):
            raise FileNotFoundError(f"Logits file missing: {logits_path}")

        payload = torch.load(logits_path, map_location='cpu')

        logits = None
        targets = None

        if isinstance(payload, dict):
            # 提取 logits
            for k in ['logits', 'outputs', 'predictions']:
                if k in payload: logits = payload[k]; break
            # 提取 targets
            for k in ['targets', 'labels', 'ground_truth']:
                if k in payload: targets = payload[k]; break

            # 兜底
            if logits is None or targets is None:
                keys = list(payload.keys())
                logits = payload[keys[0]]
                targets = payload[keys[1]]

        elif isinstance(payload, (tuple, list)):
            logits = payload[0]
            targets = payload[1]

        if isinstance(logits, np.ndarray): logits = torch.from_numpy(logits)
        if isinstance(targets, np.ndarray): targets = torch.from_numpy(targets)

        # ----------------------------------------------------
        # 3. 最终对齐验证
        # ----------------------------------------------------
        print(f"[DEBUG] Final Check -> Features: {features.shape}, Logits: {logits.shape}")

        if features.shape[0] != logits.shape[0]:
            diff = features.shape[0] - logits.shape[0]
            # 尝试高级验证：对比 Class 分布
            # 计算 Logits 中 target 的分布
            print("[ERROR] Shape Mismatch.")
            unique, counts = torch.unique(targets, return_counts=True)
            print(f"Logits Target Distribution (first 5 classes): {counts[:5]}")

            # 如果特征是按类拼接的，我们可以尝试检查是不是有些类没对上
            raise ValueError(f"Feature count ({features.shape[0]}) != Logits count ({logits.shape[0]}). Diff: {diff}")

        return features.float(), logits.float(), targets.long()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help='SWAB root directory')
    args = parser.parse_args()

    loader = SWABDataLoader(args.root)

    test_model = "RN50_openai"
    test_dataset = "cars"

    try:
        f, l, t = loader.load_data(test_model, test_dataset)
        print(f"\n✅ SUCCESS! Loaded {f.shape[0]} samples.")

        # 简单计算一下准确率，看看特征和标签是否对齐
        # 注意：这里我们无法直接验证特征和标签是否对齐，因为LogME还没跑
        # 但我们可以验证 Logits 和 Targets 的准确率
        acc = (l.argmax(dim=1) == t).float().mean()
        print(f"Logits/Targets Accuracy: {acc:.4%}")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")