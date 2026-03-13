#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探索 caption_text_feat 的内部数据组织方式。

目标：
1. 确认 pickle 的顶层结构与键分布；
2. 递归打印有限深度的 schema 摘要；
3. 标记可能可被 VEGA 消费的文本特征候选路径；
4. 检查目标 dataset 是否存在于任意层级。

默认运行环境：/root/mxy/SWAB
"""

import argparse
import os
import pickle
from collections.abc import Mapping, Sequence

import numpy as np
import torch


DEFAULT_DATA_DIR = '/root/mxy/SWAB'


def summarize_scalar(value, max_len=80):
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + '...'
    return text


def is_string_sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def describe_array(value):
    if isinstance(value, torch.Tensor):
        return 'torch.Tensor shape=%s dtype=%s' % (tuple(value.shape), value.dtype)
    if isinstance(value, np.ndarray):
        return 'numpy.ndarray shape=%s dtype=%s' % (value.shape, value.dtype)
    return type(value).__name__


def is_embedding_vector(value):
    return (
        isinstance(value, (np.ndarray, torch.Tensor))
        and len(value.shape) == 1
        and value.shape[0] > 0
    )


def is_embedding_matrix(value):
    return (
        isinstance(value, (np.ndarray, torch.Tensor))
        and len(value.shape) == 2
        and value.shape[0] > 0
        and value.shape[1] > 0
    )


def inspect_candidate(value):
    """返回 (kind, detail) 或 None。"""
    if is_embedding_matrix(value):
        shape = tuple(value.shape)
        dtype = value.dtype
        return 'direct-matrix', '可直接作为 [K, D] 使用: shape=%s dtype=%s' % (shape, dtype)

    if isinstance(value, Mapping) and value:
        items = list(value.items())
        vectors = []
        matrices = []
        for _, item in items:
            if is_embedding_vector(item):
                vectors.append(item)
            elif is_embedding_matrix(item):
                matrices.append(item)

        if vectors and len(vectors) == len(items):
            dims = sorted({int(v.shape[0]) for v in vectors})
            return 'dict-of-vectors', '可堆叠为 [K, D]: entries=%d dims=%s' % (len(vectors), dims)

        if matrices and len(matrices) == len(items):
            shapes = sorted({tuple(m.shape) for m in matrices})
            return 'dict-of-matrices', '每个键下是 2D 特征，需要聚合: entries=%d shapes=%s' % (len(matrices), shapes)

    if is_string_sequence(value) and value:
        vectors = [item for item in value if is_embedding_vector(item)]
        matrices = [item for item in value if is_embedding_matrix(item)]
        if vectors and len(vectors) == len(value):
            dims = sorted({int(v.shape[0]) for v in vectors})
            return 'sequence-of-vectors', '可堆叠为 [K, D]: entries=%d dims=%s' % (len(vectors), dims)
        if matrices and len(matrices) == len(value):
            shapes = sorted({tuple(m.shape) for m in matrices})
            return 'sequence-of-matrices', '每个元素是 2D 特征，需要聚合: entries=%d shapes=%s' % (len(matrices), shapes)

    return None


def key_matches_dataset(key, dataset_name):
    if dataset_name is None:
        return False
    return dataset_name.lower() in str(key).lower()


def object_summary(value):
    if isinstance(value, Mapping):
        return 'dict(len=%d)' % len(value)
    if isinstance(value, torch.Tensor):
        return 'torch.Tensor(shape=%s, dtype=%s)' % (tuple(value.shape), value.dtype)
    if isinstance(value, np.ndarray):
        return 'numpy.ndarray(shape=%s, dtype=%s)' % (value.shape, value.dtype)
    if is_string_sequence(value):
        return '%s(len=%d)' % (type(value).__name__, len(value))
    return '%s(%s)' % (type(value).__name__, summarize_scalar(value))


def explore_node(
    value,
    path,
    dataset_name,
    max_depth,
    max_items,
    lines,
    dataset_hits,
    candidate_hits,
    visited,
    depth=0,
):
    object_id = id(value)
    if object_id in visited:
        lines.append('%s%s -> %s [visited]' % ('  ' * depth, path, object_summary(value)))
        return
    visited.add(object_id)

    summary = object_summary(value)
    lines.append('%s%s -> %s' % ('  ' * depth, path, summary))

    candidate = inspect_candidate(value)
    if candidate is not None:
        candidate_hits.append({
            'path': path,
            'kind': candidate[0],
            'detail': candidate[1],
        })
        lines.append('%s  [candidate] %s' % ('  ' * depth, candidate[1]))

    if dataset_name is not None:
        path_lower = path.lower()
        if dataset_name.lower() in path_lower:
            dataset_hits.append({'path': path, 'reason': 'path contains dataset name'})

    if depth >= max_depth:
        return

    if isinstance(value, Mapping):
        items = list(value.items())
        if not items:
            return
        for idx, (key, item) in enumerate(items[:max_items]):
            child_path = '%s[%r]' % (path, key)
            if key_matches_dataset(key, dataset_name):
                dataset_hits.append({'path': child_path, 'reason': 'dict key matches dataset'})
            explore_node(
                item,
                child_path,
                dataset_name,
                max_depth,
                max_items,
                lines,
                dataset_hits,
                candidate_hits,
                visited,
                depth=depth + 1,
            )
        remaining = len(items) - min(len(items), max_items)
        if remaining > 0:
            lines.append('%s  ... %d more dict entries omitted' % ('  ' * depth, remaining))
        return

    if is_string_sequence(value):
        if not value:
            return
        for idx, item in enumerate(list(value)[:max_items]):
            child_path = '%s[%d]' % (path, idx)
            if dataset_name is not None and isinstance(item, str) and dataset_name.lower() in item.lower():
                dataset_hits.append({'path': child_path, 'reason': 'sequence item matches dataset'})
            explore_node(
                item,
                child_path,
                dataset_name,
                max_depth,
                max_items,
                lines,
                dataset_hits,
                candidate_hits,
                visited,
                depth=depth + 1,
            )
        remaining = len(value) - min(len(value), max_items)
        if remaining > 0:
            lines.append('%s  ... %d more sequence entries omitted' % ('  ' * depth, remaining))


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def print_header(title):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def print_section(title):
    print('\n' + '-' * 80)
    print(title)
    print('-' * 80)


def resolve_file_path(data_dir, model_name):
    return os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', '%s.pkl' % model_name)


def print_top_level_summary(payload, dataset_name, max_items):
    print_section('顶层摘要')
    print('类型: %s' % type(payload))
    print('摘要: %s' % object_summary(payload))

    if isinstance(payload, Mapping):
        keys = list(payload.keys())
        print('顶层键数量: %d' % len(keys))
        print('顶层前 %d 个键:' % min(len(keys), max_items))
        for key in keys[:max_items]:
            print('  - %r' % key)
        if dataset_name is not None:
            print('目标数据集 %r 是否在顶层: %s' % (dataset_name, dataset_name in payload))
    elif is_string_sequence(payload):
        print('顶层长度: %d' % len(payload))
        print('顶层前 %d 个元素类型:' % min(len(payload), max_items))
        for item in list(payload)[:max_items]:
            print('  - %s' % object_summary(item))
    else:
        print('顶层值: %s' % summarize_scalar(payload))


def print_dataset_hits(dataset_name, dataset_hits):
    print_section('数据集命中报告')
    if dataset_name is None:
        print('未提供 --dataset，跳过数据集命中检查。')
        return

    if not dataset_hits:
        print('未发现与数据集 %r 直接相关的键名、路径或字符串命中。' % dataset_name)
        print('这通常意味着：该文件可能不是按 dataset 显式索引，或 dataset 信息被编码在其他字段中。')
        return

    print('与数据集 %r 相关的命中路径:' % dataset_name)
    seen = set()
    for hit in dataset_hits:
        key = (hit['path'], hit['reason'])
        if key in seen:
            continue
        seen.add(key)
        print('  - %s | %s' % (hit['path'], hit['reason']))


def print_candidates(candidate_hits):
    print_section('VEGA 候选路径')
    if not candidate_hits:
        print('未发现明显可转为 [K, D] 的候选节点。')
        print('这说明后续很可能不是简单换路径，而是需要额外聚合或重建。')
        return

    unique_hits = []
    seen = set()
    for hit in candidate_hits:
        key = (hit['path'], hit['kind'], hit['detail'])
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append(hit)

    for hit in unique_hits:
        print('  - %s' % hit['path'])
        print('    kind: %s' % hit['kind'])
        print('    detail: %s' % hit['detail'])


def print_conclusion(payload, dataset_name, candidate_hits):
    print_section('结论提示')
    if isinstance(payload, Mapping) and dataset_name is not None and dataset_name in payload:
        print('当前文件在顶层直接包含目标 dataset 键，v6 loader 也许只需要更细的子层适配。')
    elif dataset_name is not None:
        print('当前文件在顶层不包含目标 dataset 键，run_benchmark_v6.py 的直取假设不成立。')
    else:
        print('未指定 dataset，仅能给出结构级结论。')

    direct = [hit for hit in candidate_hits if hit['kind'] == 'direct-matrix']
    aggregate = [hit for hit in candidate_hits if hit['kind'] != 'direct-matrix']
    if direct:
        print('存在可直接使用的 2D 候选路径，后续优先检查这些路径与类别数 K 是否匹配。')
    elif aggregate:
        print('存在需要聚合后才能用的候选路径，后续需要决定聚合策略，而不是只改字典索引。')
    else:
        print('没有发现明显候选路径，caption_text_feat 可能并不直接面向 VEGA 的类别级文本输入。')


def parse_args():
    parser = argparse.ArgumentParser(description='探索 caption_text_feat 的内部数据结构')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='SWAB 根目录')
    parser.add_argument('--model', default='RN50_openai', help='模型名称')
    parser.add_argument('--dataset', default=None, help='要检查的数据集名称')
    parser.add_argument('--max-depth', type=int, default=3, help='递归打印深度')
    parser.add_argument('--max-items', type=int, default=8, help='每层最多展示的键或元素数量')
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = resolve_file_path(args.data_dir, args.model)

    print_header('Caption Text Feature Schema Explorer')
    print('data_dir : %s' % args.data_dir)
    print('model    : %s' % args.model)
    print('dataset  : %s' % args.dataset)
    print('file     : %s' % file_path)
    print('max_depth: %d' % args.max_depth)
    print('max_items: %d' % args.max_items)

    if not os.path.exists(file_path):
        print('\n文件不存在: %s' % file_path)
        raise SystemExit(1)

    payload = load_pickle(file_path)
    print_top_level_summary(payload, args.dataset, args.max_items)

    lines = []
    dataset_hits = []
    candidate_hits = []
    visited = set()

    print_section('递归 Schema 摘要')
    explore_node(
        payload,
        'root',
        args.dataset,
        args.max_depth,
        args.max_items,
        lines,
        dataset_hits,
        candidate_hits,
        visited,
        depth=0,
    )
    for line in lines:
        print(line)

    print_dataset_hits(args.dataset, dataset_hits)
    print_candidates(candidate_hits)
    print_conclusion(payload, args.dataset, candidate_hits)


if __name__ == '__main__':
    main()