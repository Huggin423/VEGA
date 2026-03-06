#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEGA vs LogME Benchmark Script (Improved Version)
Compare VEGA and LogME methods on model selection tasks

Environment: Lab server /root/mxy/VEGA (symlink to SWAB data)

Changelog:
- 2026-03-06: Use VEGAScorer class from methods/baseline/vega.py
- 2026-03-06: Add PCA whitening, diagonal covariance, vectorized computation
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
import time
from datetime import datetime
import json
import hashlib
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import optimized VEGA implementation
from methods.baseline.vega import VEGAScorer

# ============================================================================
# Configuration
# ============================================================================

CACHE_DIR = project_root / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ENABLE_CACHE = True

# ============================================================================
# Progress Display Tools
# ============================================================================

class ProgressBar:
    """Simple progress bar display"""
    
    def __init__(self, total, desc="Progress", width=50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, n=1, info=""):
        self.current += n
        elapsed = time.time() - self.start_time
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        bar = '#' * filled + '-' * (self.width - filled)
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = "ETA: %.0fs" % eta
        else:
            eta_str = "ETA: --"
        status = "%s: |%s| %d/%d [%.1f%%] %s" % (self.desc, bar, self.current, self.total, progress*100, eta_str)
        if info:
            status += " | %s" % info
        print("\r%s" % status, end='', flush=True)
        if self.current >= self.total:
            print(" | Done: %.1fs" % elapsed)
    
    def close(self):
        if self.current < self.total:
            print()


def print_header(title, width=70):
    print("\n" + "=" * width)
    print(" %s" % title)
    print("=" * width)


def print_subheader(title, width=70):
    print("\n" + "-" * width)
    print(" %s" % title)
    print("-" * width)


def print_status(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print("[%s] [%s] %s" % (timestamp, level, msg))


def print_detail(msg, indent=2):
    print(" " * indent + msg)


# ============================================================================
# Cache System
# ============================================================================

def get_cache_key(model_name, dataset_name, method):
    key_str = "%s_%s_%s" % (model_name, dataset_name, method)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(model_name, dataset_name, method):
    cache_key = get_cache_key(model_name, dataset_name, method)
    return CACHE_DIR / ("%s.pkl" % cache_key)


def save_cache(model_name, dataset_name, method, data):
    if not ENABLE_CACHE:
        return
    cache_path = get_cache_path(model_name, dataset_name, method)
    cache_data = {
        'model': model_name,
        'dataset': dataset_name,
        'method': method,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print_detail("[Cache] Saved %s result" % method, indent=4)


def load_cache(model_name, dataset_name, method):
    if not ENABLE_CACHE:
        return None
    cache_path = get_cache_path(model_name, dataset_name, method)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print_detail("[Cache] Hit %s result (time: %s)" % (method, cache_data['timestamp']), indent=4)
        return cache_data['data']
    except Exception as e:
        print_detail("[Cache] Load failed: %s" % e, indent=4)
        return None


def clear_cache():
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        print_status("Cache cleared")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_logits_data(data_dir, model_name, dataset_name):
    """Load model logits data"""
    logits_path = os.path.join(data_dir, 'ptm_stats/logits', '%s__%s.pth' % (model_name, dataset_name))
    if not os.path.exists(logits_path):
        return None
    data = torch.load(logits_path, map_location='cpu')
    result = {}
    if isinstance(data, dict):
        if 'logits' in data:
            logits = data['logits']
            result['logits'] = logits.numpy() if isinstance(logits, torch.Tensor) else logits
        if 'labels' in data:
            labels = data['labels']
            result['labels'] = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        if 'acc1' in data:
            result['acc1'] = data['acc1']
    elif isinstance(data, torch.Tensor):
        result['logits'] = data.numpy()
    elif isinstance(data, np.ndarray):
        result['logits'] = data
    return result if 'logits' in result else None


def load_image_features(data_dir, model_name, dataset_name, verbose=False):
    """Load model image features"""
    feat_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/img_feat', '%s.pkl' % model_name)
    if not os.path.exists(feat_path):
        if verbose:
            print_detail("[!] Image feature file not found: %s" % feat_path)
        return None
    with open(feat_path, 'rb') as f:
        img_feats = pickle.load(f)
    if not isinstance(img_feats, dict) or dataset_name not in img_feats:
        if verbose:
            print_detail("[!] Dataset %s not in image features" % dataset_name)
        return None
    dataset_feats = img_feats[dataset_name]
    if isinstance(dataset_feats, dict):
        all_features = []
        for class_name, feat in dataset_feats.items():
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            if isinstance(feat, np.ndarray):
                if len(feat.shape) == 1:
                    all_features.append(feat)
                elif len(feat.shape) == 2:
                    all_features.extend(feat)
                elif len(feat.shape) == 3:
                    all_features.extend(feat.reshape(-1, feat.shape[-1]))
        if all_features:
            return np.array(all_features)
    elif isinstance(dataset_feats, (torch.Tensor, np.ndarray)):
        feat = dataset_feats.numpy() if isinstance(dataset_feats, torch.Tensor) else dataset_feats
        if len(feat.shape) == 2:
            return feat
        elif len(feat.shape) == 3:
            return feat.reshape(-1, feat.shape[-1])
    return None


def load_text_features(data_dir, model_name, dataset_name, verbose=False):
    """Load model text features (class embeddings)"""
    search_paths = [
        os.path.join(data_dir, 'ptm_stats/class_text_feat', '%s.pkl' % model_name),
        os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/caption_text_feat', '%s.pkl' % model_name),
        os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/syn_text_feat', '%s.pkl' % model_name),
    ]
    feat_path = None
    for path in search_paths:
        if os.path.exists(path):
            feat_path = path
            break
    if feat_path is None:
        if verbose:
            print_detail("[!] Cannot load text features")
        return None
    with open(feat_path, 'rb') as f:
        text_feats = pickle.load(f)
    if not isinstance(text_feats, dict) or dataset_name not in text_feats:
        if verbose:
            print_detail("[!] Dataset %s not in text features" % dataset_name)
        return None
    dataset_feats = text_feats[dataset_name]
    if isinstance(dataset_feats, (torch.Tensor, np.ndarray)):
        emb = dataset_feats.cpu().numpy() if isinstance(dataset_feats, torch.Tensor) else dataset_feats
        if verbose:
            print_detail("Text features (array format): %s" % str(emb.shape), indent=4)
        return emb
    if isinstance(dataset_feats, dict):
        embeddings = []
        for class_name, emb in dataset_feats.items():
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            if isinstance(emb, np.ndarray):
                if len(emb.shape) == 1:
                    embeddings.append(emb)
                elif len(emb.shape) == 2 and emb.shape[0] == 1:
                    embeddings.append(emb.flatten())
        if embeddings:
            result = np.array(embeddings)
            if verbose:
                print_detail("Text features (dict format): %s" % str(result.shape), indent=4)
            return result
    return None


def load_ground_truth_accuracy(data_dir, model_name, dataset_name):
    """Load model ground truth accuracy on dataset"""
    logits_data = load_logits_data(data_dir, model_name, dataset_name)
    if logits_data and 'acc1' in logits_data:
        return logits_data['acc1']
    acc_path = os.path.join(data_dir, 'ptm_stats/stats_on_hist_task/class_level_acc', '%s.pkl' % model_name)
    if not os.path.exists(acc_path):
        return None
    with open(acc_path, 'rb') as f:
        acc_data = pickle.load(f)
    if isinstance(acc_data, dict) and dataset_name in acc_data:
        class_acc = acc_data[dataset_name]
        if isinstance(class_acc, dict):
            return np.mean(list(class_acc.values()))
        elif isinstance(class_acc, (list, np.ndarray)):
            return np.mean(class_acc)
    return None


# ============================================================================
# VEGA Score Computation (with progress log)
# ============================================================================

def compute_vega_score_with_progress(img_features, text_features, logits, model_name=""):
    """
    Compute VEGA score using optimized VEGAScorer class
    
    Optimizations:
    1. PCA dimensionality reduction and whitening
    2. Diagonal covariance approximation (avoids matrix inversion)
    3. Vectorized Bhattacharyya distance computation (no double for loop)
    4. Reused Cosine Similarity matrix
    5. Numerical stability with epsilon
    """
    print_detail("Computing VEGA score (optimized version):", indent=4)
    print_detail("- Image features: %s" % str(img_features.shape), indent=6)
    print_detail("- Text features: %s" % str(text_features.shape), indent=6)
    print_detail("- Logits: %s" % str(logits.shape), indent=6)
    
    start_time = time.time()
    
    try:
        # Use optimized VEGAScorer class
        vega = VEGAScorer(
            temperature=0.05,
            min_samples_per_class=1,
            use_pca=True,
            pca_dim=256,
            pca_whiten=True
        )
        
        # Compute score with detailed results
        result = vega.compute_score(
            features=img_features,
            text_embeddings=text_features,
            logits=logits,
            return_details=True
        )
        
        elapsed = time.time() - start_time
        
        # Extract results
        vega_score = result.get('score', 0.0)
        node_sim = result.get('node_similarity', 0.0)
        edge_sim = result.get('edge_similarity', 0.0)
        valid_classes = result.get('valid_classes', 0)
        pca_dim = result.get('pca_dim', img_features.shape[1])
        
        print_detail("VEGA total score: %.4f (time: %.2fs)" % (vega_score, elapsed), indent=4)
        print_detail("  - Node similarity: %.4f" % node_sim, indent=6)
        print_detail("  - Edge similarity: %.4f" % edge_sim, indent=6)
        print_detail("  - Valid classes: %d" % valid_classes, indent=6)
        print_detail("  - PCA dimension: %d" % pca_dim, indent=6)
        
        return_result = {
            'score': vega_score,
            'node_similarity': node_sim,
            'edge_similarity': edge_sim,
            'valid_classes': valid_classes,
            'computation_time': elapsed,
            'pca_dim': pca_dim,
            'diagonal_covariance': True,
            'vectorized_computation': True
        }
        
        return vega_score, return_result
        
    except Exception as e:
        print_detail("[!] VEGA computation error: %s" % e, indent=4)
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


def compute_logme_score(features, pseudo_labels):
    """Compute LogME score using LogME_official library"""
    print_detail("Computing LogME score:", indent=4)
    print_detail("- Features: %s" % str(features.shape), indent=6)
    print_detail("- Pseudo labels: %s, unique classes: %d" % (str(pseudo_labels.shape), len(np.unique(pseudo_labels))), indent=6)
    
    start_time = time.time()
    
    try:
        from LogME_official.LogME import LogME as LogMEOfficial
        logme = LogMEOfficial(regression=False)
        score = logme.fit(features.astype(np.float64), pseudo_labels.astype(np.int64))
        elapsed = time.time() - start_time
        print_detail("LogME score: %.4f (time: %.2fs)" % (score, elapsed), indent=4)
        return score, {'score': score, 'computation_time': elapsed}
    except Exception as e:
        print_detail("[!] LogME computation error: %s" % e, indent=4)
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(predicted_scores, ground_truth, verbose=True):
    """Compute evaluation metrics"""
    common_models = set(predicted_scores.keys()) & set(ground_truth.keys())
    common_models = [m for m in common_models if predicted_scores[m] is not None and ground_truth[m] is not None]
    
    if len(common_models) < 3:
        if verbose:
            print_detail("[!] Insufficient valid models (%d < 3)" % len(common_models), indent=2)
        return {'error': 'Insufficient data: only %d common models' % len(common_models)}
    
    pred = np.array([predicted_scores[m] for m in common_models])
    gt = np.array([ground_truth[m] for m in common_models])
    
    if verbose:
        print("\n  Model ranking details (%d models):" % len(common_models))
        pred_order = np.argsort(pred)[::-1]
        print("\n  Ranked by predicted score:")
        for rank, idx in enumerate(pred_order):
            model = common_models[idx]
            print("    %d. %s: pred=%.4f, gt_acc=%.4f" % (rank+1, model, pred[idx], gt[idx]))
        gt_order = np.argsort(gt)[::-1]
        print("\n  Ranked by ground truth accuracy:")
        for rank, idx in enumerate(gt_order):
            model = common_models[idx]
            print("    %d. %s: gt_acc=%.4f, pred=%.4f" % (rank+1, model, gt[idx], pred[idx]))
    
    tau, p_value = stats.kendalltau(pred, gt)
    spearman, sp_pvalue = stats.spearmanr(pred, gt)
    pearson, pp_pvalue = stats.pearsonr(pred, gt)
    
    top5_gt = set(np.argsort(gt)[-5:])
    top5_pred = set(np.argsort(pred)[-5:])
    top5_recall = len(top5_gt & top5_pred) / 5
    
    top1_pred_idx = np.argmax(pred)
    top1_pred_model = common_models[top1_pred_idx]
    top1_accuracy = gt[top1_pred_idx]
    
    oracle_idx = np.argmax(gt)
    oracle_accuracy = gt[oracle_idx]
    
    return {
        'kendall_tau': tau,
        'kendall_p': p_value,
        'spearman': spearman,
        'spearman_p': sp_pvalue,
        'pearson': pearson,
        'pearson_p': pp_pvalue,
        'top5_recall': top5_recall,
        'top1_accuracy': top1_accuracy,
        'top1_model': top1_pred_model,
        'oracle_accuracy': oracle_accuracy,
        'oracle_model': common_models[oracle_idx],
        'num_models': len(common_models)
    }


# ============================================================================
# Main Benchmark Function
# ============================================================================

def run_single_dataset_benchmark(data_dir, dataset_name, model_list, verbose=True):
    """Run benchmark on a single dataset"""
    print_subheader("Dataset: %s" % dataset_name)
    
    vega_scores = {}
    logme_scores = {}
    ground_truth = {}
    failed_models = []
    
    pbar = ProgressBar(len(model_list), desc="  %s" % dataset_name)
    
    for idx, model_name in enumerate(model_list):
        pbar.update(1, "Processing %s..." % model_name[:25])
        print("\n  Processing model: %s" % model_name)
        
        cached_vega = load_cache(model_name, dataset_name, 'vega')
        cached_logme = load_cache(model_name, dataset_name, 'logme')
        
        print_detail("Loading data...", indent=4)
        
        logits_data = load_logits_data(data_dir, model_name, dataset_name)
        if logits_data is None:
            print_detail("[!] Cannot load logits", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no logits'})
            continue
        logits = logits_data['logits']
        print_detail("Logits: %s" % str(logits.shape), indent=6)
        
        img_feat = load_image_features(data_dir, model_name, dataset_name)
        if img_feat is None:
            print_detail("[!] Cannot load image features", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no image features'})
            continue
        print_detail("Image features: %s" % str(img_feat.shape), indent=6)
        
        text_feat = load_text_features(data_dir, model_name, dataset_name)
        if text_feat is None:
            print_detail("[!] Cannot load text features", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no text features'})
            continue
        print_detail("Text features: %s" % str(text_feat.shape), indent=6)
        
        gt_acc = load_ground_truth_accuracy(data_dir, model_name, dataset_name)
        if gt_acc is None:
            print_detail("[!] Cannot load accuracy", indent=4)
            failed_models.append({'model': model_name, 'reason': 'no accuracy'})
            continue
        ground_truth[model_name] = gt_acc
        print_detail("Ground truth accuracy: %.4f" % gt_acc, indent=6)
        
        n_samples, n_classes = logits.shape
        n_text_classes = text_feat.shape[0]
        
        if n_text_classes != n_classes:
            print_detail("[!] Class count mismatch: logits=%d, text_feat=%d" % (n_classes, n_text_classes), indent=4)
            min_classes = min(n_classes, n_text_classes)
            logits = logits[:, :min_classes]
            text_feat = text_feat[:min_classes]
            print_detail("Using first %d classes" % min_classes, indent=6)
        
        if img_feat.shape[0] != logits.shape[0]:
            print_detail("[!] Sample count mismatch: img_feat=%d, logits=%d" % (img_feat.shape[0], logits.shape[0]), indent=4)
            min_samples = min(img_feat.shape[0], logits.shape[0])
            img_feat = img_feat[:min_samples]
            logits = logits[:min_samples]
            print_detail("Using first %d samples" % min_samples, indent=6)
        
        if cached_vega is not None:
            vega_score = cached_vega.get('score')
            vega_details = cached_vega
        else:
            vega_score, vega_details = compute_vega_score_with_progress(img_feat, text_feat, logits, model_name)
            if vega_score is not None:
                save_cache(model_name, dataset_name, 'vega', vega_details)
        if vega_score is not None:
            vega_scores[model_name] = vega_score
        
        pseudo_labels = np.argmax(logits, axis=1)
        if cached_logme is not None:
            logme_score = cached_logme.get('score')
            logme_details = cached_logme
        else:
            logme_score, logme_details = compute_logme_score(img_feat, pseudo_labels)
            if logme_score is not None:
                save_cache(model_name, dataset_name, 'logme', logme_details)
        if logme_score is not None:
            logme_scores[model_name] = logme_score
    
    if failed_models:
        print("\n  Failed models summary (%d/%d):" % (len(failed_models), len(model_list)))
        for fail in failed_models:
            print("    - %s: %s" % (fail['model'], fail['reason']))
    
    print("\n" + "=" * 70)
    print("Evaluation results: %s" % dataset_name)
    print("=" * 70)
    
    vega_metrics = compute_metrics(vega_scores, ground_truth, verbose=verbose)
    logme_metrics = compute_metrics(logme_scores, ground_truth, verbose=verbose)
    
    return {
        'dataset': dataset_name,
        'vega_scores': vega_scores,
        'logme_scores': logme_scores,
        'ground_truth': ground_truth,
        'vega_metrics': vega_metrics,
        'logme_metrics': logme_metrics,
        'failed_models': failed_models
    }


def print_final_results(all_results):
    """Print final summary results"""
    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    
    valid_vega = [r for r in all_results if 'error' not in r['vega_metrics']]
    valid_logme = [r for r in all_results if 'error' not in r['logme_metrics']]
    
    if valid_vega:
        avg_vega_tau = np.mean([r['vega_metrics']['kendall_tau'] for r in valid_vega])
        avg_vega_spearman = np.mean([r['vega_metrics']['spearman'] for r in valid_vega])
        avg_vega_top5 = np.mean([r['vega_metrics']['top5_recall'] for r in valid_vega])
        print("\nVEGA (on %d datasets):" % len(valid_vega))
        print("  Average Kendall tau: %.4f" % avg_vega_tau)
        print("  Average Spearman: %.4f" % avg_vega_spearman)
        print("  Average Top-5 Recall: %.2f" % avg_vega_top5)
    else:
        print("\nVEGA: No valid results")
    
    if valid_logme:
        avg_logme_tau = np.mean([r['logme_metrics']['kendall_tau'] for r in valid_logme])
        avg_logme_spearman = np.mean([r['logme_metrics']['spearman'] for r in valid_logme])
        avg_logme_top5 = np.mean([r['logme_metrics']['top5_recall'] for r in valid_logme])
        print("\nLogME (on %d datasets):" % len(valid_logme))
        print("  Average Kendall tau: %.4f" % avg_logme_tau)
        print("  Average Spearman: %.4f" % avg_logme_spearman)
        print("  Average Top-5 Recall: %.2f" % avg_logme_top5)
    else:
        print("\nLogME: No valid results")
    
    print("\nDetailed results by dataset:")
    print("%-25s %10s %10s %10s %10s" % ("Dataset", "VEGA tau", "LogME tau", "VEGA Top5", "LogME Top5"))
    print("-" * 70)
    for r in all_results:
        dataset = r['dataset']
        vega_tau = r['vega_metrics'].get('kendall_tau', float('nan'))
        logme_tau = r['logme_metrics'].get('kendall_tau', float('nan'))
        vega_top5 = r['vega_metrics'].get('top5_recall', float('nan'))
        logme_top5 = r['logme_metrics'].get('top5_recall', float('nan'))
        print("%-25s %10.4f %10.4f %10.2f %10.2f" % (dataset, vega_tau, logme_tau, vega_top5, logme_top5))
    
    print("\nFailed models summary:")
    for r in all_results:
        if r.get('failed_models'):
            print("  %s: %d failed" % (r['dataset'], len(r['failed_models'])))
            for fail in r['failed_models']:
                print("    - %s: %s" % (fail['model'], fail['reason']))


def main():
    """Main function"""
    data_dir = '/root/mxy/SWAB'
    if not os.path.exists(data_dir):
        data_dir = '/root/mxy/VEGA/ptm_stats'
        if not os.path.exists(data_dir):
            print("Error: Data directory not found")
            print("Please run this script on the server")
            return
    
    # 31 CLIP family models (with both logits and class_text_feat)
    # Based on ptm_stats/class_text_feat/ and ptm_stats/logits/
    test_models = [
        # OpenAI CLIP (5 models)
        'RN50_openai',
        'RN101_openai',
        'ViT-B-32_openai',
        'ViT-B-16_openai',
        'ViT-L-14_openai',
        # LAION 400M (7 models)
        'ViT-B-32_laion400m_e31',
        'ViT-B-32_laion400m_e32',
        'ViT-B-32-quickgelu_laion400m_e32',
        'ViT-B-16_laion400m_e32',
        'ViT-B-16-plus-240_laion400m_e32',
        'ViT-L-14_laion400m_e32',
        # LAION 2B (4 models)
        'ViT-B-32_laion2b_e16',
        'ViT-B-32_laion2b_s34b_b79k',
        # SigLIP (2 models)
        'SigLIP_base',
        'SigLIP_so400m',
        # Other architectures (8 models)
        'ALIGN',
        'AltCLIP',
        'GroupViT',
        'MetaCLIP',
        'StreetCLIP',
        'RN50x4_openai',
        'RN50x16_openai',
        'RN50x64_openai',
        # Additional large models (5 models)
        'ViT-L-14-336_openai',
    ]
    
    # 10 datasets (from 22 available, excluding imagenet1k and clevr_closest_object_distance)
    test_datasets = [
        'cars',
        'cifar100',
        'flowers',
        'pets',
        'dtd',
        'eurosat',
        'food101',  # Note: may not have all models
        'gtsrb',
        'mnist',
        'sun397',
    ]
    
    print("=" * 70)
    print("VEGA vs LogME Benchmark (Improved Version)")
    print("=" * 70)
    print("Data directory: %s" % data_dir)
    print("Cache directory: %s" % CACHE_DIR)
    print("Cache enabled: %s" % ENABLE_CACHE)
    print("Number of test models: %d" % len(test_models))
    print("Test datasets: %s" % test_datasets)
    
    all_results = []
    for dataset in test_datasets:
        result = run_single_dataset_benchmark(data_dir, dataset, test_models, verbose=True)
        all_results.append(result)
    
    print_final_results(all_results)
    
    result_file = CACHE_DIR / "benchmark_results.json"
    with open(result_file, 'w') as f:
        serializable_results = []
        for r in all_results:
            sr = {
                'dataset': r['dataset'],
                'vega_scores': r['vega_scores'],
                'logme_scores': r['logme_scores'],
                'ground_truth': r['ground_truth'],
                'vega_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r['vega_metrics'].items()},
                'logme_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r['logme_metrics'].items()},
                'failed_models': r.get('failed_models', [])
            }
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to: %s" % result_file)


if __name__ == '__main__':
    main()