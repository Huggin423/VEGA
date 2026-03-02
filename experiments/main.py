"""
Main entry point for VLM model selection experiments.
Usage: python experiments/main.py [options]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from data import PTMDataLoader
from methods.baseline.logme import LogME
from methods.baseline.vega import VEGAScorer, compute_vega_score
from evaluation.metrics import compute_full_metrics, print_metrics
from configs.dataset_config import get_dataset_list, VALID_DATASETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_dataset(
    loader: PTMDataLoader,
    dataset_name: str,
    methods: List[str] = ['logme', 'vega'],
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """
    Run experiments on a single dataset.
    
    Args:
        loader: PTMDataLoader instance
        dataset_name: Name of the dataset
        methods: List of methods to run
        output_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results for all models
    """
    results = {}
    
    # Get available models
    models = loader.get_available_models()
    
    if not models:
        logger.warning(f"No models available for dataset {dataset_name}")
        return results
    
    if verbose:
        logger.info(f"Running experiments on {dataset_name} with {len(models)} models")
    
    # Collect data for all models
    features_dict = {}
    logits_dict = {}
    ground_truth_acc = {}
    text_embeddings = None
    labels = None
    
    for model_name in models:
        try:
            features, logits, targets = loader.load_data(model_name, dataset_name)
            
            features_dict[model_name] = features.numpy()
            logits_dict[model_name] = logits.numpy()
            
            # Compute ground truth accuracy
            acc = (logits.argmax(dim=1) == targets).float().mean().item()
            ground_truth_acc[model_name] = acc
            
            # Store labels (same for all models)
            if labels is None:
                labels = targets.numpy()
            
            # Load text embeddings (same for all models)
            if text_embeddings is None:
                try:
                    text_embeddings = loader.load_text_classifier(model_name, dataset_name).numpy()
                except Exception as e:
                    logger.warning(f"Could not load text embeddings for {model_name}: {e}")
            
        except Exception as e:
            logger.warning(f"Error loading data for {model_name} on {dataset_name}: {e}")
            continue
    
    if not features_dict:
        logger.warning(f"No valid data loaded for dataset {dataset_name}")
        return results
    
    # Run LogME
    if 'logme' in methods:
        logme_scores = {}
        for model_name in features_dict:
            try:
                logme = LogME()
                score = logme.fit(features_dict[model_name], labels)
                logme_scores[model_name] = score
            except Exception as e:
                logger.warning(f"LogME failed for {model_name}: {e}")
                logme_scores[model_name] = float('-inf')
        
        results['LogME'] = compute_full_metrics(logme_scores, ground_truth_acc)
        if verbose:
            print_metrics(results['LogME'], f"LogME Results - {dataset_name}")
    
    # Run VEGA
    if 'vega' in methods and text_embeddings is not None:
        vega_scores = {}
        vega = VEGAScorer(temperature=0.05)
        
        for model_name in features_dict:
            try:
                score = vega.compute_score(
                    features_dict[model_name],
                    text_embeddings,
                    logits_dict[model_name]
                )
                vega_scores[model_name] = score
            except Exception as e:
                logger.warning(f"VEGA failed for {model_name}: {e}")
                vega_scores[model_name] = 0.0
        
        results['VEGA'] = compute_full_metrics(vega_scores, ground_truth_acc)
        if verbose:
            print_metrics(results['VEGA'], f"VEGA Results - {dataset_name}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f'{dataset_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def run_all_datasets(
    root_path: str,
    datasets: List[str] = None,
    methods: List[str] = ['logme', 'vega'],
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """
    Run experiments on all datasets.
    
    Args:
        root_path: Project root directory
        datasets: List of datasets to run (None = all valid)
        methods: List of methods to run
        output_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results for all datasets
    """
    loader = PTMDataLoader(root_path)
    
    if datasets is None:
        datasets = get_dataset_list(exclude_failed=True)
    
    all_results = {}
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            results = run_single_dataset(
                loader, dataset_name, methods, output_dir, verbose
            )
            all_results[dataset_name] = results
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Save aggregated results
    if output_dir:
        summary_path = os.path.join(output_dir, 'all_results.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nAll results saved to {summary_path}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run VLM model selection experiments'
    )
    parser.add_argument(
        '--root', type=str, default='.',
        help='Project root directory'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset name (default: run all)'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+',
        default=['logme', 'vega'],
        help='Methods to run'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(args.root, 'results', timestamp)
    
    # Run experiments
    if args.dataset:
        loader = PTMDataLoader(args.root)
        run_single_dataset(
            loader, args.dataset, args.methods, args.output,
            verbose=not args.quiet
        )
    else:
        run_all_datasets(
            args.root, None, args.methods, args.output,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()