"""
Methods module for VLM model selection.
Contains baseline methods and our proposed method.
"""
from .baseline.logme import LogME
from .baseline.vega_v3 import VEGAv3Scorer, compute_vega_v3_score
from .baseline.vega_v2 import VEGAOptimizedScorer, compute_vega_score_optimized
from .baseline.vega_v1 import VEGAOriginalScorer, compute_vega_score_original



__all__ = ['LogME', 'VEGAv3Scorer', 'compute_vega_v3_score', 'VEGAOptimizedScorer', 'compute_vega_score_optimized', 'VEGAOriginalScorer', 'compute_vega_score_original']
    