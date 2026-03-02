"""
Methods module for VLM model selection.
Contains baseline methods and our proposed method.
"""

from .baseline.logme import LogME
from .baseline.vega import VEGAScorer

__all__ = ['LogME', 'VEGAScorer']