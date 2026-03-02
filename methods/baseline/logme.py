"""
LogME wrapper - uses official LogME implementation.
"""

import sys
import os
from pathlib import Path

# Add LogME_official to path
logme_path = Path(__file__).parent.parent.parent / 'LogME_official'
if str(logme_path) not in sys.path:
    sys.path.insert(0, str(logme_path))

from LogME import LogME

__all__ = ['LogME']