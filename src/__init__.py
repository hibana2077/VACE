"""
VACE source package

Contains the main implementation of VACE (Variance-Adaptive Cross-Entropy) loss function,
dataset utilities, and supporting modules.
"""

from .loss import VACE, EMAStats, VACEHead

__version__ = "0.1.0"

__all__ = [
    'VACE',
    'EMAStats', 
    'VACEHead',
]
