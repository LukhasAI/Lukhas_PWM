"""
Hippocampal Memory System
Fast episodic memory encoding with pattern separation and completion
"""

from .hippocampal_buffer import HippocampalBuffer, EpisodicMemory, HippocampalState
from .pattern_separator import PatternSeparator
from .theta_oscillator import ThetaOscillator, OscillationPhase

__all__ = [
    'HippocampalBuffer',
    'EpisodicMemory',
    'HippocampalState',
    'PatternSeparator',
    'ThetaOscillator',
    'OscillationPhase'
]