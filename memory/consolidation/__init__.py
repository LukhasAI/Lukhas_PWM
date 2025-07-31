"""
Memory Consolidation System
Orchestrates transfer from hippocampus to neocortex during sleep cycles
"""

from .consolidation_orchestrator import (
    ConsolidationOrchestrator,
    SleepStage,
    ConsolidationMode
)
from .sleep_cycle_manager import (
    SleepCycleManager,
    CircadianPhase,
    SleepPressure
)
from .ripple_generator import (
    RippleGenerator,
    RippleType,
    ReplayDirection
)

__all__ = [
    'ConsolidationOrchestrator',
    'SleepStage',
    'ConsolidationMode',
    'SleepCycleManager',
    'CircadianPhase',
    'SleepPressure',
    'RippleGenerator',
    'RippleType',
    'ReplayDirection'
]