"""
═══════════════════════════════════════════════════════════════════════════════
║ 🧠 LUKHAS AI - MEMORY CORE SYMBOLIC DRIFT TRACKER ALIAS
║ Provides compatibility alias to the core symbolic drift tracker implementation
╚═══════════════════════════════════════════════════════════════════════════════
"""
# ΛTAGS: ΛALIAS, ΛMEMORY_CORE, ΛSYMBOLIC_DRIFT
import structlog
from core.symbolic.drift.symbolic_drift_tracker import SymbolicDriftTracker

logger = structlog.get_logger(__name__)
logger.debug("memory.core_memory.symbolic_drift_tracker alias loaded")

__all__ = ["SymbolicDriftTracker"]
