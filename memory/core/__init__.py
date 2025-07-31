"""
Core Module
Auto-generated module initialization file
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .unified_memory_orchestrator import UnifiedMemoryOrchestrator
    logger.debug("Imported UnifiedMemoryOrchestrator from .unified_memory_orchestrator")
except ImportError as e:
    logger.warning(f"Could not import UnifiedMemoryOrchestrator: {e}")
    UnifiedMemoryOrchestrator = None

try:
    from .colony_memory_validator import ColonyMemoryValidator
    logger.debug("Imported ColonyMemoryValidator from .colony_memory_validator")
except ImportError as e:
    logger.warning(f"Could not import ColonyMemoryValidator: {e}")
    ColonyMemoryValidator = None

try:
    from .hybrid_memory_fold import HybridMemoryFold, create_hybrid_memory_fold
    logger.debug("Imported HybridMemoryFold and create_hybrid_memory_fold from .hybrid_memory_fold")
except ImportError as e:
    logger.warning(f"Could not import HybridMemoryFold and create_hybrid_memory_fold: {e}")
    HybridMemoryFold = None
    create_hybrid_memory_fold = None

__all__ = [
    'UnifiedMemoryOrchestrator',
    'ColonyMemoryValidator',
    'HybridMemoryFold',
    'create_hybrid_memory_fold',
]

# Filter out None values from __all__ if imports failed
__all__ = [name for name in __all__ if globals().get(name) is not None]

logger.info(f"core module initialized. Available components: {__all__}")
