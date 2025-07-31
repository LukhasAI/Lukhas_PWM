"""
Memory Interface Definitions
Standard interfaces for all memory types with colony compatibility
"""

from .memory_interface import (
    BaseMemoryInterface,
    MemoryType,
    MemoryState,
    MemoryMetadata,
    MemoryOperation,
    MemoryResponse,
    ValidationResult,
    MemoryInterfaceRegistry,
    memory_registry
)

from .episodic_interface import (
    EpisodicMemoryInterface,
    EpisodicContext,
    EpisodicMemoryContent
)

from .semantic_interface import (
    SemanticMemoryInterface,
    SemanticRelationType,
    SemanticRelation,
    ConceptNode,
    SemanticMemoryContent
)

__all__ = [
    # Base interfaces
    'BaseMemoryInterface',
    'MemoryType',
    'MemoryState',
    'MemoryMetadata',
    'MemoryOperation',
    'MemoryResponse',
    'ValidationResult',
    'MemoryInterfaceRegistry',
    'memory_registry',

    # Episodic interface
    'EpisodicMemoryInterface',
    'EpisodicContext',
    'EpisodicMemoryContent',

    # Semantic interface
    'SemanticMemoryInterface',
    'SemanticRelationType',
    'SemanticRelation',
    'ConceptNode',
    'SemanticMemoryContent'
]