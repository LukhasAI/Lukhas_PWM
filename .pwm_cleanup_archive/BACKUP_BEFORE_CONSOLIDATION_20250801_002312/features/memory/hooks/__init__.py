"""Memory Hook Interface

This module provides extensibility points for memory management,
allowing plugins to process memory items before storage and after recall.

ΛTAG: memory_hooks_interface
"""

from .base import (
    MemoryItem,
    MemoryHook,
    HookExecutionError
)
from .registry import (
    HookRegistry,
    HookPriority,
    HookRegistrationError
)

__all__ = [
    'MemoryItem',
    'MemoryHook',
    'HookExecutionError',
    'HookRegistry',
    'HookPriority',
    'HookRegistrationError'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'LUKHAS AGI Team'