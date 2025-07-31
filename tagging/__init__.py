"""Tagging system module.

Provides tag specifications and basic interfaces for symbolic deduplication.
"""

from .tagging_system import (
    Tag,
    TagSchema,
    TagResolver,
    SimpleTagResolver,
    DeduplicationCache,
)

__all__ = [
    "Tag",
    "TagSchema",
    "TagResolver",
    "SimpleTagResolver",
    "DeduplicationCache",
]
