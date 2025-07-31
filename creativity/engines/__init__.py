"""
🎨 LUKHAS CREATE - AGI-Powered Content Generation Engine

Core Design Philosophy: "Simplicity is the ultimate sophistication"
Advanced AGI Vision: "AGI should augment human creativity, not replace it"

This module provides the flagship content creation capabilities for the LUKHAS AGI system,
designed with future-proof architecture and elegant user experience.
"""

from .engine import LucasCreateEngine
from .templates import ContentTemplateManager
from .processors import (
    TextProcessor,
    ImageProcessor,
    VideoProcessor,
    AudioProcessor,
    MultiModalProcessor
)
from .adapters import CreativeAdapter
from .api import LucasCreateAPI

__version__ = "1.0.0"
__author__ = "LUKHAS AGI Team"

# Jobs-Level UX: One-line content creation
def create(prompt: str, type: str = "auto", **kwargs):
    """
    One-click content creation - the core vision realized.

    Args:
        prompt: Natural language description of desired content
        type: Content type ('text', 'image', 'video', 'audio', 'auto')
        **kwargs: Additional parameters for fine-tuning

    Returns:
        Generated content with metadata

    Example:
        >>> create("Write a haiku about AI consciousness")
        >>> create("Design a logo for sustainable tech startup", type="image")
        >>> create("Compose background music for meditation app", type="audio")
    """
    engine = LucasCreateEngine()
    return engine.create(prompt, type, **kwargs)

# Export main interface
__all__ = [
    "LucasCreateEngine",
    "ContentTemplateManager",
    "TextProcessor",
    "ImageProcessor",
    "VideoProcessor",
    "AudioProcessor",
    "MultiModalProcessor",
    "CreativeAdapter",
    "LucasCreateAPI",
    "create"
]
