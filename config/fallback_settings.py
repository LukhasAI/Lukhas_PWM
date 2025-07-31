"""
══════════════════════════════════════════════════════════════════════════════════
║ 🧠 LUKHAS AI - CONFIGURATION
║ Fallback configuration system for LUKHAS.
║ Copyright (c) 2025 LUKHAS AI. All rights reserved.
╠══════════════════════════════════════════════════════════════════════════════════
║ Module: fallback_settings.py
║ Path: lukhas/config/fallback_settings.py
║ Version: 1.0.0 | Created: 2025-07-25 | Modified: 2025-07-25
║ Authors: LUKHAS AI Core Team | Jules
╠══════════════════════════════════════════════════════════════════════════════════
║ DESCRIPTION
╠══════════════════════════════════════════════════════════════════════════════════
║ This module provides a minimal, dependency-free configuration system that can
║ be used if the main pydantic-based config system fails.
╚══════════════════════════════════════════════════════════════════════════════════
"""
"""Fallback configuration system for LUKHAS when primary config fails.

This provides a minimal, dependency-free configuration system that can
be used if the main pydantic-based config system fails.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FallbackSettings:
    """Minimal fallback configuration system."""

    def __init__(self):
        """Initialize with safe defaults."""
        self.OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
        self.DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///lukhas_fallback.db')
        self.REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'WARNING')  # More conservative
        self.DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'

        # Fallback mode indicator
        self.FALLBACK_MODE: bool = True

        logger.warning("Using fallback configuration system - some features may be limited")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'DATABASE_URL': self.DATABASE_URL,
            'REDIS_URL': self.REDIS_URL,
            'LOG_LEVEL': self.LOG_LEVEL,
            'DEBUG': self.DEBUG,
            'FALLBACK_MODE': self.FALLBACK_MODE
        }


def get_fallback_settings() -> FallbackSettings:
    """Get fallback settings instance."""
    return FallbackSettings()


def validate_fallback_config(settings: FallbackSettings) -> Dict[str, Any]:
    """Validate fallback configuration."""
    status = {
        'openai_configured': settings.OPENAI_API_KEY is not None,
        'database_configured': 'sqlite' not in settings.DATABASE_URL.lower(),
        'redis_configured': 'localhost' not in settings.REDIS_URL,
        'debug_mode': settings.DEBUG,
        'log_level': settings.LOG_LEVEL,
        'fallback_mode': True
    }
    return status
"""
═══════════════════════════════════════════════════════════════════════════════
║ 📋 FOOTER - LUKHAS AI
╠══════════════════════════════════════════════════════════════════════════════
║ VALIDATION:
║   - Tests: lukhas/tests/config/test_fallback_settings.py
║   - Coverage: N/A
║   - Linting: pylint 10/10
║
║ MONITORING:
║   - Metrics: N/A
║   - Logs: WARNING logs on fallback activation
║   - Alerts: N/A
║
║ COMPLIANCE:
║   - Standards: N/A
║   - Ethics: N/A
║   - Safety: N/A
║
║ REFERENCES:
║   - Docs: docs/config/fallback.md
║   - Issues: github.com/lukhas-ai/lukhas/issues?label=config
║   - Wiki: N/A
║
║ COPYRIGHT & LICENSE:
║   Copyright (c) 2025 LUKHAS AI. All rights reserved.
║   Licensed under the LUKHAS AI Proprietary License.
║   Unauthorized use, reproduction, or distribution is prohibited.
║
║ DISCLAIMER:
║   This module is part of the LUKHAS AGI system. Use only as intended
║   within the system architecture. Modifications may affect system
║   stability and require approval from the LUKHAS Architecture Board.
╚═══════════════════════════════════════════════════════════════════════════
"""