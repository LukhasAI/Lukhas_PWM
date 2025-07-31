"""
══════════════════════════════════════════════════════════════════════════════════
║ 🧠 LUKHAS AI - CONFIGURATION
║ Configuration package for LUKHAS.
║ Copyright (c) 2025 LUKHAS AI. All rights reserved.
╠══════════════════════════════════════════════════════════════════════════════════
║ Module: __init__.py
║ Path: lukhas/config/__init__.py
║ Version: 1.0.0 | Created: 2025-07-25 | Modified: 2025-07-25
║ Authors: LUKHAS AI Core Team | Jules
╠══════════════════════════════════════════════════════════════════════════════════
║ DESCRIPTION
╠══════════════════════════════════════════════════════════════════════════════════
║ This module initializes the configuration for LUKHAS, with a fallback system.
╚══════════════════════════════════════════════════════════════════════════════════
"""
try:
    from .settings import settings, Settings
    from .validators import validate_config, validate_optional_config

    # Test that the settings can be accessed
    _ = settings.DATABASE_URL
    fallback_mode = False

except Exception as e:
    # Fallback to minimal configuration system
    from .fallback_settings import (
        get_fallback_settings,
        validate_fallback_config,
        FallbackSettings
    )

    settings = get_fallback_settings()
    Settings = FallbackSettings
    validate_config = lambda s: None  # No strict validation in fallback mode
    validate_optional_config = validate_fallback_config
    fallback_mode = True

    import logging
    logging.getLogger(__name__).error(f"Config system failed, using fallback: {e}")

__all__ = ["settings", "Settings", "validate_config", "validate_optional_config", "fallback_mode"]
"""
═══════════════════════════════════════════════════════════════════════════════
║ 📋 FOOTER - LUKHAS AI
╠══════════════════════════════════════════════════════════════════════════════
║ VALIDATION:
║   - Tests: N/A
║   - Coverage: N/A
║   - Linting: N/A
║
║ MONITORING:
║   - Metrics: N/A
║   - Logs: ERROR log on fallback activation
║   - Alerts: N/A
║
║ COMPLIANCE:
║   - Standards: N/A
║   - Ethics: N/A
║   - Safety: N/A
║
║ REFERENCES:
║   - Docs: docs/config/init.md
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
