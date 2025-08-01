"""
Compatibility Module for Bio Orchestrator Migration

This module provides import redirections to support the gradual migration
from old import *  # TODO: Specify imports

Usage:
    # Old import (deprecated):
    from orchestration.brain.bio_symbolic.bio_orchestrator import BioOrchestrator

    # New import:
    from bio.core import BioOrchestrator

Created: 2025-07-26
"""

import warnings
import sys

# New module location
NEW_MODULE = 'lukhas.bio.systems.orchestration'

# Map old import paths to new location
IMPORT_REDIRECTS = {
    'lukhas.orchestration.brain.bio_symbolic.bio_orchestrator': NEW_MODULE,
    'lukhas.orchestration_src.brain.bio_symbolic.bio_orchestrator': NEW_MODULE,
    'lukhas.core.bio.bio_orchestrator': NEW_MODULE,
    'lukhas.bio.symbolic.bio_orchestrator': NEW_MODULE,
    'lukhas.voice.bio_core.oscillator.bio_orchestrator': NEW_MODULE,
    'lukhas.orchestration.specialized.bio_orchestrator': NEW_MODULE + '.oscillator_orchestrator'
}

def setup_import_redirects():
    """Set up import redirections for backward compatibility"""
    for old_path, new_path in IMPORT_REDIRECTS.items():
        warnings.warn(
            f"Import path '{old_path}' is deprecated. "
            f"Please use '{new_path}' instead.",
            DeprecationWarning,
            stacklevel=2
        )

# Note: Actual import redirection would require more complex sys.modules manipulation
# This is a documentation/reference file for the migration