#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LUKHAS (Logical Unified Knowledge Hyper-Adaptable System) - Core Monitoring Module

Copyright (c) 2025 LUKHAS AGI Development Team
All rights reserved.

This file is part of the LUKHAS AGI system, an enterprise artificial general
intelligence platform combining symbolic reasoning, emotional intelligence,
quantum integration, and bio-inspired architecture.

Mission: To illuminate complex reality through rigorous logic, adaptive
intelligence, and human-centred ethics—turning data into understanding,
understanding into foresight, and foresight into shared benefit for people
and planet.

Core monitoring module providing unified drift detection and system health
monitoring capabilities. Exports the UnifiedDriftMonitor system and related
components for comprehensive AGI system monitoring.

For more information, visit: https://lukhas.ai
"""

# ΛTRACE: Core monitoring module initialization
# ΛORIGIN_AGENT: Claude Code
# ΛTASK_ID: Task 12 - Drift Detection Integration

__version__ = "1.0.0"
__author__ = "LUKHAS Development Team"
__email__ = "dev@lukhas.ai"
__status__ = "Production"

from .drift_monitor import (
    UnifiedDriftMonitor,
    DriftType,
    InterventionType,
    UnifiedDriftScore,
    DriftAlert,
    create_drift_monitor
)

__all__ = [
    'UnifiedDriftMonitor',
    'DriftType',
    'InterventionType',
    'UnifiedDriftScore',
    'DriftAlert',
    'create_drift_monitor'
]