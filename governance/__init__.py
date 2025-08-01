"""
🛡️ LUKHAS PWM Governance Module
===============================

Pack-What-Matters governance system providing ethical oversight 
for workspace management operations.

Key Components:
- Guardian System v1.0.0 (core.py)
- PWM-specific ethical principles
- Workspace safety protocols
- File operation governance

Purpose: Intelligent workspace management with ethical guardrails.
"""

from .core import (
    LucasGovernanceModule,
    EthicalDecision, 
    RemediatorAlert,
    GovernanceAction,
    EthicalSeverity
)

__version__ = "1.0.0"
__all__ = [
    "LucasGovernanceModule",
    "EthicalDecision", 
    "RemediatorAlert",
    "GovernanceAction",
    "EthicalSeverity"
]
