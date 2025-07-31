"""
📋 LUKHAS DAST - Dynamic Attention & Symbolic Tagging System (Enhanced)

Steve Jobs Design Philosophy: "Technology should anticipate your needs"
Sam Altman AGI Vision: "AI should understand context and provide proactive assistance"

This module provides the enhanced DAST (Dynamic Attention & Symbolic Tagging) system
with AI-powered task management, intelligent prioritization, and symbolic reasoning.
"""

from .engine import LucasDASTEngine
from .intelligence import (
    TaskIntelligence,
    PriorityOptimizer,
    ContextTracker,
    SymbolicReasoner,
    WorkflowAnalyzer
)
from .processors import (
    TaskProcessor,
    TagProcessor,
    AttentionProcessor,
    SolutionProcessor
)
from .adapters import DASTAdapter
from .api import LucasDASTAPI

__version__ = "2.0.0"
__author__ = "LUKHAS AGI Team"

# Jobs-Level UX: One-line task management
def track(task: str, context: dict = None, **kwargs):
    """
    Intelligent task tracking with AI-powered optimization.

    Args:
        task: Description of task or problem to track
        context: Optional context for better understanding
        **kwargs: Additional parameters

    Returns:
        Enhanced task with AI insights and recommendations

    Example:
        >>> track("Build user authentication system")
        >>> track("Debug performance issue", context={"severity": "high"})
        >>> track("Plan team meeting agenda", priority="medium")
    """
    engine = LucasDASTEngine()
    return engine.track(task, context, **kwargs)

# Altman AGI Vision: Proactive AI assistance
def optimize_workflow(workflow: str, constraints: dict = None, **kwargs):
    """
    AI-powered workflow optimization with predictive insights.

    Args:
        workflow: Description of workflow to optimize
        constraints: Optional constraints and requirements
        **kwargs: Optimization parameters

    Returns:
        Optimized workflow with AI recommendations
    """
    engine = LucasDASTEngine()
    return engine.optimize_workflow(workflow, constraints, **kwargs)

# Export main interface
__all__ = [
    "LucasDASTEngine",
    "TaskIntelligence",
    "PriorityOptimizer",
    "ContextTracker",
    "SymbolicReasoner",
    "WorkflowAnalyzer",
    "TaskProcessor",
    "TagProcessor",
    "AttentionProcessor",
    "SolutionProcessor",
    "DASTAdapter",
    "LucasDASTAPI",
    "track",
    "optimize_workflow"
]
