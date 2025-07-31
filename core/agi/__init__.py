"""
LUKHAS AGI Core Systems
Advanced AGI capabilities for autonomous operation
"""

from .self_improvement import (
    SelfImprovementEngine,
    AGIGoalAlignment,
    ImprovementDomain,
    ImprovementGoal
)

from .consciousness_stream import (
    ConsciousnessStreamServer,
    ConsciousnessStreamClient,
    ConsciousnessFrame,
    StreamType
)

from .autonomous_learning import (
    AutonomousLearningPipeline,
    LearningStrategy,
    KnowledgeType,
    LearningGoal
)

__all__ = [
    # Self-improvement
    'SelfImprovementEngine',
    'AGIGoalAlignment',
    'ImprovementDomain',
    'ImprovementGoal',
    
    # Consciousness streaming
    'ConsciousnessStreamServer',
    'ConsciousnessStreamClient',
    'ConsciousnessFrame',
    'StreamType',
    
    # Autonomous learning
    'AutonomousLearningPipeline',
    'LearningStrategy',
    'KnowledgeType',
    'LearningGoal'
]