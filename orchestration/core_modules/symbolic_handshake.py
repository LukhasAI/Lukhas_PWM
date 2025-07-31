# ΛTAG: orchestration, communication, symbolic_ai
# ΛLOCKED: true - Symbolic handshake protocol is frozen

"""
Symbolic Communication Protocol for Lukhas AGI System

This module defines the symbolic handshake patterns and communication signals
between the orchestrator and sub-agents (dream, intent, emotion, memory).
"""

from typing import Dict, Any, Optional, Callable, List
import logging
import time
from orchestration.signal_router import route_signal
from orchestration.signals import SymbolicSignal, SignalType

logger = logging.getLogger("SymbolicHandshake")


class SymbolicHandshake:
    """
    ΛTAG: orchestration, communication, handshake_protocol
    ΛLOCKED: true

    Manages symbolic handshake patterns between orchestrator and sub-agents.
    Ensures semantic consistency across modules.
    """

    def __init__(self):
        self.registered_modules: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[SymbolicSignal] = []
        self.emotion_state_history: List[Dict[str, Any]] = []
        self.handshake_callbacks: Dict[str, Callable] = {}
        self.brain_integrator = None  # Set via set_brain_integrator method

        # Define symbolic handshake patterns
        self.handshake_patterns = {
            "dream_engine": {
                "init_signal": SignalType.DREAM_INVOKE,
                "response_expected": "dream:state",
                "timeout": 5.0,
                "retry_count": 3,
            },
            "memory_fold": {
                "init_signal": SignalType.MEMORY_PULL,
                "response_expected": "memory:data",
                "timeout": 3.0,
                "retry_count": 5,
            },
            "intent_processor": {
                "init_signal": SignalType.INTENT_PROCESS,
                "response_expected": "intent:result",
                "timeout": 2.0,
                "retry_count": 2,
            },
            "emotion_sync": {
                "init_signal": SignalType.EMOTION_SYNC,
                "response_expected": "emotion:state",
                "timeout": 1.0,
                "retry_count": 1,
            },
        }

        logger.info("Symbolic handshake protocol initialized")

    def register_module(self, module_name: str, module_info: Dict[str, Any]):
        """
        ΛTAG: orchestration, module_registration
        ΛLOCKED: true

        Register a module with the symbolic handshake system.
        """
        self.registered_modules[module_name] = {
            "info": module_info,
            "last_handshake": None,
            "active": True,
            "symbolic_tags": module_info.get("symbolic_tags", []),
        }

        logger.info(
            f"Module registered: {module_name} with tags: {module_info.get('symbolic_tags', [])}"
        )

    def create_signal(
        self, signal_type: SignalType, source: str, target: str, payload: Dict[str, Any]
    ) -> SymbolicSignal:
        """
        ΛTAG: orchestration, signal_creation
        ΛLOCKED: true

        Create a new symbolic signal with proper validation.
        """
        signal = SymbolicSignal(
            signal_type=signal_type,
            source_module=source,
            target_module=target,
            payload=payload,
            timestamp=time.time(),
        )

        self.signal_history.append(signal)
        route_signal(signal)

        if signal.signal_type == SignalType.EMOTION_SYNC and "emotion_state" in payload:
            self.emotion_state_history.append(
                {
                    "timestamp": signal.timestamp,
                    "emotion_state": payload["emotion_state"],
                }
            )

        return signal

    def validate_handshake(self, module_name: str, signal: SymbolicSignal) -> bool:
        """
        ΛTAG: orchestration, handshake_validation
        ΛLOCKED: true

        Validate that a handshake follows the symbolic protocol.
        """
        if module_name not in self.handshake_patterns:
            logger.warning(f"No handshake pattern defined for {module_name}")
            return False

        pattern = self.handshake_patterns[module_name]

        # Check signal type matches expected pattern
        if signal.signal_type != pattern["init_signal"]:
            logger.error(
                f"Signal type mismatch for {module_name}: expected {pattern['init_signal']}, got {signal.signal_type}"
            )
            return False

        # Check for required symbolic elements
        if signal.signal_type == SignalType.LUKHAS_RECALL:
            if "memory_fold" not in signal.payload:
                logger.error("lukhas:recall signal missing memory_fold payload")
                return False

        logger.info(f"Handshake validated for {module_name}")
        return True

    def get_signal_history(
        self, module_name: Optional[str] = None
    ) -> List[SymbolicSignal]:
        """
        ΛTAG: orchestration, signal_tracing

        Get signal history for debugging symbolic interactions.
        """
        if module_name:
            return [
                s
                for s in self.signal_history
                if s.source_module == module_name or s.target_module == module_name
            ]
        return self.signal_history.copy()

    def get_emotion_state_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of emotion states.
        """
        return self.emotion_state_history.copy()
