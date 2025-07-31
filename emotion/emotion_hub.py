#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
██╗     ██╗   ██╗██╗  ██╗██╗  ██╗ █████╗ ███████╗
██║     ██║   ██║██║ ██╔╝██║  ██║██╔══██╗██╔════╝
██║     ██║   ██║█████╔╝ ███████║███████║███████╗
██║     ██║   ██║██╔═██╗ ██╔══██║██╔══██║╚════██║
███████╗╚██████╔╝██║  ██╗██║  ██║██║  ██║███████║
╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

Emotion Hub - Central coordination for emotion subsystem
=======================================================

Advanced emotional intelligence coordination, affect management,
and emotional state processing for human-AI empathetic interactions.

Agent 10 Advanced Systems Implementation
"""

import asyncio
from typing import Any, Dict, List, Optional

import structlog

# Import priority emotion components
from .affect_stagnation_detector import AffectStagnationDetector
from .dreamseed_upgrade import DreamSeedEmotionEngine
from .mood_regulator import MoodRegulator
from .recurring_emotion_tracker import RecurringEmotionTracker

logger = structlog.get_logger(__name__)


class EmotionHub:
    """Central hub for emotion system coordination"""

    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.emotional_state: Dict[str, Any] = {}
        self.initialized = False

        # Initialize core emotion services
        self._initialize_core_services()

    def _initialize_core_services(self) -> None:
        """Initialize core emotion services"""
        try:
            # Create a simple emotional memory if needed
            self.emotional_memory = self._create_emotional_memory()

            # Affect stagnation detection
            self.affect_detector = AffectStagnationDetector(self.emotional_memory)
            self.register_service("affect_detection", self.affect_detector)

            # Recurring emotion tracking
            self.emotion_tracker = RecurringEmotionTracker(self.emotional_memory)
            self.register_service("emotion_tracking", self.emotion_tracker)

            # Mood regulation system
            self.mood_regulator = MoodRegulator(self.emotional_memory)
            self.register_service("mood_regulation", self.mood_regulator)

            # DREAMSEED emotion engine
            self.dreamseed_engine = DreamSeedEmotionEngine(self.emotional_memory)
            self.register_service("dreamseed", self.dreamseed_engine)

            logger.info("emotion_core_services_initialized")

        except Exception as e:
            logger.error("emotion_service_initialization_failed", error=str(e))

    def _create_emotional_memory(self) -> Any:
        """Create a basic emotional memory instance"""
        try:
            # Try to create proper emotional memory
            from memory.emotional_memory import EmotionalMemory

            return EmotionalMemory()
        except ImportError:
            # Fallback to simple dict-based memory
            logger.warning("EmotionalMemory not available, using simple memory")
            return {"emotions": [], "patterns": {}}

    def register_service(self, name: str, service: Any) -> None:
        """Register an emotion service"""
        self.services[name] = service
        logger.debug("emotion_service_registered", service=name)

    async def initialize(self) -> None:
        """Initialize emotion hub"""
        if self.initialized:
            return

        try:
            # Initialize emotional state
            self.emotional_state = {
                "primary_emotion": "neutral",
                "intensity": 0.5,
                "context": {},
                "history": [],
            }

            # Initialize async emotion services
            await self._initialize_async_services()

            # Connect with other hubs
            await self._connect_external_integrations()

            self.initialized = True
            logger.info("emotion_hub_initialized")

        except Exception as e:
            logger.warning("emotion_hub_initialization_failed", error=str(e))

    async def _initialize_async_services(self) -> None:
        """Initialize asynchronous emotion services"""
        init_tasks = []

        for service in self.services.values():
            if hasattr(service, "initialize"):
                init_tasks.append(service.initialize())

        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)

    async def _connect_external_integrations(self) -> None:
        """Connect emotion hub with other system hubs"""
        try:
            # Connect with voice hub for emotional speech
            from voice.voice_hub import get_voice_hub

            voice_hub = get_voice_hub()
            if hasattr(voice_hub, "register_service"):
                voice_hub.register_service("emotion_engine", self)

            # Connect with API hub for emotion endpoints
            from api.api_hub import get_api_hub

            api_hub = get_api_hub()
            if hasattr(api_hub, "register_service"):
                api_hub.register_service("emotion_hub", self)

            logger.info("emotion_external_integrations_connected")

        except Exception as e:
            logger.debug("emotion_integration_connection_failed", error=str(e))

    async def process_emotional_input(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process emotional input and update emotional state"""
        try:
            results = {}

            # Analyze for affect stagnation
            if "affect_detection" in self.services:
                affect_result = await self._analyze_affect_stagnation(input_data)
                results["affect_analysis"] = affect_result

            # Track recurring emotions
            if "emotion_tracking" in self.services:
                tracking_result = await self._track_emotions(input_data)
                results["emotion_tracking"] = tracking_result

            # Mood regulation
            if "mood_regulation" in self.services:
                mood_result = await self._regulate_mood(input_data)
                results["mood_regulation"] = mood_result

            # Update emotional state
            self._update_emotional_state(results)

            results["current_state"] = self.emotional_state
            results["timestamp"] = asyncio.get_event_loop().time()

            return results

        except Exception as e:
            logger.error("emotional_processing_failed", error=str(e))
            return {"error": str(e)}

    async def _analyze_affect_stagnation(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze input for affect stagnation patterns"""
        try:
            detector = self.services["affect_detection"]
            if hasattr(detector, "detect"):
                return await detector.detect(input_data)
            else:
                return {"status": "detector_unavailable"}
        except Exception as e:
            return {"error": str(e)}

    async def _track_emotions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track emotional patterns and recurrence"""
        try:
            tracker = self.services["emotion_tracking"]
            if hasattr(tracker, "track"):
                return await tracker.track(input_data)
            else:
                return {"status": "tracker_unavailable"}
        except Exception as e:
            return {"error": str(e)}

    async def _regulate_mood(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mood regulation algorithms"""
        try:
            regulator = self.services["mood_regulation"]
            if hasattr(regulator, "regulate"):
                return await regulator.regulate(input_data)
            else:
                return {"status": "regulator_unavailable"}
        except Exception as e:
            return {"error": str(e)}

    def _update_emotional_state(self, analysis_results: Dict[str, Any]) -> None:
        """Update the current emotional state based on analysis"""
        try:
            # Extract emotional indicators from analysis
            if "emotion_tracking" in analysis_results:
                tracking = analysis_results["emotion_tracking"]
                if "primary_emotion" in tracking:
                    self.emotional_state["primary_emotion"] = tracking[
                        "primary_emotion"
                    ]

            if "mood_regulation" in analysis_results:
                regulation = analysis_results["mood_regulation"]
                if "intensity" in regulation:
                    self.emotional_state["intensity"] = regulation["intensity"]

            # Add to history
            self.emotional_state["history"].append(
                {
                    "timestamp": asyncio.get_event_loop().time(),
                    "analysis": analysis_results,
                }
            )

            # Keep history manageable
            if len(self.emotional_state["history"]) > 100:
                self.emotional_state["history"] = self.emotional_state["history"][-50:]

        except Exception as e:
            logger.error("emotional_state_update_failed", error=str(e))

    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered emotion service"""
        return self.services.get(name)

    def get_emotional_state(self) -> Dict[str, Any]:
        """Get the current emotional state"""
        return self.emotional_state.copy()

    def list_services(self) -> List[str]:
        """List all registered emotion services"""
        return list(self.services.keys())

    async def shutdown(self) -> None:
        """Gracefully shutdown emotion services"""
        shutdown_tasks = []

        for service in self.services.values():
            if hasattr(service, "shutdown"):
                shutdown_tasks.append(service.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        logger.info("emotion_hub_shutdown_complete")


# Singleton pattern for emotion hub
_emotion_hub_instance: Optional[EmotionHub] = None


def get_emotion_hub() -> EmotionHub:
    """Get the global emotion hub instance"""
    global _emotion_hub_instance
    if _emotion_hub_instance is None:
        _emotion_hub_instance = EmotionHub()
    return _emotion_hub_instance


# Export for Agent 10 integration
__all__ = ["EmotionHub", "get_emotion_hub"]
__all__ = ["EmotionHub", "get_emotion_hub"]
