# ═══════════════════════════════════════════════════════════════════════════
# FILENAME: emotion/recurring_emotion_tracker.py
# MODULE: emotion.recurring_emotion_tracker
# DESCRIPTION: Tracks recurring emotional states and generates symbolic prompts.
# DEPENDENCIES: datetime, numpy
# LICENSE: PROPRIETARY - LUKHAS AI SYSTEMS - DO NOT DISTRIBUTE
# ═══════════════════════════════════════════════════════════════════════════
# {AIM}{emotion}
# {ΛDRIFT}
# {ΛTRACE}

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from emotion.affect_stagnation_detector import AffectStagnationDetector
from memory.emotional import EmotionalMemory, EmotionVector

log = logging.getLogger(__name__)


class RecurringEmotionTracker:
    """
    Compares current affect vectors to historic mood states and
    triggers symbolic prompts on emotional echo or stagnation.
    """

    def __init__(
        self,
        emotional_memory: EmotionalMemory,
        bio_oscillator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.emotional_memory = emotional_memory
        self.bio_oscillator = bio_oscillator
        self.config = config or {}
        self.history_window = self.config.get("history_window_days", 7)
        self.recurrence_threshold = self.config.get("recurrence_threshold", 3)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.9)
        self.stagnation_detector = AffectStagnationDetector(
            self.emotional_memory, config
        )

    def check_for_recurrence(self) -> Optional[Dict[str, Any]]:
        """
        Checks for recurring emotional states and stagnation.

        Returns:
            Optional[Dict[str, Any]]: A symbolic prompt if recurrence or stagnation is detected.
        """
        # ΛTRACE: Checking for emotion recurrence and stagnation.
        log.info("Checking for emotion recurrence and stagnation.")

        stagnation_detected = self.stagnation_detector.check_for_stagnation()
        if stagnation_detected:
            return stagnation_detected

        emotional_history = self.emotional_memory.get_emotional_history(
            hours_ago=self.history_window * 24
        )
        if not emotional_history:
            return None

        current_emotion = self.emotional_memory.get_current_emotional_state()[
            "current_emotion_vector"
        ]
        recurrence_detected = self._check_recurrence(current_emotion, emotional_history)
        if recurrence_detected:
            return recurrence_detected

        return None

    def _check_recurrence(
        self, current_emotion: Dict[str, Any], emotional_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Checks for recurring emotional patterns.
        """
        similar_emotions = []
        current_vector = np.array(list(current_emotion["dimensions"].values()))

        for entry in emotional_history:
            history_vector = np.array(list(entry["emotion_vec"]["dimensions"].values()))
            if (
                np.linalg.norm(current_vector) > 0
                and np.linalg.norm(history_vector) > 0
            ):
                similarity = np.dot(current_vector, history_vector) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(history_vector)
                )
                if similarity > self.similarity_threshold:
                    similar_emotions.append(entry)

        if len(similar_emotions) >= self.recurrence_threshold:
            primary_emotion = current_emotion["primary_emotion"]
            log.warning(f"Recurring emotion detected: {primary_emotion}")

            # Check if this is stagnation (same emotion for a long time) vs recurrence (repeating pattern)
            is_stagnation = (
                len(similar_emotions) >= 10
            )  # More occurrences indicate stagnation
            symbol = "⏳" if is_stagnation else "🔄"
            trigger_type = (
                "Emotional stagnation" if is_stagnation else "Recurring emotion"
            )

            # Find the dream associated with the first occurrence
            origin_dream = self._find_origin_dream(similar_emotions[0])

            return {
                "recurrence": True,
                "symbol": symbol,
                "origin_dream": origin_dream,
                "trigger": f"{trigger_type} detected: {primary_emotion} appeared {len(similar_emotions)} times recently.",
            }
        return None

    def _find_origin_dream(self, emotion_entry: Dict[str, Any]) -> Optional[str]:
        """
        Finds the dream that originated a specific emotional state.
        This is a conceptual implementation and needs to be connected to the actual dream log.
        """
        # This is a placeholder. In a real implementation, we would need to
        # search the dream log for a dream that occurred around the same time
        # as the emotion_entry and has a similar emotional context.
        return "Dream log search not yet implemented."

    # LUKHAS_TAG: symbolic_affect_convergence
    def update_bio_oscillator(self):
        """
        Updates the BioOscillator with the current emotional state.
        """
        if not self.bio_oscillator:
            return

        current_emotion_vector = self.emotional_memory.current_emotion

        # Map emotional state to oscillator parameters
        # This is a conceptual mapping and can be refined
        frequency = 10 + (
            current_emotion_vector.arousal * 20
        )  # Beta range for active processing

        if hasattr(self.bio_oscillator, "adjust_frequency"):
            self.bio_oscillator.adjust_frequency(frequency)
            self.bio_oscillator.adjust_frequency(frequency)
