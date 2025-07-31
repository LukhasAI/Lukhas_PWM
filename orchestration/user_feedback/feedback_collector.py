"""
feedback_collector.py
----------------------

Collects symbolic and natural feedback from LUKHAS users following
introspective modules (dreams, reflections, memory recalls).

This module supports future symbolic fine-tuning through compliant channels.
All feedback is timestamped and stored securely.

# ΛTAG: feedback
"""

import json
import os
from datetime import datetime
from pathlib import Path

FEEDBACK_LOG_PATH = Path("logs/symbolic_feedback_log.json")
FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def collect_feedback(user_id: str, module_name: str, rating: int, comment: str) -> dict:
    """Capture symbolic feedback from a user"""
    feedback = {
        "user_id": user_id,
        "module": module_name,
        "rating": rating,
        "comment": comment,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return feedback


def store_feedback(feedback: dict) -> None:
    """Append feedback to the central feedback log"""
    existing = []
    if FEEDBACK_LOG_PATH.exists():
        with open(FEEDBACK_LOG_PATH, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(feedback)

    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    # Example usage (test):
    example = collect_feedback(
        "user_001",
        "dream_reflection_loop",
        4,
        "Felt emotionally resonant but a bit too recursive.",
    )
    store_feedback(example)
    print("✅ Feedback stored!")
    print(f"Feedback log path: {FEEDBACK_LOG_PATH}")
    print(f"Current feedback count: {len(json.load(open(FEEDBACK_LOG_PATH)))}")
