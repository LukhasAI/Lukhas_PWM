from __future__ import annotations

"""Dream Sandbox for divergent recursive dream simulations."""

from typing import List, Dict, Any, Callable
import uuid
import json
import sys
import types
from dataclasses import dataclass, field

# Provide lightweight stubs to satisfy heavy dependencies during testing
for _mod in ("openai", "anthropic", "aiohttp", "speech_recognition", "pydub"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.SimpleNamespace()


@dataclass
class MediaInput:
    type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIGeneratedDream:
    narrative: str
    dream_themes: List[str]


@dataclass
class SimpleInterpretation:
    main_themes: List[str]
    emotional_tone: str
    symbols: List[Dict[str, str]]
    personal_insight: str
    guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "main_themes": self.main_themes,
            "emotional_tone": self.emotional_tone,
            "symbols": self.symbols,
            "personal_insight": self.personal_insight,
            "guidance": self.guidance,
        }


class DreamInterpreter:
    """Simplified dream interpreter used for sandbox testing."""

    def __init__(self):
        self.dream_text = ""

    def set_dream_text(self, text: str) -> None:
        self.dream_text = text

    def interpret_dream_with_ai(self, ai_complete_function) -> SimpleInterpretation | None:
        if not self.dream_text:
            return None
        data = json.loads(ai_complete_function(self.dream_text))
        return SimpleInterpretation(
            main_themes=data.get("mainThemes", []),
            emotional_tone=data.get("emotionalTone", ""),
            symbols=data.get("symbols", []),
            personal_insight=data.get("personalInsight", ""),
            guidance=data.get("guidance", ""),
        )


def mock_generate_ai_dream(media_inputs: List[MediaInput]) -> AIGeneratedDream:
    text_parts = [m.content for m in media_inputs if m.type == "text"]
    narrative = " ".join(text_parts)
    return AIGeneratedDream(narrative=narrative, dream_themes=["mock"])


class SimpleRLCycle:
    """Minimal Q-learning style loop for dream mutation."""

    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9):
        self.q_table: Dict[str, float] = {}
        self.learning_rate = learning_rate
        self.gamma = gamma

    @staticmethod
    def _drift_score(d1: Dict[str, Any], d2: Dict[str, Any]) -> float:
        a = d1.get("narrative", "")
        b = d2.get("narrative", "")
        return min(abs(len(a) - len(b)) / 100.0, 1.0)

    def mutate(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        mutated = dict(dream)
        mutated.setdefault("symbols", [])
        mutated["symbols"].append("ETHICAL_CHECK")
        mutated["dream_id"] = uuid.uuid4().hex
        return mutated

    def step(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        mutated = self.mutate(dream)
        drift = self._drift_score(dream, mutated)
        state = dream["dream_id"]
        next_state = mutated["dream_id"]
        future_q = self.q_table.get(next_state, 0.0)
        current_q = self.q_table.get(state, 0.0)
        updated_q = current_q + self.learning_rate * (1.0 - drift + self.gamma * future_q - current_q)
        self.q_table[state] = updated_q
        mutated["driftScore"] = drift
        return mutated


# ΛTAG: dream_sandbox
class DreamSandbox:
    """Run divergent recursive dreams using an RL-driven loop."""

    def __init__(self, iterations: int = 3, ai_complete: Callable[[str], str] | None = None):
        self.iterations = iterations
        self.interpreter = DreamInterpreter()
        self.rl_cycle = SimpleRLCycle()
        self.ai_complete = ai_complete or self._default_ai
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _default_ai(prompt: str) -> str:
        """Fallback AI completion returning a minimal interpretation."""
        return json.dumps({
            "mainThemes": ["recursion"],
            "emotionalTone": "curious",
            "symbols": [{"symbol": "loop", "meaning": "repetition"}],
            "personalInsight": "exploring recursion",
            "guidance": "continue iterating",
        })

    def _generate_dream(self, text: str) -> Dict[str, Any]:
        """Generate a simple dream state from text using mock AI."""
        inputs = [MediaInput(type="text", content=text)]
        ai_dream = mock_generate_ai_dream(inputs)
        return {
            "dream_id": uuid.uuid4().hex,
            "narrative": ai_dream.narrative,
            "symbols": ai_dream.dream_themes,
        }

    def run_recursive(self, seed_text: str) -> List[Dict[str, Any]]:
        """Run the sandbox for a number of iterations."""
        current = self._generate_dream(seed_text)
        self.history.append(current)

        for _ in range(self.iterations):
            self.interpreter.set_dream_text(current["narrative"])
            interp = self.interpreter.interpret_dream_with_ai(self.ai_complete)
            if interp:
                current["interpretation"] = interp.to_dict()
            mutated = self.rl_cycle.step(current)
            self.history.append(mutated)
            current = mutated
        return self.history
