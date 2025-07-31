import pytest
from reasoning.reasoning_hub import get_reasoning_hub
from reasoning.LBot_reasoning_processed import ΛBotAdvancedReasoningOrchestrator


def test_reasoning_hub_registration():
    hub = get_reasoning_hub()
    service = hub.get_service("advanced_orchestrator")
    assert isinstance(service, ΛBotAdvancedReasoningOrchestrator)
