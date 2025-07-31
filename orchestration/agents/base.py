"""Abstract base class for orchestration agents.

ΛTAG: orchestration_agent_interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from .types import AgentCapability, AgentContext, AgentResponse


class OrchestrationAgent(ABC):
    """Base class for orchestration agents."""

    @abstractmethod
    def get_agent_id(self) -> str:
        """Return a unique agent identifier."""
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the list of capabilities supported by this agent."""
        raise NotImplementedError

    @abstractmethod
    def process(self, context: AgentContext) -> AgentResponse:
        """Process a task in the given context."""
        raise NotImplementedError

    @abstractmethod
    def validate_context(self, context: AgentContext) -> bool:
        """Return True if the agent can handle the provided context."""
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, any]:
        """Return metadata describing the agent."""
        return {
            "id": self.get_agent_id(),
            "capabilities": [c.value for c in self.get_capabilities()],
            "version": "1.0.0",
        }
