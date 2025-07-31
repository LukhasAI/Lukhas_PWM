#!/usr/bin/env python3
"""
Contracts Layer
Defines interfaces for cross-module communication to break circular dependencies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


# Base data structures used across modules

@dataclass
class AgentContext:
    """Context information for an agent"""
    agent_id: str
    tier: int = 0
    permissions: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Standard result from processing operations"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


# Module interfaces

class IMemoryModule(ABC):
    """Interface for memory module operations"""

    @abstractmethod
    async def store(self, agent_id: str, data: Dict[str, Any], memory_type: str) -> ProcessingResult:
        """Store data in memory"""
        pass

    @abstractmethod
    async def retrieve(self, agent_id: str, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Retrieve memories based on query"""
        pass

    @abstractmethod
    async def consolidate(self, agent_id: str, strategy: str) -> ProcessingResult:
        """Consolidate memories using specified strategy"""
        pass


class ILearningModule(ABC):
    """Interface for learning module operations"""

    @abstractmethod
    async def train(self, agent_id: str, data: Dict[str, Any], config: Dict[str, Any]) -> ProcessingResult:
        """Train a model"""
        pass

    @abstractmethod
    async def predict(self, agent_id: str, input_data: Dict[str, Any]) -> ProcessingResult:
        """Make predictions"""
        pass

    @abstractmethod
    async def evaluate(self, agent_id: str, test_data: Dict[str, Any]) -> ProcessingResult:
        """Evaluate model performance"""
        pass


class IConsciousnessModule(ABC):
    """Interface for consciousness module operations"""

    @abstractmethod
    async def process_awareness(self, agent_id: str, stimulus: Dict[str, Any]) -> ProcessingResult:
        """Process awareness of stimulus"""
        pass

    @abstractmethod
    async def integrate_experience(self, agent_id: str, experience: Dict[str, Any]) -> ProcessingResult:
        """Integrate experience into consciousness"""
        pass

    @abstractmethod
    async def get_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current consciousness state"""
        pass


class IIdentityModule(ABC):
    """Interface for identity module operations"""

    @abstractmethod
    async def verify_access(self, agent_id: str, resource: str) -> bool:
        """Verify agent has access to resource"""
        pass

    @abstractmethod
    async def get_agent_tier(self, agent_id: str) -> int:
        """Get agent's security tier"""
        pass

    @abstractmethod
    async def log_audit(self, agent_id: str, action: str, data: Dict[str, Any]) -> None:
        """Log audit trail"""
        pass


class IOrchestrationModule(ABC):
    """Interface for orchestration module operations"""

    @abstractmethod
    async def coordinate_task(self, task: Dict[str, Any]) -> ProcessingResult:
        """Coordinate a complex task across modules"""
        pass

    @abstractmethod
    async def schedule_operation(self, operation: Dict[str, Any], when: datetime) -> str:
        """Schedule an operation for future execution"""
        pass

    @abstractmethod
    async def monitor_status(self, operation_id: str) -> Dict[str, Any]:
        """Monitor status of an operation"""
        pass


class IBioModule(ABC):
    """Interface for bio-inspired processing module"""

    @abstractmethod
    async def ground_symbol(self, symbol: Dict[str, Any], modality: str) -> ProcessingResult:
        """Ground abstract symbol in biological process"""
        pass

    @abstractmethod
    async def process_relation(self, state1: Dict[str, Any], relation: str, state2: Dict[str, Any]) -> ProcessingResult:
        """Process relationship between states"""
        pass

    @abstractmethod
    async def find_coherent_clusters(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find coherent clusters in states"""
        pass


class IQuantumModule(ABC):
    """Interface for quantum-inspired processing module"""

    @abstractmethod
    async def process_symbolic_state(self, state: Dict[str, Any]) -> ProcessingResult:
        """Process state using quantum-inspired methods"""
        pass

    @abstractmethod
    async def entangle_states(self, state1: Dict[str, Any], state2: Dict[str, Any], relation_type: str) -> ProcessingResult:
        """Create entanglement between states"""
        pass

    @abstractmethod
    async def find_correlations(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find quantum correlations between states"""
        pass


# Cross-module communication contracts

class IEventBus(ABC):
    """Interface for event-based communication"""

    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        pass

    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> str:
        """Subscribe to an event type"""
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        pass


# Factory interfaces for dependency injection

class IServiceFactory(ABC):
    """Interface for service factories"""

    @abstractmethod
    def create_service(self, service_type: str, config: Dict[str, Any]) -> Any:
        """Create a service instance"""
        pass

    @abstractmethod
    def get_service_info(self, service_type: str) -> Dict[str, Any]:
        """Get information about a service type"""
        pass


# Module communication patterns

class ModuleCommunicationPattern:
    """Defines allowed communication patterns between modules"""

    # Define layer hierarchy
    LAYERS = {
        "core": 0,
        "memory": 1,
        "identity": 1,
        "bio": 2,
        "quantum": 2,
        "consciousness": 3,
        "learning": 3,
        "orchestration": 4,
        "api": 5
    }

    @classmethod
    def is_allowed(cls, from_module: str, to_module: str) -> bool:
        """Check if communication from one module to another is allowed"""
        from_layer = cls.LAYERS.get(from_module, 999)
        to_layer = cls.LAYERS.get(to_module, 999)

        # Higher layers can communicate with lower layers
        # Same layer can communicate
        # Lower layers should not directly communicate with higher layers
        return from_layer >= to_layer


# Export all interfaces
__all__ = [
    # Data structures
    'AgentContext',
    'ProcessingResult',

    # Module interfaces
    'IMemoryModule',
    'ILearningModule',
    'IConsciousnessModule',
    'IIdentityModule',
    'IOrchestrationModule',
    'IBioModule',
    'IQuantumModule',

    # Communication interfaces
    'IEventBus',
    'IServiceFactory',

    # Patterns
    'ModuleCommunicationPattern'
]