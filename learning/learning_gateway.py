#!/usr/bin/env python3
"""
Learning Gateway - Dependency Firewall for Learning Services
This abstraction layer ensures core modules remain agnostic to learning implementation details.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import asyncio
from datetime import datetime


@dataclass
class LearningRequest:
    """Standard request format for learning operations"""
    agent_id: str
    operation: str
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LearningResponse:
    """Standard response format from learning operations"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LearningGatewayInterface(ABC):
    """
    Abstract interface for learning operations.
    Implementations can swap between RL, supervised, symbolic, or hybrid approaches.
    """

    @abstractmethod
    async def process_learning_request(self, request: LearningRequest) -> LearningResponse:
        """Process a learning request through the appropriate learning system"""
        pass

    @abstractmethod
    async def get_learning_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the current learning status for an agent"""
        pass

    @abstractmethod
    async def update_learning_parameters(self, agent_id: str, parameters: Dict[str, Any]) -> bool:
        """Update learning parameters for a specific agent"""
        pass

    @abstractmethod
    async def get_learning_metrics(self, agent_id: str, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve learning metrics and performance data"""
        pass


class LearningGateway(LearningGatewayInterface):
    """
    Concrete implementation of the learning gateway.
    This serves as the single entry point for all learning operations.
    """

    def __init__(self):
        self._service = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """Lazy initialization of the actual learning service"""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    # Lazy import to avoid circular dependencies
                    from learning.service import LearningService
                    self._service = LearningService()
                    self._initialized = True

    async def process_learning_request(self, request: LearningRequest) -> LearningResponse:
        """Process a learning request through the appropriate learning system"""
        await self._ensure_initialized()

        try:
            # Route to appropriate learning method based on operation
            if request.operation == "train":
                result = await self._service.train(
                    agent_id=request.agent_id,
                    data=request.data,
                    context=request.context
                )
            elif request.operation == "predict":
                result = await self._service.predict(
                    agent_id=request.agent_id,
                    data=request.data,
                    context=request.context
                )
            elif request.operation == "evaluate":
                result = await self._service.evaluate(
                    agent_id=request.agent_id,
                    data=request.data,
                    context=request.context
                )
            else:
                return LearningResponse(
                    success=False,
                    error=f"Unknown operation: {request.operation}"
                )

            return LearningResponse(
                success=True,
                result=result,
                metadata={"timestamp": datetime.now().isoformat()}
            )

        except Exception as e:
            return LearningResponse(
                success=False,
                error=str(e),
                metadata={"timestamp": datetime.now().isoformat()}
            )

    async def get_learning_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the current learning status for an agent"""
        await self._ensure_initialized()
        return await self._service.get_status(agent_id)

    async def update_learning_parameters(self, agent_id: str, parameters: Dict[str, Any]) -> bool:
        """Update learning parameters for a specific agent"""
        await self._ensure_initialized()
        return await self._service.update_parameters(agent_id, parameters)

    async def get_learning_metrics(self, agent_id: str, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve learning metrics and performance data"""
        await self._ensure_initialized()
        return await self._service.get_metrics(agent_id, metric_type)


# Singleton instance for the gateway
_gateway_instance = None


def get_learning_gateway() -> LearningGateway:
    """
    Get the singleton learning gateway instance.
    This ensures all modules use the same gateway.
    """
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LearningGateway()
    return _gateway_instance


# Export only the interface and gateway factory
__all__ = [
    'LearningGatewayInterface',
    'LearningRequest',
    'LearningResponse',
    'get_learning_gateway'
]