from typing import Dict, List, Any, Callable, Optional
import asyncio

from core.colonies.base_colony import BaseColony
from bridge.message_bus import MessageBus


class InterColonyBridge:
    """Facilitates communication between different colony types."""

    def __init__(self):
        self.message_bus = MessageBus()
        self.colony_registry: Dict[str, BaseColony] = {}
        self.protocol_handlers: Dict[str, Callable] = {}
        self.routing_table: Dict[str, List[str]] = {}

    def register_colony(self, colony: BaseColony):
        self.colony_registry[colony.colony_id] = colony
        self.message_bus.subscribe(
            f"colony.{colony.colony_id}.*",
            lambda msg: asyncio.create_task(self._route_to_colony(colony.colony_id, msg)),
        )
        self._register_protocol_handlers(colony)

    def _register_protocol_handlers(self, colony: BaseColony):
        if "reasoning" in colony.capabilities:
            self.protocol_handlers[f"{colony.colony_id}.reason"] = lambda m: m

    async def broadcast_across_colonies(self, message: Dict[str, Any], source_colony_id: str, target_capabilities: Optional[List[str]] = None):
        tasks = []
        for cid, colony in self.colony_registry.items():
            if cid == source_colony_id:
                continue
            if target_capabilities and not any(cap in colony.capabilities for cap in target_capabilities):
                continue
            tasks.append(self.message_bus.publish(f"colony.{cid}.broadcast", message))
        await asyncio.gather(*tasks)

    async def _route_to_colony(self, colony_id: str, message: Dict[str, Any]):
        pass
