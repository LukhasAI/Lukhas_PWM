from fastapi import APIRouter, Depends, Request
from typing import Any, Dict

from pydantic import BaseModel
from core.event_bus import EventBus

router = APIRouter()


def get_event_bus(request: Request) -> EventBus:
    return request.app.state.event_bus


class CapabilityAnnouncement(BaseModel):
    agent_id: str
    capability: Dict[str, Any]


class TaskAnnouncement(BaseModel):
    agent_id: str
    task: Dict[str, Any]


@router.post("/announce-task")
async def announce_task(
    payload: TaskAnnouncement, bus: EventBus = Depends(get_event_bus)
) -> Dict[str, Any]:
    bus.announce_task(payload.model_dump())
    return {"status": "announced"}


@router.post("/announce-capability")
async def announce_capability(
    payload: CapabilityAnnouncement, bus: EventBus = Depends(get_event_bus)
) -> Dict[str, Any]:
    bus.announce_capability(payload.agent_id, payload.capability)
    return {"status": "registered"}
