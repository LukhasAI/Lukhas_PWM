from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ProcessingMode(str, Enum):
    SYMBOLIC = "symbolic"
    CAUSAL = "causal"
    HYBRID = "hybrid"


class ProcessRequest(BaseModel):
    """Main processing request model."""

    input_text: str = Field(..., min_length=1, max_length=10000)
    mode: ProcessingMode = ProcessingMode.HYBRID
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

    @field_validator("input_text")
    def validate_input(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input text cannot be empty")
        return v


class SymbolicState(BaseModel):
    """Symbolic state representation."""

    glyphs: List[str]
    resonance: float = Field(..., ge=0.0, le=1.0)
    drift_score: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(..., ge=0.0, le=1.0)


class ProcessResponse(BaseModel):
    """Processing response model."""

    request_id: str
    timestamp: datetime
    result: Dict[str, Any]
    symbolic_state: Optional[SymbolicState] = None
    metadata: Dict[str, Any] = {}
    processing_time_ms: float


class HealthStatus(BaseModel):
    """System health status."""

    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    uptime_seconds: float
    components: Dict[str, bool]


class MetricsResponse(BaseModel):
    """System metrics."""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    drift_metrics: Dict[str, float]
    request_count: int
    average_response_time_ms: float


class CapabilityAnnouncement(BaseModel):
    """Agent capability announcement payload."""

    agent_id: str
    capability: Dict[str, Any]


class TaskAnnouncement(BaseModel):
    """Task announcement payload."""

    agent_id: str
    task: Dict[str, Any]
