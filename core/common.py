"""
Common utilities and shared components for the core module
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

# Common enums
class ComponentStatus(Enum):
    """Status of a component"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

class MessageType(Enum):
    """Common message types"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    INFO = "info"

# Common utilities
def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def validate_component_id(component_id: str) -> bool:
    """Validate component ID format"""
    if not component_id:
        return False
    if not isinstance(component_id, str):
        return False
    if len(component_id) < 3:
        return False
    return True

# Common base classes
class BaseComponent:
    """Base class for all components"""

    def __init__(self, component_id: str, component_type: str = "generic"):
        self.component_id = component_id
        self.component_type = component_type
        self.status = ComponentStatus.INITIALIZING
        self.created_at = get_timestamp()
        self.metadata = {}

    def set_status(self, status: ComponentStatus):
        """Update component status"""
        self.status = status
        logger.info(f"Component {self.component_id} status changed to {status.value}")

    def add_metadata(self, key: str, value: Any):
        """Add metadata to component"""
        self.metadata[key] = value

class BaseMessage:
    """Base class for messages"""

    def __init__(self,
                 message_type: MessageType,
                 source: str,
                 target: Optional[str] = None,
                 payload: Optional[Dict[str, Any]] = None):
        self.message_type = message_type
        self.source = source
        self.target = target
        self.payload = payload or {}
        self.timestamp = get_timestamp()
        self.message_id = f"{source}_{datetime.now().timestamp()}"

# Common exceptions
class ComponentError(Exception):
    """Base exception for component errors"""
    pass

class ValidationError(ComponentError):
    """Validation error"""
    pass

class CommunicationError(ComponentError):
    """Communication error between components"""
    pass

# Common configuration
DEFAULT_CONFIG = {
    "timeout": 30,
    "retry_attempts": 3,
    "log_level": "INFO",
    "enable_metrics": True,
    "enable_tracing": True
}

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return DEFAULT_CONFIG.get(key, default)

# Export all public symbols
__all__ = [
    'ComponentStatus',
    'MessageType',
    'get_timestamp',
    'validate_component_id',
    'BaseComponent',
    'BaseMessage',
    'ComponentError',
    'ValidationError',
    'CommunicationError',
    'DEFAULT_CONFIG',
    'get_config',
    'logger'
]