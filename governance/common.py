"""
Common base classes and utilities for governance module
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


class BaseLukhasModule(ABC):
    """Base class for all LUKHAS modules"""
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass


class BaseLucasConfig:
    """Base configuration class"""
    
    def __init__(self):
        self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value


class BaseLucasHealth:
    """Base health monitoring class"""
    
    def __init__(self):
        self.status = "healthy"
        self.metrics = {}
    
    def update_metric(self, name: str, value: Any):
        self.metrics[name] = value
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics
        }


def symbolic_vocabulary() -> Dict[str, str]:
    """Returns the symbolic vocabulary for GLYPH tokens"""
    return {
        "SAFE": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "INFO": "ℹ️",
        "SUCCESS": "✨",
        "PROCESSING": "⚙️"
    }


def symbolic_message(cls):
    """Decorator for symbolic message classes"""
    return cls


def ethical_validation(func):
    """Decorator for functions requiring ethical validation"""
    def wrapper(*args, **kwargs):
        # In production, this would validate through the ethics engine
        return func(*args, **kwargs)
    return wrapper