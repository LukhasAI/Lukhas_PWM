"""
Embodiment System - Auto-generated entity exports
Generated from entity activation scan
Total entities: 3
"""

# Lazy imports to avoid circular dependencies
import importlib
import logging

logger = logging.getLogger(__name__)

# Entity registry for lazy loading
_ENTITY_REGISTRY = {
    "ProprioceptiveState": ("body_state", "ProprioceptiveState"),
}

# Function registry
_FUNCTION_REGISTRY = {
    "update_joint": ("body_state", "update_joint"),
    "to_dict": ("body_state", "to_dict"),
}


def __getattr__(name):
    """Lazy import entities on access"""
    # Check class registry first
    if name in _ENTITY_REGISTRY:
        module_path, attr_name = _ENTITY_REGISTRY[name]
        try:
            module = importlib.import_module(f".{module_path}", package=__package__)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to import {attr_name} from {module_path}: {e}")
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    # Check function registry
    if name in _FUNCTION_REGISTRY:
        module_path, attr_name = _FUNCTION_REGISTRY[name]
        try:
            module = importlib.import_module(f".{module_path}", package=__package__)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to import {attr_name} from {module_path}: {e}")
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    """List all available entities"""
    return list(_ENTITY_REGISTRY.keys()) + list(_FUNCTION_REGISTRY.keys())


# Export commonly used entities directly for better IDE support
__all__ = [
    "ProprioceptiveState",
]

# System metadata
__system__ = "embodiment"
__total_entities__ = 3
__classes__ = 1
__functions__ = 2
