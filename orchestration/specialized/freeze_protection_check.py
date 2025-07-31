# LUKHAS_TAG: freeze_protection, core_trace
import inspect
import logging

logger = logging.getLogger(__name__)

def is_locked(obj):
    """
    Checks if an object is locked.
    """
    if hasattr(obj, "__doc__") and obj.__doc__ and "ΛLOCKED: true" in obj.__doc__:
        return True
    return False

def freeze_protection_check(module):
    """
    Verifies that the ΛLOCKED tags are respected in a module.
    """
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if is_locked(obj):
                logger.info(f"Object {name} is locked.")
            else:
                logger.warning(f"Object {name} is not locked.")

if __name__ == "__main__":
    from lukhas_personality.brain_orchestrator import BrainOrchestrator agi_brain_orchestrator
    freeze_protection_check(orchestration.agi_brain_orchestrator)
