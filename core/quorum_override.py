"""Quorum-based override system for sensitive actions."""
from typing import List
import structlog

log = structlog.get_logger(__name__)

class QuorumOverride:
    """Simple multi-agent consensus check."""
    def __init__(self, required: int = 2):
        self.required = required

    def request_access(self, approvers: List[str]) -> bool:
        """Return True if approvers reach required quorum."""
        approved = len(set(approvers)) >= self.required
        log.info("Quorum check", approvers=approvers, approved=approved)
        return approved
