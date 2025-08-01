import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ΛTAG: tag_debug
def trace_tag_flow(dream_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Log tag flow with drift and convergence metrics."""
    tags: List[str] = dream_payload.get("tags", [])
    metrics: Dict[str, float] = dream_payload.get("metrics", {})
    drift = metrics.get("driftScore", 0.0)
    convergence = metrics.get("convergence", 0.0)
    logger.info(
        "TAG_FLOW %s | drift=%.2f | convergence=%.2f",
        tags,
        drift,
        convergence,
    )
    return {"tags": tags, "driftScore": drift, "convergence": convergence}
