import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Simple witness chain anchoring mechanism
ANCHOR_LOG = Path(os.environ.get("LUKHAS_BLOCKCHAIN_ANCHOR", "anchor_log.json"))


def anchor_hash(data_hash: str) -> str:
    """Anchor a data hash to the witness chain (simulated blockchain)."""
    tx = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_hash": data_hash,
    }
    if ANCHOR_LOG.exists():
        anchors = json.loads(ANCHOR_LOG.read_text())
    else:
        anchors = []
    anchors.append(tx)
    ANCHOR_LOG.write_text(json.dumps(anchors, indent=2))
    return tx["timestamp"]
