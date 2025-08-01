import importlib
SymbolicAuditMode = importlib.import_module("lukhas.identity.backend.verifold.compliance.symbolic_audit_mode").SymbolicAuditMode

def test_secure_replay_with_audit_basic():
    audit = SymbolicAuditMode()
    audit.consent_checkpoints = {"cp123": {"lukhas_id": "L-001"}}
    audit.consent_database = {"mem789": {"tier_level": 2, "purpose": "replay"}}

    request = {
        "memory_hash": "mem789",
        "lukhas_id": "L-001",
        "checkpoint_id": "cp123",
        "consent_scope": {"tier_level": 2, "purpose": "replay"}
    }

    result = audit.secure_replay_with_audit(request)
    assert result["consent_valid"] is True
    assert result["ethics_ok"] is True
    assert "event_id" in result
