import importlib
ZKNarrativeProofAdapter = importlib.import_module("lukhas.identity.backend.verifold.cryptography.zkNarrativeProof_adapter").ZKNarrativeProofAdapter

def test_zk_proof_roundtrip():
    adapter = ZKNarrativeProofAdapter()
    payload = adapter.create_experience_proof("abc123", "LUKHAS-X")
    assert adapter.verify_narrative_proof(payload) is True
