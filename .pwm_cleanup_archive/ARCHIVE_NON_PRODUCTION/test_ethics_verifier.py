import importlib
EthicsVerifier = importlib.import_module("lukhas.identity.backend.verifold.identity.ethics_verifier").EthicsVerifier

def test_ethics_replay_approval():
    verifier = EthicsVerifier()
    verifier.consent_database = {"hash123": {"tier_level": 2, "purpose": "replay"}}
    result = verifier.verify_replay_ethics("hash123", {"tier_level": 2, "purpose": "replay"})
    assert result is True
