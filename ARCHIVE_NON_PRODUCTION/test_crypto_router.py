import importlib
CryptoRouter = importlib.import_module("lukhas.identity.backend.verifold.cryptography.crypto_router").CryptoRouter
SecurityTier = importlib.import_module("lukhas.identity.backend.verifold.cryptography.crypto_router").SecurityTier

def test_select_signature_scheme_default():
    router = CryptoRouter()
    sig = router.select_signature_scheme(SecurityTier.TIER_2)
    assert sig in {"falcon", "sphincs+"}
