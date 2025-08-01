from ethics.fallback import FallbackEthicsLayer


def test_symbol_allowed():
    layer = FallbackEthicsLayer(banned_symbols={"☠"})
    assert layer.is_allowed("α")
    assert not layer.is_allowed("☠")
