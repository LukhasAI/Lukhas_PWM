import importlib
GlyphStegoEncoder = importlib.import_module("lukhas.identity.backend.verifold.visual.glyph_stego_encoder").GlyphStegoEncoder

def test_validate_glyph_integrity_png():
    encoder = GlyphStegoEncoder()
    with open("tests/assets/sample_glyph.png", "rb") as f:
        data = f.read()
    assert encoder.validate_glyph_integrity(data) is True
