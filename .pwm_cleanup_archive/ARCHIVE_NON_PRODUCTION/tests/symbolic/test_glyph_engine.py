from symbolic.glyph_engine import evaluate_entropy, generate_glyph


def test_generate_glyph_returns_string():
    result = generate_glyph({"state": 1})
    assert isinstance(result, str)


def test_evaluate_entropy_basic():
    ent = evaluate_entropy([1, 1, 2])
    assert ent > 0
