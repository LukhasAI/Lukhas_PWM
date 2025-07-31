from symbolic.symbolic_glyph_hash import (
    compute_glyph_hash,
    entropy_delta,
)


def test_compute_glyph_hash_stable():
    h1 = compute_glyph_hash("α")
    h2 = compute_glyph_hash("α")
    assert h1 == h2


def test_entropy_delta():
    h1 = compute_glyph_hash("α")
    h2 = compute_glyph_hash("β")
    delta = entropy_delta(h1, h2)
    assert 0.0 <= delta <= 1.0
    assert delta > 0
