from orchestration.init import bootstrap_seed_chain


def test_bootstrap_seed_chain_length():
    chain = bootstrap_seed_chain("seed", depth=4)
    assert len(chain) == 4
    assert chain[0] == "seed"
