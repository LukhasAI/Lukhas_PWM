from trace.vdf import generate_vdf_proof, verify_vdf_proof


def test_vdf_cycle():
    data = b"test-data"
    proof = generate_vdf_proof(data, iterations=10)
    assert verify_vdf_proof(proof, data)
