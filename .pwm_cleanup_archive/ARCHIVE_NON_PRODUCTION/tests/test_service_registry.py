from hub.service_registry import register_all_providers, get_service


def test_quantum_bio_optimizer_provider():
    register_all_providers()
    optimizer = get_service('quantum_bio_optimizer')
    assert optimizer is not None
