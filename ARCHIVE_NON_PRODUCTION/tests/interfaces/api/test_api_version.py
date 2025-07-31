from interfaces.api import API_VERSION, API_PREFIX


def test_api_version_constants():
    assert API_VERSION == "v1"
    assert API_PREFIX == f"/api/{API_VERSION}"
