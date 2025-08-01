# Test Environment Checklist

This checklist should be used to verify that the test environment is complete and ready to run the test suite.

## Dependencies

*   [x] `numpy`
*   [x] `structlog`
*   [x] `pyyaml`
*   [x] `testcontainers`
*   [x] `joblib`
*   [x] `streamlit`
*   [x] `python-dotenv`
*   [x] `openai`
*   [x] `websockets`
*   [x] `torch`
*   [x] `scikit-learn`
*   [x] `aiohttp`
*   [x] `matplotlib`
*   [x] `expecttest`
*   [x] `pytest`

## Configuration

*   [x] `tests/conftest.py` exists and is correctly configured.
*   [x] `.vscode/settings.json` is correctly configured.

## Test Scripts

*   [x] `tests/test_runner.py` exists and is correctly configured.

## Known Issues

*   [ ] `core/bio_core/dream/test_dream.py`: This test is failing due to a `ModuleNotFoundError`.
*   [ ] `core/interaction/test_symptom_reporter.py`: This test is failing due to a `ModuleNotFoundError`.
*   [ ] `core/test_logger.py`: This test is failing due to a `ModuleNotFoundError`.
*   [ ] `integration/simple_lukhas_integration_test.py`: This test is failing due to a `SyntaxError`.
*   [ ] `integration/test_integration_communication.py`: This test is failing due to a `SyntaxError`.
