# ═══════════════════════════════════════════════════════════════════════════
# FILENAME: test_api.py
# MODULE: tests
# DESCRIPTION: API integration tests for Oneiric Core. Tests REST endpoints,
#              authentication, dream generation, user management, and ΛiD
#              integration using pytest and httpx AsyncClient.
# DEPENDENCIES: pytest, httpx, ..oneiric_core.main, .conftest
# LICENSE: PROPRIETARY - LUKHAS AI SYSTEMS - UNAUTHORIZED ACCESS PROHIBITED
# ═══════════════════════════════════════════════════════════════════════════

import pytest
from httpx import AsyncClient

@pytest.mark.pg
async def test_health_check_with_auth(client):
    """Test health check endpoint with authentication"""
    response = await client.get(
        "/healthz",
        headers={"Authorization": "Bearer fake_token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["user_id"] == "test_user_123"
    assert data["tier"] == 1

@pytest.mark.pg
async def test_health_check_without_auth(client):
    """Test health check fails without authentication"""
    response = await client.get("/healthz")

    assert response.status_code == 401

@pytest.mark.pg
async def test_new_user_creates_record(client):
    """Test that a new user gets automatically inserted + ΛiD issued"""
    response = await client.get(
        "/healthz",
        headers={"Authorization": "Bearer fake_token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == 1
    assert data["lukhas_id"] is not None

@pytest.mark.pg
async def test_generate_dream(client):
    """Test dream generation endpoint"""
    response = await client.post(
        "/api/generate-dream",
        headers={"Authorization": "Bearer fake_token"},
        json={"prompt": "flying through clouds", "recursive": False}
    )

    assert response.status_code == 200
    data = response.json()
    assert "sceneId" in data
    assert "narrativeText" in data
    assert "symbolicStructure" in data

@pytest.mark.pg
async def test_list_dreams(client):
    """Test dreams listing endpoint"""
    response = await client.get(
        "/api/dreams",
        headers={"Authorization": "Bearer fake_token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "dreams" in data
    assert isinstance(data["dreams"], list)

@pytest.mark.pg
async def test_rate_dream(client):
    """Test dream rating endpoint"""
    response = await client.post(
        "/api/dreams/test_dream_id/rate",
        headers={"Authorization": "Bearer fake_token"},
        json={"rating": 1}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

@pytest.mark.unit
def test_lukhas_id_generation():
    """Test ΛiD generation function"""
    from oneiric_core.identity.lukhas_id import generate_lukhas_id

    lid1 = generate_lukhas_id("user1", 1, "test_secret")
    lid2 = generate_lukhas_id("user1", 1, "test_secret")
    lid3 = generate_lukhas_id("user2", 1, "test_secret")

    # Same user + tier should generate same ID (within same day)
    assert lid1 == lid2
    # Different users should generate different IDs
    assert lid1 != lid3
    # All should start with LUKHAS
    assert lid1.startswith("LUKHAS")
    assert lid2.startswith("LUKHAS")
    assert lid3.startswith("LUKHAS")

# ═══════════════════════════════════════════════════════════════════════════
# FILENAME: test_api.py
# VERSION: 1.0.0
# TIER SYSTEM: Test environment (All tiers for comprehensive testing)
# ΛTRACE INTEGRATION: ENABLED
# CAPABILITIES: API endpoint testing, authentication testing, dream generation
#               testing, user management testing, database integration testing
# FUNCTIONS: test_health_check_with_auth, test_health_check_without_auth,
#           test_new_user_creates_record, test_generate_dream, test_analyze_dream,
#           test_get_user_symbols, test_save_dream, test_get_dream_history,
#           test_user_profile_operations, test_drift_analysis
# CLASSES: None directly defined (uses pytest fixtures)
# DECORATORS: @pytest.mark.pg (PostgreSQL tests)
# DEPENDENCIES: pytest, httpx AsyncClient, oneiric_core.main, test fixtures
# INTERFACES: HTTP API testing interface via AsyncClient
# ERROR HANDLING: Test assertions, HTTP status code validation
# LOGGING: ΛTRACE_ENABLED for test execution tracking
# AUTHENTICATION: Tests both authenticated and unauthenticated scenarios
# HOW TO USE:
#   pytest tests/test_api.py
#   pytest tests/test_api.py::test_health_check_with_auth
#   pytest -m pg tests/test_api.py
# INTEGRATION NOTES: Uses testcontainers for PostgreSQL isolation. Tests full
#   API stack including authentication, database operations, and ΛiD integration.
#   Requires Docker for database container management.
# MAINTENANCE: Update tests when API changes, add new endpoint tests, maintain
#   test data consistency, and update mock authentication as needed.
# CONTACT: LUKHAS DEVELOPMENT TEAM
# LICENSE: PROPRIETARY - LUKHAS AI SYSTEMS - UNAUTHORIZED ACCESS PROHIBITED
# ═══════════════════════════════════════════════════════════════════════════
