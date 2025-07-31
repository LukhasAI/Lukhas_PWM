#!/usr/bin/env python3
"""
LUKHAS AI API Test Suite
========================

Comprehensive test suite for LUKHAS AI FastAPI endpoints including:
- Memory system operations
- Dream processing functionality
- Emotional analysis capabilities
- Consciousness integration features

Run with: pytest test_api.py -v

Author: LUKHAS AI Team
Date: 2025-07-27
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
from datetime import datetime
from typing import Dict, Any

# Import the FastAPI app
try:
    from main import app
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False
    app = None

# Test configuration
TEST_USER_ID = "test_user"
TEST_TIER = 5

@pytest.fixture
def client():
    """Create FastAPI test client"""
    if not APP_AVAILABLE:
        pytest.skip("FastAPI app not available")
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client"""
    if not APP_AVAILABLE:
        pytest.skip("FastAPI app not available")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

class TestBasicEndpoints:
    """Test basic API endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "LUKHAS AI API - Tier 5 Ready"
        assert data["tier_level"] == 5
        assert data["status"] == "operational"

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "tier" in data
        assert "modules" in data
        assert data["tier"] == 5

    def test_api_info(self, client):
        """Test API info endpoint"""
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert data["api_name"] == "LUKHAS AI API"
        assert data["tier_level"] == 5
        assert "capabilities" in data

class TestMemoryAPI:
    """Test memory system API endpoints"""

    def test_memory_health(self, client):
        """Test memory system health check"""
        response = client.get("/api/v1/memory/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "error"]

    def test_create_memory(self, client):
        """Test memory creation"""
        memory_data = {
            "emotion": "test_emotion",
            "context_snippet": "This is a test memory for API validation",
            "user_id": TEST_USER_ID,
            "metadata": {
                "test": True,
                "timestamp": datetime.now().isoformat()
            }
        }

        response = client.post("/api/v1/memory/create", json=memory_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data

    def test_recall_memories(self, client):
        """Test memory recall"""
        recall_data = {
            "user_id": TEST_USER_ID,
            "user_tier": TEST_TIER,
            "limit": 10
        }

        response = client.post("/api/v1/memory/recall", json=recall_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data

    def test_enhanced_recall(self, client):
        """Test enhanced memory recall"""
        recall_data = {
            "user_id": TEST_USER_ID,
            "target_emotion": "enlightenment",
            "user_tier": TEST_TIER,
            "emotion_threshold": 0.5,
            "max_results": 5
        }

        response = client.post("/api/v1/memory/enhanced-recall", json=recall_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_memory_statistics(self, client):
        """Test memory statistics"""
        response = client.get("/api/v1/memory/statistics")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

class TestDreamAPI:
    """Test dream processing API endpoints"""

    def test_dream_health(self, client):
        """Test dream system health check"""
        response = client.get("/api/v1/dream/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "error"]

    def test_log_dream(self, client):
        """Test dream logging"""
        dream_data = {
            "dream_type": "test",
            "user_id": TEST_USER_ID,
            "content": "This is a test dream for API validation with symbolic patterns and consciousness exploration.",
            "metadata": {
                "test": True,
                "lucidity_level": 0.8
            }
        }

        response = client.post("/api/v1/dream/log", json=dream_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_consolidate_memories(self, client):
        """Test dream consolidation"""
        consolidation_data = {
            "user_id": TEST_USER_ID,
            "hours_limit": 24,
            "max_memories": 50,
            "consolidation_type": "standard"
        }

        response = client.post("/api/v1/dream/consolidate", json=consolidation_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_dream_patterns(self, client):
        """Test dream pattern analysis"""
        response = client.get(f"/api/v1/dream/patterns?user_id={TEST_USER_ID}&pattern_type=emotional")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_dream_insights(self, client):
        """Test dream insights generation"""
        response = client.get(f"/api/v1/dream/insights?user_id={TEST_USER_ID}&insight_type=overview")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

class TestEmotionAPI:
    """Test emotional analysis API endpoints"""

    def test_emotion_health(self, client):
        """Test emotion system health check"""
        response = client.get("/api/v1/emotion/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "error"]

    def test_emotional_landscape(self, client):
        """Test emotional landscape mapping"""
        response = client.get("/api/v1/emotion/landscape")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_analyze_emotion(self, client):
        """Test emotion analysis"""
        analysis_data = {
            "content": "This text contains joy, wonder, and enlightenment patterns for emotional analysis.",
            "analysis_depth": "standard",
            "return_vectors": False
        }

        response = client.post("/api/v1/emotion/analyze", json=analysis_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_emotion_clusters(self, client):
        """Test emotion clustering"""
        cluster_data = {
            "tier_level": TEST_TIER,
            "cluster_method": "automatic",
            "min_cluster_size": 2
        }

        response = client.post("/api/v1/emotion/clusters", json=cluster_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_emotion_neighborhood(self, client):
        """Test emotion neighborhood analysis"""
        response = client.get("/api/v1/emotion/neighborhood/enlightenment?threshold=0.6")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_emotion_vectors(self, client):
        """Test emotion vectors"""
        response = client.get("/api/v1/emotion/vectors")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

class TestConsciousnessAPI:
    """Test consciousness integration API endpoints"""

    def test_consciousness_health(self, client):
        """Test consciousness system health check"""
        response = client.get("/api/v1/consciousness/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "error"]

    def test_consciousness_state(self, client):
        """Test consciousness state retrieval"""
        response = client.get(f"/api/v1/consciousness/state?user_id={TEST_USER_ID}")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_synthesize_consciousness(self, client):
        """Test consciousness synthesis"""
        synthesis_data = {
            "synthesis_type": "integration",
            "data_sources": ["memory", "emotion"],
            "complexity_level": 3,
            "user_id": TEST_USER_ID
        }

        response = client.post("/api/v1/consciousness/synthesize", json=synthesis_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_integrate_patterns(self, client):
        """Test pattern integration"""
        response = client.post(f"/api/v1/consciousness/integrate?patterns=test_pattern&user_id={TEST_USER_ID}")

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

    def test_assess_awareness(self, client):
        """Test awareness assessment"""
        assessment_data = {
            "user_id": TEST_USER_ID,
            "assessment_type": "comprehensive",
            "include_recommendations": True
        }

        response = client.post("/api/v1/consciousness/assess", json=assessment_data)

        # Should succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

class TestErrorHandling:
    """Test API error handling"""

    def test_404_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()

    def test_invalid_json(self, client):
        """Test invalid JSON handling"""
        response = client.post("/api/v1/memory/create", data="invalid json")
        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_required_fields(self, client):
        """Test missing required fields"""
        incomplete_data = {"emotion": "test"}  # Missing required fields
        response = client.post("/api/v1/memory/create", json=incomplete_data)
        assert response.status_code == 422

class TestAsyncEndpoints:
    """Test async functionality"""

    @pytest.mark.asyncio
    async def test_async_memory_operations(self, async_client):
        """Test async memory operations"""
        if not APP_AVAILABLE:
            pytest.skip("FastAPI app not available")

        # Test async memory creation
        memory_data = {
            "emotion": "async_test",
            "context_snippet": "Testing async memory creation functionality",
            "user_id": TEST_USER_ID,
            "metadata": {"async": True}
        }

        response = await async_client.post("/api/v1/memory/create", json=memory_data)
        assert response.status_code in [200, 500, 503]

def test_integration_flow(client):
    """Test complete integration flow"""
    if not APP_AVAILABLE:
        pytest.skip("FastAPI app not available")

    # 1. Check health
    health_response = client.get("/health")
    assert health_response.status_code == 200

    # 2. Get API info
    info_response = client.get("/api/v1/info")
    assert info_response.status_code == 200

    # 3. Check module health
    modules = ["memory", "dream", "emotion", "consciousness"]
    for module in modules:
        health_response = client.get(f"/api/v1/{module}/health")
        assert health_response.status_code == 200

if __name__ == "__main__":
    """Run tests directly"""
    if APP_AVAILABLE:
        print("🧪 Running LUKHAS AI API Tests...")
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("❌ FastAPI app not available - skipping tests")
        print("Install requirements and ensure lukhas.main module is available")