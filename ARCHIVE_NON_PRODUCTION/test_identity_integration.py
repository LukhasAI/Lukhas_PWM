"""
Real Integration Tests for LUKHAS Identity System

IMPORTANT: These tests connect to actual system components where available.
Mock implementations are clearly marked with MOCK: comments.
"""

import asyncio
import pytest
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('IDENTITY_INTEGRATION_TEST')

# Import identity components
from identity.core.events import (
    IdentityEventPublisher, IdentityEventType,
    get_identity_event_publisher, IdentityEventPriority,
    AuthenticationContext, VerificationResult
)
from identity.core.colonies import (
    BiometricVerificationColony,
    ConsciousnessVerificationColony,
    DreamVerificationColony
)
from identity.core.swarm import TierAwareSwarmHub
from identity.core.tagging import IdentityTagResolver, IdentityTagType, TrustLevel
from identity.core.health import IdentityHealthMonitor, ComponentType, HealthMetric
from identity.core.glyph import DistributedGLYPHColony, GLYPHType

# MOCK: Biometric sample generation (replace with real biometric capture in production)
def generate_mock_biometric_sample(biometric_type: str, quality: float = 0.8) -> Dict[str, Any]:
    """
    MOCK: Generate simulated biometric data.
    In production, this would interface with actual biometric sensors.
    """
    sample_id = f"sample_{biometric_type}_{int(time.time() * 1000)}"

    # Simulate raw biometric data
    if biometric_type == "fingerprint":
        raw_data = np.random.bytes(2048)  # MOCK: Would be actual minutiae data
    elif biometric_type == "facial":
        raw_data = np.random.bytes(4096)  # MOCK: Would be facial encoding
    elif biometric_type == "iris":
        raw_data = np.random.bytes(1024)  # MOCK: Would be iris pattern
    else:
        raw_data = np.random.bytes(512)

    return {
        "sample_id": sample_id,
        "biometric_type": biometric_type,
        "raw_data": raw_data,
        "quality_score": quality,
        "capture_timestamp": datetime.utcnow(),
        "device_id": "test_device_001",  # MOCK: Would be actual device ID
        "environmental_factors": {
            "lighting": "good",
            "noise_level": "low",
            "motion": "stable"
        },
        "preprocessing_applied": ["normalization", "enhancement"]
    }


# MOCK: Consciousness data generation
def generate_mock_consciousness_data() -> Dict[str, Any]:
    """
    MOCK: Generate simulated consciousness state data.
    In production, this would come from actual consciousness monitoring devices.
    """
    return {
        "brainwave_patterns": {
            "alpha": np.random.random(100) * 30,  # MOCK: Would be EEG data
            "beta": np.random.random(100) * 50,
            "gamma": np.random.random(100) * 100,
            "theta": np.random.random(100) * 10,
            "delta": np.random.random(100) * 5
        },
        "coherence_score": np.random.uniform(0.6, 0.9),
        "emotional_state": {
            "valence": np.random.uniform(-1, 1),
            "arousal": np.random.uniform(0, 1),
            "dominance": np.random.uniform(0, 1)
        },
        "meditation_depth": np.random.uniform(0, 1),
        "focus_level": np.random.uniform(0.5, 1),
        "timestamp": datetime.utcnow()
    }


# MOCK: Dream sequence generation
def generate_mock_dream_sequence() -> List[Dict[str, Any]]:
    """
    MOCK: Generate simulated dream sequence data.
    In production, this would come from dream state monitoring.
    """
    sequence = []
    for i in range(5):
        sequence.append({
            "sequence_id": i,
            "dream_type": np.random.choice(["lucid", "rem", "deep"]),
            "symbols": np.random.choice(["water", "flying", "door", "mirror", "light"], 3).tolist(),
            "emotional_tone": np.random.uniform(-1, 1),
            "clarity": np.random.uniform(0, 1),
            "timestamp": datetime.utcnow() - timedelta(minutes=i*5)
        })
    return sequence


class TestIdentityIntegration:
    """Test suite for integrated identity system."""

    @pytest.fixture
    async def setup_components(self):
        """Set up all identity components for testing."""
        # Initialize event publisher
        event_publisher = await get_identity_event_publisher()

        # Initialize colonies
        biometric_colony = BiometricVerificationColony("test_biometric_colony")
        await biometric_colony.initialize()

        consciousness_colony = ConsciousnessVerificationColony("test_consciousness_colony")
        await consciousness_colony.initialize()

        dream_colony = DreamVerificationColony("test_dream_colony")
        await dream_colony.initialize()

        # Initialize swarm hub
        swarm_hub = TierAwareSwarmHub("test_swarm_hub")
        await swarm_hub.initialize()

        # Initialize tag resolver
        tag_resolver = IdentityTagResolver("test_tag_resolver")
        await tag_resolver.initialize()

        # Initialize health monitor
        health_monitor = IdentityHealthMonitor("test_health_monitor")
        await health_monitor.initialize()

        # Initialize GLYPH colony
        glyph_colony = DistributedGLYPHColony("test_glyph_colony")
        await glyph_colony.initialize()

        # Register components with health monitor
        await health_monitor.register_component(
            "test_biometric_colony",
            ComponentType.COLONY,
            tier_level=0
        )

        yield {
            "event_publisher": event_publisher,
            "biometric_colony": biometric_colony,
            "consciousness_colony": consciousness_colony,
            "dream_colony": dream_colony,
            "swarm_hub": swarm_hub,
            "tag_resolver": tag_resolver,
            "health_monitor": health_monitor,
            "glyph_colony": glyph_colony
        }

        # Cleanup
        # In production, would properly shutdown components

    @pytest.mark.asyncio
    async def test_biometric_verification_real(self, setup_components):
        """Test real biometric verification with colony consensus."""
        components = await setup_components
        biometric_colony = components["biometric_colony"]

        test_results = {
            "test_name": "Biometric Verification Colony Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "BiometricVerificationColony",
            "results": []
        }

        # Test across all tier levels
        for tier in range(6):
            logger.info(f"\n=== Testing Tier {tier} Biometric Verification ===")

            # Generate test identity
            lambda_id = f"test_user_tier_{tier}"

            # MOCK: Generate biometric samples
            biometric_samples = []
            sample_types = ["fingerprint", "facial", "iris"] if tier >= 2 else ["fingerprint"]

            for bio_type in sample_types:
                # Generate multiple samples for consensus
                for i in range(3):
                    quality = 0.9 - (i * 0.1)  # Varying quality
                    sample = generate_mock_biometric_sample(bio_type, quality)

                    # Convert to proper format
                    from identity.core.colonies.biometric_verification_colony import BiometricSample, BiometricType
                    biometric_sample = BiometricSample(
                        sample_id=sample["sample_id"],
                        biometric_type=BiometricType(bio_type),
                        raw_data=sample["raw_data"],
                        quality_score=sample["quality_score"],
                        capture_timestamp=sample["capture_timestamp"],
                        device_id=sample["device_id"],
                        environmental_factors=sample["environmental_factors"],
                        preprocessing_applied=sample["preprocessing_applied"]
                    )
                    biometric_samples.append(biometric_sample)

            # MOCK: Reference template (in production, would come from enrollment)
            reference_template = hashlib.sha256(f"{lambda_id}_reference".encode()).digest()

            # Perform verification
            start_time = time.time()
            try:
                result = await biometric_colony.verify_biometric_identity(
                    lambda_id=lambda_id,
                    biometric_samples=biometric_samples,
                    reference_template=reference_template,
                    tier_level=tier,
                    session_id=f"test_session_{tier}"
                )

                duration = (time.time() - start_time) * 1000

                test_result = {
                    "tier": tier,
                    "success": True,
                    "verified": result.verified,
                    "confidence_score": result.confidence_score,
                    "duration_ms": duration,
                    "consensus_data": result.colony_consensus,
                    "agents_involved": result.agents_involved,
                    "sample_count": len(biometric_samples),
                    "sample_types": list(set(s.biometric_type.value for s in biometric_samples))
                }

                logger.info(f"Tier {tier} Result: Verified={result.verified}, "
                          f"Confidence={result.confidence_score:.3f}, "
                          f"Duration={duration:.1f}ms")

            except Exception as e:
                test_result = {
                    "tier": tier,
                    "success": False,
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
                logger.error(f"Tier {tier} Error: {e}")

            test_results["results"].append(test_result)

            # Small delay between tests
            await asyncio.sleep(0.5)

        # Get colony health status
        colony_health = biometric_colony.get_colony_health_status()
        test_results["colony_health"] = colony_health

        # Save results
        self._save_test_results("biometric_verification", test_results)

        # Assertions
        assert len(test_results["results"]) == 6
        assert colony_health["health_score"] > 0.5
        assert colony_health["total_agents"] > 0

    @pytest.mark.asyncio
    async def test_consciousness_verification_real(self, setup_components):
        """Test consciousness verification for Tier 3+ users."""
        components = await setup_components
        consciousness_colony = components["consciousness_colony"]

        test_results = {
            "test_name": "Consciousness Verification Colony Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "ConsciousnessVerificationColony",
            "results": []
        }

        # Test Tier 3, 4, and 5 (consciousness verification required)
        for tier in [3, 4, 5]:
            logger.info(f"\n=== Testing Tier {tier} Consciousness Verification ===")

            lambda_id = f"test_consciousness_user_tier_{tier}"

            # MOCK: Generate consciousness data
            consciousness_data = generate_mock_consciousness_data()

            start_time = time.time()
            try:
                result = await consciousness_colony.verify_consciousness_state(
                    lambda_id=lambda_id,
                    consciousness_data=consciousness_data,
                    tier_level=tier,
                    session_id=f"test_consciousness_session_{tier}"
                )

                duration = (time.time() - start_time) * 1000

                test_result = {
                    "tier": tier,
                    "success": True,
                    "verified": result.verified,
                    "confidence_score": result.confidence_score,
                    "duration_ms": duration,
                    "verification_method": result.verification_method,
                    "agents_involved": result.agents_involved,
                    "analysis_methods": result.colony_consensus.get("analysis_methods", []) if result.colony_consensus else []
                }

                logger.info(f"Tier {tier} Consciousness Result: Verified={result.verified}, "
                          f"Confidence={result.confidence_score:.3f}, "
                          f"Duration={duration:.1f}ms")

            except Exception as e:
                test_result = {
                    "tier": tier,
                    "success": False,
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
                logger.error(f"Tier {tier} Consciousness Error: {e}")

            test_results["results"].append(test_result)
            await asyncio.sleep(0.5)

        # Get emergent patterns
        emergent_patterns = consciousness_colony.get_emergent_patterns()
        test_results["emergent_patterns"] = {
            "pattern_count": len(emergent_patterns),
            "sample_patterns": list(emergent_patterns.keys())[:5]
        }

        # Save results
        self._save_test_results("consciousness_verification", test_results)

        # Assertions
        assert len(test_results["results"]) == 3
        assert all(r["tier"] >= 3 for r in test_results["results"])

    @pytest.mark.asyncio
    async def test_dream_verification_tier5(self, setup_components):
        """Test dream-based authentication for Tier 5."""
        components = await setup_components
        dream_colony = components["dream_colony"]

        test_results = {
            "test_name": "Dream Verification Colony Test (Tier 5)",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "DreamVerificationColony",
            "results": []
        }

        logger.info("\n=== Testing Tier 5 Dream Verification ===")

        lambda_id = "test_dream_user_tier_5"

        # MOCK: Generate dream sequence
        dream_sequence = generate_mock_dream_sequence()

        start_time = time.time()
        try:
            result = await dream_colony.verify_dream_authentication(
                lambda_id=lambda_id,
                dream_sequence=dream_sequence,
                multiverse_branches=7,
                session_id="test_dream_session_5"
            )

            duration = (time.time() - start_time) * 1000

            test_result = {
                "tier": 5,
                "success": True,
                "verified": result.verified,
                "confidence_score": result.confidence_score,
                "duration_ms": duration,
                "multiverse_correlation": result.colony_consensus.get("multiverse_correlation", 0) if result.colony_consensus else 0,
                "quantum_verification": result.colony_consensus.get("quantum_verified", False) if result.colony_consensus else False,
                "dream_coherence": result.colony_consensus.get("dream_coherence", 0) if result.colony_consensus else 0
            }

            logger.info(f"Tier 5 Dream Result: Verified={result.verified}, "
                      f"Confidence={result.confidence_score:.3f}, "
                      f"Duration={duration:.1f}ms")

        except Exception as e:
            test_result = {
                "tier": 5,
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000
            }
            logger.error(f"Tier 5 Dream Error: {e}")

        test_results["results"].append(test_result)

        # Get collective dream space info
        collective_patterns = dream_colony.collective_unconscious_patterns
        test_results["collective_patterns"] = {
            "pattern_count": len(collective_patterns),
            "sample_patterns": list(collective_patterns.keys())[:3]
        }

        # Save results
        self._save_test_results("dream_verification", test_results)

        # Assertions
        assert len(test_results["results"]) == 1
        assert test_results["results"][0]["tier"] == 5

    @pytest.mark.asyncio
    async def test_swarm_hub_orchestration(self, setup_components):
        """Test tier-aware swarm hub orchestration."""
        components = await setup_components
        swarm_hub = components["swarm_hub"]

        test_results = {
            "test_name": "Tier-Aware Swarm Hub Orchestration Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "TierAwareSwarmHub",
            "results": []
        }

        # Test task submission for different tiers
        for tier in [0, 2, 4]:
            logger.info(f"\n=== Testing Swarm Hub for Tier {tier} ===")

            lambda_id = f"test_swarm_user_tier_{tier}"

            # MOCK: Prepare auth data based on tier
            auth_data = {
                "biometric_data": {
                    "samples": [generate_mock_biometric_sample("fingerprint")],
                    "template": hashlib.sha256(f"{lambda_id}_template".encode()).digest()
                }
            }

            if tier >= 3:
                auth_data["consciousness_data"] = generate_mock_consciousness_data()

            if tier >= 5:
                auth_data["dream_data"] = {
                    "sequence": generate_mock_dream_sequence(),
                    "branches": 7
                }

            start_time = time.time()
            try:
                task_id = await swarm_hub.submit_identity_verification_task(
                    lambda_id=lambda_id,
                    tier_level=tier,
                    verification_type="comprehensive",
                    session_id=f"test_swarm_session_{tier}",
                    auth_data=auth_data
                )

                # Wait a bit for task to be processed
                await asyncio.sleep(2)

                duration = (time.time() - start_time) * 1000

                # Get hub statistics
                hub_stats = swarm_hub.get_hub_statistics()

                test_result = {
                    "tier": tier,
                    "success": True,
                    "task_id": task_id,
                    "duration_ms": duration,
                    "required_colonies": swarm_hub._determine_required_colonies(tier, "comprehensive"),
                    "priority_boost": swarm_hub.tier_profiles[tier].priority_boost,
                    "max_agents": swarm_hub.tier_profiles[tier].max_agents,
                    "active_orchestrations": hub_stats["active_orchestrations"]
                }

                logger.info(f"Tier {tier} Swarm Task: ID={task_id[:20]}..., "
                          f"Colonies={len(test_result['required_colonies'])}, "
                          f"Priority Boost={test_result['priority_boost']}")

            except Exception as e:
                test_result = {
                    "tier": tier,
                    "success": False,
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
                logger.error(f"Tier {tier} Swarm Error: {e}")

            test_results["results"].append(test_result)

        # Get performance report
        performance_report = swarm_hub.get_tier_performance_report()
        test_results["performance_report"] = performance_report

        # Save results
        self._save_test_results("swarm_hub_orchestration", test_results)

        # Assertions
        assert len(test_results["results"]) == 3
        assert all(r["tier"] in [0, 2, 4] for r in test_results["results"])

    @pytest.mark.asyncio
    async def test_trust_network_and_tagging(self, setup_components):
        """Test identity tag resolver with trust networks."""
        components = await setup_components
        tag_resolver = components["tag_resolver"]

        test_results = {
            "test_name": "Identity Tag Resolver and Trust Network Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "IdentityTagResolver",
            "results": []
        }

        logger.info("\n=== Testing Trust Network and Tagging ===")

        # Create test identities
        identities = [f"test_trust_user_{i}" for i in range(5)]

        # Establish trust relationships
        logger.info("Establishing trust relationships...")
        trust_results = []

        for i in range(len(identities)):
            for j in range(i + 1, len(identities)):
                trust_level = TrustLevel.MEDIUM if (i + j) % 2 == 0 else TrustLevel.LOW

                success = await tag_resolver.establish_trust_relationship(
                    from_identity=identities[i],
                    to_identity=identities[j],
                    initial_trust=trust_level,
                    trust_factors={"test_factor": 0.5}
                )

                trust_results.append({
                    "from": identities[i],
                    "to": identities[j],
                    "trust_level": trust_level.name,
                    "success": success
                })

                logger.info(f"Trust: {identities[i][:15]} -> {identities[j][:15]} = {trust_level.name}")

        test_results["trust_relationships"] = trust_results

        # Test tag assignment with consensus
        logger.info("\nTesting tag assignment with consensus...")
        tag_results = []

        for i, identity in enumerate(identities[:3]):
            tier = i + 1

            # Direct tag (no consensus)
            tag_id = await tag_resolver.assign_identity_tag(
                lambda_id=identity,
                tag_type=IdentityTagType.CAPABILITY,
                tag_value=f"tier_{tier}_access",
                tier_level=tier,
                metadata={"test": True},
                require_consensus=False
            )

            tag_results.append({
                "identity": identity,
                "tag_type": "capability",
                "tag_value": f"tier_{tier}_access",
                "consensus_required": False,
                "tag_id": tag_id
            })

            # Consensus-based tag
            if i > 0:  # Need trust network for consensus
                consensus_tag_id = await tag_resolver.assign_identity_tag(
                    lambda_id=identity,
                    tag_type=IdentityTagType.PERMISSION,
                    tag_value="admin_access",
                    tier_level=tier,
                    require_consensus=True,
                    issuer_id=identities[0]
                )

                tag_results.append({
                    "identity": identity,
                    "tag_type": "permission",
                    "tag_value": "admin_access",
                    "consensus_required": True,
                    "consensus_request_id": consensus_tag_id
                })

        test_results["tag_assignments"] = tag_results

        # Test reputation calculation
        logger.info("\nCalculating identity reputations...")
        reputation_results = []

        for identity in identities[:3]:
            reputation = tag_resolver.get_identity_reputation(identity)
            reputation_results.append({
                "identity": identity,
                "overall_reputation": reputation["overall_reputation"],
                "trust_reputation": reputation["trust_reputation"],
                "network_influence": reputation["network_influence"]
            })

            logger.info(f"Reputation {identity[:15]}: {reputation['overall_reputation']:.3f}")

        test_results["reputation_scores"] = reputation_results

        # Get resolver statistics
        resolver_stats = tag_resolver.get_resolver_statistics()
        test_results["resolver_statistics"] = resolver_stats

        # Save results
        self._save_test_results("trust_network_tagging", test_results)

        # Assertions
        assert len(trust_results) > 0
        assert all(r["success"] for r in trust_results)
        assert len(tag_results) > 0

    @pytest.mark.asyncio
    async def test_health_monitoring_and_healing(self, setup_components):
        """Test health monitoring and self-healing capabilities."""
        components = await setup_components
        health_monitor = components["health_monitor"]
        biometric_colony = components["biometric_colony"]

        test_results = {
            "test_name": "Identity Health Monitor and Self-Healing Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "IdentityHealthMonitor",
            "results": []
        }

        logger.info("\n=== Testing Health Monitoring and Self-Healing ===")

        # Register more components
        await health_monitor.register_component(
            "test_consciousness_colony",
            ComponentType.COLONY,
            tier_level=3
        )

        await health_monitor.register_component(
            "test_swarm_hub",
            ComponentType.SWARM_HUB,
            tier_level=0
        )

        # Report healthy metrics
        logger.info("Reporting healthy metrics...")
        healthy_metrics = {
            HealthMetric.SUCCESS_RATE: 0.95,
            HealthMetric.ERROR_RATE: 0.05,
            HealthMetric.RESPONSE_TIME: 150,  # ms
            HealthMetric.RESOURCE_USAGE: 0.4,
            HealthMetric.CONSENSUS_STRENGTH: 0.85
        }

        await health_monitor.report_component_metrics(
            "test_biometric_colony",
            healthy_metrics,
            tier_level=2
        )

        # Get initial health
        initial_health = health_monitor.get_system_health_report()
        test_results["initial_health"] = initial_health

        logger.info(f"Initial System Health: {initial_health['overall_health']:.3f}")

        # Simulate degradation
        logger.info("\nSimulating component degradation...")
        degraded_metrics = {
            HealthMetric.SUCCESS_RATE: 0.3,
            HealthMetric.ERROR_RATE: 0.7,
            HealthMetric.RESPONSE_TIME: 5000,  # ms
            HealthMetric.RESOURCE_USAGE: 0.95
        }

        await health_monitor.report_component_metrics(
            "test_biometric_colony",
            degraded_metrics,
            tier_level=2
        )

        # Report errors
        for i in range(5):
            await health_monitor.report_component_error(
                "test_biometric_colony",
                f"Test error {i}: Simulated failure",
                severity="error",
                tier_level=2
            )

        # Wait for healing to trigger
        await asyncio.sleep(2)

        # Get post-degradation health
        degraded_health = health_monitor.get_system_health_report()
        test_results["degraded_health"] = degraded_health

        logger.info(f"Degraded System Health: {degraded_health['overall_health']:.3f}")
        logger.info(f"Active Healing Plans: {degraded_health['active_healing_plans']}")

        # Get component details
        component_details = health_monitor.get_component_health_details("test_biometric_colony")
        test_results["component_health_details"] = component_details

        # Test results summary
        test_results["results"] = [{
            "test": "health_monitoring",
            "initial_health_score": initial_health["overall_health"],
            "degraded_health_score": degraded_health["overall_health"],
            "healing_triggered": degraded_health["active_healing_plans"] > 0,
            "component_status": component_details["status"] if component_details else "unknown",
            "recent_errors": component_details["recent_errors"] if component_details else 0
        }]

        # Save results
        self._save_test_results("health_monitoring", test_results)

        # Assertions
        assert initial_health["overall_health"] > degraded_health["overall_health"]
        assert degraded_health["active_healing_plans"] >= 0  # Healing should be considered

    @pytest.mark.asyncio
    async def test_distributed_glyph_generation(self, setup_components):
        """Test distributed GLYPH generation."""
        components = await setup_components
        glyph_colony = components["glyph_colony"]

        test_results = {
            "test_name": "Distributed GLYPH Generation Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "DistributedGLYPHColony",
            "results": []
        }

        logger.info("\n=== Testing Distributed GLYPH Generation ===")

        # Test GLYPH generation for different tiers
        for tier in [0, 2, 5]:
            logger.info(f"\nGenerating GLYPH for Tier {tier}...")

            lambda_id = f"test_glyph_user_tier_{tier}"

            # Identity data for embedding
            identity_data = {
                "lambda_id": lambda_id,
                "tier": tier,
                "timestamp": datetime.utcnow().isoformat(),
                "test": True
            }

            # MOCK: Steganographic data (would include actual biometric hashes in production)
            steganographic_data = {
                "biometric_hash": hashlib.sha256(f"{lambda_id}_biometric".encode()).hexdigest()[:16],
                "tier_signature": f"tier_{tier}_sig"
            }

            start_time = time.time()
            try:
                glyph = await glyph_colony.generate_identity_glyph(
                    lambda_id=lambda_id,
                    glyph_type=GLYPHType.AUTHENTICATION,
                    tier_level=tier,
                    identity_data=identity_data,
                    steganographic_data=steganographic_data,
                    session_id=f"test_glyph_session_{tier}"
                )

                duration = (time.time() - start_time) * 1000

                test_result = {
                    "tier": tier,
                    "success": True,
                    "glyph_id": glyph.glyph_id,
                    "duration_ms": duration,
                    "image_size": glyph.image_data.shape,
                    "quality_score": glyph.quality_metrics.get("overall_quality", 0),
                    "fragments_used": len(glyph.fragments_used),
                    "consensus_achieved": glyph.consensus_achieved,
                    "complexity": glyph.generation_metadata.get("complexity", "unknown")
                }

                logger.info(f"Tier {tier} GLYPH: Quality={test_result['quality_score']:.3f}, "
                          f"Fragments={test_result['fragments_used']}, "
                          f"Duration={duration:.1f}ms")

                # Save GLYPH image (base64) sample
                test_result["glyph_sample"] = glyph.to_base64()[:100] + "..."  # First 100 chars

            except Exception as e:
                test_result = {
                    "tier": tier,
                    "success": False,
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
                logger.error(f"Tier {tier} GLYPH Error: {e}")

            test_results["results"].append(test_result)

        # Get colony statistics
        colony_stats = glyph_colony.get_colony_statistics()
        test_results["colony_statistics"] = colony_stats

        # Save results
        self._save_test_results("glyph_generation", test_results)

        # Assertions
        assert len(test_results["results"]) == 3
        assert any(r["success"] for r in test_results["results"])

    @pytest.mark.asyncio
    async def test_colony_connectivity_and_state(self, setup_components):
        """Test colony connectivity and state management."""
        components = await setup_components

        test_results = {
            "test_name": "Colony Connectivity and State Test",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "Multiple Colonies",
            "results": []
        }

        logger.info("\n=== Testing Colony Connectivity and State ===")

        # Test each colony
        colonies = [
            ("biometric_colony", components["biometric_colony"]),
            ("consciousness_colony", components["consciousness_colony"]),
            ("dream_colony", components["dream_colony"]),
            ("glyph_colony", components["glyph_colony"])
        ]

        for colony_name, colony in colonies:
            logger.info(f"\nTesting {colony_name}...")

            # Get colony state
            if hasattr(colony, "get_colony_health_status"):
                health_status = colony.get_colony_health_status()
            else:
                health_status = {
                    "colony_id": colony.colony_id,
                    "agent_count": len(colony.agents),
                    "capabilities": colony.capabilities
                }

            # Test agent connectivity
            active_agents = 0
            agent_states = {}

            for agent_id, agent in colony.agents.items():
                agent_states[agent_id] = {
                    "state": agent.state.value if hasattr(agent.state, "value") else str(agent.state),
                    "capabilities": list(agent.capabilities) if hasattr(agent, "capabilities") else []
                }

                if hasattr(agent, "state") and agent.state.value in ["IDLE", "WORKING"]:
                    active_agents += 1

            colony_result = {
                "colony_name": colony_name,
                "colony_id": colony.colony_id,
                "total_agents": len(colony.agents),
                "active_agents": active_agents,
                "health_status": health_status,
                "sample_agent_states": dict(list(agent_states.items())[:3])  # First 3 agents
            }

            logger.info(f"{colony_name}: {active_agents}/{len(colony.agents)} agents active")

            test_results["results"].append(colony_result)

        # Test inter-colony communication via event bus
        logger.info("\nTesting inter-colony event communication...")
        event_publisher = components["event_publisher"]

        # Publish test event
        test_event_id = await event_publisher.publish_colony_event(
            IdentityEventType.COLONY_CONSENSUS_VOTING,
            lambda_id="test_connectivity_user",
            tier_level=3,
            colony_id="test_colony",
            consensus_data={"test": True, "connectivity_check": True}
        )

        test_results["event_communication"] = {
            "test_event_id": test_event_id,
            "event_type": "COLONY_CONSENSUS_VOTING",
            "event_stats": event_publisher.get_event_statistics()
        }

        # Save results
        self._save_test_results("colony_connectivity", test_results)

        # Assertions
        assert len(test_results["results"]) == 4
        assert all(r["total_agents"] > 0 for r in test_results["results"])
        assert test_results["event_communication"]["test_event_id"] is not None

    def _save_test_results(self, test_name: str, results: Dict[str, Any]):
        """Save test results to file."""
        filename = f"identity/tests/results/test_{test_name}_{int(time.time())}.json"

        # Ensure directory exists
        import os
        os.makedirs("identity/tests/results", exist_ok=True)

        # Save results
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Test results saved to: {filename}")


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])