#!/usr/bin/env python3
"""
Test script for migrated orchestrators
Tests basic functionality and compares with expected behavior
"""

import asyncio
import sys
import traceback
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MigrationTest")

# Test results storage
test_results = {
    "brain_orchestrator": {},
    "memory_orchestrator": {},
    "ethics_orchestrator": {},
    "summary": {}
}


async def test_brain_orchestrator():
    """Test the migrated Brain Orchestrator"""
    logger.info("=" * 60)
    logger.info("Testing Brain Orchestrator Migration")
    logger.info("=" * 60)

    results = {
        "initialization": False,
        "startup": False,
        "processing": False,
        "health_check": False,
        "shutdown": False,
        "errors": []
    }

    try:
        # Import the migrated orchestrator
        from brain_orchestrator import BrainOrchestrator, BrainOrchestratorConfig, SystemMode

        # Test 1: Initialization
        logger.info("Test 1: Initialization")
        config = BrainOrchestratorConfig(
            name="TestBrainOrchestrator",
            description="Test instance of migrated Brain Orchestrator",
            mode=SystemMode.ADAPTIVE,
            crista_enabled=True,
            meta_learning_enabled=True,
            quantum_enabled=True
        )

        orchestrator = BrainOrchestrator(config)
        logger.info("✓ Orchestrator created successfully")

        init_success = await orchestrator.initialize()
        results["initialization"] = init_success
        logger.info(f"✓ Initialization: {'SUCCESS' if init_success else 'FAILED'}")

        # Test 2: Startup
        logger.info("\nTest 2: Startup")
        start_success = await orchestrator.start()
        results["startup"] = start_success
        logger.info(f"✓ Startup: {'SUCCESS' if start_success else 'FAILED'}")

        # Test 3: Processing
        logger.info("\nTest 3: Processing")
        test_data = {
            "input_type": "test_data",
            "priority": "high",
            "data": {"value": 42}
        }

        processing_result = await orchestrator.orchestrate_processing(test_data)
        results["processing"] = processing_result.get("error") is None
        logger.info(f"✓ Processing: {'SUCCESS' if results['processing'] else 'FAILED'}")
        logger.info(f"  Processing time: {processing_result.get('processing_time', 0):.2f}s")
        logger.info(f"  Current stage: {processing_result.get('current_stage', 'unknown')}")

        # Test 4: Health Check
        logger.info("\nTest 4: Health Check")
        status = orchestrator.get_status()
        results["health_check"] = status is not None
        logger.info(f"✓ Health Check: {'SUCCESS' if results['health_check'] else 'FAILED'}")
        logger.info(f"  State: {status.get('state', 'unknown')}")
        logger.info(f"  Modules: {status.get('modules', {})}")

        # Test 5: Shutdown
        logger.info("\nTest 5: Shutdown")
        stop_success = await orchestrator.stop()
        results["shutdown"] = stop_success
        logger.info(f"✓ Shutdown: {'SUCCESS' if stop_success else 'FAILED'}")

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    return results


async def test_memory_orchestrator():
    """Test the migrated Memory Orchestrator"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Memory Orchestrator Migration")
    logger.info("=" * 60)

    results = {
        "initialization": False,
        "startup": False,
        "category_processing": {},
        "cache_functionality": False,
        "health_check": False,
        "shutdown": False,
        "errors": []
    }

    try:
        # Import the migrated orchestrator
        from memory_orchestrator import MemoryOrchestrator, MemoryOrchestratorConfig

        # Test 1: Initialization
        logger.info("Test 1: Initialization")
        config = MemoryOrchestratorConfig(
            name="TestMemoryOrchestrator",
            description="Test instance of migrated Memory Orchestrator",
            module_name="memory",
            enable_memory_cache=True,
            cache_size_mb=10
        )

        orchestrator = MemoryOrchestrator(config)
        logger.info("✓ Orchestrator created successfully")

        init_success = await orchestrator.initialize()
        results["initialization"] = init_success
        logger.info(f"✓ Initialization: {'SUCCESS' if init_success else 'FAILED'}")

        # Test 2: Startup
        logger.info("\nTest 2: Startup")
        start_success = await orchestrator.start()
        results["startup"] = start_success
        logger.info(f"✓ Startup: {'SUCCESS' if start_success else 'FAILED'}")

        # Test 3: Category Processing
        logger.info("\nTest 3: Category-based Processing")
        test_cases = [
            {"data": "consciousness test", "expected": "consciousness"},
            {"data": "governance policy", "expected": "governance"},
            {"data": "voice processing", "expected": "voice"},
            {"data": "identity verification", "expected": "identity"},
            {"data": "quantum-like state", "expected": "quantum"},
            {"data": "random data", "expected": "generic"}
        ]

        for test in test_cases:
            result = await orchestrator.process(test["data"])
            success = result.get("status") == "success"
            category = result.get("category", "unknown")
            results["category_processing"][test["expected"]] = success and category == test["expected"]
            logger.info(f"  {test['expected']}: {'✓' if results['category_processing'][test['expected']] else '✗'} (got: {category})")

        # Test 4: Cache Functionality
        logger.info("\nTest 4: Cache Functionality")
        cache_stats = orchestrator.get_cache_stats()
        results["cache_functionality"] = cache_stats.get("cache_enabled", False)
        logger.info(f"✓ Cache Enabled: {cache_stats.get('cache_enabled')}")
        logger.info(f"  Cache Entries: {cache_stats.get('entries_count', 0)}")

        # Test 5: Health Check
        logger.info("\nTest 5: Health Check")
        valid = await orchestrator.validate()
        status = orchestrator.get_status()
        results["health_check"] = valid and status is not None
        logger.info(f"✓ Health Check: {'SUCCESS' if results['health_check'] else 'FAILED'}")
        logger.info(f"  Validation: {'PASSED' if valid else 'FAILED'}")
        logger.info(f"  Active Categories: {len(status.get('categories', {}))}")

        # Test 6: Shutdown
        logger.info("\nTest 6: Shutdown")
        stop_success = await orchestrator.stop()
        results["shutdown"] = stop_success
        logger.info(f"✓ Shutdown: {'SUCCESS' if stop_success else 'FAILED'}")

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    return results


async def test_ethics_orchestrator():
    """Test the migrated Ethics Orchestrator"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Ethics Orchestrator Migration")
    logger.info("=" * 60)

    results = {
        "initialization": False,
        "startup": False,
        "decision_evaluation": {},
        "audit_trail": False,
        "mode_configuration": False,
        "health_check": False,
        "shutdown": False,
        "errors": []
    }

    try:
        # Import the migrated orchestrator
        from ethics_orchestrator import (
            UnifiedEthicsOrchestrator,
            UnifiedEthicsOrchestratorConfig,
            EthicsMode,
            Decision
        )

        # Test 1: Initialization
        logger.info("Test 1: Initialization")
        config = UnifiedEthicsOrchestratorConfig(
            name="TestEthicsOrchestrator",
            description="Test instance of migrated Ethics Orchestrator",
            mode=EthicsMode.ENHANCED,
            enable_human_escalation=False,  # Disable for testing
            enable_meg_evaluation=True,
            enable_drift_detection=True,
            enable_audit_trail=True
        )

        orchestrator = UnifiedEthicsOrchestrator(config)
        logger.info("✓ Orchestrator created successfully")

        init_success = await orchestrator.initialize()
        results["initialization"] = init_success
        logger.info(f"✓ Initialization: {'SUCCESS' if init_success else 'FAILED'}")

        # Test 2: Startup
        logger.info("\nTest 2: Startup")
        start_success = await orchestrator.start()
        results["startup"] = start_success
        logger.info(f"✓ Startup: {'SUCCESS' if start_success else 'FAILED'}")

        # Test 3: Decision Evaluation
        logger.info("\nTest 3: Decision Evaluation")
        test_decisions = [
            Decision("read_file", {"file_type": "public", "user_authorized": True}),
            Decision("delete_data", {"data_type": "user_data", "consent": False}),
            Decision("help_user", {"intent": "educational", "safe": True})
        ]

        for i, decision in enumerate(test_decisions):
            allowed, audit = await orchestrator.evaluate_decision(decision)
            results["decision_evaluation"][decision.action] = {
                "allowed": allowed,
                "confidence": audit.confidence,
                "components": len(audit.components_used)
            }
            logger.info(f"  Decision {i+1} ({decision.action}):")
            logger.info(f"    Allowed: {allowed}")
            logger.info(f"    Confidence: {audit.confidence:.2f}")
            logger.info(f"    Components Used: {', '.join(audit.components_used)}")

        # Test 4: Audit Trail
        logger.info("\nTest 4: Audit Trail")
        audit_trail = orchestrator.get_audit_trail(limit=5)
        results["audit_trail"] = len(audit_trail) > 0
        logger.info(f"✓ Audit Trail: {len(audit_trail)} entries recorded")

        # Test 5: Mode Configuration
        logger.info("\nTest 5: Mode Configuration")
        orchestrator.configure(mode=EthicsMode.PARANOID)
        status = orchestrator.get_ethics_status()
        results["mode_configuration"] = status.get("mode") == "paranoid"
        logger.info(f"✓ Mode Configuration: {'SUCCESS' if results['mode_configuration'] else 'FAILED'}")
        logger.info(f"  Current Mode: {status.get('mode', 'unknown')}")

        # Test 6: Health Check
        logger.info("\nTest 6: Health Check")
        status = orchestrator.get_status()
        results["health_check"] = status is not None
        logger.info(f"✓ Health Check: {'SUCCESS' if results['health_check'] else 'FAILED'}")
        logger.info(f"  State: {status.get('state', 'unknown')}")
        logger.info(f"  Active Modules: {sum(1 for m in status.get('modules', {}).values() if m == 'healthy')}")

        # Test 7: Shutdown
        logger.info("\nTest 7: Shutdown")
        stop_success = await orchestrator.stop()
        results["shutdown"] = stop_success
        logger.info(f"✓ Shutdown: {'SUCCESS' if stop_success else 'FAILED'}")

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    return results


async def run_all_tests():
    """Run all orchestrator tests"""
    logger.info("Starting Orchestrator Migration Tests")
    logger.info(f"Test Date: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Run tests
    test_results["brain_orchestrator"] = await test_brain_orchestrator()
    test_results["memory_orchestrator"] = await test_memory_orchestrator()
    test_results["ethics_orchestrator"] = await test_ethics_orchestrator()

    # Calculate summary
    total_tests = 0
    passed_tests = 0

    for orchestrator, results in test_results.items():
        if orchestrator != "summary":
            for test, result in results.items():
                if test != "errors":
                    if isinstance(result, dict):
                        for sub_test, sub_result in result.items():
                            total_tests += 1
                            if sub_result:
                                passed_tests += 1
                    else:
                        total_tests += 1
                        if result:
                            passed_tests += 1

    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        "test_date": datetime.now().isoformat()
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {test_results['summary']['success_rate']:.1f}%")

    # Save results
    import json
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    logger.info("\nTest results saved to test_results.json")

    return test_results


if __name__ == "__main__":
    asyncio.run(run_all_tests())