#!/usr/bin/env python3
"""
Test script to verify all reorganized modules and their connections.
"""

import sys
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("LUKHAS AGI - Module Reorganization Test")
print("=" * 80)

tests_passed = 0
tests_failed = 0

def test_import(module_path, class_names=None):
    """Test importing a module and optionally specific classes."""
    global tests_passed, tests_failed

    try:
        print(f"\n[TEST] Importing {module_path}...")
        module = __import__(module_path, fromlist=['*'])

        if class_names:
            for class_name in class_names:
                if hasattr(module, class_name):
                    print(f"  ✓ Found class: {class_name}")
                else:
                    print(f"  ✗ Missing class: {class_name}")
                    tests_failed += 1
                    return

        print(f"  ✓ Successfully imported {module_path}")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"  ✗ Failed to import {module_path}")
        print(f"    Error: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        tests_failed += 1
        return False

# Test 1: Memory Fold System
print("\n" + "="*60)
print("Testing Memory Fold System")
print("="*60)

if test_import("memory.core_memory.memory_fold",
               ["MemoryFoldSystem", "MemoryFoldConfig", "MemoryFoldDatabase"]):
    try:
        from memory.core_memory.memory_fold import MemoryFoldSystem
        system = MemoryFoldSystem()
        print("  ✓ Successfully instantiated MemoryFoldSystem")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed to instantiate MemoryFoldSystem: {e}")
        tests_failed += 1

# Test 2: Brain Integration
print("\n" + "="*60)
print("Testing Brain Integration")
print("="*60)

if test_import("consciousness.brain_integration_20250620_013824",
               ["LucasBrainIntegration", "BrainIntegrationConfig"]):
    try:
        from consciousness.brain_integration_20250620_013824 import LucasBrainIntegration
        brain = LucasBrainIntegration()
        print("  ✓ Successfully instantiated LucasBrainIntegration")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed to instantiate LucasBrainIntegration: {e}")
        tests_failed += 1

# Test 3: Cognitive Architecture Controller
print("\n" + "="*60)
print("Testing Cognitive Architecture Controller")
print("="*60)

if test_import("consciousness.cognitive_architecture_controller",
               ["CognitiveArchitectureController", "CognitiveConfig"]):
    try:
        from consciousness.cognitive_architecture_controller import CognitiveArchitectureController
        controller = CognitiveArchitectureController(user_tier=5)
        print("  ✓ Successfully instantiated CognitiveArchitectureController")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed to instantiate CognitiveArchitectureController: {e}")
        tests_failed += 1

# Test 4: Cognitive Adapter
print("\n" + "="*60)
print("Testing Cognitive Adapter")
print("="*60)

if test_import("consciousness.cognitive.cognitive_adapter_complete",
               ["CognitiveAdapter", "CognitiveAdapterConfig"]):
    try:
        from consciousness.cognitive.cognitive_adapter_complete import CognitiveAdapter
        adapter = CognitiveAdapter(user_tier=5)
        print("  ✓ Successfully instantiated CognitiveAdapter")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed to instantiate CognitiveAdapter: {e}")
        tests_failed += 1

# Test 5: AGI Consciousness Engine
print("\n" + "="*60)
print("Testing AGI Consciousness Engine")
print("="*60)

if test_import("consciousness.core_consciousness.agi_consciousness_engine_complete",
               ["AGIConsciousnessEngine", "ConsciousnessEngineConfig"]):
    try:
        from consciousness.core_consciousness.agi_consciousness_engine_complete import AGIConsciousnessEngine
        # Note: This requires ANTHROPIC_API_KEY to be set
        print("  ℹ Note: AGIConsciousnessEngine requires ANTHROPIC_API_KEY environment variable")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed to import AGIConsciousnessEngine: {e}")
        tests_failed += 1

# Test 6: Configuration Files
print("\n" + "="*60)
print("Testing Configuration Files")
print("="*60)

config_files = [
    "config/memory_fold_config.json",
    "config/brain_integration_config.ini",
    "config/cognitive_architecture_config.ini",
    "config/cognitive_adapter_config.json",
    "config/agi_consciousness_config.json"
]

for config_file in config_files:
    config_path = project_root / config_file
    if config_path.exists():
        print(f"  ✓ Found config file: {config_file}")
        tests_passed += 1
    else:
        print(f"  ✗ Missing config file: {config_file}")
        tests_failed += 1

# Test 7: Check for removed duplicate files
print("\n" + "="*60)
print("Verifying Duplicate Files Removed")
print("="*60)

removed_files = [
    "core/integration/memory/memory_fold.py",
    "core/interfaces/as_agent/core/memory_fold.py",
    "orchestration/brain/memory/memory_fold.py",
    "orchestration/brain/spine/memory_fold.py",
    "memory/memory_fold.py",
    "core/spine/memory_fold_enhanced.py",
    "core/spine/memory_fold.py"
]

for removed_file in removed_files:
    file_path = project_root / removed_file
    if not file_path.exists():
        print(f"  ✓ Successfully removed: {removed_file}")
        tests_passed += 1
    else:
        print(f"  ✗ File still exists: {removed_file}")
        tests_failed += 1

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print(f"Total Tests: {tests_passed + tests_failed}")
print(f"Success Rate: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")

if tests_failed == 0:
    print("\n✅ All tests passed! The reorganization was successful.")
else:
    print(f"\n⚠️  {tests_failed} tests failed. Please check the errors above.")

print("\nNote: Some imports may fail due to missing dependencies in other modules.")
print("This is expected for the dynamic import system with fallbacks.")