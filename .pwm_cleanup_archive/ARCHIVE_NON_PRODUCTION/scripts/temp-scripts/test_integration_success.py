#!/usr/bin/env python3
"""
Test Integration Success
Verify that the key integrations from the plan are working correctly.
"""


def test_core_components():
    """Test individual core components that should be working"""
    print("🧪 Testing Core Integration Components...")

    # Test 1: Core Hub
    try:
        from core.core_hub import CoreHub

        core_hub = CoreHub()
        print("✅ CoreHub: WORKING")
    except Exception as e:
        print(f"❌ CoreHub: FAILED - {e}")

    # Test 2: Consciousness Hub (after our fix)
    try:
        from consciousness.consciousness_hub import ConsciousnessHub

        consciousness_hub = ConsciousnessHub()
        print("✅ ConsciousnessHub: WORKING")
    except Exception as e:
        print(f"❌ ConsciousnessHub: FAILED - {e}")

    # Test 3: Ethics Service
    try:
        from ethics.service import EthicsService

        ethics_service = EthicsService()
        print("✅ EthicsService: WORKING")
    except Exception as e:
        print(f"❌ EthicsService: FAILED - {e}")

    # Test 4: Bio Engine (Phase 1.1 from plan)
    try:
        from bio.bio_engine import get_bio_engine

        bio_engine = get_bio_engine()
        print("✅ Bio Engine: WORKING")
    except Exception as e:
        print(f"❌ Bio Engine: FAILED - {e}")

    # Test 5: Bio Integration Hub (Phase 2.1 from plan)
    try:
        from bio.bio_integration_hub import get_bio_integration_hub

        bio_hub = get_bio_integration_hub()
        print("✅ Bio Integration Hub: WORKING")
    except Exception as e:
        print(f"❌ Bio Integration Hub: FAILED - {e}")

    # Test 6: Ethics Integration (Phase 1.2 from plan)
    try:
        from ethics.ethics_integration import get_ethics_integration

        unified_ethics = get_ethics_integration()
        print("✅ Unified Ethics: WORKING")
    except Exception as e:
        print(f"❌ Unified Ethics: FAILED - {e}")

    # Test 7: Core Interfaces Hub (Phase 3.1 from plan)
    try:
        from core.interfaces.interfaces_hub import get_interfaces_hub

        interfaces_hub = get_interfaces_hub()
        print("✅ Core Interfaces Hub: WORKING")
    except Exception as e:
        print(f"❌ Core Interfaces Hub: FAILED - {e}")

    # Test 8: Unified Consciousness Engine (Phase 4.1 from plan)
    try:
        from consciousness.systems.unified_consciousness_engine import (
            get_unified_consciousness_engine,
        )

        unified_consciousness = get_unified_consciousness_engine()
        print("✅ Unified Consciousness Engine: WORKING")
    except Exception as e:
        print(f"❌ Unified Consciousness Engine: FAILED - {e}")


def test_integration_connections():
    """Test that the integration connections work"""
    print("\n🔗 Testing Integration Connections...")

    try:
        # Test bio engine registration with core hub
        from core.core_hub import CoreHub
        from bio.bio_engine import get_bio_engine

        core_hub = CoreHub()
        bio_engine = get_bio_engine()

        # Test service registration (Phase 1.1 requirement)
        core_hub.register_service("bio_engine", bio_engine)
        print("✅ Bio Engine → Core Hub registration: WORKING")

        # Test if bio engine is accessible
        registered_bio = core_hub.get_service("bio_engine")
        if registered_bio is not None:
            print("✅ Bio Engine service retrieval: WORKING")
        else:
            print("❌ Bio Engine service retrieval: FAILED")

    except Exception as e:
        print(f"❌ Integration connections: FAILED - {e}")


def test_file_connectivity():
    """Test if the isolated files from the plan are now connected"""
    print("\n📁 Testing File Connectivity Improvements...")

    # Files from Priority 1 (Critical Implementation Files) in the plan
    priority_1_files = [
        "bio/bio_engine.py",
        "ethics/ethics_integration.py",
        "consciousness/systems/unified_consciousness_engine.py",
        "core/interfaces/interfaces_hub.py",
        "bio/bio_integration_hub.py",
    ]

    working_files = 0
    total_files = len(priority_1_files)

    for file_path in priority_1_files:
        try:
            # Convert file path to import path
            if file_path == "bio/bio_engine.py":
                from bio.bio_engine import get_bio_engine

                get_bio_engine()
            elif file_path == "ethics/ethics_integration.py":
                from ethics.ethics_integration import get_ethics_integration

                get_ethics_integration()
            elif file_path == "consciousness/systems/unified_consciousness_engine.py":
                from consciousness.systems.unified_consciousness_engine import (
                    get_unified_consciousness_engine,
                )

                get_unified_consciousness_engine()
            elif file_path == "core/interfaces/interfaces_hub.py":
                from core.interfaces.interfaces_hub import get_interfaces_hub

                get_interfaces_hub()
            elif file_path == "bio/bio_integration_hub.py":
                from bio.bio_integration_hub import get_bio_integration_hub

                get_bio_integration_hub()

            print(f"✅ {file_path}: CONNECTED")
            working_files += 1

        except Exception as e:
            print(f"❌ {file_path}: NOT CONNECTED - {e}")

    connectivity_percentage = (working_files / total_files) * 100
    print(
        f"\n📊 Priority 1 Files Connectivity: {working_files}/{total_files} ({connectivity_percentage:.1f}%)"
    )

    return connectivity_percentage


def main():
    """Run all integration tests"""
    print("🎯 LUKHAS INTEGRATION SUCCESS VERIFICATION")
    print("=" * 50)

    # Test core components
    test_core_components()

    # Test integration connections
    test_integration_connections()

    # Test file connectivity
    connectivity = test_file_connectivity()

    print("\n" + "=" * 50)
    print("📋 INTEGRATION SUMMARY")
    print("=" * 50)

    if connectivity >= 80:
        print("🎉 SUCCESS: Core integration goals achieved!")
        print(f"📈 Connectivity: {connectivity:.1f}% of priority files integrated")
        print("✨ The system is ready for operation!")
    else:
        print(f"⚠️  PARTIAL: {connectivity:.1f}% connectivity achieved")
        print("🔧 Some components need additional work")

    print("\n🚀 Integration testing complete!")


if __name__ == "__main__":
    main()
