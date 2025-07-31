#!/usr/bin/env python3
"""
Agent 1 Task 9 Simple Integration Test: Grid Size Calculator
Testing grid calculator integration without complex hub dependencies.
"""

import sys

sys.path.insert(0, "/Users/agi_dev/Downloads/Consolidation-Repo")


def test_grid_calculator_service_integration():
    """Test grid calculator service integration"""
    print("=" * 60)
    print("🔗 GRID SIZE CALCULATOR SERVICE INTEGRATION TEST")
    print("=" * 60)

    try:
        # Test grid calculator availability flag
        from identity.identity_hub import GRID_SIZE_CALCULATOR_AVAILABLE

        if not GRID_SIZE_CALCULATOR_AVAILABLE:
            print("❌ Grid Size Calculator not available in identity hub")
            return False

        print("✅ Grid Size Calculator available in identity hub")

        # Test grid calculator import
        from identity.auth_utils.grid_size_calculator import GridSizeCalculator

        calculator = GridSizeCalculator()

        print("✅ Grid Size Calculator instantiated successfully")

        # Test basic functionality
        result = calculator.calculate_optimal_grid_size(
            content_count=9, cognitive_load_level="moderate"
        )

        if not result or result.grid_size <= 0:
            print("❌ Grid calculation failed")
            return False

        print(f"✅ Grid calculation working: {result.grid_size} cells")

        # Test service registration code is present
        try:
            # This checks if the import path works
            from identity.auth_utils.grid_size_calculator import (
                GridSizeCalculator,
                GridPattern,
                SizingMode,
                ScreenDimensions,
                GridConstraints,
                GridCalculationResult,
            )

            print(
                "✅ All grid calculator components importable for service registration"
            )
        except ImportError as e:
            print(f"❌ Service registration import failed: {e}")
            return False

        # Test interface method signatures that the hub expects
        calculator = GridSizeCalculator()

        # Test calculate_optimal_grid_size method signature
        try:
            result = calculator.calculate_optimal_grid_size(
                content_count=9, cognitive_load_level="moderate"
            )
            if not hasattr(result, "grid_size"):
                print("❌ Grid calculation result missing grid_size attribute")
                return False
            print("✅ calculate_optimal_grid_size interface validated")
        except Exception as e:
            print(f"❌ calculate_optimal_grid_size interface failed: {e}")
            return False

        # Test get_grid_status method
        try:
            status = calculator.get_grid_status()
            if not isinstance(status, dict):
                print("❌ get_grid_status should return dict")
                return False
            print("✅ get_grid_status interface validated")
        except Exception as e:
            print(f"❌ get_grid_status interface failed: {e}")
            return False

        # Test calculate_adaptive_grid_size method
        try:
            performance_data = {
                "accuracy": 0.8,
                "avg_response_time_ms": 1000,
                "error_rate": 0.1,
            }
            adaptive_size = calculator.calculate_adaptive_grid_size(performance_data)
            if not isinstance(adaptive_size, int) or adaptive_size <= 0:
                print("❌ calculate_adaptive_grid_size should return positive int")
                return False
            print("✅ calculate_adaptive_grid_size interface validated")
        except Exception as e:
            print(f"❌ calculate_adaptive_grid_size interface failed: {e}")
            return False

        print("\n🎉 Grid Size Calculator service integration VALIDATED!")
        print("✅ Service availability flag: True")
        print("✅ All interfaces working")
        print("✅ Service registration ready")

        return True

    except Exception as e:
        print(f"❌ Grid calculator service integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_grid_calculator_service_integration()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}")
    exit(0 if success else 1)
