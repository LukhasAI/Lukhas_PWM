#!/usr/bin/env python3
"""
Test script for Phase 3C Module Introspection
# LUKHAS_TAG: test_introspection_system
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.introspection import analyze_module, report_symbolic_state


def test_introspection_system():
    """Test the introspection system with various modules"""

    print("🔍 Testing Phase 3C Module Introspection System")
    print("=" * 60)

    # Test modules
    test_modules = [
        "core/introspection/introspector.py",
        "oneiric/oneiric_core/memory/dream_memory_fold.py",
        "core/core.py",
    ]

    results = []

    for module_path in test_modules:
        if os.path.exists(module_path):
            print(f"\n📂 Analyzing: {module_path}")
            try:
                analysis = analyze_module(module_path)
                report = report_symbolic_state(analysis)
                results.append((module_path, analysis, report))
                print("✅ Success")
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append((module_path, None, f"Error: {e}"))
        else:
            print(f"\n⚠️  Module not found: {module_path}")

    print(
        f"\n📊 Summary: Analyzed {len([r for r in results if r[1] is not None])} modules"
    )

    # Test symbolic tag detection
    print("\n🏷️  Testing Symbolic Tag Detection:")
    for module_path, analysis, report in results:
        if analysis:
            tags = analysis.get("symbolic_tags", {})
            locked = analysis.get("locked_status", False)
            print(
                f"  {module_path}: {len(tags)} tags, {'LOCKED' if locked else 'UNLOCKED'}"
            )

    return results


if __name__ == "__main__":
    test_introspection_system()
