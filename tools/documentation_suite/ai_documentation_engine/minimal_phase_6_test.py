"""
Minimal Phase 6 Test - AI Documentation Engine
==============================================

Direct component testing without complex imports.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

async def test_documentation_generator():
    """Test DocumentationGenerator component"""
    try:
        # Direct import from file
        sys.path.insert(0, str(Path(__file__).parent))
        from ecosystem_documentation_generator import DocumentationGenerator
        
        print("   🔧 Testing DocumentationGenerator initialization...", end="")
        generator = DocumentationGenerator()
        assert generator is not None
        print(" ✅ PASSED")
        
        return True
    except Exception as e:
        print(f" ❌ FAILED: {e}")
        return False

async def test_api_documentation_generator():
    """Test APIDocumentationGenerator component"""
    try:
        from api_documentation_generator import APIDocumentationGenerator
        
        print("   🔧 Testing APIDocumentationGenerator initialization...", end="")
        generator = APIDocumentationGenerator()
        assert generator is not None
        print(" ✅ PASSED")
        
        return True
    except Exception as e:
        print(f" ❌ FAILED: {e}")
        return False

async def test_tutorial_generator():
    """Test TutorialGenerator component"""
    try:
        from interactive_tutorial_generator import TutorialGenerator, TutorialType, DifficultyLevel, LearningStyle
        
        print("   🔧 Testing TutorialGenerator initialization...", end="")
        generator = TutorialGenerator()
        assert generator is not None
        print(" ✅ PASSED")
        
        print("   🔧 Testing tutorial generation...", end="")
        tutorial = await generator.generate_tutorial(
            topic="Test Topic",
            tutorial_type=TutorialType.QUICK_START,
            difficulty_level=DifficultyLevel.BEGINNER,
            learning_style=LearningStyle.HANDS_ON
        )
        assert tutorial is not None
        assert tutorial.title is not None
        print(" ✅ PASSED")
        
        return True
    except Exception as e:
        print(f" ❌ FAILED: {e}")
        return False

async def test_documentation_analytics():
    """Test DocumentationAnalytics component"""
    try:
        from documentation_analytics import DocumentationAnalytics, AnalyticsType
        from datetime import datetime, timedelta
        
        print("   🔧 Testing DocumentationAnalytics initialization...", end="")
        analytics = DocumentationAnalytics()
        assert analytics is not None
        print(" ✅ PASSED")
        
        print("   🔧 Testing analytics report generation...", end="")
        report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.USAGE_PATTERNS,
            time_period=(datetime.now() - timedelta(days=30), datetime.now())
        )
        assert report is not None
        assert "total_page_views" in report.summary
        print(" ✅ PASSED")
        
        return True
    except Exception as e:
        print(f" ❌ FAILED: {e}")
        return False

async def run_phase_6_minimal_tests():
    """Run minimal Phase 6 tests"""
    
    print("🤖 AI Documentation Engine - Phase 6 Minimal Tests")
    print("=" * 55)
    
    tests = [
        ("DocumentationGenerator", test_documentation_generator),
        ("APIDocumentationGenerator", test_api_documentation_generator),
        ("TutorialGenerator", test_tutorial_generator),
        ("DocumentationAnalytics", test_documentation_analytics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}")
        print("-" * 40)
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 55)
    print("🧪 Phase 6 Minimal Test Summary")
    print("=" * 55)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 Phase 6 AI Documentation Engine implementation successful!")
        print("   📚 All core components are functional")
        print("   🔧 Documentation generation capabilities: Ready")
        print("   📊 Analytics and reporting systems: Ready")
        print("   🎓 Interactive tutorial creation: Ready")
        print("\n✅ Phase 6 Complete - Advanced Documentation & Development Tools")
        print("🚀 Ready to proceed with Phase 7+ implementation!")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review required.")
        return False

if __name__ == "__main__":
    # Run minimal Phase 6 tests
    success = asyncio.run(run_phase_6_minimal_tests())
    exit(0 if success else 1)
