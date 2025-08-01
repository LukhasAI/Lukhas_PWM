"""
Simple Test Runner for AI Documentation Engine
=============================================

Comprehensive tests for the AI documentation engine without external dependencies.
"""

import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import AI documentation engine components
try:
    from tools.documentation_suite.ai_documentation_engine import (
        DocumentationGenerator,
        APIDocumentationGenerator,
        TutorialGenerator,
        DocumentationAnalytics,
        TutorialType,
        DifficultyLevel,
        LearningStyle,
        AnalyticsType,
        OutputFormat
    )
    print("✅ Successfully imported AI documentation engine components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

class SimpleTestRunner:
    """Simple test runner without external dependencies"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    async def run_test(self, test_name: str, test_func):
        """Run a single test"""
        self.total_tests += 1
        
        try:
            print(f"   🔧 {test_name}...", end="")
            await test_func()
            print(" ✅ PASSED")
            self.passed_tests += 1
            self.test_results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f" ❌ FAILED: {str(e)}")
            self.failed_tests += 1
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🧪 AI Documentation Engine Test Summary")
        print("=" * 60)
        print(f"📊 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 All tests passed! AI Documentation Engine is working correctly.")
            return True
        else:
            print(f"\n⚠️ {self.failed_tests} tests failed.")
            return False

async def test_ecosystem_documentation_generator():
    """Test ecosystem documentation generation"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create sample files
        (workspace / "main.py").write_text("""
def hello_world():
    '''Simple hello world function'''
    return "Hello, World!"

class ExampleClass:
    '''Example class for testing'''
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
""")
        
        (workspace / "README.md").write_text("""
# Test Project

This is a test project for documentation generation.

## Features
- Hello world functionality
- Example class implementation
""")
        
        generator = DocumentationGenerator()
        
        # Test workspace structure discovery
        structure = await generator.discover_workspace_structure(str(workspace))
        assert structure is not None, "Workspace structure should not be None"
        assert "files" in structure, "Structure should contain files"
        assert len(structure["files"]) > 0, "Should find files in workspace"
        
        # Test code file analysis
        python_file = workspace / "main.py"
        analysis = await generator.analyze_code_file(str(python_file))
        assert analysis is not None, "Code analysis should not be None"
        assert len(analysis.functions) > 0, "Should find functions in code"
        assert len(analysis.classes) > 0, "Should find classes in code"
        
        # Test comprehensive documentation generation
        documentation = await generator.generate_comprehensive_documentation(str(workspace))
        assert documentation is not None, "Documentation should not be None"
        assert documentation.title is not None, "Documentation should have a title"
        assert len(documentation.sections) > 0, "Documentation should have sections"
        
        # Test documentation export
        output_path = await generator.export_documentation(
            documentation=documentation,
            output_path=str(workspace),
            format="markdown"
        )
        assert output_path is not None, "Export should return output path"
        assert Path(output_path).exists(), "Exported file should exist"

async def test_api_documentation_generator():
    """Test API documentation generation"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create sample FastAPI application
        (workspace / "api.py").write_text("""
from fastapi import FastAPI

app = FastAPI(title="Test API", version="1.0.0")

@app.get("/")
async def root():
    '''Root endpoint'''
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    '''Get user by ID'''
    return {"id": user_id, "name": "John Doe"}

@app.post("/users")
async def create_user(user_data: dict):
    '''Create a new user'''
    return {"message": "User created", "user": user_data}
""")
        
        generator = APIDocumentationGenerator()
        
        # Test API discovery
        apis = await generator.discover_apis(str(workspace))
        assert apis is not None, "API discovery should return results"
        assert len(apis) > 0, "Should discover at least one API"
        
        api = apis[0]
        assert api.name is not None, "API should have a name"
        assert len(api.endpoints) > 0, "API should have endpoints"
        
        # Test endpoint analysis
        endpoints = await generator.analyze_fastapi_endpoints(str(workspace / "api.py"))
        assert endpoints is not None, "Endpoint analysis should return results"
        assert len(endpoints) >= 3, "Should find at least 3 endpoints"
        
        # Test OpenAPI spec generation
        openapi_spec = await generator.generate_openapi_spec(api)
        assert openapi_spec is not None, "OpenAPI spec should not be None"
        assert "openapi" in openapi_spec, "Should contain OpenAPI version"
        assert "info" in openapi_spec, "Should contain API info"
        assert "paths" in openapi_spec, "Should contain API paths"

async def test_tutorial_generator():
    """Test interactive tutorial generation"""
    
    generator = TutorialGenerator()
    
    # Test basic tutorial generation
    tutorial = await generator.generate_tutorial(
        topic="API Integration",
        tutorial_type=TutorialType.QUICK_START,
        difficulty_level=DifficultyLevel.BEGINNER,
        learning_style=LearningStyle.HANDS_ON
    )
    
    assert tutorial is not None, "Tutorial should not be None"
    assert tutorial.title is not None, "Tutorial should have a title"
    assert "API Integration" in tutorial.title, "Tutorial title should contain topic"
    assert tutorial.tutorial_type == TutorialType.QUICK_START, "Tutorial type should match"
    assert tutorial.difficulty_level == DifficultyLevel.BEGINNER, "Difficulty should match"
    assert len(tutorial.steps) > 0, "Tutorial should have steps"
    assert len(tutorial.learning_objectives) > 0, "Tutorial should have learning objectives"
    
    # Test compliance tutorial
    compliance_tutorial = await generator.generate_tutorial(
        topic="Compliance Validation",
        tutorial_type=TutorialType.COMPLIANCE_TUTORIAL,
        difficulty_level=DifficultyLevel.INTERMEDIATE
    )
    
    assert compliance_tutorial is not None, "Compliance tutorial should not be None"
    assert "compliance" in compliance_tutorial.title.lower(), "Should be compliance tutorial"
    assert len(compliance_tutorial.steps) > 0, "Should have steps"
    
    # Test security tutorial
    security_tutorial = await generator.generate_tutorial(
        topic="Security Testing",
        tutorial_type=TutorialType.SECURITY_TUTORIAL,
        difficulty_level=DifficultyLevel.ADVANCED
    )
    
    assert security_tutorial is not None, "Security tutorial should not be None"
    assert "security" in security_tutorial.title.lower(), "Should be security tutorial"
    assert len(security_tutorial.steps) > 0, "Should have steps"

async def test_documentation_analytics():
    """Test documentation analytics"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        
        # Create sample documentation files
        (docs_dir / "good_doc.md").write_text("""
# Good Documentation

## Introduction
This is a well-structured document with proper headings.

## Usage
Here's how to use this feature:

```python
# Example code
def example():
    return "Hello World"
```

## Examples
Multiple examples are provided here.

## API Reference
Detailed API documentation.
""")
        
        (docs_dir / "poor_doc.md").write_text("""
# Poor Doc
Some text without structure.
No examples or proper sections.
TODO: Add more content
FIXME: Fix this section
""")
        
        temp_docs = [str(f) for f in docs_dir.glob("*.md")]
        analytics = DocumentationAnalytics()
        
        # Test quality analysis report
        quality_report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.QUALITY_ANALYSIS,
            content_paths=temp_docs
        )
        
        assert quality_report is not None, "Quality report should not be None"
        assert quality_report.analytics_type == AnalyticsType.QUALITY_ANALYSIS, "Report type should match"
        assert "overall_quality_score" in quality_report.summary, "Should have overall quality score"
        
        # Test usage patterns report
        usage_report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.USAGE_PATTERNS,
            time_period=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        assert usage_report is not None, "Usage report should not be None"
        assert usage_report.analytics_type == AnalyticsType.USAGE_PATTERNS, "Report type should match"
        assert "total_page_views" in usage_report.summary, "Should have page views data"
        
        # Test content gaps report
        gaps_report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.CONTENT_GAPS,
            time_period=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        assert gaps_report is not None, "Gaps report should not be None"
        assert gaps_report.analytics_type == AnalyticsType.CONTENT_GAPS, "Report type should match"
        assert "total_gaps_identified" in gaps_report.summary, "Should have gaps data"

async def test_integration_scenario():
    """Test end-to-end integration scenario"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create sample project with API
        (workspace / "main.py").write_text("""
from fastapi import FastAPI

app = FastAPI(title="Integration Test API")

@app.get("/")
def root():
    '''Root endpoint for testing'''
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    '''Health check endpoint'''
    return {"status": "healthy"}
""")
        
        (workspace / "README.md").write_text("""
# Integration Test Project
API project for testing complete documentation workflow
""")
        
        # Test ecosystem documentation
        eco_generator = DocumentationGenerator()
        eco_docs = await eco_generator.generate_comprehensive_documentation(str(workspace))
        assert eco_docs is not None, "Ecosystem docs should be generated"
        
        # Test API documentation
        api_generator = APIDocumentationGenerator()
        apis = await api_generator.discover_apis(str(workspace))
        assert len(apis) > 0, "Should discover APIs"
        
        # Test tutorial generation
        tutorial_generator = TutorialGenerator()
        tutorial = await tutorial_generator.generate_tutorial(
            topic="API Integration",
            tutorial_type=TutorialType.QUICK_START,
            difficulty_level=DifficultyLevel.BEGINNER
        )
        assert tutorial is not None, "Tutorial should be generated"
        
        # Test documentation analytics
        analytics = DocumentationAnalytics()
        
        # Export ecosystem docs first to create files for analysis
        eco_output = await eco_generator.export_documentation(
            documentation=eco_docs,
            output_path=str(workspace),
            format="markdown"
        )
        
        quality_report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.QUALITY_ANALYSIS,
            content_paths=[eco_output]
        )
        assert quality_report is not None, "Quality report should be generated"

async def run_all_tests():
    """Run all tests for the AI documentation engine"""
    
    print("🤖 Initializing AI Documentation Engine...")
    print("🧪 Running AI Documentation Engine Comprehensive Tests")
    print("=" * 60)
    
    runner = SimpleTestRunner()
    
    # Define test cases
    test_cases = [
        ("Ecosystem Documentation Generator", test_ecosystem_documentation_generator),
        ("API Documentation Generator", test_api_documentation_generator),
        ("Tutorial Generator", test_tutorial_generator),
        ("Documentation Analytics", test_documentation_analytics),
        ("Integration Scenario", test_integration_scenario)
    ]
    
    # Run all tests
    for test_name, test_func in test_cases:
        print(f"\n📋 Testing {test_name}")
        print("-" * 40)
        await runner.run_test(test_name, test_func)
    
    # Print summary and return result
    success = runner.print_summary()
    
    if success:
        print("\n🎉 Phase 6 AI Documentation Engine implementation completed successfully!")
        print("   🔧 All core components are functional and tested")
        print("   📚 Ecosystem documentation generation: ✅")
        print("   🔌 API documentation generation: ✅") 
        print("   🎓 Interactive tutorial creation: ✅")
        print("   📊 Documentation analytics: ✅")
        print("   🔗 Integration capabilities: ✅")
    
    return success

if __name__ == "__main__":
    # Run all tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🚀 Ready to proceed with Phase 7+ implementation!")
    else:
        print("\n⚠️ Please address test failures before proceeding.")
    
    exit(0 if success else 1)
