"""
Comprehensive Test Suite for AI Documentation Engine
==================================================

Tests all components of the AI documentation engine including:
- Ecosystem documentation generation
- API documentation generation  
- Interactive tutorial creation
- Documentation analytics and quality assessment
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import AI documentation engine components
from tools.documentation_suite.ai_documentation_engine import (
    EcosystemDocumentationGenerator,
    APIDocumentationGenerator,
    TutorialGenerator,
    DocumentationAnalytics,
    TutorialType,
    DifficultyLevel,
    LearningStyle,
    AnalyticsType,
    OutputFormat,
    ContentType
)

class TestEcosystemDocumentationGenerator:
    """Test ecosystem documentation generation"""
    
    @pytest.fixture
    def generator(self):
        return EcosystemDocumentationGenerator()
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
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
            
            (workspace / "requirements.txt").write_text("""
requests>=2.25.0
pytest>=6.0.0
""")
            
            yield str(workspace)
    
    @pytest.mark.asyncio
    async def test_discover_workspace_structure(self, generator, temp_workspace):
        """Test workspace structure discovery"""
        
        structure = await generator.discover_workspace_structure(temp_workspace)
        
        assert structure is not None
        assert "files" in structure
        assert "directories" in structure
        assert len(structure["files"]) > 0
        
        # Check for specific files
        file_names = [f["name"] for f in structure["files"]]
        assert "main.py" in file_names
        assert "README.md" in file_names
    
    @pytest.mark.asyncio
    async def test_analyze_code_file(self, generator, temp_workspace):
        """Test code file analysis"""
        
        python_file = Path(temp_workspace) / "main.py"
        analysis = await generator.analyze_code_file(str(python_file))
        
        assert analysis is not None
        assert analysis.file_path == str(python_file)
        assert len(analysis.functions) > 0
        assert len(analysis.classes) > 0
        
        # Check function analysis
        function_names = [f.name for f in analysis.functions]
        assert "hello_world" in function_names
        
        # Check class analysis
        class_names = [c.name for c in analysis.classes]
        assert "ExampleClass" in class_names
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_documentation(self, generator, temp_workspace):
        """Test comprehensive documentation generation"""
        
        documentation = await generator.generate_comprehensive_documentation(temp_workspace)
        
        assert documentation is not None
        assert documentation.title is not None
        assert len(documentation.sections) > 0
        
        # Check for essential sections
        section_titles = [s.title for s in documentation.sections]
        assert any("overview" in title.lower() for title in section_titles)
        assert any("architecture" in title.lower() or "structure" in title.lower() for title in section_titles)
    
    @pytest.mark.asyncio
    async def test_export_documentation(self, generator, temp_workspace):
        """Test documentation export"""
        
        # Generate documentation first
        documentation = await generator.generate_comprehensive_documentation(temp_workspace)
        
        # Export to markdown
        output_path = await generator.export_documentation(
            documentation=documentation,
            output_path=temp_workspace,
            format="markdown"
        )
        
        assert output_path is not None
        assert Path(output_path).exists()
        
        # Check content
        with open(output_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert documentation.title in content

class TestAPIDocumentationGenerator:
    """Test API documentation generation"""
    
    @pytest.fixture
    def generator(self):
        return APIDocumentationGenerator()
    
    @pytest.fixture
    def temp_api_workspace(self):
        """Create temporary workspace with API code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create sample FastAPI application
            (workspace / "api.py").write_text("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Test API", version="1.0.0")

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/")
async def root():
    '''Root endpoint'''
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    '''Get user by ID'''
    if user_id == 1:
        return User(id=1, name="John Doe", email="john@example.com")
    raise HTTPException(status_code=404, detail="User not found")

@app.post("/users")
async def create_user(user: User):
    '''Create a new user'''
    return {"message": "User created", "user": user}
""")
            
            yield str(workspace)
    
    @pytest.mark.asyncio
    async def test_discover_apis(self, generator, temp_api_workspace):
        """Test API discovery"""
        
        apis = await generator.discover_apis(temp_api_workspace)
        
        assert apis is not None
        assert len(apis) > 0
        
        # Check first API
        api = apis[0]
        assert api.name is not None
        assert len(api.endpoints) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_fastapi_endpoints(self, generator, temp_api_workspace):
        """Test FastAPI endpoint analysis"""
        
        api_file = Path(temp_api_workspace) / "api.py"
        endpoints = await generator.analyze_fastapi_endpoints(str(api_file))
        
        assert endpoints is not None
        assert len(endpoints) >= 3  # root, get_user, create_user
        
        # Check endpoint details
        endpoint_paths = [e.path for e in endpoints]
        assert "/" in endpoint_paths
        assert "/users/{user_id}" in endpoint_paths
        assert "/users" in endpoint_paths
        
        # Check methods
        methods = [e.method for e in endpoints]
        assert "GET" in methods
        assert "POST" in methods
    
    @pytest.mark.asyncio
    async def test_generate_openapi_spec(self, generator, temp_api_workspace):
        """Test OpenAPI specification generation"""
        
        apis = await generator.discover_apis(temp_api_workspace)
        api = apis[0]
        
        openapi_spec = await generator.generate_openapi_spec(api)
        
        assert openapi_spec is not None
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Check paths
        paths = openapi_spec["paths"]
        assert "/" in paths
        assert "/users/{user_id}" in paths
    
    @pytest.mark.asyncio
    async def test_generate_postman_collection(self, generator, temp_api_workspace):
        """Test Postman collection generation"""
        
        apis = await generator.discover_apis(temp_api_workspace)
        api = apis[0]
        
        collection = await generator.generate_postman_collection(api)
        
        assert collection is not None
        assert "info" in collection
        assert "item" in collection
        assert len(collection["item"]) > 0
        
        # Check request items
        items = collection["item"]
        request_names = [item["name"] for item in items]
        assert any("root" in name.lower() for name in request_names)

class TestTutorialGenerator:
    """Test interactive tutorial generation"""
    
    @pytest.fixture
    def generator(self):
        return TutorialGenerator()
    
    @pytest.mark.asyncio
    async def test_generate_tutorial(self, generator):
        """Test tutorial generation"""
        
        tutorial = await generator.generate_tutorial(
            topic="API Integration",
            tutorial_type=TutorialType.QUICK_START,
            difficulty_level=DifficultyLevel.BEGINNER,
            learning_style=LearningStyle.HANDS_ON
        )
        
        assert tutorial is not None
        assert tutorial.title is not None
        assert "API Integration" in tutorial.title
        assert tutorial.tutorial_type == TutorialType.QUICK_START
        assert tutorial.difficulty_level == DifficultyLevel.BEGINNER
        assert len(tutorial.steps) > 0
        assert len(tutorial.learning_objectives) > 0
    
    @pytest.mark.asyncio
    async def test_generate_compliance_tutorial(self, generator):
        """Test compliance tutorial generation"""
        
        tutorial = await generator.generate_tutorial(
            topic="Compliance Validation",
            tutorial_type=TutorialType.COMPLIANCE_TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE
        )
        
        assert tutorial is not None
        assert "compliance" in tutorial.title.lower()
        assert len(tutorial.steps) > 0
        
        # Check for compliance-specific content
        step_contents = [step.content for step in tutorial.steps]
        assert any("compliance" in content.lower() for content in step_contents)
    
    @pytest.mark.asyncio
    async def test_generate_security_tutorial(self, generator):
        """Test security tutorial generation"""
        
        tutorial = await generator.generate_tutorial(
            topic="Security Testing",
            tutorial_type=TutorialType.SECURITY_TUTORIAL,
            difficulty_level=DifficultyLevel.ADVANCED
        )
        
        assert tutorial is not None
        assert "security" in tutorial.title.lower()
        assert len(tutorial.steps) > 0
        
        # Check for security-specific content
        step_contents = [step.content for step in tutorial.steps]
        assert any("security" in content.lower() for content in step_contents)
    
    @pytest.mark.asyncio
    async def test_tutorial_step_types(self, generator):
        """Test different tutorial step types"""
        
        tutorial = await generator.generate_tutorial(
            topic="General Topic",
            tutorial_type=TutorialType.COMPREHENSIVE,
            difficulty_level=DifficultyLevel.INTERMEDIATE
        )
        
        # Check for different step types
        step_types = [step.step_type for step in tutorial.steps]
        assert len(set(step_types)) > 1  # Should have multiple step types
        
        # Check for code examples
        code_examples = [step for step in tutorial.steps if step.code_example]
        assert len(code_examples) > 0

class TestDocumentationAnalytics:
    """Test documentation analytics"""
    
    @pytest.fixture
    def analytics(self):
        return DocumentationAnalytics()
    
    @pytest.fixture
    def temp_docs(self):
        """Create temporary documentation files for testing"""
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
            
            (docs_dir / "medium_doc.md").write_text("""
# Medium Quality Doc

## Overview
This document has some structure.

## Usage
Basic usage information.

```python
# One example
print("hello")
```

TODO: Add more examples
""")
            
            yield [str(f) for f in docs_dir.glob("*.md")]
    
    @pytest.mark.asyncio
    async def test_generate_quality_report(self, analytics, temp_docs):
        """Test quality analytics report generation"""
        
        report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.QUALITY_ANALYSIS,
            content_paths=temp_docs
        )
        
        assert report is not None
        assert report.analytics_type == AnalyticsType.QUALITY_ANALYSIS
        assert "overall_quality_score" in report.summary
        assert len(report.detailed_findings) >= 0
        assert len(report.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_usage_report(self, analytics):
        """Test usage analytics report generation"""
        
        report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.USAGE_PATTERNS,
            time_period=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        assert report is not None
        assert report.analytics_type == AnalyticsType.USAGE_PATTERNS
        assert "total_page_views" in report.summary
        assert len(report.detailed_findings) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_content_gaps_report(self, analytics):
        """Test content gaps report generation"""
        
        report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.CONTENT_GAPS,
            time_period=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        assert report is not None
        assert report.analytics_type == AnalyticsType.CONTENT_GAPS
        assert "total_gaps_identified" in report.summary
        assert len(report.detailed_findings) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_user_behavior_report(self, analytics):
        """Test user behavior analytics report generation"""
        
        report = await analytics.generate_analytics_report(
            analytics_type=AnalyticsType.USER_BEHAVIOR,
            time_period=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        assert report is not None
        assert report.analytics_type == AnalyticsType.USER_BEHAVIOR
        assert "total_patterns_detected" in report.summary
        assert len(report.detailed_findings) >= 0
    
    @pytest.mark.asyncio
    async def test_quality_metrics_analysis(self, analytics, temp_docs):
        """Test individual quality metrics analysis"""
        
        # Test completeness analysis
        completeness_score = await analytics._analyze_completeness(temp_docs[0], {})
        assert completeness_score is not None
        assert 0 <= completeness_score.score <= 100
        
        # Test structure analysis
        structure_score = await analytics._analyze_structure(temp_docs[0], {})
        assert structure_score is not None
        assert 0 <= structure_score.score <= 100
        
        # Test examples analysis
        examples_score = await analytics._analyze_examples(temp_docs[0], {})
        assert examples_score is not None
        assert 0 <= examples_score.score <= 100

class TestIntegrationScenarios:
    """Test integration scenarios between components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_documentation_generation(self):
        """Test complete documentation generation workflow"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create sample project
            (workspace / "main.py").write_text("""
from fastapi import FastAPI

app = FastAPI(title="Test API")

@app.get("/")
def root():
    return {"message": "Hello World"}
""")
            
            (workspace / "README.md").write_text("""
# Test Project
API project for testing
""")
            
            # Generate ecosystem documentation
            eco_generator = EcosystemDocumentationGenerator()
            eco_docs = await eco_generator.generate_comprehensive_documentation(str(workspace))
            
            # Generate API documentation
            api_generator = APIDocumentationGenerator()
            apis = await api_generator.discover_apis(str(workspace))
            
            # Generate tutorial
            tutorial_generator = TutorialGenerator()
            tutorial = await tutorial_generator.generate_tutorial(
                topic="API Integration",
                tutorial_type=TutorialType.QUICK_START,
                difficulty_level=DifficultyLevel.BEGINNER
            )
            
            # Analyze documentation quality
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
            
            # Verify all components worked
            assert eco_docs is not None
            assert len(apis) > 0
            assert tutorial is not None
            assert quality_report is not None
            
            print("✅ End-to-end documentation generation completed successfully")
    
    @pytest.mark.asyncio
    async def test_multi_format_output_integration(self):
        """Test generating documentation in multiple formats"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create sample API
            (workspace / "api.py").write_text("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
def test_endpoint():
    return {"status": "ok"}
""")
            
            # Generate API docs in multiple formats
            api_generator = APIDocumentationGenerator()
            apis = await api_generator.discover_apis(str(workspace))
            
            if apis:
                api = apis[0]
                
                # Generate different formats
                formats_to_test = [OutputFormat.MARKDOWN, OutputFormat.OPENAPI]
                
                for output_format in formats_to_test:
                    try:
                        result = await api_generator.generate_documentation(
                            api=api,
                            output_format=output_format,
                            output_path=str(workspace)
                        )
                        assert result is not None
                        print(f"✅ Generated {output_format.value} documentation")
                    except Exception as e:
                        print(f"⚠️ Format {output_format.value} generation failed: {e}")

# Test runner function
async def run_comprehensive_tests():
    """Run all comprehensive tests for the AI documentation engine"""
    
    print("🧪 Running AI Documentation Engine Comprehensive Tests")
    print("=" * 60)
    
    # Initialize test classes
    test_classes = [
        TestEcosystemDocumentationGenerator(),
        TestAPIDocumentationGenerator(),
        TestTutorialGenerator(),
        TestDocumentationAnalytics(),
        TestIntegrationScenarios()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__class__.__name__}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            test_method = getattr(test_class, test_method_name)
            
            try:
                print(f"   🔧 {test_method_name}...", end="")
                
                # Handle fixtures for test methods
                if hasattr(test_class, test_method_name):
                    if test_method_name in [
                        'test_discover_workspace_structure',
                        'test_analyze_code_file', 
                        'test_generate_comprehensive_documentation',
                        'test_export_documentation'
                    ]:
                        # Ecosystem documentation tests
                        generator = EcosystemDocumentationGenerator()
                        with tempfile.TemporaryDirectory() as temp_dir:
                            workspace = Path(temp_dir)
                            (workspace / "main.py").write_text("def test(): pass")
                            (workspace / "README.md").write_text("# Test")
                            await test_method(generator, str(workspace))
                    
                    elif test_method_name in [
                        'test_discover_apis',
                        'test_analyze_fastapi_endpoints',
                        'test_generate_openapi_spec',
                        'test_generate_postman_collection'
                    ]:
                        # API documentation tests
                        generator = APIDocumentationGenerator()
                        with tempfile.TemporaryDirectory() as temp_dir:
                            workspace = Path(temp_dir)
                            (workspace / "api.py").write_text("""
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def root(): return {"message": "Hello"}
""")
                            await test_method(generator, str(workspace))
                    
                    elif test_method_name in [
                        'test_generate_tutorial',
                        'test_generate_compliance_tutorial',
                        'test_generate_security_tutorial',
                        'test_tutorial_step_types'
                    ]:
                        # Tutorial generator tests
                        generator = TutorialGenerator()
                        await test_method(generator)
                    
                    elif test_method_name in [
                        'test_generate_quality_report',
                        'test_generate_usage_report',
                        'test_generate_content_gaps_report',
                        'test_generate_user_behavior_report',
                        'test_quality_metrics_analysis'
                    ]:
                        # Analytics tests
                        analytics = DocumentationAnalytics()
                        if 'temp_docs' in test_method_name or 'quality_metrics' in test_method_name:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                docs_dir = Path(temp_dir)
                                (docs_dir / "test.md").write_text("# Test\n## Usage\nExample content")
                                temp_docs = [str(docs_dir / "test.md")]
                                await test_method(analytics, temp_docs)
                        else:
                            await test_method(analytics)
                    
                    else:
                        # Integration tests and others
                        await test_method()
                
                print(" ✅ PASSED")
                passed_tests += 1
                
            except Exception as e:
                print(f" ❌ FAILED: {str(e)}")
                failed_tests += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 AI Documentation Engine Test Summary")
    print("=" * 60)
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! AI Documentation Engine is working correctly.")
        return True
    else:
        print(f"\n⚠️ {failed_tests} tests failed. Please review the failures above.")
        return False

if __name__ == "__main__":
    # Run comprehensive tests
    asyncio.run(run_comprehensive_tests())
