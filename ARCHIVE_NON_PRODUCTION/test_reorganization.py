#!/usr/bin/env python3
"""
LUKHAS Reorganization Test Suite
Verifies critical functionality after reorganization
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

class ReorganizationTester:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'errors': []
        }
        
        # Critical imports to test
        self.critical_imports = [
            # Dream system
            ('dream.core.dream_utils', 'Core dream utilities'),
            ('dream.core.dream_cli', 'Dream CLI'),
            ('dream.core.nias_dream_bridge', 'Dream bridge'),
            ('dream.core.dream_memory_manager', 'Dream memory manager'),
            ('dream.engine.dream_engine', 'Dream engine'),
            ('dream.visualization.dream_log_viewer', 'Dream visualization'),
            
            # Memory system
            ('memory.core.quantum_memory_manager', 'Quantum memory manager'),
            ('memory.core.base_manager', 'Base memory manager'),
            ('memory.fold_system.enhanced_memory_fold', 'Memory fold system'),
            ('memory.episodic.episodic_memory', 'Episodic memory'),
            ('memory.consolidation.memory_consolidation', 'Memory consolidation'),
            
            # Personality system
            ('lukhas_personality.brain.brain', 'LUKHAS brain'),
            ('lukhas_personality.voice.voice_narrator', 'Voice narrator'),
            ('lukhas_personality.creative_core.creative_core', 'Creative core'),
            ('lukhas_personality.narrative_engine.dream_narrator_queue', 'Dream narrator'),
        ]
        
        # Test basic functionality
        self.functionality_tests = [
            ('test_dream_import_chain', self.test_dream_import_chain),
            ('test_memory_import_chain', self.test_memory_import_chain),
            ('test_personality_preservation', self.test_personality_preservation),
            ('test_cross_module_imports', self.test_cross_module_imports),
        ]

    def run_all_tests(self):
        """Run all tests and report results"""
        print("🧪 LUKHAS Reorganization Test Suite")
        print("=" * 50)
        
        # Test critical imports
        print("\n📦 Testing Critical Imports...")
        self._test_imports()
        
        # Test functionality
        print("\n🔧 Testing Functionality...")
        self._test_functionality()
        
        # Report results
        self._report_results()

    def _test_imports(self):
        """Test that critical modules can be imported"""
        for module_path, description in self.critical_imports:
            try:
                importlib.import_module(module_path)
                self.results['passed'].append(f"✅ Import: {module_path} - {description}")
            except ImportError as e:
                self.results['failed'].append(f"❌ Import: {module_path} - {description}")
                self.results['errors'].append(f"  Error: {str(e)}")
            except Exception as e:
                self.results['failed'].append(f"💥 Import: {module_path} - {description}")
                self.results['errors'].append(f"  Unexpected error: {str(e)}")

    def _test_functionality(self):
        """Test basic functionality"""
        for test_name, test_func in self.functionality_tests:
            try:
                result = test_func()
                if result:
                    self.results['passed'].append(f"✅ Function: {test_name}")
                else:
                    self.results['failed'].append(f"❌ Function: {test_name}")
            except Exception as e:
                self.results['failed'].append(f"💥 Function: {test_name}")
                self.results['errors'].append(f"  Error: {str(e)}")
                self.results['errors'].append(f"  Traceback: {traceback.format_exc()}")

    def test_dream_import_chain(self) -> bool:
        """Test that dream modules can import from each other"""
        try:
            # This tests the import chain within dream system
            dream_core = importlib.import_module('dream.core')
            dream_engine = importlib.import_module('dream.engine')
            
            # Check if modules have expected attributes
            return hasattr(dream_core, '__file__') and hasattr(dream_engine, '__file__')
        except:
            return False

    def test_memory_import_chain(self) -> bool:
        """Test that memory modules can import from each other"""
        try:
            # Test memory system imports
            memory_core = importlib.import_module('memory.core')
            memory_episodic = importlib.import_module('memory.episodic')
            
            return True
        except:
            return False

    def test_personality_preservation(self) -> bool:
        """Test that personality files are accessible"""
        try:
            # Check if personality files exist
            personality_path = Path('lukhas_personality')
            
            required_files = [
                'brain/brain.py',
                'voice/voice_narrator.py',
                'creative_core/creative_core.py',
                'narrative_engine/dream_narrator_queue.py'
            ]
            
            for file in required_files:
                if not (personality_path / file).exists():
                    return False
                    
            return True
        except:
            return False

    def test_cross_module_imports(self) -> bool:
        """Test imports between different modules"""
        try:
            # Test that dream can import from memory (common pattern)
            sys.path.insert(0, str(Path.cwd()))
            
            # Create a test module that imports cross-module
            test_code = '''
import dream.core.dream_memory_manager
import memory.core.quantum_memory_manager
result = True
'''
            exec(test_code, {'__name__': '__main__'})
            return True
        except:
            return False

    def _report_results(self):
        """Generate test report"""
        total_tests = len(self.results['passed']) + len(self.results['failed'])
        
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {len(self.results['passed'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        
        if self.results['failed']:
            print("\n❌ Failed Tests:")
            for failure in self.results['failed']:
                print(f"  {failure}")
            
            if self.results['errors']:
                print("\n🔍 Error Details:")
                for error in self.results['errors']:
                    print(f"  {error}")
        
        # Save detailed report
        with open('REORGANIZATION_TEST_REPORT.md', 'w') as f:
            f.write("# LUKHAS Reorganization Test Report\n\n")
            
            f.write(f"## Summary\n")
            f.write(f"- Total Tests: {total_tests}\n")
            f.write(f"- Passed: {len(self.results['passed'])}\n")
            f.write(f"- Failed: {len(self.results['failed'])}\n\n")
            
            if self.results['passed']:
                f.write("## ✅ Passed Tests\n\n")
                for test in self.results['passed']:
                    f.write(f"- {test}\n")
                f.write("\n")
            
            if self.results['failed']:
                f.write("## ❌ Failed Tests\n\n")
                for i, test in enumerate(self.results['failed']):
                    f.write(f"- {test}\n")
                    # Find corresponding error
                    if i < len(self.results['errors']):
                        f.write(f"{self.results['errors'][i]}\n")
                f.write("\n")
            
            f.write("## Next Steps\n\n")
            if self.results['failed']:
                f.write("1. Fix the failed imports by updating import paths\n")
                f.write("2. Ensure all moved files have correct __init__.py files\n")
                f.write("3. Update any hardcoded paths in the codebase\n")
            else:
                f.write("✅ All tests passed! The reorganization was successful.\n")
                f.write("\nNext steps:\n")
                f.write("1. Continue with remaining system consolidation\n")
                f.write("2. Create commercial API abstractions\n")
                f.write("3. Document the new architecture\n")
        
        # Return success/failure
        return len(self.results['failed']) == 0


def main():
    """Run the reorganization tests"""
    tester = ReorganizationTester()
    success = tester.run_all_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Reorganization successful.")
    else:
        print("❌ Some tests failed. Check REORGANIZATION_TEST_REPORT.md for details.")
    print("=" * 50)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()