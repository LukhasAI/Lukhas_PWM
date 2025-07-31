# LUKHAS Unified Grammar Test Suite

## Overview

This test suite validates the implementation of the LUKHAS Unified Grammar v1.0.0, ensuring all modules comply with the standardized framework and patterns.

## Test Categories

### 1. Base Module Tests (`test_base_module.py`)
Tests the foundational `BaseLukhasModule` class that all modules extend:
- Module initialization and lifecycle management
- State transitions (stopped → running → stopped)
- Health status reporting
- Configuration management
- Tier-based access control
- Symbolic logging functionality

### 2. Symbolic Vocabulary Tests (`test_symbolic_vocabulary.py`)
Validates the symbolic vocabulary system:
- Vocabulary schema compliance
- Symbol uniqueness across vocabularies
- Helper function correctness
- Vocabulary completeness for each module
- JSON serialization capability
- Documentation existence

### 3. Module Integration Tests (`test_module_integration.py`)
Tests inter-module communication and integration:
- Multi-module startup sequences
- Module communication patterns
- Registry operations
- Error handling and recovery
- Performance metrics collection
- Security and isolation

### 4. Grammar Compliance Tests (`test_grammar_compliance.py`)
Ensures all modules follow Unified Grammar specifications:
- Directory structure compliance
- Required method implementation
- Naming convention adherence
- Configuration pattern compliance
- Documentation standards
- Example code validation

## Running Tests

### Run All Tests
```bash
cd /Users/agi_dev/Downloads/Consolidation-Repo/tests/unified_grammar
python run_tests.py
```

### Run Specific Test Category
```bash
python run_tests.py --test test_base_module.py
python run_tests.py --test test_symbolic_vocabulary.py
python run_tests.py --test test_module_integration.py
python run_tests.py --test test_grammar_compliance.py
```

### Quick Smoke Test
```bash
python run_tests.py --quick
```

### Generate Coverage Report
```bash
python run_tests.py --coverage
```

## Test Requirements

The tests require the following to be properly set up:
- Python 3.8+
- pytest
- pytest-asyncio
- pytest-cov (for coverage reports)
- Access to `/lukhas_unified_grammar/` modules
- Access to `/lukhas/symbolic/vocabularies/`

## Expected Results

All tests should pass for a compliant Unified Grammar implementation:

```
✅ PASSED - Base Module Tests
✅ PASSED - Symbolic Vocabulary Tests  
✅ PASSED - Module Integration Tests
✅ PASSED - Grammar Compliance Tests
```

## Adding New Tests

When adding new modules to the Unified Grammar:

1. **Update Grammar Compliance Tests**: Add the new module to the required modules list
2. **Add Vocabulary Tests**: Include vocabulary validation for the new module
3. **Update Integration Tests**: Add integration scenarios with other modules
4. **Create Module-Specific Tests**: Add tests for module-specific functionality

## Continuous Integration

These tests should be run:
- Before merging any changes to Unified Grammar modules
- When adding new modules
- When updating vocabulary definitions
- As part of the regular CI/CD pipeline

## Known Limitations

- Module registry tests are currently skipped pending full implementation
- Some integration tests use mocked inter-module communication
- Performance tests have generous timeout thresholds for CI environments

## Troubleshooting

### Import Errors
Ensure your Python path includes the repository root:
```bash
export PYTHONPATH=/Users/agi_dev/Downloads/Consolidation-Repo:$PYTHONPATH
```

### Missing Dependencies
Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov
```

### Permission Errors
Some tests may create temporary files. Ensure write permissions in the test directory.

---

*LUKHAS Unified Grammar Test Suite v1.0.0*
*Last Updated: 2025-07-25*