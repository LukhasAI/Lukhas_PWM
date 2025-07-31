# Agent 1 Integration Tests

This directory contains integration tests for Agent 1's task implementations.

## Test Organization

Tests are organized by task number with descriptive naming:

- `test_task06_integration.py` - Task 6: [Integration test description]
- `test_task07_integration.py` - Task 7: [Integration test description]  
- `test_task08_integration.py` - Task 8: [Integration test description]
- `test_task09_integration.py` - Task 9: [Integration test description]
- `test_task09_direct.py` - Task 9: Direct test variant
- `test_task10_integration.py` - Task 10: Unified Emotional Memory Manager
- `test_task11_integration.py` - Task 11: [Integration test description]
- `test_task12_integration.py` - Task 12: [Integration test description]
- `test_task13_integration.py` - Task 13: [Integration test description]
- `test_task14_integration.py` - Task 14: TraumaLockSystem Integration
- `test_task14_simple.py` - Task 14: TraumaLockSystem Simple Tests
- `test_task15_integration.py` - Task 15: [Integration test description]
- `test_task16_integration.py` - Task 16: [Integration test description]

## Running Tests

To run all Agent 1 tests:
```bash
pytest tests/agents/agent1/
```

To run specific task tests:
```bash
pytest tests/agents/agent1/test_task10_integration.py
```

## Test Patterns

Each test follows these patterns:
- **Integration Tests**: Test component integration with memory hub and other systems
- **Direct Tests**: Test component functionality directly without external dependencies
- **Simple Tests**: Simplified versions focusing on core functionality

## Priority Scores

Tests include priority scores indicating their importance in the overall system validation.