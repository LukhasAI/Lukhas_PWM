# MEMORY Module

## Overview
Memory management, storage, and retrieval systems

## Module Structure
- **Python Files**: 156
- **Submodules**: 11
- **Last Updated**: 2025-07-28

## Directory Structure
```
memory/
├── adapters/
├── compression/
├── convergence/
├── docs/
├── episodic/
├── governance/
├── memory_systems/
├── protection/
├── repair/
├── systems/
└── ... (1 more directories)
```

## Internal Dependencies
- `bio`
- `bridge`
- `consciousness`
- `core`
- `ethics`
- `features`
- `identity`
- `learning`
- `memory`
- `symbolic`

## ⚠️ Import Issues to Fix
- lukhas_replayer.py: Old import 'symbolic_ai.personas.lukhas.lukhas_visualizer'
- lukhas_dreams.py: Old import 'symbolic_ai.personas.lukhas.memory.lukhas_memory'

## Usage

```python
# Import example
from memory import ...

# Module initialization
# Add specific examples based on module content
```

## Key Components

### `enhanced_memory_manager`
Module containing enhanced memory manager functionality.

### `service`
Module containing service functionality.

### `dream_memory_manager`
Module containing dream memory manager functionality.

### `voice_memory_manager`
Module containing voice memory manager functionality.

### `memory_manager`
Module containing memory manager functionality.

## Development

### Running Tests
```bash
pytest tests/test_{module_name}*.py
```

### Adding New Features
1. Create new module file in appropriate subdirectory
2. Update `__init__.py` exports
3. Add tests in `tests/` directory
4. Update this README

## Related Documentation
- [Main README](../README.md)
- [Architecture Overview](../docs/architecture.md)
- [API Reference](../docs/api_reference.md)
