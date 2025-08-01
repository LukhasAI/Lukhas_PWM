# TOOLS Module

## Overview
Development tools and utilities

## Module Structure
- **Python Files**: 28
- **Submodules**: 5
- **Last Updated**: 2025-07-28

## Directory Structure
```
tools/
├── cli/
├── dev/
├── parsers/
├── prediction/
├── vision/
```

## Internal Dependencies
- `core`
- `ethics`
- `tools`
- `trace`

## ⚠️ Import Issues to Fix
- import_path_fixer.py: Old import 'lukhas.{service_pattern}'

## Usage

```python
# Import example
from tools import ...

# Module initialization
# Add specific examples based on module content
```

## Key Components

### `digest_extractor`
Module containing digest extractor functionality.

### `symbolic_cli_test`
Module containing symbolic cli test functionality.

### `test_knowledge_integration`
Module containing test knowledge integration functionality.

### `fix_lukhas_headers`
Module containing fix lukhas headers functionality.

### `cleanup_and_organize`
Module containing cleanup and organize functionality.

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
