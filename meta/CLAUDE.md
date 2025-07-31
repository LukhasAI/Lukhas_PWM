# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LUKHAS is a symbolic cognition prototype exploring AGI-adjacent functionality through modular consciousness architecture. It combines symbolic reasoning, emotional intelligence, quantum-inspired processing, and ethical governance to create an experimental platform for researching advanced AI capabilities. The codebase is organized into 13+ specialized modules, each exploring different aspects of cognitive processing and intelligent behavior.

## Common Development Tasks

### Building and Installing

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install LUKHAS package in development mode
pip install -e .

# Set up OpenAI integration (required for LLM features)
export OPENAI_API_KEY="your-api-key-here"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests for a specific module
pytest tests/memory/
pytest tests/core/

# Run with coverage
pytest --cov=lukhas tests/

# Run specific test markers (from creativity/dream/oneiric_engine)
pytest -m unit      # Fast unit tests
pytest -m integration  # Integration tests
pytest -m pg        # PostgreSQL tests
```

### Code Quality

```bash
# Format code
black lukhas/

# Run linting
flake8 lukhas/

# Type checking
mypy lukhas/

# Run all quality checks
black lukhas/ && flake8 lukhas/ && mypy lukhas/
```

## High-Level Architecture

### Core Philosophy
LUKHAS embodies a "living architecture" where modules interact like organs in a consciousness system. Each module has poetic, verbose documentation that explains both technical implementation and philosophical purpose.

### Module Organization

**Foundation Layer:**
- `core/` - System foundation with GLYPH engine, symbolic processing, drift detection
- `config/` - Configuration management and validators
- `trace/` - System monitoring, drift detection, and causality tracking

**Cognitive Layer:**
- `consciousness/` - Awareness systems with superposition-like state and reflection
- `reasoning/` - Symbolic logic, causal inference, and goal processing
- `learning/` - Meta-adaptive systems and pattern recognition
- `memory/` - Revolutionary fold-based memory with causal archaeology

**Affective Layer:**
- `emotion/` - VAD affect detection, mood regulation, cascade prevention
- `creativity/` - Dream engine with 40+ modules for creative synthesis
- `identity/` - Tier-based access, quantum-resistant auth, cultural awareness

**Integration Layer:**
- `bridge/` - External API connections with unified OpenAI client
- `orchestration/` - Multi-agent coordination and workflow management
- `ethics/` - Multi-tiered policy engines and ethical governance

**Advanced Layer:**
- `quantum/` - Post-quantum cryptography and quantum-inspired processing
- `bio/` - Biological consciousness bridge with oscillator networks
- `symbolic/` - GLYPH-based universal language system

### Key Design Patterns

1. **Symbolic Communication**: All modules communicate through GLYPH tokens - quantum-like symbols that carry compressed meaning
2. **Fold-Based Memory**: Memory is stored in "folds" that preserve causal chains and emotional context
3. **Drift Detection**: Continuous monitoring for ethical and behavioral drift
4. **Emotional Continuity**: Every action carries emotional weight tracked through the system
5. **Dream-Driven Innovation**: Creative solutions emerge from controlled chaos in the dream engine

### Module Dependencies

Critical path modules (required for basic operation):
1. `core/` - Universal foundation
2. `config/` - Configuration management  
3. `memory/` - State persistence
4. `ethics/` - Governance layer
5. `orchestration/` - System coordination

Dependency hierarchy:
- All modules depend on `core/`
- Basic services depend only on core
- Complex services can depend on basic services
- Circular dependencies are strictly avoided

## Development Guidelines

### Adding New Features

1. **Module Selection**: Choose the appropriate module based on functionality
2. **Symbolic Integration**: Use GLYPH tokens for cross-module communication
3. **Ethical Validation**: All new features must pass through ethics engine
4. **Memory Integration**: Significant actions should create memory folds
5. **Documentation**: Write poetic, comprehensive documentation matching existing style

### Testing Strategy

1. **Unit Tests**: Fast, isolated tests for individual components
2. **Integration Tests**: Cross-module interaction testing
3. **Ethical Tests**: Ensure compliance with governance policies
4. **Drift Tests**: Monitor for behavioral changes
5. **Dream Tests**: Validate creative output quality

### Security Considerations

- All external inputs go through `bridge/` validation
- Authentication handled by `identity/` module
- Ethical constraints enforced by `ethics/` engine
- Quantum-resistant cryptography in preparation
- Complete audit logging with causality chains

## Important Files and Locations

- **Module Documentation**: Each module has comprehensive README.md
- **Configuration**: `config/` directory and module-specific config files
- **Tests**: `tests/` directory mirrors module structure
- **Utilities**: `tools/` directory for development utilities
- **Monitoring**: `trace/` for system health and drift detection

## Common Pitfalls to Avoid

1. **Bypassing Ethics Engine**: Never skip ethical validation for performance
2. **Breaking Symbolic Unity**: Always use GLYPHs for cross-module communication
3. **Ignoring Drift**: Monitor and respond to drift detection warnings
4. **Memory Leaks**: Properly close memory folds after use
5. **Circular Dependencies**: Maintain strict dependency hierarchy

## Module-Specific Notes

### Dream Engine (`creativity/dream/`)
- Complex 40+ module system for creative synthesis
- Uses hyperspace navigation for solution exploration
- Test coverage target: 98%+

### Memory System (`memory/`)
- Fold-based architecture preserves causality
- Emotional context stored alongside facts
- 99.7% cascade prevention rate

### Ethics Engine (`ethics/`)
- Multi-tiered policy validation
- Drift governance and detection
- Cannot be bypassed or disabled

### Identity System (`identity/`)
- Quantum-resistant authentication
- Tier-based access control
- Cultural awareness features

## Debugging Tips

1. **Trace Logs**: Check `trace/` for system-wide debugging
2. **Drift Metrics**: Monitor `drift_score` for behavioral changes
3. **Memory Visualization**: Use memory fold visualizers
4. **Ethics Violations**: Check ethics engine logs for constraints
5. **GLYPH Translation**: Use symbolic parser for GLYPH debugging