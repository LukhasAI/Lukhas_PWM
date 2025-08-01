# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LUKHAS PWM (Pack-What-Matters) is a lean, production-ready AGI system that distills essential LUKHAS components into working, functional modules. It combines consciousness, memory, identity, quantum processing, biological adaptation, and ethical governance to create a sophisticated AI research platform. The codebase achieves 99.9% system connectivity with comprehensive Guardian System protection.

## Common Development Tasks

### Building and Running

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing

# Set up environment
cp .env.example .env
# Edit .env with OpenAI API key and other settings

# Run main orchestrator
python main.py

# Run specific modules
python orchestration/brain/primary_hub.py
python consciousness/unified/auto_consciousness.py
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
python test_governance.py              # Basic governance tests
python test_enhanced_governance.py     # Enhanced governance tests
python test_comprehensive_governance.py # Full Guardian Reflector suite

# Run with coverage
pytest --cov=lukhas tests/

# Run functional analysis
python PWM_FUNCTIONAL_ANALYSIS.py
python PWM_OPERATIONAL_SUMMARY.py
```

### Code Quality

```bash
# Format code
black lukhas/ tests/

# Linting
flake8 lukhas/

# Type checking
mypy lukhas/
```

## High-Level Architecture

### System Organization

The codebase follows a modular architecture with 53 root systems organized into functional domains:

**Core Infrastructure:**
- `core/` - GLYPH engine, symbolic processing, system foundations (396 files)
- `orchestration/` - Brain integration, multi-agent coordination (349 files)
- `governance/` - Guardian System v1.0.0 ethical oversight

**Cognitive Systems:**
- `consciousness/` - Awareness, reflection, decision-making (55 files, 70.9% functional)
- `memory/` - Fold-based memory with causal chains (222 files, 72.1% functional)
- `reasoning/` - Logic, causal inference, goal processing (117 files)
- `identity/` - Quantum-resistant auth, tier access (147 files, 66.0% functional)

**Advanced Processing:**
- `quantum/` - Quantum-inspired algorithms (93 files, 82.8% functional)
- `bio/` - Biological adaptation systems (49 files, 65.3% functional)
- `emotion/` - VAD affect, mood regulation (26 files, 64.7% functional)
- `creativity/` - Dream engine with 40+ modules (89 files)

**Integration:**
- `api/` - FastAPI endpoints (12 files, 33.3% functional)
- `bridge/` - External API connections (98 files)
- `ethics/` - Multi-tiered policy engines (17 files)

### Key Design Patterns

1. **Symbolic Communication**: All modules use GLYPH tokens for cross-module messaging
2. **Guardian Protection**: Every operation validated by ethics engine
3. **Fold-Based Memory**: Preserves causal chains and emotional context
4. **Dream-Driven Innovation**: Creative solutions from controlled chaos
5. **Bio-Symbolic Coherence**: 102.22% coherence between biological and symbolic systems

### Module Communication

Critical dependencies:
- All modules depend on `core/` for GLYPH processing
- `orchestration/brain/` coordinates cross-module actions
- `governance/` validates all operations
- `memory/` provides persistence across modules
- Circular dependencies strictly avoided

## Guardian System Integration

The Guardian System v1.0.0 provides comprehensive ethical oversight:

### Protection Features
- **Remediator Agent**: Symbolic immune system for threat detection
- **Reflection Layer**: Ethical reasoning for all operations
- **Symbolic Firewall**: Multi-layered security protection
- **Drift Detection**: Monitors for ethical and behavioral drift
- **Decision Justification**: Philosophical reasoning engine

### Testing with Guardian Reflector
- Multi-framework moral reasoning (virtue ethics, deontological, consequentialist)
- SEEDRA-v3 model for deep ethical analysis
- Real-time consciousness protection
- Comprehensive enterprise-grade validation

## Development Guidelines

### Adding Features

1. **Choose Module**: Select appropriate module based on functionality
2. **Use GLYPHs**: Always use symbolic tokens for cross-module communication
3. **Ethics First**: All features must pass Guardian System validation
4. **Memory Integration**: Significant actions create memory folds
5. **Test Coverage**: Aim for 98%+ coverage (like dream engine)

### Working with Specific Modules

**Memory System (`memory/`)**
- Use fold-based architecture
- Preserve emotional context
- 99.7% cascade prevention rate
- Close memory folds after use

**Quantum Processing (`quantum/`)**
- 82.8% functional - most reliable module
- Post-quantum cryptography ready
- Multi-stage transformation pipelines

**API Module (`api/`)**
- Only 33.3% functional - needs improvement
- FastAPI-based endpoints
- Focus on stability over features

**Orchestration (`orchestration/`)**
- 60.5% functional - refactoring needed
- Brain hub coordinates all modules
- Avoid tight coupling

### Security Considerations

- All inputs validated through `bridge/`
- Identity module handles authentication
- Ethics engine cannot be bypassed
- Complete audit trail with causality chains
- Quantum-resistant cryptography throughout

## Common Pitfalls

1. **Never bypass Guardian System** - All operations must be validated
2. **Always use GLYPHs** - Direct module communication breaks symbolic unity
3. **Monitor drift scores** - Respond to behavioral changes
4. **Close memory folds** - Prevent memory leaks
5. **Check functional status** - Some modules only partially operational

## Debugging Tips

1. **Check Trace Logs**: `trace/` directory for system-wide debugging
2. **Monitor Drift**: Watch `drift_score` in governance metrics
3. **Memory Visualization**: Use fold visualizers for memory debugging
4. **Ethics Violations**: Check Guardian System logs
5. **GLYPH Translation**: Use symbolic parser for token debugging
6. **Functional Analysis**: Run `PWM_FUNCTIONAL_ANALYSIS.py` for module status

## Production Deployment

The system includes 3 microservice deployments:
- Consciousness Platform (`deployments/consciousness_platform/`)
- Dream Commerce (`deployments/dream_commerce/`)
- Memory Services (`deployments/memory_services/`)

Each service is Docker-containerized with FastAPI endpoints.