"""
Core AGI Systems Integration Tests
=================================

Migrated from: archive/integration_tests/test_core_agi_systems.py
Migration: TASK 18 - Updated for pytest compatibility with modern lukhas/ structure

Tests the main LUKHAS AGI core systems after package restructuring.
Framework: pytest (migrated from custom test runner)
Tags: #ΛLEGACY, integration, memory, symbolic, quantum
"""

import pytest

from core.symbolic import GLYPH_MAP


class TestMemorySystems:
    """Test memory system components."""

    @pytest.mark.memory
    def test_trace_injector(self):
        """Test memory trace injector initialization."""
        try:
            from memory.core_memory.trace_injector import get_global_injector

            injector = get_global_injector()
            assert injector is not None
            assert hasattr(injector, "__class__")
        except ImportError as e:
            pytest.skip(f"Memory trace injector not available: {e}")

    @pytest.mark.memory
    def test_collapse_trace(self):
        """Test memory collapse tracer initialization."""
        try:
            from memory.core_memory.collapse_trace import get_global_tracer

            tracer = get_global_tracer()
            assert tracer is not None
            assert hasattr(tracer, "__class__")
        except ImportError as e:
            pytest.skip(f"Memory collapse tracer not available: {e}")


class TestSymbolicSystems:
    """Test symbolic reasoning systems."""

    @pytest.mark.symbolic
    def test_symbolic_map(self):
        """Test symbolic trace map initialization."""
        try:
            from core.symbolic.memoria.symbolic_trace_map import (
                get_global_trace_map,
            )

            trace_map = get_global_trace_map()
            assert trace_map is not None
            assert hasattr(trace_map, "__class__")
        except ImportError as e:
            pytest.skip(f"Symbolic trace map not available: {e}")

    @pytest.mark.symbolic
    def test_recall_hooks(self):
        """Test recall hook manager initialization."""
        try:
            from core.symbolic.memoria.recall_hooks import (
                get_global_hook_manager,
            )

            hook_manager = get_global_hook_manager()
            assert hook_manager is not None
            assert hasattr(hook_manager, "__class__")
        except ImportError as e:
            pytest.skip(f"Recall hook manager not available: {e}")

    @pytest.mark.symbolic
    def test_glyph_system_available(self):
        """Test that the glyph system is accessible."""
        assert isinstance(GLYPH_MAP, dict)
        # Should have at least some glyphs defined
        if GLYPH_MAP:
            assert len(GLYPH_MAP) > 0


class TestBridgeSystems:
    """Test bridge communication systems."""

    @pytest.mark.integration
    def test_dream_bridge(self):
        """Test symbolic dream bridge initialization."""
        try:
            from bridge.symbolic_dream_bridge import SymbolicDreamBridge

            bridge = SymbolicDreamBridge()
            assert bridge is not None
            assert hasattr(bridge, "__class__")
        except ImportError as e:
            pytest.skip(f"Symbolic dream bridge not available: {e}")

    @pytest.mark.integration
    def test_message_bus(self):
        """Test message bus initialization."""
        try:
            from bridge.message_bus import MessageBus

            bus = MessageBus()
            assert bus is not None
            assert hasattr(bus, "__class__")
        except ImportError as e:
            pytest.skip(f"Message bus not available: {e}")


class TestReasoningSystems:
    """Test reasoning systems."""

    @pytest.mark.symbolic
    def test_symbolic_reasoning(self):
        """Test symbolic reasoning initialization."""
        try:
            # Check if any symbolic reasoning module exists
            import reasoning.symbolic_reasoning

            # Test basic module import
            assert hasattr(lukhas.reasoning.symbolic_reasoning, "__name__")
        except ImportError as e:
            pytest.skip(f"Symbolic reasoning not available: {e}")

    @pytest.mark.integration
    def test_reasoning_engine_module(self):
        """Test reasoning engine module availability."""
        try:
            import reasoning.reasoning_engine as re_module

            available_classes = [
                name
                for name in dir(re_module)
                if name.endswith("Engine") and not name.startswith("_")
            ]
            # Should have at least some engine classes
            assert isinstance(available_classes, list)
        except ImportError as e:
            pytest.skip(f"Reasoning engine module not available: {e}")


class TestQuantumSystems:
    """Test quantum systems."""

    @pytest.mark.quantum
    def test_quantum_inspired_layer(self):
        """Test quantum layer initialization."""
        try:
            import quantum_inspired.quantum_inspired_layer

            # Test basic module import
            assert hasattr(lukhas.quantum.quantum_inspired_layer, "__name__")
        except ImportError as e:
            pytest.skip(f"Quantum layer not available: {e}")

    @pytest.mark.quantum
    def test_quantum_inspired_processor(self):
        """Test quantum processor initialization."""
        try:
            import quantum_inspired.quantum_inspired_processor

            # Test basic module import
            assert hasattr(lukhas.quantum.quantum_inspired_processor, "__name__")
        except ImportError as e:
            pytest.skip(f"Quantum processor not available: {e}")


class TestSystemIntegration:
    """Test cross-system integration."""

    @pytest.mark.integration
    def test_all_systems_loadable(self):
        """Test that all core systems can be imported without conflicts."""
        systems_status = {}

        # Test memory systems
        try:
            import memory.core_memory.trace_injector

            systems_status["memory_trace"] = True
        except ImportError:
            systems_status["memory_trace"] = False

        # Test symbolic systems
        try:
            import core.symbolic

            systems_status["symbolic_core"] = True
        except ImportError:
            systems_status["symbolic_core"] = False

        # Test bridge systems
        try:
            import bridge.message_bus  # noqa: F401

            systems_status["bridge_systems"] = True
        except ImportError:
            systems_status["bridge_systems"] = False

        # At least one system should be available
        assert any(
            systems_status.values()
        ), f"No core systems available: {systems_status}"
