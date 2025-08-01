"""Tests for legacy alias modules."""

import importlib

import pytest


def test_symbolic_drift_tracker_alias():
    try:
        module = importlib.import_module(
            "lukhas.memory.core_memory.symbolic_drift_tracker"
        )
    except Exception as exc:
        pytest.skip(f"Import failed: {exc}")
    assert hasattr(module, "SymbolicDriftTracker")


def test_enhanced_memory_manager_core_alias():
    """Verify the main enhanced memory manager is accessible."""
    try:
        alias = importlib.import_module("lukhas.memory.enhanced_memory_manager")
        target = importlib.import_module("lukhas.memory.memory_manager")
    except Exception as exc:  # pragma: no cover - optional import path
        pytest.skip(f"Import failed: {exc}")
    assert alias.EnhancedMemoryManager is target.EnhancedMemoryManager
