"""Tests for Agents of Failure simulation."""

from simulation.agents_of_failure import FailureSimulator


def test_entropy_and_collapse_progression():
    sim = FailureSimulator(collapse_threshold=1.0, decay_rate=0.1)
    metrics1 = sim.step({"value": 1})
    metrics2 = sim.step({"value": 2})

    assert metrics2.drift_score >= 0.0
    assert metrics2.entropy < metrics1.entropy + metrics2.drift_score
    assert 0.0 <= metrics2.collapse_probability <= 1.0
