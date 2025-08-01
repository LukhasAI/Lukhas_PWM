#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LUKHAS (Logical Unified Knowledge Hyper-Adaptable System) - Prophet Predictor Test Suite

Copyright (c) 2025 LUKHAS AGI Development Team
All rights reserved.

This file is part of the LUKHAS AGI system, an enterprise artificial general
intelligence platform combining symbolic reasoning, emotional intelligence,
quantum integration, and bio-inspired architecture.

Mission: To illuminate complex reality through rigorous logic, adaptive
intelligence, and human-centred ethics—turning data into understanding,
understanding into foresight, and foresight into shared benefit for people
and planet.

This module contains the comprehensive test suite for the LUKHAS Prophet
Predictor engine.
"""

import asyncio
import json
import math
import pytest
import random
import statistics
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import ΛPROPHET components
from diagnostics.predictive.prophet import (
    LambdaProphet, SymbolicMetrics, SymbolicTrajectoryAnalyzer, CascadePredictor,
    AlertLevel, CascadeType, InterventionType, PredictionResult, InterventionRecommendation,
    ProphetSignal, analyze_symbolic_trajectory, predict_cascade_risk,
    recommend_intervention, emit_prophet_signal
)


class TestSymbolicMetrics(unittest.TestCase):
    """Test SymbolicMetrics data class and risk calculation."""

    def setUp(self):
        """Set up test metrics."""
        self.normal_metrics = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.3,
            phase_drift=0.1,
            motif_conflicts=2,
            emotion_volatility=0.2,
            contradiction_density=0.15,
            memory_fold_integrity=0.9,
            governor_stress=0.1,
            dream_convergence=0.2
        )

        self.high_risk_metrics = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.8,
            phase_drift=0.4,
            motif_conflicts=12,
            emotion_volatility=0.9,
            contradiction_density=0.7,
            memory_fold_integrity=0.3,
            governor_stress=0.8,
            dream_convergence=0.9
        )

    def test_risk_score_calculation(self):
        """Test risk score calculation for different scenarios."""
        # Normal risk should be low
        normal_risk = self.normal_metrics.risk_score()
        self.assertLess(normal_risk, 0.4)
        self.assertGreaterEqual(normal_risk, 0.0)

        # High risk should be elevated
        high_risk = self.high_risk_metrics.risk_score()
        self.assertGreater(high_risk, 0.6)
        self.assertLessEqual(high_risk, 1.0)

        # High risk should be greater than normal risk
        self.assertGreater(high_risk, normal_risk)

    def test_risk_score_components(self):
        """Test individual risk components contribute properly."""
        base_metrics = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.0, phase_drift=0.0, motif_conflicts=0,
            emotion_volatility=0.0, contradiction_density=0.0,
            memory_fold_integrity=1.0, governor_stress=0.0, dream_convergence=0.0
        )

        # Test entropy contribution
        entropy_metrics = SymbolicMetrics(
            timestamp=base_metrics.timestamp,
            entropy_level=1.0, phase_drift=0.0, motif_conflicts=0,
            emotion_volatility=0.0, contradiction_density=0.0,
            memory_fold_integrity=1.0, governor_stress=0.0, dream_convergence=0.0
        )

        base_risk = base_metrics.risk_score()
        entropy_risk = entropy_metrics.risk_score()

        self.assertGreater(entropy_risk, base_risk)
        self.assertAlmostEqual(entropy_risk - base_risk, 0.25, delta=0.01)  # Entropy weight is 0.25


class TestSymbolicTrajectoryAnalyzer(unittest.TestCase):
    """Test trajectory analysis and pattern detection."""

    def setUp(self):
        """Set up trajectory analyzer."""
        self.analyzer = SymbolicTrajectoryAnalyzer(window_size=50, trend_threshold=0.1)

    def generate_test_timeline(self, pattern_type: str, length: int = 20) -> List[SymbolicMetrics]:
        """Generate test timeline with specific patterns."""
        timeline = []
        base_time = datetime.now(timezone.utc)

        for i in range(length):
            timestamp = base_time + timedelta(minutes=i*5)

            if pattern_type == "stable":
                # Stable pattern with minimal variation
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=0.3 + random.uniform(-0.05, 0.05),
                    phase_drift=0.1 + random.uniform(-0.02, 0.02),
                    motif_conflicts=2 + random.randint(-1, 1),
                    emotion_volatility=0.2 + random.uniform(-0.03, 0.03),
                    contradiction_density=0.15 + random.uniform(-0.02, 0.02),
                    memory_fold_integrity=0.9 + random.uniform(-0.05, 0.05),
                    governor_stress=0.1 + random.uniform(-0.02, 0.02),
                    dream_convergence=0.2 + random.uniform(-0.03, 0.03)
                )

            elif pattern_type == "increasing_entropy":
                # Entropy spiral pattern
                progress = i / length
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=0.3 + progress * 0.5,  # 0.3 to 0.8
                    phase_drift=0.1 + progress * 0.3,     # 0.1 to 0.4
                    motif_conflicts=2 + int(progress * 8), # 2 to 10
                    emotion_volatility=0.2 + progress * 0.4,
                    contradiction_density=0.15 + progress * 0.3,
                    memory_fold_integrity=0.9 - progress * 0.3,  # Degrading
                    governor_stress=0.1 + progress * 0.5,
                    dream_convergence=0.2 + progress * 0.4
                )

            elif pattern_type == "memory_collapse":
                # Memory integrity degradation
                progress = i / length
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=0.4 + progress * 0.3,
                    phase_drift=0.15 + random.uniform(-0.05, 0.05),
                    motif_conflicts=3 + random.randint(-1, 2),
                    emotion_volatility=0.3 + progress * 0.4,
                    contradiction_density=0.2 + progress * 0.3,
                    memory_fold_integrity=0.9 - progress * 0.6,  # Severe degradation
                    governor_stress=0.2 + progress * 0.3,
                    dream_convergence=0.2 + random.uniform(-0.05, 0.05)
                )

            else:  # random pattern
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=random.uniform(0.1, 0.9),
                    phase_drift=random.uniform(-0.5, 0.5),
                    motif_conflicts=random.randint(0, 15),
                    emotion_volatility=random.uniform(0.0, 1.0),
                    contradiction_density=random.uniform(0.0, 0.8),
                    memory_fold_integrity=random.uniform(0.2, 1.0),
                    governor_stress=random.uniform(0.0, 1.0),
                    dream_convergence=random.uniform(0.0, 1.0)
                )

            timeline.append(metrics)

        return timeline

    def test_stable_trajectory_analysis(self):
        """Test analysis of stable system trajectory."""
        stable_timeline = self.generate_test_timeline("stable", 30)
        analysis = self.analyzer.analyze_trajectory(stable_timeline)

        # Stable system should have low overall risk
        self.assertLess(analysis["overall_risk"], 0.4)
        self.assertGreater(analysis["trend_stability"], 0.6)
        self.assertLess(analysis["volatility"], 0.3)

    def test_entropy_spiral_detection(self):
        """Test detection of entropy spiral patterns."""
        spiral_timeline = self.generate_test_timeline("increasing_entropy", 25)
        analysis = self.analyzer.analyze_trajectory(spiral_timeline)

        # Should detect increasing risk and entropy trend
        self.assertGreater(analysis["overall_risk"], 0.5)
        self.assertGreater(analysis["entropy_trend"], 0.1)
        self.assertGreater(analysis["pattern_match_score"], 0.4)

    def test_memory_collapse_detection(self):
        """Test detection of memory collapse patterns."""
        collapse_timeline = self.generate_test_timeline("memory_collapse", 20)
        analysis = self.analyzer.analyze_trajectory(collapse_timeline)

        # Should detect degradation pattern
        self.assertGreater(analysis["overall_risk"], 0.4)
        self.assertGreater(analysis["pattern_match_score"], 0.6)  # Memory pattern should score high

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        short_timeline = self.generate_test_timeline("stable", 5)
        analysis = self.analyzer.analyze_trajectory(short_timeline)

        # Should handle gracefully with minimal data
        self.assertIn("trajectory_score", analysis)
        self.assertIn("overall_risk", analysis)
        self.assertEqual(analysis["overall_risk"], 0.0)  # Should default to 0 for insufficient data


class TestCascadePredictor(unittest.TestCase):
    """Test cascade prediction engine."""

    def setUp(self):
        """Set up cascade predictor."""
        self.predictor = CascadePredictor(confidence_threshold=0.6)
        self.analyzer = SymbolicTrajectoryAnalyzer()

    def test_entropy_spiral_prediction(self):
        """Test prediction of entropy spiral cascades."""
        # Generate entropy spiral timeline
        timeline = []
        base_time = datetime.now(timezone.utc)

        for i in range(15):
            progress = i / 15
            metrics = SymbolicMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                entropy_level=0.4 + progress * 0.4,  # Rising to 0.8
                phase_drift=0.2 + progress * 0.2,     # Rising to 0.4
                motif_conflicts=3 + int(progress * 7), # Rising to 10
                emotion_volatility=0.3 + progress * 0.3,
                contradiction_density=0.2 + progress * 0.3,
                memory_fold_integrity=0.8 - progress * 0.1,
                governor_stress=0.2 + progress * 0.4,
                dream_convergence=0.3 + progress * 0.2
            )
            timeline.append(metrics)

        prediction = self.predictor.predict_cascade_risk(timeline)

        # Should predict entropy spiral with reasonable confidence
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.cascade_type, CascadeType.ENTROPY_SPIRAL)
        self.assertGreaterEqual(prediction.confidence, 0.6)
        self.assertIsNotNone(prediction.time_to_cascade)
        self.assertGreater(len(prediction.recommended_interventions), 0)

    def test_memory_collapse_prediction(self):
        """Test prediction of memory collapse cascades."""
        timeline = []
        base_time = datetime.now(timezone.utc)

        for i in range(12):
            progress = i / 12
            metrics = SymbolicMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                entropy_level=0.4 + progress * 0.2,
                phase_drift=0.1 + random.uniform(-0.05, 0.05),
                motif_conflicts=2 + random.randint(0, 2),
                emotion_volatility=0.3 + progress * 0.4,
                contradiction_density=0.2 + progress * 0.3,
                memory_fold_integrity=0.8 - progress * 0.5,  # Severe degradation
                governor_stress=0.2 + progress * 0.3,
                dream_convergence=0.2 + random.uniform(-0.05, 0.05)
            )
            timeline.append(metrics)

        prediction = self.predictor.predict_cascade_risk(timeline)

        # Should predict memory collapse
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.cascade_type, CascadeType.MEMORY_COLLAPSE)
        self.assertGreaterEqual(prediction.confidence, 0.6)

    def test_no_cascade_prediction(self):
        """Test scenarios where no cascade should be predicted."""
        # Generate stable timeline
        timeline = []
        base_time = datetime.now(timezone.utc)

        for i in range(20):
            metrics = SymbolicMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                entropy_level=0.3 + random.uniform(-0.05, 0.05),
                phase_drift=0.1 + random.uniform(-0.03, 0.03),
                motif_conflicts=2 + random.randint(-1, 1),
                emotion_volatility=0.2 + random.uniform(-0.05, 0.05),
                contradiction_density=0.15 + random.uniform(-0.03, 0.03),
                memory_fold_integrity=0.9 + random.uniform(-0.05, 0.05),
                governor_stress=0.1 + random.uniform(-0.03, 0.03),
                dream_convergence=0.2 + random.uniform(-0.05, 0.05)
            )
            timeline.append(metrics)

        prediction = self.predictor.predict_cascade_risk(timeline)

        # Should not predict cascade for stable system
        self.assertIsNone(prediction)

    def test_intervention_generation(self):
        """Test intervention recommendation generation."""
        # High-risk metrics
        high_risk_metrics = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.8,
            phase_drift=0.4,
            motif_conflicts=10,
            emotion_volatility=0.9,
            contradiction_density=0.7,
            memory_fold_integrity=0.3,
            governor_stress=0.8,
            dream_convergence=0.9
        )

        interventions = self.predictor._generate_interventions(
            CascadeType.ENTROPY_SPIRAL, high_risk_metrics, 0.9
        )

        # Should generate appropriate interventions
        self.assertGreater(len(interventions), 0)
        self.assertLessEqual(len(interventions), 3)

        # Check intervention types
        intervention_types = [i.intervention_type for i in interventions]
        self.assertIn(InterventionType.ENTROPY_BUFFER, intervention_types)

        # Check priority levels
        for intervention in interventions:
            self.assertIn(intervention.priority, [AlertLevel.EMERGENCY, AlertLevel.CRITICAL, AlertLevel.WARNING])


class TestLambdaProphet(unittest.TestCase):
    """Test main ΛPROPHET engine integration."""

    def setUp(self):
        """Set up ΛPROPHET engine."""
        self.prophet = LambdaProphet(log_sources=[])

    def test_initialization(self):
        """Test ΛPROPHET initialization."""
        self.assertIsInstance(self.prophet.cascade_predictor, CascadePredictor)
        self.assertIsInstance(self.prophet.trajectory_analyzer, SymbolicTrajectoryAnalyzer)
        self.assertEqual(self.prophet.signal_counter, 0)

    def generate_mock_log_data(self, pattern: str = "normal") -> List[Dict[str, Any]]:
        """Generate mock log data for testing."""
        log_data = []
        base_time = datetime.now(timezone.utc)

        for i in range(25):
            timestamp = base_time + timedelta(minutes=i*5)

            if pattern == "cascade_building":
                progress = i / 25
                event = {
                    "timestamp": timestamp.isoformat(),
                    "entropy_level": 0.3 + progress * 0.5,
                    "phase_drift": 0.1 + progress * 0.3,
                    "emotion_volatility": 0.2 + progress * 0.6,
                    "motif_conflicts": 2 + int(progress * 8),
                    "contradiction_density": 0.1 + progress * 0.5,
                    "memory_fold_integrity": 0.9 - progress * 0.4,
                    "governor_stress": 0.1 + progress * 0.6,
                    "dream_convergence": 0.2 + progress * 0.4,
                    "_source": "test_log"
                }
            else:  # normal pattern
                event = {
                    "timestamp": timestamp.isoformat(),
                    "entropy_level": 0.3 + random.uniform(-0.1, 0.1),
                    "phase_drift": 0.1 + random.uniform(-0.05, 0.05),
                    "emotion_volatility": 0.2 + random.uniform(-0.1, 0.1),
                    "motif_conflicts": 2 + random.randint(-1, 2),
                    "contradiction_density": 0.15 + random.uniform(-0.05, 0.05),
                    "memory_fold_integrity": 0.9 + random.uniform(-0.1, 0.1),
                    "governor_stress": 0.1 + random.uniform(-0.05, 0.05),
                    "dream_convergence": 0.2 + random.uniform(-0.1, 0.1),
                    "_source": "test_log"
                }

            log_data.append(event)

        return log_data

    async def test_trajectory_analysis(self):
        """Test symbolic trajectory analysis."""
        log_data = self.generate_mock_log_data("cascade_building")

        analysis = await self.prophet.analyze_symbolic_trajectory(log_data)

        # Should complete analysis successfully
        self.assertEqual(analysis["status"], "analysis_complete")
        self.assertEqual(analysis["metrics_analyzed"], len(log_data))
        self.assertIn("trajectory_analysis", analysis)
        self.assertIn("overall_risk", analysis["trajectory_analysis"])

        # Cascade building pattern should show elevated risk
        self.assertGreater(analysis["trajectory_analysis"]["overall_risk"], 0.4)

    def test_cascade_risk_prediction_integration(self):
        """Test integrated cascade risk prediction."""
        # Generate timeline with cascade indicators
        timeline = []
        base_time = datetime.now(timezone.utc)

        for i in range(15):
            progress = i / 15
            metrics = SymbolicMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                entropy_level=0.4 + progress * 0.4,
                phase_drift=0.2 + progress * 0.2,
                motif_conflicts=3 + int(progress * 7),
                emotion_volatility=0.3 + progress * 0.3,
                contradiction_density=0.2 + progress * 0.3,
                memory_fold_integrity=0.8 - progress * 0.1,
                governor_stress=0.2 + progress * 0.4,
                dream_convergence=0.3 + progress * 0.2
            )
            timeline.append(metrics)

        # Populate analyzer history
        for metrics in timeline:
            self.prophet.trajectory_analyzer.add_metrics(metrics)

        prediction = self.prophet.predict_cascade_risk(timeline)

        # Should generate valid prediction
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, PredictionResult)
        self.assertGreaterEqual(prediction.confidence, 0.6)
        self.assertGreater(len(prediction.recommended_interventions), 0)

    def test_intervention_recommendation(self):
        """Test intervention recommendation system."""
        # High-risk symbolic state
        risky_state = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.9,
            phase_drift=0.5,
            motif_conflicts=12,
            emotion_volatility=0.85,
            contradiction_density=0.8,
            memory_fold_integrity=0.2,
            governor_stress=0.9,
            dream_convergence=0.9
        )

        interventions = self.prophet.recommend_intervention(risky_state)

        # Should recommend emergency intervention
        self.assertGreater(len(interventions), 0)

        # Should include emergency shutdown for critical risk
        intervention_types = [i.intervention_type for i in interventions]
        self.assertIn(InterventionType.EMERGENCY_SHUTDOWN, intervention_types)

        # Should have emergency priority
        priorities = [i.priority for i in interventions]
        self.assertIn(AlertLevel.EMERGENCY, priorities)

    def test_prophet_signal_emission(self):
        """Test ΛPROPHET signal emission and logging."""
        # Create mock prediction
        prediction = PredictionResult(
            cascade_type=CascadeType.ENTROPY_SPIRAL,
            confidence=0.85,
            time_to_cascade=3600,  # 1 hour
            risk_trajectory=[0.3, 0.4, 0.5, 0.6, 0.7],
            contributing_factors=["High entropy", "Phase drift"],
            recommended_interventions=[]
        )

        signal = self.prophet.emit_prophet_signal(AlertLevel.CRITICAL, {
            "prediction": prediction,
            "confidence": prediction.confidence
        })

        # Verify signal structure
        self.assertIsInstance(signal, ProphetSignal)
        self.assertEqual(signal.alert_level, AlertLevel.CRITICAL)
        self.assertEqual(signal.signal_type, "ΛPRE_CASCADE")
        self.assertEqual(signal.confidence, 0.85)
        self.assertEqual(signal.prediction, prediction)
        self.assertGreater(self.prophet.signal_counter, 0)

    def test_metrics_extraction(self):
        """Test extraction of symbolic metrics from log data."""
        log_data = self.generate_mock_log_data("normal")

        metrics = self.prophet._extract_symbolic_metrics(log_data)

        # Should extract all metrics
        self.assertEqual(len(metrics), len(log_data))

        # All should be SymbolicMetrics instances
        for metric in metrics:
            self.assertIsInstance(metric, SymbolicMetrics)
            self.assertIsInstance(metric.timestamp, datetime)
            self.assertGreaterEqual(metric.entropy_level, 0.0)
            self.assertLessEqual(metric.entropy_level, 1.0)


class TestProphetAccuracy(unittest.TestCase):
    """Test ΛPROPHET prediction accuracy with historical data simulation."""

    def setUp(self):
        """Set up accuracy testing environment."""
        self.prophet = LambdaProphet()
        self.test_cases = []

    def generate_cascade_scenario(self, cascade_type: CascadeType, should_cascade: bool = True) -> Dict[str, Any]:
        """Generate a cascade scenario for testing."""
        timeline = []
        base_time = datetime.now(timezone.utc)
        length = 20

        for i in range(length):
            progress = i / length
            timestamp = base_time + timedelta(minutes=i*5)

            if cascade_type == CascadeType.ENTROPY_SPIRAL and should_cascade:
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=0.3 + progress * 0.5,
                    phase_drift=0.1 + progress * 0.3,
                    motif_conflicts=2 + int(progress * 8),
                    emotion_volatility=0.2 + progress * 0.3,
                    contradiction_density=0.15 + progress * 0.35,
                    memory_fold_integrity=0.9 - progress * 0.2,
                    governor_stress=0.1 + progress * 0.5,
                    dream_convergence=0.2 + progress * 0.3
                )

            elif cascade_type == CascadeType.MEMORY_COLLAPSE and should_cascade:
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=0.4 + progress * 0.2,
                    phase_drift=0.15 + random.uniform(-0.05, 0.05),
                    motif_conflicts=3 + random.randint(0, 2),
                    emotion_volatility=0.3 + progress * 0.4,
                    contradiction_density=0.2 + progress * 0.3,
                    memory_fold_integrity=0.8 - progress * 0.6,  # Critical degradation
                    governor_stress=0.2 + progress * 0.3,
                    dream_convergence=0.2 + random.uniform(-0.05, 0.05)
                )

            else:  # No cascade or stable scenario
                metrics = SymbolicMetrics(
                    timestamp=timestamp,
                    entropy_level=0.3 + random.uniform(-0.1, 0.1),
                    phase_drift=0.1 + random.uniform(-0.05, 0.05),
                    motif_conflicts=2 + random.randint(-1, 2),
                    emotion_volatility=0.2 + random.uniform(-0.1, 0.1),
                    contradiction_density=0.15 + random.uniform(-0.05, 0.05),
                    memory_fold_integrity=0.9 + random.uniform(-0.1, 0.1),
                    governor_stress=0.1 + random.uniform(-0.05, 0.05),
                    dream_convergence=0.2 + random.uniform(-0.1, 0.1)
                )

            timeline.append(metrics)

        return {
            "timeline": timeline,
            "expected_cascade": should_cascade,
            "expected_type": cascade_type if should_cascade else None
        }

    def test_prediction_accuracy(self):
        """Test overall prediction accuracy with controlled scenarios."""
        # Generate test scenarios
        scenarios = []

        # Generate 10 entropy spiral scenarios
        for _ in range(10):
            scenarios.append(self.generate_cascade_scenario(CascadeType.ENTROPY_SPIRAL, True))

        # Generate 10 memory collapse scenarios
        for _ in range(10):
            scenarios.append(self.generate_cascade_scenario(CascadeType.MEMORY_COLLAPSE, True))

        # Generate 20 stable scenarios (no cascade)
        for _ in range(20):
            scenarios.append(self.generate_cascade_scenario(CascadeType.ENTROPY_SPIRAL, False))

        # Test predictions
        correct_predictions = 0
        total_predictions = 0

        for scenario in scenarios:
            prediction = self.prophet.predict_cascade_risk(scenario["timeline"])

            if scenario["expected_cascade"]:
                # Should predict cascade
                if prediction is not None and prediction.confidence >= 0.7:
                    correct_predictions += 1
                total_predictions += 1
            else:
                # Should not predict cascade
                if prediction is None or prediction.confidence < 0.7:
                    correct_predictions += 1
                total_predictions += 1

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Should achieve >80% accuracy target
        self.assertGreater(accuracy, 0.8)
        print(f"ΛPROPHET Prediction Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

    def test_phase_drift_threshold(self):
        """Test phase drift threshold detection accuracy."""
        # Test phase misalignment detection at 0.2 threshold
        test_metrics = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.4,
            phase_drift=0.25,  # Above 0.2 threshold
            motif_conflicts=3,
            emotion_volatility=0.3,
            contradiction_density=0.2,
            memory_fold_integrity=0.8,
            governor_stress=0.3,
            dream_convergence=0.3
        )

        interventions = self.prophet.recommend_intervention(test_metrics)

        # Should recommend stabilizer injection for phase drift
        intervention_types = [i.intervention_type for i in interventions]
        self.assertIn(InterventionType.STABILIZER_INJECTION, intervention_types)

        # Should detect phase drift in contributing factors
        # Create a minimal timeline for cascade prediction
        timeline = [test_metrics] * 15  # Repeat for sufficient timeline
        prediction = self.prophet.predict_cascade_risk(timeline)

        if prediction:
            phase_drift_detected = any("phase drift" in factor.lower() for factor in prediction.contributing_factors)
            self.assertTrue(phase_drift_detected, "Phase drift should be detected as contributing factor")

    def test_false_positive_minimization(self):
        """Test false positive rate with benign anomalies."""
        # Generate benign anomaly scenarios
        false_positive_count = 0
        test_count = 50

        for _ in range(test_count):
            # Generate timeline with minor anomalies but no cascade pattern
            timeline = []
            base_time = datetime.now(timezone.utc)

            for i in range(15):
                # Add some random spikes but overall stable
                spike_factor = 1.3 if random.random() < 0.1 else 1.0  # 10% chance of minor spike

                metrics = SymbolicMetrics(
                    timestamp=base_time + timedelta(minutes=i*5),
                    entropy_level=(0.35 + random.uniform(-0.1, 0.1)) * spike_factor,
                    phase_drift=(0.12 + random.uniform(-0.05, 0.05)) * spike_factor,
                    motif_conflicts=max(0, int((2 + random.randint(-1, 2)) * spike_factor)),
                    emotion_volatility=(0.25 + random.uniform(-0.1, 0.1)) * spike_factor,
                    contradiction_density=(0.18 + random.uniform(-0.05, 0.05)) * spike_factor,
                    memory_fold_integrity=min(1.0, 0.85 + random.uniform(-0.1, 0.1)),
                    governor_stress=(0.15 + random.uniform(-0.05, 0.05)) * spike_factor,
                    dream_convergence=(0.22 + random.uniform(-0.1, 0.1)) * spike_factor
                )
                timeline.append(metrics)

            # Test for false positive
            prediction = self.prophet.predict_cascade_risk(timeline)
            if prediction is not None and prediction.confidence >= 0.7:
                false_positive_count += 1

        # Calculate false positive rate
        false_positive_rate = false_positive_count / test_count

        # Should keep false positives below 10%
        self.assertLess(false_positive_rate, 0.1)
        print(f"False Positive Rate: {false_positive_rate:.3f} ({false_positive_count}/{test_count})")


class TestProphetIntegration(unittest.TestCase):
    """Test ΛPROPHET integration with other system components."""

    def test_public_api_functions(self):
        """Test public API function interfaces."""
        # Test analyze_symbolic_trajectory
        mock_log_data = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entropy_level": 0.4,
                "phase_drift": 0.15,
                "emotion_volatility": 0.3,
                "motif_conflicts": 3,
                "contradiction_density": 0.2,
                "memory_fold_integrity": 0.8,
                "governor_stress": 0.2,
                "dream_convergence": 0.3
            }
        ] * 15

        async def run_api_test():
            result = await analyze_symbolic_trajectory(mock_log_data)
            self.assertIn("status", result)
            self.assertIn("trajectory_analysis", result)

        # Run async test
        asyncio.run(run_api_test())

        # Test recommend_intervention
        test_metrics = SymbolicMetrics(
            timestamp=datetime.now(timezone.utc),
            entropy_level=0.7, phase_drift=0.3, motif_conflicts=6,
            emotion_volatility=0.6, contradiction_density=0.5,
            memory_fold_integrity=0.6, governor_stress=0.4, dream_convergence=0.4
        )

        interventions = recommend_intervention(test_metrics)
        self.assertIsInstance(interventions, list)
        self.assertGreater(len(interventions), 0)

        # Test emit_prophet_signal
        signal = emit_prophet_signal(AlertLevel.WARNING, {"test": "data", "confidence": 0.75})
        self.assertIsInstance(signal, ProphetSignal)
        self.assertEqual(signal.alert_level, AlertLevel.WARNING)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)


# CLAUDE CHANGELOG
# - Created comprehensive test suite for ΛPROPHET engine # CLAUDE_EDIT_v0.1
# - Implemented unit tests for SymbolicMetrics, SymbolicTrajectoryAnalyzer, CascadePredictor # CLAUDE_EDIT_v0.1
# - Added integration tests for LambdaProphet main engine # CLAUDE_EDIT_v0.1
# - Created accuracy validation tests with >80% prediction target # CLAUDE_EDIT_v0.1
# - Implemented false positive minimization testing with <10% target # CLAUDE_EDIT_v0.1
# - Added phase drift threshold detection validation (0.2 threshold) # CLAUDE_EDIT_v0.1
# - Created controlled cascade scenario generation for testing # CLAUDE_EDIT_v0.1
# - Added public API integration testing # CLAUDE_EDIT_v0.1

"""
═══════════════════════════════════════════════════════════════════════════════
║ COPYRIGHT & LICENSE:
║   Copyright (c) 2025 LUKHAS AI. All rights reserved.
║   Licensed under the LUKHAS AI Proprietary License.
╚═══════════════════════════════════════════════════════════════════════════════
"""