#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LUKHAS (Logical Unified Knowledge Hyper-Adaptable System) - Knowledge Loader Test Suite

Copyright (c) 2025 LUKHAS AGI Development Team
All rights reserved.

This file is part of the LUKHAS AGI system, an enterprise artificial general
intelligence platform combining symbolic reasoning, emotional intelligence,
quantum integration, and bio-inspired architecture.

Mission: To illuminate complex reality through rigorous logic, adaptive
intelligence, and human-centred ethics—turning data into understanding,
understanding into foresight, and foresight into shared benefit for people
and planet.

This module contains the comprehensive test suite for the LUKHAS symbolic
knowledge loader system.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import asyncio
from typing import Dict, Any

# Import the modules to test
from .knowledge_loader import (SymbolicKnowledgeLoader,
    SymbolicConcept,
    load_symbolic_ontology,
    normalize_knowledge_structure,
    merge_knowledge_bases
)

class TestSymbolicKnowledgeLoader(unittest.TestCase):
    """Test suite for SymbolicKnowledgeLoader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = SymbolicKnowledgeLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Create test knowledge base
        self.test_knowledge = {
            "consciousness": {
                "definition": "The state of being aware of existence",
                "related": ["awareness", "cognition", "experience"],
                "importance": 9.0,
                "affect_tag": "wonder"
            },
            "creativity": {
                "definition": "The use of imagination to create something new",
                "related": ["innovation", "inspiration", "art"],
                "importance": 7.5,
                "affect_tag": "inspiration"
            },
            "fear": {
                "definition": "An emotional response to perceived danger",
                "related": ["anxiety", "worry", "threat"],
                "importance": 6.0,
                "affect_tag": "anxiety"
            }
        }

        # Create test knowledge file
        self.test_file_path = Path(self.temp_dir) / "test_knowledge.json"
        with open(self.test_file_path, 'w') as f:
            json.dump(self.test_knowledge, f)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        os.rmdir(self.temp_dir)

    def test_symbolic_concept_creation(self):
        """Test SymbolicConcept creation and properties."""
        concept = SymbolicConcept(
            concept="test_concept",
            definition="A test concept for validation",
            affect_tag="neutral",
            importance=5.0,
            related=["test", "validation"]
        )

        # Verify basic properties
        self.assertEqual(concept.concept, "test_concept")
        self.assertEqual(concept.definition, "A test concept for validation")
        self.assertEqual(concept.affect_tag, "neutral")
        self.assertEqual(concept.importance, 5.0)
        self.assertEqual(concept.related, ["test", "validation"])

        # Verify auto-generated properties
        self.assertIsNotNone(concept.symbolic_hash)
        self.assertIsNotNone(concept.temporal_stamp)
        self.assertTrue(concept.symbolic_hash.startswith("concept_"))

    def test_knowledge_base_loading(self):
        """Test loading knowledge base from JSON file."""
        concepts = self.loader.load_symbolic_ontology(self.test_file_path)

        # Verify correct number of concepts loaded
        self.assertEqual(len(concepts), 3)
        self.assertIn("consciousness", concepts)
        self.assertIn("creativity", concepts)
        self.assertIn("fear", concepts)

        # Verify concept properties
        consciousness = concepts["consciousness"]
        self.assertEqual(consciousness.concept, "consciousness")
        self.assertEqual(consciousness.definition, "The state of being aware of existence")
        self.assertEqual(consciousness.affect_tag, "wonder")
        self.assertEqual(consciousness.importance, 9.0)
        self.assertEqual(consciousness.related, ["awareness", "cognition", "experience"])

    def test_affect_tag_inference(self):
        """Test automatic affect tag inference."""
        # Test direct mapping
        self.assertEqual(self.loader._infer_affect_tag("consciousness"), "wonder")
        self.assertEqual(self.loader._infer_affect_tag("emotion"), "resonance")

        # Test keyword-based inference
        self.assertEqual(self.loader._infer_affect_tag("fearful_situation"), "anxiety")
        self.assertEqual(self.loader._infer_affect_tag("joyful_moment"), "joy")
        self.assertEqual(self.loader._infer_affect_tag("creative_process"), "inspiration")

        # Test fallback
        self.assertEqual(self.loader._infer_affect_tag("unknown_concept"), "neutral")

    def test_knowledge_normalization(self):
        """Test knowledge structure normalization."""
        concepts = self.loader.load_symbolic_ontology(self.test_file_path)
        normalized = self.loader.normalize_knowledge_structure(concepts)

        # Verify structure
        self.assertEqual(len(normalized), 3)

        for concept_name, concept_data in normalized.items():
            # Verify required fields
            self.assertIn("concept", concept_data)
            self.assertIn("definition", concept_data)
            self.assertIn("affect_tag", concept_data)
            self.assertIn("importance", concept_data)
            self.assertIn("system_integrations", concept_data)

            # Verify system integrations
            integrations = concept_data["system_integrations"]
            self.assertIn("memory", integrations)
            self.assertIn("narrative", integrations)
            self.assertIn("ethics", integrations)
            self.assertIn("reasoning", integrations)

            # Verify integration logic
            if concept_data["importance"] >= 7.0:
                self.assertTrue(integrations["ethics"])

            if len(concept_data["related"]) > 2:
                self.assertTrue(integrations["reasoning"])

    def test_knowledge_base_merging(self):
        """Test merging multiple knowledge bases."""
        # Create second knowledge base
        second_knowledge = {
            "consciousness": {  # Duplicate with different importance
                "definition": "Alternative definition of consciousness",
                "related": ["mind", "soul"],
                "importance": 8.0,
                "affect_tag": "mystery"
            },
            "memory": {  # New concept
                "definition": "The faculty of encoding and retrieving information",
                "related": ["recall", "storage", "forgetting"],
                "importance": 7.0,
                "affect_tag": "nostalgia"
            }
        }

        second_file_path = Path(self.temp_dir) / "second_knowledge.json"
        with open(second_file_path, 'w') as f:
            json.dump(second_knowledge, f)

        # Load both knowledge bases
        first_concepts = self.loader.load_symbolic_ontology(self.test_file_path)
        second_concepts = self.loader.load_symbolic_ontology(second_file_path)

        # Merge them
        merged = self.loader.merge_knowledge_bases(first_concepts, second_concepts)

        # Verify merge results
        self.assertEqual(len(merged), 4)  # 3 from first + 1 new from second
        self.assertIn("memory", merged)  # New concept added

        # Verify conflict resolution (first has higher importance)
        consciousness = merged["consciousness"]
        self.assertEqual(consciousness.importance, 9.0)  # Should keep higher importance
        self.assertIn("mind", consciousness.related)  # Should merge related terms

        # Clean up
        os.remove(second_file_path)

    def test_concept_search(self):
        """Test concept search functionality."""
        self.loader.load_symbolic_ontology(self.test_file_path)

        # Test name search
        results = self.loader.search_concepts("consciousness", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].concept, "consciousness")

        # Test definition search
        results = self.loader.search_concepts("imagination", limit=5)
        self.assertTrue(any(r.concept == "creativity" for r in results))

        # Test related terms search
        results = self.loader.search_concepts("anxiety", limit=5)
        self.assertTrue(any(r.concept == "fear" for r in results))

    def test_affect_based_retrieval(self):
        """Test retrieval of concepts by affect tag."""
        self.loader.load_symbolic_ontology(self.test_file_path)

        # Test specific affect retrieval
        wonder_concepts = self.loader.get_concepts_by_affect("wonder")
        self.assertEqual(len(wonder_concepts), 1)
        self.assertEqual(wonder_concepts[0].concept, "consciousness")

        anxiety_concepts = self.loader.get_concepts_by_affect("anxiety")
        self.assertEqual(len(anxiety_concepts), 1)
        self.assertEqual(anxiety_concepts[0].concept, "fear")

    def test_memory_system_export(self):
        """Test export format for memory system integration."""
        self.loader.load_symbolic_ontology(self.test_file_path)
        export = self.loader.export_for_memory_system()

        # Verify structure
        self.assertIn("symbolic_enrichment", export)
        self.assertIn("metadata", export)

        # Verify content
        enrichment = export["symbolic_enrichment"]
        self.assertIn("consciousness", enrichment)

        consciousness_data = enrichment["consciousness"]
        self.assertIn("definition", consciousness_data)
        self.assertIn("affect_context", consciousness_data)
        self.assertIn("symbolic_weight", consciousness_data)
        self.assertIn("associative_links", consciousness_data)

        # Verify metadata
        metadata = export["metadata"]
        self.assertIn("total_concepts", metadata)
        self.assertIn("generation_timestamp", metadata)
        self.assertTrue(metadata["integration_ready"])

    def test_narrative_system_export(self):
        """Test export format for narrative system integration."""
        self.loader.load_symbolic_ontology(self.test_file_path)
        export = self.loader.export_for_narrative_system()

        # Verify structure
        self.assertIn("archetypal_concepts", export)
        self.assertIn("concept_relationships", export)

        # Verify high-importance concepts are included
        archetypal = export["archetypal_concepts"]
        self.assertIn("consciousness", archetypal)  # importance 9.0 >= 6.0
        self.assertIn("creativity", archetypal)     # importance 7.5 >= 6.0

        # Verify concept structure
        consciousness_data = archetypal["consciousness"]
        self.assertIn("symbolic_meaning", consciousness_data)
        self.assertIn("narrative_weight", consciousness_data)
        self.assertIn("emotional_resonance", consciousness_data)

    def test_ethics_system_export(self):
        """Test export format for ethics system integration."""
        self.loader.load_symbolic_ontology(self.test_file_path)
        export = self.loader.export_for_ethics_system()

        # Verify structure
        self.assertIn("ethical_grounding", export)
        self.assertIn("policy_concepts", export)
        self.assertIn("compliance_keywords", export)

        # Verify only high-importance concepts are included
        ethical_grounding = export["ethical_grounding"]
        self.assertIn("consciousness", ethical_grounding)  # importance 9.0 >= 7.0
        self.assertIn("creativity", ethical_grounding)     # importance 7.5 >= 7.0
        self.assertNotIn("fear", ethical_grounding)        # importance 6.0 < 7.0

    def test_convenience_functions(self):
        """Test convenience functions for direct usage."""
        # Test load_symbolic_ontology function
        concepts = load_symbolic_ontology(self.test_file_path)
        self.assertEqual(len(concepts), 3)
        self.assertIn("consciousness", concepts)

        # Test normalize_knowledge_structure function
        normalized = normalize_knowledge_structure(concepts)
        self.assertEqual(len(normalized), 3)
        self.assertIn("system_integrations", normalized["consciousness"])

        # Test merge_knowledge_bases function
        # Create a small second knowledge base
        concepts2 = {
            "memory": SymbolicConcept(
                concept="memory",
                definition="Information storage and retrieval",
                importance=6.0
            )
        }

        merged = merge_knowledge_bases(concepts, concepts2)
        self.assertEqual(len(merged), 4)
        self.assertIn("memory", merged)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            self.loader.load_symbolic_ontology("/non/existent/path.json")

        # Test loading invalid JSON
        invalid_json_path = Path(self.temp_dir) / "invalid.json"
        with open(invalid_json_path, 'w') as f:
            f.write("invalid json content")

        with self.assertRaises(json.JSONDecodeError):
            self.loader.load_symbolic_ontology(invalid_json_path)

        # Clean up
        os.remove(invalid_json_path)


class TestKnowledgeIntegration(unittest.TestCase):
    """Test knowledge integration functionality."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock foundational knowledge
        self.mock_knowledge = {
            "test_concept": {
                "definition": "A concept for testing integration",
                "related": ["testing", "validation"],
                "importance": 8.0
            }
        }

        self.knowledge_file = Path(self.temp_dir) / "foundational_knowledge.json"
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.mock_knowledge, f)

    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.knowledge_file):
            os.remove(self.knowledge_file)
        os.rmdir(self.temp_dir)

    def test_knowledge_loading_integration(self):
        """Test integration of knowledge loading across system."""
        # Import and test the integration module if available
        try:
            from ...config.knowledge.symbolic_knowledge_integration import (
                SymbolicKnowledgeIntegrator,
                get_knowledge_integrator
            )

            # Test integrator initialization
            integrator = SymbolicKnowledgeIntegrator(self.temp_dir)
            self.assertIsNotNone(integrator)

            # Test global integrator
            global_integrator = get_knowledge_integrator()
            self.assertIsNotNone(global_integrator)

        except ImportError:
            # Integration module not available, skip test
            self.skipTest("Integration module not available")


def run_knowledge_loader_tests():
    """
    Run the complete knowledge loader test suite.

    Returns:
        Test results summary
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSymbolicKnowledgeLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return summary
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        "success": result.wasSuccessful()
    }


if __name__ == "__main__":
    # Run tests when executed directly
    print("🧪 ΛTRACE: Running LUKHAS Knowledge Loader Test Suite")
    results = run_knowledge_loader_tests()

    print(f"\n📊 Test Results Summary:")
    print(f"   Tests Run: {results['tests_run']}")
    print(f"   Failures: {results['failures']}")
    print(f"   Errors: {results['errors']}")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    print(f"   Overall Success: {'✅' if results['success'] else '❌'}")

# CLAUDE CHANGELOG
# - Created comprehensive test suite for symbolic knowledge loader # CLAUDE_EDIT_v1.0
# - Added tests for loading, normalization, merging, and system export functionality # CLAUDE_EDIT_v1.1
# - Implemented error handling and integration testing # CLAUDE_EDIT_v1.2

"""
═══════════════════════════════════════════════════════════════════════════════
║ COPYRIGHT & LICENSE:
║   Copyright (c) 2025 LUKHAS AI. All rights reserved.
║   Licensed under the LUKHAS AI Proprietary License.
╚═══════════════════════════════════════════════════════════════════════════════
"""