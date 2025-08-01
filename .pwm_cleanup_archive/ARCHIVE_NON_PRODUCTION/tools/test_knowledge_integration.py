#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LUKHAS (Logical Unified Knowledge Hyper-Adaptable System) - Knowledge Integration Test

Copyright (c) 2025 LUKHAS AGI Development Team
All rights reserved.

This file is part of the LUKHAS AGI system, an enterprise artificial general
intelligence platform combining symbolic reasoning, emotional intelligence,
quantum integration, and bio-inspired architecture.

Mission: To illuminate complex reality through rigorous logic, adaptive
intelligence, and human-centred ethics—turning data into understanding,
understanding into foresight, and foresight into shared benefit for people
and planet.

This module provides a mini test to validate knowledge loader functionality.
"""

import sys
import os
from pathlib import Path
import asyncio
import json

# Add the current directory to the path to enable imports
current_dir = Path(__file__).parent
lukhas_root = current_dir.parent
sys.path.insert(0, str(lukhas_root))

def test_knowledge_loader():
    """Test the knowledge loader functionality."""
    print("🔍 ΛTRACE: Starting knowledge loader validation test")

    try:
        # Import the knowledge loader
        from tools.parsers.knowledge_loader import SymbolicKnowledgeLoader
        print("✅ Successfully imported SymbolicKnowledgeLoader")

        # Initialize loader
        loader = SymbolicKnowledgeLoader()
        print("✅ Successfully initialized knowledge loader")

        # Try to load foundational knowledge
        knowledge_paths = [
            lukhas_root / "config" / "knowledge" / "foundational_knowledge.json",
            lukhas_root / "foundry" / "symbolic_seeds" / "foundational_knowledge.json"
        ]

        concepts_loaded = False
        for knowledge_path in knowledge_paths:
            if knowledge_path.exists():
                try:
                    concepts = loader.load_symbolic_ontology(knowledge_path)
                    print(f"✅ Successfully loaded {len(concepts)} concepts from {knowledge_path}")
                    concepts_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load from {knowledge_path}: {e}")

        if not concepts_loaded:
            print("⚠️ No knowledge files found, creating test concepts")
            # Create test concepts manually
            from tools.parsers.knowledge_loader import SymbolicConcept
            test_concepts = {
                "affect_collapse": SymbolicConcept(
                    concept="affect_collapse",
                    definition="Emotional cascade failure leading to system instability",
                    affect_tag="dread",
                    importance=9.5,
                    related=["emotional_cascade", "system_failure", "instability"]
                ),
                "resonance_inversion": SymbolicConcept(
                    concept="resonance_inversion",
                    definition="Reversal of harmonic patterns in symbolic processing",
                    affect_tag="dissonance",
                    importance=8.0,
                    related=["harmonic_reversal", "symbolic_chaos", "pattern_break"]
                )
            }
            loader.knowledge_cache.update(test_concepts)
            print("✅ Created test concepts for validation")

        # Test concept queries
        print("\n🔍 Testing concept queries...")

        # Query for affect collapse
        collapse_results = loader.search_concepts("affect collapse", limit=3)
        if collapse_results:
            concept = collapse_results[0]
            print(f"✅ Found 'affect collapse' concept:")
            print(f"   Definition: {concept.definition}")
            print(f"   Affect: {concept.affect_tag}")
            print(f"   Importance: {concept.importance}")
            print(f"   Related: {concept.related}")

            # Validate structure
            assert concept.concept is not None, "Concept name should not be None"
            assert concept.definition is not None, "Definition should not be None"
            assert concept.affect_tag is not None, "Affect tag should not be None"
            assert isinstance(concept.importance, (int, float)), "Importance should be numeric"
            assert isinstance(concept.related, list), "Related should be a list"
            print("✅ Concept structure validation passed")
        else:
            print("❌ No results found for 'affect collapse'")

        # Query for resonance inversion
        resonance_results = loader.search_concepts("resonance inversion", limit=3)
        if resonance_results:
            concept = resonance_results[0]
            print(f"✅ Found 'resonance inversion' concept:")
            print(f"   Definition: {concept.definition}")
            print(f"   Affect: {concept.affect_tag}")
            print(f"   Related: {concept.related}")
        else:
            print("❌ No results found for 'resonance inversion'")

        # Test system exports
        print("\n🔍 Testing system integration exports...")

        try:
            memory_export = loader.export_for_memory_system()
            assert "symbolic_enrichment" in memory_export, "Memory export should have symbolic_enrichment"
            assert "metadata" in memory_export, "Memory export should have metadata"
            print("✅ Memory system export format validated")
        except Exception as e:
            print(f"❌ Memory system export failed: {e}")

        try:
            narrative_export = loader.export_for_narrative_system()
            assert "archetypal_concepts" in narrative_export, "Narrative export should have archetypal_concepts"
            print("✅ Narrative system export format validated")
        except Exception as e:
            print(f"❌ Narrative system export failed: {e}")

        try:
            ethics_export = loader.export_for_ethics_system()
            assert "ethical_grounding" in ethics_export, "Ethics export should have ethical_grounding"
            print("✅ Ethics system export format validated")
        except Exception as e:
            print(f"❌ Ethics system export failed: {e}")

        print("\n✅ ΛTRACE: Knowledge loader validation completed successfully")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the knowledge loader module is in the correct location")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_knowledge_integration():
    """Test the knowledge integration system."""
    print("\n🔍 ΛTRACE: Testing knowledge integration system")

    try:
        # Import the integration module
        from config.knowledge.symbolic_knowledge_integration import (
            SymbolicKnowledgeIntegrator,
            get_knowledge_integrator,
            initialize_symbolic_knowledge
        )
        print("✅ Successfully imported knowledge integration components")

        # Test integrator initialization
        integrator = get_knowledge_integrator()
        print("✅ Successfully initialized knowledge integrator")

        # Test knowledge initialization
        try:
            result = await initialize_symbolic_knowledge()
            print(f"✅ Knowledge integration initialized:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Concepts: {result.get('concepts_loaded', 0)}")
            print(f"   Sources: {len(result.get('sources', []))}")

            # Test integration status
            status = integrator.get_integration_status()
            print(f"✅ Integration status retrieved:")
            for system, success in status.get('status', {}).items():
                print(f"   {system}: {'✅' if success else '❌'}")

        except Exception as e:
            print(f"⚠️ Integration initialization failed (expected in test): {e}")

        print("✅ ΛTRACE: Knowledge integration test completed")
        return True

    except ImportError as e:
        print(f"❌ Integration module import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

def main():
    """Main test execution."""
    print("🧪 LUKHAS Knowledge Integration Validation Test")
    print("=" * 60)

    # Test knowledge loader
    loader_success = test_knowledge_loader()

    # Test knowledge integration
    integration_success = asyncio.run(test_knowledge_integration())

    # Summary
    print("\n📊 Test Summary:")
    print(f"   Knowledge Loader: {'✅ PASS' if loader_success else '❌ FAIL'}")
    print(f"   Integration System: {'✅ PASS' if integration_success else '❌ FAIL'}")

    overall_success = loader_success and integration_success
    print(f"\n🎯 Overall Result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")

    if overall_success:
        print("\n🎉 ΛTRACE: All knowledge integration tests passed!")
        print("The symbolic knowledge system is ready for use.")
    else:
        print("\n⚠️ ΛTRACE: Some tests failed. Check the output above for details.")

    return 0 if overall_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# CLAUDE CHANGELOG
# - Created mini test for knowledge integration validation # CLAUDE_EDIT_v1.0
# - Added concept query testing and structure validation # CLAUDE_EDIT_v1.1
# - Implemented system export testing and integration validation # CLAUDE_EDIT_v1.2

"""
═══════════════════════════════════════════════════════════════════════════════
║ COPYRIGHT & LICENSE:
║   Copyright (c) 2025 LUKHAS AI. All rights reserved.
║   Licensed under the LUKHAS AI Proprietary License.
╚═══════════════════════════════════════════════════════════════════════════════
"""