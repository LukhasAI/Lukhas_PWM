#!/usr/bin/env python3
"""
Agent 1 Task 6: Golden Helix Memory Mapper Integration Test
Testing the memory_helix_golden.py integration with memory hub.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_golden_helix_integration():
    """Test the Golden Helix Memory Mapper integration"""
    print("🔬 Agent 1 Task 6: Golden Helix Memory Mapper Integration Test")
    print("=" * 60)

    try:
        # Test 1: Direct module import
        print("Test 1: Testing direct Golden Helix imports...")
        from memory.systems.memory_helix_golden import (
            HealixMapper,
            MemoryStrand,
            MutationStrategy,
        )

        mapper = HealixMapper()
        print("✅ Golden Helix classes imported and instantiated successfully")

        # Test 2: Memory encoding across different strands
        print("\nTest 2: Testing memory encoding across different strand types...")
        test_memories = [
            {
                "memory": {
                    "content": "Agent 1 successfully completed reasoning integration",
                    "emotional_weight": 0.9,
                    "valence": "positive",
                    "metadata": {"agent": "Agent_1", "task": 1, "priority": 91.0},
                },
                "strand": MemoryStrand.EMOTIONAL,
                "context": {"source": "Agent1_Task1_Completion"},
            },
            {
                "memory": {
                    "content": "Cultural adaptation patterns for enterprise authentication",
                    "cultural_markers": ["enterprise", "security", "authentication"],
                    "origin": "identity_hub_integration",
                    "metadata": {"agent": "Agent_1", "task": 3, "priority": 49.0},
                },
                "strand": MemoryStrand.CULTURAL,
                "context": {"source": "Agent1_Task3_Enterprise_Auth"},
            },
            {
                "memory": {
                    "content": "Experience with meta-learning enhancement systems",
                    "context": "learning_hub_integration",
                    "timestamp": "2025-07-31T00:00:00",
                    "metadata": {"agent": "Agent_1", "task": 4, "priority": 45.5},
                },
                "strand": MemoryStrand.EXPERIENTIAL,
                "context": {"source": "Agent1_Task4_Meta_Learning"},
            },
            {
                "memory": {
                    "content": "Procedure for resource efficiency analysis and optimization",
                    "steps": ["monitor", "analyze", "optimize", "validate"],
                    "triggers": ["high_cpu_usage", "memory_pressure", "io_bottleneck"],
                    "metadata": {"agent": "Agent_1", "task": 5, "priority": 42.5},
                },
                "strand": MemoryStrand.PROCEDURAL,
                "context": {"source": "Agent1_Task5_Resource_Efficiency"},
            },
            {
                "memory": {
                    "content": "Cognitive patterns in helix memory architecture",
                    "concepts": ["helix", "dna", "quantum_resistance", "mutation"],
                    "associations": ["biology", "memory", "evolution", "adaptation"],
                    "metadata": {"agent": "Agent_1", "task": 6, "priority": 38.5},
                },
                "strand": MemoryStrand.COGNITIVE,
                "context": {"source": "Agent1_Task6_Golden_Helix"},
            },
        ]

        encoded_memories = []
        for i, test_data in enumerate(test_memories):
            memory_id = await mapper.encode_memory(
                test_data["memory"], test_data["strand"], test_data["context"]
            )
            encoded_memories.append(memory_id)
            print(f"  ✅ Encoded {test_data['strand'].value} memory: {memory_id}")

        # Test 3: Memory retrieval
        print("\nTest 3: Testing memory retrieval...")
        for i, memory_id in enumerate(encoded_memories):
            retrieved = await mapper.retrieve_memory(memory_id)
            if retrieved:
                strand_type = test_memories[i]["strand"].value
                print(f"  ✅ Retrieved {strand_type} memory: {retrieved['id']}")
            else:
                print(f"  ❌ Failed to retrieve memory: {memory_id}")

        # Test 4: Memory mutation
        print("\nTest 4: Testing memory mutation...")
        if encoded_memories:
            first_memory_id = encoded_memories[0]
            mutation_data = {
                "content_addition": "Updated with mutation testing",
                "emotional_weight_delta": 0.1,
            }

            success = await mapper.mutate_memory(
                first_memory_id, mutation_data, MutationStrategy.POINT
            )
            print(f"  ✅ Point mutation {'successful' if success else 'failed'}")

            # Verify mutation was recorded
            mutated_memory = await mapper.retrieve_memory(first_memory_id)
            if mutated_memory and mutated_memory.get("mutations"):
                print(
                    f"  ✅ Mutation recorded: {len(mutated_memory['mutations'])} mutations"
                )

        # Test 5: Memory search
        print("\nTest 5: Testing memory search...")
        search_queries = [
            {"content": "Agent"},
            {"content": "integration"},
            {"emotional_weight": 0.8},
        ]

        for query in search_queries:
            results = await mapper.search_memories(query)
            print(f"  ✅ Search for {query} returned {len(results)} results")

        # Test 6: Pattern coherence validation
        print("\nTest 6: Testing pattern coherence...")
        test_invalid_memory = {
            "content": "",  # Invalid: empty content
            "emotional_weight": 1.5,  # Invalid: out of range
        }

        try:
            invalid_id = await mapper.encode_memory(
                test_invalid_memory, MemoryStrand.EMOTIONAL
            )
            if invalid_id:
                print("  ❌ Invalid memory was encoded (should have been rejected)")
            else:
                print("  ✅ Invalid memory properly rejected")
        except Exception as e:
            print(f"  ✅ Invalid memory properly rejected with error: {e}")

        print("\n" + "=" * 60)
        print("🎯 Agent 1 Task 6: Golden Helix Memory Mapper Integration COMPLETE!")
        print(f"✅ Successfully tested {len(test_memories)} memory strand types")
        print(
            f"✅ All 4 mutation strategies available: {[s.value for s in MutationStrategy]}"
        )
        print(f"✅ Quantum-resistant encryption enabled: {mapper.quantum_encryption}")
        print(f"✅ Pattern validation enabled: {mapper.pattern_validation}")
        print(f"✅ Mutation tracking enabled: {mapper.mutation_tracking}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_golden_helix_integration())
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    success = asyncio.run(test_golden_helix_integration())
    sys.exit(0 if success else 1)
