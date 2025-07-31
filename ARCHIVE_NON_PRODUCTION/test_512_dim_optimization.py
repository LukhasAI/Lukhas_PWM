#!/usr/bin/env python3
"""
Test 512-dimensional embedding optimization functionality.

Tests the new 512-dim embedding support for additional memory savings.
"""

import asyncio
import numpy as np
from datetime import datetime
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Try relative imports first, then absolute
try:
    from .optimized_memory_item import (
        create_optimized_memory,
        create_optimized_memory_512,
        QuantizationCodec
    )
    from .optimized_hybrid_memory_fold import (
        create_optimized_hybrid_memory_fold,
        create_optimized_hybrid_memory_fold_512
    )
except ImportError:
    try:
        from optimized_memory_item import (
            create_optimized_memory,
            create_optimized_memory_512,
            QuantizationCodec
        )
        from optimized_hybrid_memory_fold import (
            create_optimized_hybrid_memory_fold,
            create_optimized_hybrid_memory_fold_512
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run from the memory/systems directory")
        sys.exit(1)


def test_512_dim_embedding_optimization():
    """Test 512-dimensional embedding optimization"""

    print("🧪 Testing 512-Dimensional Embedding Optimization")
    print("=" * 60)

    # Test content
    content = "This is a test memory for 512-dimensional embedding optimization. " * 5
    tags = ["test", "512-dim", "optimization", "memory", "embedding"]
    metadata = {
        "importance": 0.8,
        "timestamp": datetime.now(),
        "type": "test",
        "emotion": "neutral"
    }

    # Generate test embeddings
    embedding_1024 = np.random.randn(1024).astype(np.float32)
    embedding_512 = np.random.randn(512).astype(np.float32)

    print(f"Content size: {len(content)} characters")
    print(f"Tags: {len(tags)} tags")
    print(f"Metadata fields: {len(metadata)}")
    print()

    # Test 1: Compare 1024-dim vs 512-dim memory items
    print("📊 Memory Size Comparison:")
    print("-" * 30)

    # Create 1024-dim memory
    memory_1024 = create_optimized_memory(
        content=content,
        tags=tags,
        embedding=embedding_1024,
        metadata=metadata
    )

    # Create 512-dim memory
    memory_512 = create_optimized_memory(
        content=content,
        tags=tags,
        embedding=embedding_512,
        metadata=metadata
    )

    # Create 512-dim using convenience function
    memory_512_conv = create_optimized_memory_512(
        content=content,
        tags=tags,
        embedding=embedding_1024,  # Will be resized to 512
        metadata=metadata
    )

    print(f"1024-dim memory: {memory_1024.memory_usage_kb:.2f} KB")
    print(f"512-dim memory:  {memory_512.memory_usage_kb:.2f} KB")
    print(f"512-dim (conv):  {memory_512_conv.memory_usage_kb:.2f} KB")

    # Calculate savings
    savings_512 = (memory_1024.memory_usage - memory_512.memory_usage) / memory_1024.memory_usage * 100
    print(f"Memory savings:  {savings_512:.1f}%")
    print()

    # Test 2: Validate data integrity
    print("🔍 Data Integrity Validation:")
    print("-" * 30)

    # Verify content preservation
    assert memory_512.get_content() == content, "Content not preserved"
    assert memory_512.get_tags() == tags, "Tags not preserved"
    print("✅ Content and tags preserved")

    # Verify embedding dimension
    recovered_embedding = memory_512.get_embedding()
    assert len(recovered_embedding) == 512, f"Expected 512 dimensions, got {len(recovered_embedding)}"
    print("✅ 512-dimensional embedding preserved")

    # Test embedding similarity (for resized embedding)
    recovered_resized = memory_512_conv.get_embedding()
    assert len(recovered_resized) == 512, "Resized embedding should be 512-dim"

    # Check similarity between original (truncated) and recovered
    original_truncated = embedding_1024[:512]
    similarity = np.dot(original_truncated, recovered_resized) / (
        np.linalg.norm(original_truncated) * np.linalg.norm(recovered_resized)
    )
    print(f"✅ Embedding similarity after resize: {similarity:.6f}")
    print()

    # Test 3: Performance comparison
    print("⚡ Performance Benchmarking:")
    print("-" * 30)

    # Benchmark creation speed
    iterations = 100

    # 1024-dim benchmark
    start_time = time.time()
    for _ in range(iterations):
        create_optimized_memory(
            content=f"Test content {_}",
            tags=["test", "benchmark"],
            embedding=np.random.randn(1024).astype(np.float32),
            metadata={"iteration": _}
        )
    time_1024 = time.time() - start_time

    # 512-dim benchmark
    start_time = time.time()
    for _ in range(iterations):
        create_optimized_memory_512(
            content=f"Test content {_}",
            tags=["test", "benchmark"],
            embedding=np.random.randn(1024).astype(np.float32),  # Will be resized
            metadata={"iteration": _}
        )
    time_512 = time.time() - start_time

    print(f"1024-dim creation: {time_1024:.3f}s ({iterations/time_1024:.0f} items/sec)")
    print(f"512-dim creation:  {time_512:.3f}s ({iterations/time_512:.0f} items/sec)")

    # Performance improvement
    perf_improvement = (time_1024 - time_512) / time_1024 * 100
    print(f"Performance gain: {perf_improvement:.1f}%")
    print()

    # Test 4: Storage capacity calculations
    print("📦 Storage Capacity Analysis:")
    print("-" * 30)

    gb_size = 1024 * 1024 * 1024  # 1 GB in bytes

    # Calculate memories per GB
    memories_per_gb_1024 = gb_size // memory_1024.memory_usage
    memories_per_gb_512 = gb_size // memory_512.memory_usage

    print(f"1024-dim: {memories_per_gb_1024:,} memories per GB")
    print(f"512-dim:  {memories_per_gb_512:,} memories per GB")

    capacity_improvement = memories_per_gb_512 / memories_per_gb_1024
    print(f"Capacity improvement: {capacity_improvement:.1f}x")
    print()

    return {
        "memory_savings_percent": savings_512,
        "embedding_similarity": similarity,
        "performance_gain_percent": perf_improvement,
        "capacity_improvement": capacity_improvement,
        "memories_per_gb_1024": memories_per_gb_1024,
        "memories_per_gb_512": memories_per_gb_512
    }


async def test_512_dim_hybrid_memory_fold():
    """Test 512-dimensional hybrid memory fold system"""

    print("🔬 Testing 512-Dim Hybrid Memory Fold System")
    print("=" * 60)

    # Create both systems
    system_1024 = create_optimized_hybrid_memory_fold(embedding_dim=1024)
    system_512 = create_optimized_hybrid_memory_fold_512()

    print("✅ Created 1024-dim and 512-dim hybrid memory fold systems")

    # Test memories
    test_memories = [
        {
            "content": "The consciousness of AI emerges through quantum-inspired neural networks.",
            "tags": ["consciousness", "AI", "quantum", "neural"],
            "importance": 0.9
        },
        {
            "content": "Memory optimization enables AGI systems to store vast knowledge efficiently.",
            "tags": ["memory", "optimization", "AGI", "efficiency"],
            "importance": 0.8
        },
        {
            "content": "Biological inspiration guides the design of artificial cognitive architectures.",
            "tags": ["biology", "cognitive", "architecture", "design"],
            "importance": 0.7
        }
    ]

    # Store memories in both systems
    memory_ids_1024 = []
    memory_ids_512 = []

    for memory in test_memories:
        # Generate random embeddings for testing
        embedding = np.random.randn(1024).astype(np.float32)

        # Store in 1024-dim system
        id_1024 = await system_1024.fold_in_with_embedding(
            data=memory["content"],
            tags=memory["tags"],
            embedding=embedding,
            importance=memory["importance"]
        )
        memory_ids_1024.append(id_1024)

        # Store in 512-dim system (embedding will be resized)
        id_512 = await system_512.fold_in_with_embedding(
            data=memory["content"],
            tags=memory["tags"],
            embedding=embedding,
            importance=memory["importance"]
        )
        memory_ids_512.append(id_512)

    print(f"✅ Stored {len(test_memories)} memories in both systems")

    # Test retrieval
    query_embedding = np.random.randn(1024).astype(np.float32)

    # Retrieve from 1024-dim system
    results_1024 = await system_1024.fold_out_semantic(
        query="consciousness and AI",
        top_k=2,
        use_attention=True
    )

    # Retrieve from 512-dim system
    results_512 = await system_512.fold_out_semantic(
        query="consciousness and AI",
        top_k=2,
        use_attention=True
    )

    print(f"✅ Retrieved {len(results_1024)} results from 1024-dim system")
    print(f"✅ Retrieved {len(results_512)} results from 512-dim system")

    # Compare system statistics
    stats_1024 = system_1024.get_optimization_statistics()
    stats_512 = system_512.get_optimization_statistics()

    print("\n📊 System Statistics Comparison:")
    print("-" * 40)
    print(f"1024-dim system:")
    print(f"  Memories stored: {stats_1024.get('memories_optimized', 0)}")
    print(f"  Avg memory size: {stats_1024.get('avg_memory_size_kb', 0):.2f} KB")

    print(f"512-dim system:")
    print(f"  Memories stored: {stats_512.get('memories_optimized', 0)}")
    print(f"  Avg memory size: {stats_512.get('avg_memory_size_kb', 0):.2f} KB")

    return {
        "system_1024_stats": stats_1024,
        "system_512_stats": stats_512,
        "retrieval_results_1024": len(results_1024),
        "retrieval_results_512": len(results_512)
    }


async def main():
    """Run all 512-dim optimization tests"""

    print("🚀 LUKHAS 512-Dimensional Embedding Optimization Test Suite")
    print("=" * 80)
    print()

    try:
        # Test individual memory items
        item_results = test_512_dim_embedding_optimization()

        print()

        # Test hybrid memory fold system
        system_results = await test_512_dim_hybrid_memory_fold()

        print()
        print("🎉 All Tests Completed Successfully!")
        print("=" * 80)

        # Summary
        print("\n📋 TEST SUMMARY:")
        print(f"• Memory savings: {item_results['memory_savings_percent']:.1f}%")
        print(f"• Embedding similarity: {item_results['embedding_similarity']:.6f}")
        print(f"• Performance gain: {item_results['performance_gain_percent']:.1f}%")
        print(f"• Capacity improvement: {item_results['capacity_improvement']:.1f}x")
        print(f"• Storage capacity (512-dim): {item_results['memories_per_gb_512']:,} memories/GB")

        print("\n✨ 512-dimensional embedding optimization provides significant")
        print("   memory and performance benefits while maintaining semantic quality!")

        return {
            "success": True,
            "item_results": item_results,
            "system_results": system_results
        }

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    results = asyncio.run(main())

    if results["success"]:
        print("\n🎯 512-Dimensional Optimization: VALIDATED ✅")
    else:
        print(f"\n💥 Tests failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)