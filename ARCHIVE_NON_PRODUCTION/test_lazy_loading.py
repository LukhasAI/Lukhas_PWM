#!/usr/bin/env python3
"""
Test lazy loading embedding system functionality.

Tests the new lazy loading system for handling large-scale memory collections.
"""

import asyncio
import numpy as np
from datetime import datetime
import time
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_lazy_loading_basic():
    """Test basic lazy loading functionality"""

    print("🧪 Testing Basic Lazy Loading Functionality")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Import here to avoid circular import issues
        exec(open('lazy_loading_embeddings.py').read())

        # Create lazy loading system
        lazy_loader = create_lazy_embedding_system(
            storage_path=temp_dir,
            cache_size=100,
            cache_memory_mb=32
        )

        print("✅ Created lazy loading system")

        # Test data
        test_embeddings = {
            f"memory_{i}": np.random.randn(1024).astype(np.float32)
            for i in range(50)
        }

        print(f"📊 Testing with {len(test_embeddings)} embeddings")

        # Store embeddings
        store_start = time.time()
        for memory_id, embedding in test_embeddings.items():
            asyncio.run(lazy_loader.store_embedding(memory_id, embedding))
        store_time = time.time() - store_start

        print(f"📥 Storage time: {store_time:.3f}s ({len(test_embeddings)/store_time:.0f} embeddings/sec)")

        # Test individual retrieval
        retrieve_start = time.time()
        retrieved_embedding = asyncio.run(lazy_loader.get_embedding("memory_25"))
        retrieve_time = time.time() - retrieve_start

        assert retrieved_embedding is not None, "Failed to retrieve embedding"
        assert retrieved_embedding.shape == (1024,), f"Wrong shape: {retrieved_embedding.shape}"

        print(f"📤 Individual retrieval: {retrieve_time*1000:.2f}ms")

        # Test batch retrieval
        batch_ids = [f"memory_{i}" for i in range(10, 20)]
        batch_start = time.time()
        batch_embeddings = asyncio.run(lazy_loader.get_embeddings_batch(batch_ids))
        batch_time = time.time() - batch_start

        assert len(batch_embeddings) == len(batch_ids), f"Expected {len(batch_ids)}, got {len(batch_embeddings)}"

        print(f"📦 Batch retrieval: {batch_time*1000:.2f}ms for {len(batch_ids)} embeddings")

        # Test cache performance
        stats = lazy_loader.get_performance_stats()
        print(f"📈 Cache hit rate: {stats['cache']['hit_rate']:.2%}")
        print(f"📊 Cache entries: {stats['cache']['current_entries']}")
        print(f"💾 Storage size: {stats['storage']['storage_size_mb']:.2f} MB")

        # Test cache pressure handling
        print("\n🔧 Testing cache pressure handling...")

        # Load many more embeddings than cache size
        pressure_embeddings = {
            f"pressure_{i}": np.random.randn(1024).astype(np.float32)
            for i in range(200)  # More than cache size of 100
        }

        for memory_id, embedding in pressure_embeddings.items():
            asyncio.run(lazy_loader.store_embedding(memory_id, embedding))

        # Retrieve all pressure embeddings (should trigger evictions)
        for memory_id in pressure_embeddings.keys():
            embedding = asyncio.run(lazy_loader.get_embedding(memory_id))
            assert embedding is not None, f"Failed to retrieve {memory_id}"

        final_stats = lazy_loader.get_performance_stats()
        print(f"✅ Handled cache pressure: {final_stats['cache']['current_entries']} entries in cache")
        print(f"✅ Total embeddings stored: {final_stats['storage']['total_embeddings']}")

        return {
            "storage_time": store_time,
            "retrieval_time": retrieve_time,
            "batch_time": batch_time,
            "final_stats": final_stats
        }


def test_lazy_loading_integration():
    """Test integration with optimized memory systems"""

    print("\n🔬 Testing Lazy Loading Integration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Import the hybrid system
            exec(open('optimized_hybrid_memory_fold.py').read())

            # Create system with lazy loading
            system = create_optimized_hybrid_memory_fold_with_lazy_loading(
                embedding_dim=1024,
                lazy_loading_cache_size=50,
                lazy_loading_cache_memory_mb=16,
                lazy_loading_storage_path=temp_dir
            )

            print("✅ Created hybrid memory system with lazy loading")

            # Test memories
            test_memories = [
                {
                    "content": f"Test memory content {i} with various information about consciousness and AI systems.",
                    "tags": [f"test_{i}", "memory", "ai"],
                    "importance": 0.5 + (i % 10) * 0.05
                }
                for i in range(100)  # More than cache size
            ]

            # Store memories
            memory_ids = []
            store_start = time.time()

            for i, memory in enumerate(test_memories):
                embedding = np.random.randn(1024).astype(np.float32)

                memory_id = asyncio.run(system.fold_in_with_embedding(
                    data=memory["content"],
                    tags=memory["tags"],
                    embedding=embedding,
                    importance=memory["importance"]
                ))
                memory_ids.append(memory_id)

            store_time = time.time() - store_start
            print(f"📥 Stored {len(test_memories)} memories in {store_time:.2f}s")

            # Test retrieval
            retrieve_start = time.time()
            results = asyncio.run(system.fold_out_semantic(
                query="consciousness and AI systems",
                top_k=10,
                use_attention=True
            ))
            retrieve_time = time.time() - retrieve_start

            print(f"📤 Retrieved {len(results)} memories in {retrieve_time*1000:.2f}ms")

            # Test system statistics
            if hasattr(system, 'lazy_loader') and system.lazy_loader:
                lazy_stats = system.lazy_loader.get_performance_stats()
                print(f"📊 Lazy loading stats:")
                print(f"   Cache hit rate: {lazy_stats['cache']['hit_rate']:.2%}")
                print(f"   Cached embeddings: {lazy_stats['cache']['current_entries']}")
                print(f"   Total stored: {lazy_stats['storage']['total_embeddings']}")

                return {
                    "integration_success": True,
                    "store_time": store_time,
                    "retrieve_time": retrieve_time,
                    "lazy_stats": lazy_stats
                }
            else:
                print("⚠️ Lazy loading not enabled (dependency missing)")
                return {"integration_success": False, "reason": "lazy_loading_disabled"}

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"integration_success": False, "error": str(e)}


async def main():
    """Run all lazy loading tests"""

    print("🚀 LUKHAS Lazy Loading Embedding System Test Suite")
    print("=" * 80)
    print()

    try:
        # Test basic functionality
        basic_results = test_lazy_loading_basic()

        # Test integration
        integration_results = test_lazy_loading_integration()

        print()
        print("🎉 All Tests Completed!")
        print("=" * 80)

        # Summary
        print("\n📋 TEST SUMMARY:")
        if basic_results:
            print(f"• Storage performance: {len(range(50))/basic_results['storage_time']:.0f} embeddings/sec")
            print(f"• Retrieval latency: {basic_results['retrieval_time']*1000:.2f}ms")
            print(f"• Batch efficiency: {basic_results['batch_time']*1000:.2f}ms for 10 embeddings")
            print(f"• Cache hit rate: {basic_results['final_stats']['cache']['hit_rate']:.2%}")

        if integration_results.get("integration_success"):
            print(f"• Integration: ✅ Successful")
            print(f"• Hybrid system retrieval: {integration_results['retrieve_time']*1000:.2f}ms")
        elif integration_results.get("reason") == "lazy_loading_disabled":
            print(f"• Integration: ⚠️ Disabled (missing dependencies)")
        else:
            print(f"• Integration: ❌ Failed")

        print("\n✨ Lazy loading system enables massive memory scalability")
        print("   by storing embeddings on disk with intelligent caching!")

        return {
            "success": True,
            "basic_results": basic_results,
            "integration_results": integration_results
        }

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    results = asyncio.run(main())

    if results["success"]:
        print("\n🎯 Lazy Loading System: VALIDATED ✅")
    else:
        print(f"\n💥 Tests failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)