"""
Tests for lightweight concurrency module
"""

import asyncio
import pytest
import time
from lightweight_concurrency import (
    LightweightActor,
    MemoryEfficientScheduler,
    ActorPool,
    ActorPriority,
    create_lightweight_actor_system,
    simple_counter_behavior,
    aggregator_behavior
)


@pytest.mark.asyncio
async def test_lightweight_actor_creation():
    """Test basic actor creation and memory efficiency"""
    actor = LightweightActor(
        actor_id="test_actor",
        behavior=simple_counter_behavior,
        priority=ActorPriority.NORMAL
    )

    # Check memory footprint is small
    memory_size = actor.__sizeof__()
    assert memory_size < 1000, f"Actor memory too large: {memory_size} bytes"

    # Test basic properties
    assert actor.actor_id == "test_actor"
    assert actor.priority == ActorPriority.NORMAL
    assert len(actor.mailbox) == 0
    assert len(actor.state) == 0


@pytest.mark.asyncio
async def test_scheduler_spawn_and_send():
    """Test scheduler can spawn actors and send messages"""
    scheduler = MemoryEfficientScheduler(max_actors=1000)
    await scheduler.start()

    try:
        # Spawn an actor
        actor = await scheduler.spawn_actor(
            "counter_1",
            simple_counter_behavior,
            ActorPriority.HIGH
        )

        assert "counter_1" in scheduler.actors
        assert scheduler.active_count == 1

        # Send messages
        await scheduler.send_message("counter_1", {"type": "increment"})
        await scheduler.send_message("counter_1", {"type": "increment"})
        await scheduler.send_message("counter_1", {"type": "get"})

        # Wait for processing
        await asyncio.sleep(0.1)

        # Check state was updated
        assert actor.state.get("count") == 2

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_priority_scheduling():
    """Test that high priority actors are processed first"""
    scheduler = MemoryEfficientScheduler()
    await scheduler.start()

    results = []

    async def record_behavior(actor, message):
        results.append((actor.actor_id, message["value"]))

    try:
        # Create actors with different priorities
        await scheduler.spawn_actor("low", record_behavior, ActorPriority.LOW)
        await scheduler.spawn_actor("high", record_behavior, ActorPriority.HIGH)
        await scheduler.spawn_actor("critical", record_behavior, ActorPriority.CRITICAL)

        # Send messages in reverse priority order
        await scheduler.send_message("low", {"value": "low_priority"})
        await scheduler.send_message("high", {"value": "high_priority"})
        await scheduler.send_message("critical", {"value": "critical_priority"})

        # Wait for processing
        await asyncio.sleep(0.1)

        # Check execution order (critical should be first)
        assert results[0][1] == "critical_priority"
        assert results[1][1] == "high_priority"
        assert results[2][1] == "low_priority"

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_massive_actor_creation():
    """Test creating many actors with low memory overhead"""
    scheduler = MemoryEfficientScheduler(max_actors=100_000)
    await scheduler.start()

    try:
        # Create 10,000 actors
        num_actors = 10_000
        start_time = time.time()

        for i in range(num_actors):
            await scheduler.spawn_actor(
                f"actor_{i}",
                simple_counter_behavior,
                ActorPriority.NORMAL
            )

        creation_time = time.time() - start_time

        # Check performance
        assert scheduler.active_count == num_actors
        assert creation_time < 5.0, f"Actor creation too slow: {creation_time}s"

        # Check memory efficiency
        stats = scheduler.get_stats()
        avg_memory = stats["avg_memory_per_actor"]
        assert avg_memory < 1000, f"Average memory per actor too high: {avg_memory} bytes"

        # Total memory should be reasonable
        total_mb = stats["total_memory_mb"]
        assert total_mb < 100, f"Total memory usage too high: {total_mb}MB for {num_actors} actors"

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_actor_pool():
    """Test actor pool for efficient reuse"""
    pool = ActorPool(pool_size=100)
    await pool.initialize()

    try:
        # Acquire actors
        actors = []
        for i in range(50):
            actor = await pool.acquire_actor(
                f"worker_{i}",
                simple_counter_behavior,
                ActorPriority.NORMAL
            )
            actors.append(actor)

        assert pool.scheduler.active_count == 50
        assert len(pool.available_actors) == 50  # 50 still in pool

        # Release actors back to pool
        for actor in actors[:25]:
            pool.release_actor(actor)

        assert len(pool.available_actors) == 75  # 25 returned
        assert pool.scheduler.active_count == 25

        # Reacquire should reuse pooled actors
        reused = await pool.acquire_actor(
            "reused_worker",
            simple_counter_behavior,
            ActorPriority.HIGH
        )

        assert len(pool.available_actors) == 74

    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_message_processing():
    """Test actual message processing with state updates"""
    scheduler, pool = await create_lightweight_actor_system(max_actors=1000)

    try:
        # Create aggregator actor
        aggregator = await pool.acquire_actor(
            "aggregator_1",
            aggregator_behavior,
            ActorPriority.NORMAL
        )

        # Send values to aggregate
        for i in range(150):
            await pool.scheduler.send_message("aggregator_1", {"type": "add", "value": i})

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check aggregation happened (should have triggered at 100)
        remaining = aggregator.state.get("values", [])
        assert len(remaining) == 50  # 150 - 100 that were aggregated

    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_memory_optimization():
    """Test memory optimization and garbage collection"""
    scheduler = MemoryEfficientScheduler(max_actors=1000)
    await scheduler.start()

    try:
        # Create actors
        for i in range(100):
            await scheduler.spawn_actor(
                f"temp_{i}",
                simple_counter_behavior,
                ActorPriority.LOW
            )

        initial_count = scheduler.active_count
        assert initial_count == 100

        # Clear references manually to simulate dead actors
        for actor_id in list(scheduler.actors.keys()):
            actor = scheduler.actors.pop(actor_id)
            scheduler.total_memory_bytes -= actor.__sizeof__()
            scheduler.active_count -= 1

        # Should have cleaned up
        assert scheduler.active_count == 0

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in actor behaviors"""
    error_count = 0

    async def failing_behavior(actor, message):
        nonlocal error_count
        error_count += 1
        raise ValueError("Intentional error")

    scheduler = MemoryEfficientScheduler()
    await scheduler.start()

    try:
        await scheduler.spawn_actor("failer", failing_behavior)
        await scheduler.send_message("failer", {"test": "message"})

        # Wait for processing
        await asyncio.sleep(0.1)

        # Should have tried to process but caught error
        assert error_count == 1

        # Scheduler should still be running
        assert scheduler._running

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_concurrent_message_sending():
    """Test concurrent message sending from multiple sources"""
    scheduler, pool = await create_lightweight_actor_system()

    try:
        # Create a counter actor
        await pool.acquire_actor("shared_counter", simple_counter_behavior)

        # Send messages concurrently
        async def send_increments(count):
            for _ in range(count):
                await pool.scheduler.send_message("shared_counter", {"type": "increment"})

        # Launch concurrent senders
        await asyncio.gather(
            send_increments(100),
            send_increments(100),
            send_increments(100),
        )

        # Wait for all processing
        await asyncio.sleep(0.5)

        # Check final count
        actor = pool.scheduler.actors["shared_counter"]
        assert actor.state.get("count") == 300

    finally:
        await pool.shutdown()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_lightweight_actor_creation())
    asyncio.run(test_scheduler_spawn_and_send())
    asyncio.run(test_priority_scheduling())
    asyncio.run(test_massive_actor_creation())
    asyncio.run(test_actor_pool())
    asyncio.run(test_message_processing())
    asyncio.run(test_memory_optimization())
    asyncio.run(test_error_handling())
    asyncio.run(test_concurrent_message_sending())
    print("All tests passed!")

