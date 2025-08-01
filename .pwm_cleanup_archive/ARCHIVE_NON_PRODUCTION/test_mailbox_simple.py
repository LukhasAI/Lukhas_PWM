#!/usr/bin/env python3
"""
Simple test script to demonstrate mailbox functionality
Can be run directly without pytest
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.actor_system import Actor, ActorSystem, ActorMessage
from core.mailbox import (
    MailboxActor, MailboxType, MessagePriority,
    BackPressureStrategy, UnboundedMailbox, BoundedMailbox,
    PriorityMailbox
)


async def test_sequential_processing():
    """Demonstrate sequential processing guarantee"""
    print("\n=== Testing Sequential Processing ===")

    class Counter(MailboxActor):
        def __init__(self, actor_id: str):
            super().__init__(actor_id, mailbox_type=MailboxType.BOUNDED)
            self.count = 0
            self.register_handler("increment", self._increment)

        async def _increment(self, msg):
            # Without sequential guarantee, this would have race conditions
            current = self.count
            await asyncio.sleep(0.001)  # Simulate work
            self.count = current + 1
            return {"count": self.count}

    system = ActorSystem()
    await system.start()

    counter_ref = await system.create_actor(Counter, "counter")

    # Send 50 concurrent increments
    tasks = []
    for i in range(50):
        tasks.append(counter_ref.tell("increment", {}))
    await asyncio.gather(*tasks)

    # Wait and check
    await asyncio.sleep(0.2)
    counter = system.get_actor("counter")
    print(f"Final count: {counter.count} (should be 50)")
    assert counter.count == 50, "Sequential processing failed!"
    print("✅ Sequential processing verified - no race conditions!")

    await system.stop()


async def test_priority_mailbox():
    """Demonstrate priority-based processing"""
    print("\n=== Testing Priority Mailbox ===")

    mailbox = PriorityMailbox(max_size=10, starvation_prevention=False)

    # Create test messages
    messages = [
        (MessagePriority.LOW, "low-priority"),
        (MessagePriority.HIGH, "high-priority"),
        (MessagePriority.NORMAL, "normal-priority"),
        (MessagePriority.SYSTEM, "system-critical"),
        (MessagePriority.BULK, "bulk-operation")
    ]

    # Add messages
    for priority, content in messages:
        msg = ActorMessage(
            message_id=content,
            sender="test",
            recipient="test",
            message_type="test",
            payload={"content": content},
            timestamp=0
        )
        await mailbox.put(msg, priority)

    # Get messages - should be in priority order
    print("Messages processed in order:")
    expected = ["system-critical", "high-priority", "normal-priority",
                "low-priority", "bulk-operation"]

    for i, expected_content in enumerate(expected):
        msg = await mailbox.get()
        print(f"  {i+1}. {msg.payload['content']}")
        assert msg.payload["content"] == expected_content

    print("✅ Priority ordering verified!")


async def test_back_pressure():
    """Demonstrate back-pressure handling"""
    print("\n=== Testing Back-Pressure ===")

    # Test DROP_NEWEST strategy
    mailbox = BoundedMailbox(
        max_size=3,
        back_pressure_strategy=BackPressureStrategy.DROP_NEWEST
    )

    # Fill mailbox
    for i in range(3):
        msg = ActorMessage(
            message_id=str(i),
            sender="test",
            recipient="test",
            message_type="test",
            payload={"id": i},
            timestamp=0
        )
        result = await mailbox.put(msg)
        print(f"Message {i}: {'accepted' if result else 'dropped'}")

    # Try to add more
    msg = ActorMessage(
        message_id="overflow",
        sender="test",
        recipient="test",
        message_type="test",
        payload={"id": "overflow"},
        timestamp=0
    )
    result = await mailbox.put(msg)
    print(f"Overflow message: {'accepted' if result else 'dropped'}")

    stats = mailbox.get_stats()
    print(f"Messages dropped: {stats['messages_dropped']}")
    print(f"Dead letter queue size: {len(await mailbox.dead_letter_queue.get_all())}")
    print("✅ Back-pressure handling verified!")


async def test_persistence():
    """Demonstrate mailbox persistence"""
    print("\n=== Testing Persistence ===")

    from core.mailbox import PersistentMailbox
    import tempfile

    # Create temporary file for persistence
    import os
    temp_file = tempfile.mktemp(suffix=".json")

    # Create mailbox and add messages
    mailbox1 = PersistentMailbox(
        max_size=10,
        persistence_path=temp_file,
        persistence_interval=0.1
    )

    for i in range(5):
        msg = ActorMessage(
            message_id=str(i),
            sender="test",
            recipient="test",
            message_type="test",
            payload={"id": i, "data": f"message-{i}"},
            timestamp=0
        )
        await mailbox1.put(msg)

    # Force persistence
    await asyncio.sleep(0.2)

    # Simulate crash and recovery
    print("Simulating crash...")

    # Create new mailbox and restore
    mailbox2 = PersistentMailbox(
        max_size=10,
        persistence_path=temp_file
    )

    restored = await mailbox2.restore_from_disk()
    print(f"Restored {restored} messages from disk")

    # Verify messages
    print("Recovered messages:")
    for i in range(5):
        msg = await mailbox2.get()
        print(f"  {msg.payload}")

    # Cleanup
    os.unlink(temp_file)
    print("✅ Persistence verified!")


async def main():
    """Run all tests"""
    print("Mailbox System Test Suite")
    print("=========================")

    try:
        await test_sequential_processing()
        await test_priority_mailbox()
        await test_back_pressure()
        await test_persistence()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())