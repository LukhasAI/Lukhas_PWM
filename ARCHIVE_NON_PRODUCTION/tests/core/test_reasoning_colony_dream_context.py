"""Integration test for ReasoningColony with SymbolAwareTieredMemory."""

import asyncio

from core.colonies.reasoning_colony import ReasoningColony
from memory.symbol_aware_tiered_memory import SymbolAwareTieredMemory


async def _run(colony, task):
    await colony.start()
    result = await colony.execute_task("t1", task)
    await colony.stop()
    return result


def test_high_stakes_uses_dream_context():
    mem = SymbolAwareTieredMemory()
    mem.store("d1", {"info": "dream"}, is_dream=True)
    colony = ReasoningColony("c1", memory_system=mem)
    res = asyncio.run(_run(colony, {"high_stakes": True}))
    assert res["context"]["dream_memories_used"] == 1


def test_non_high_stakes_ignores_dream_context():
    mem = SymbolAwareTieredMemory()
    mem.store("d1", {"info": "dream"}, is_dream=True)
    colony = ReasoningColony("c2", memory_system=mem)
    res = asyncio.run(_run(colony, {}))
    assert res["context"]["dream_memories_used"] == 0
