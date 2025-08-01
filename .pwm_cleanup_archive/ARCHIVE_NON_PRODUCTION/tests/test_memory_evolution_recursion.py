import unittest
from unittest.mock import MagicMock

from memory.core_memory.memory_evolution import MemoryEvolution


class TestMultiCycleRecursion(unittest.TestCase):
    def test_run_multi_cycle_recursion(self):
        mem = MemoryEvolution()
        mem.maintenance_cycle = MagicMock()
        mem.run_multi_cycle_recursion(cycles=3)
        self.assertEqual(mem.maintenance_cycle.call_count, 3)


if __name__ == "__main__":
    unittest.main()
