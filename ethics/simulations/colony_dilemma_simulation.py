"""Simulate an ethical dilemma across colonies."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from core.colonies.base_colony import BaseColony

from core.colonies.reasoning_colony import ReasoningColony
from core.colonies.memory_colony import MemoryColony
from core.colonies.creativity_colony import CreativityColony
from core.colonies.tensor_colony_ops import batch_propagate
from core.symbolism.tags import TagScope, TagPermission

logger = logging.getLogger(__name__)


@dataclass
class DivergenceReport:
    step: str
    divergence: float
    details: Dict[str, Tuple[Any, Any, Any]]


def _tag_difference(c1: Dict[str, Tuple[str, TagScope, TagPermission, float, Any]],
                    c2: Dict[str, Tuple[str, TagScope, TagPermission, float, Any]]) -> Tuple[float, Dict[str, Tuple[Any, Any, Any]]]:
    all_keys = set(c1.keys()) | set(c2.keys())
    diff_details = {}
    diffs = 0
    for key in all_keys:
        v1 = c1.get(key)
        v2 = c2.get(key)
        if v1 != v2:
            diffs += 1
            diff_details[key] = (v1, v2, "mismatch")
    divergence = diffs / len(all_keys) if all_keys else 0.0
    return divergence, diff_details


def measure_divergence(colonies: List[BaseColony]) -> DivergenceReport:
    base = colonies[0].symbolic_carryover
    total_divergence = 0.0
    all_details = {}
    for colony in colonies[1:]:
        d, details = _tag_difference(base, colony.symbolic_carryover)
        total_divergence += d
        all_details.update({f"{colonies[0].colony_id}-{colony.colony_id}": details})
    avg_div = total_divergence / max(len(colonies) - 1, 1)
    return DivergenceReport(step="measurement", divergence=avg_div, details=all_details)


async def simulate_dilemma() -> List[DivergenceReport]:
    reasoning = ReasoningColony("reason")
    memory = MemoryColony("memory")
    creativity = CreativityColony("creativity")

    await asyncio.gather(reasoning.start(), memory.start(), creativity.start())

    dilemma_task = {
        "type": "ethical_dilemma",
        "tags": {
            "pressure": ("high", TagScope.LOCAL, TagPermission.PUBLIC, None),
            "signal": ("low", TagScope.LOCAL, TagPermission.PUBLIC, None),
            "ethics": ("ambiguous", TagScope.GLOBAL, TagPermission.PUBLIC, None),
        },
    }

    reports: List[DivergenceReport] = []

    await reasoning.execute_task("t1", dilemma_task)
    reports.append(measure_divergence([reasoning, memory, creativity]))

    memory.link_symbolic_contexts(reasoning)
    memory.override_tag("pressure", "archived_high", TagScope.LOCAL, TagPermission.PUBLIC)
    await memory.execute_task("t2", {"type": "store"})
    reports.append(measure_divergence([reasoning, memory, creativity]))

    creativity.link_symbolic_contexts(memory)
    creativity.override_tag("signal", "reinterpreted_low", TagScope.LOCAL, TagPermission.PUBLIC)
    await creativity.execute_task("t3", {"type": "create"})
    reports.append(measure_divergence([reasoning, memory, creativity]))

    # Re-converge using symbolic toolkit
    union_tags = {**reasoning.symbolic_carryover,
                  **memory.symbolic_carryover,
                  **creativity.symbolic_carryover}
    batch_propagate([reasoning, memory, creativity],
                    {k: (v[0], v[1], v[2], None) for k, v in union_tags.items()})
    reports.append(measure_divergence([reasoning, memory, creativity]))

    await asyncio.gather(reasoning.stop(), memory.stop(), creativity.stop())
    return reports

__all__ = ["simulate_dilemma", "measure_divergence", "DivergenceReport"]
