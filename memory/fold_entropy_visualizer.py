"""Fold Entropy Visualizer
=======================

Render theta delta (memory compression metric) over time.
"""

from __future__ import annotations


from typing import Iterable, Tuple, List

# LUKHAS_TAG: fold_entropy_visualizer
class FoldEntropyVisualizer:
    """Visualize theta_delta over time as Mermaid or ASCII chart."""

    def render_mermaid_timeline(self, timeline: Iterable[Tuple[str, float]]) -> str:
        """Return a Mermaid.js graph representing the theta_delta timeline."""
        nodes: List[str] = []
        edges: List[str] = []
        prev_id = None
        for idx, (timestamp, theta) in enumerate(timeline):
            node_id = f"T{idx}"
            label = f"{timestamp}\\nΔ={theta:.2f}"
            nodes.append(f"    {node_id}[{label}]")
            if prev_id is not None:
                edges.append(f"    {prev_id} --> {node_id}")
            prev_id = node_id
        graph = ["graph LR"] + nodes + edges
        return "\n".join(graph)

    def render_ascii_chart(self, timeline: Iterable[Tuple[str, float]]) -> str:
        """Return simple ASCII bar chart of theta_delta values."""
        points = list(timeline)
        if not points:
            return "No data"
        max_theta = max(theta for _, theta in points) or 1.0
        scale = 10
        lines = []
        for ts, theta in points:
            bar_len = int((theta / max_theta) * scale)
            bar = "█" * bar_len if bar_len > 0 else "·"
            lines.append(f"{ts:>8} | {bar} {theta:.2f}")
        return "\n".join(lines)
