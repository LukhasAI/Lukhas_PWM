"""Dream Memoryscape Viewport
===========================

Render dream data into a 3D symbolic scene. Identity drift across dreams is
tracked via :func:`compute_drift_score` and displayed in the figure title.
"""

from __future__ import annotations


from typing import List, Dict, Any, Tuple

import plotly.graph_objects as go

from trace.drift_metrics import compute_drift_score


class DreamMemoryscapeViewport:
    """Create an interactive 3D viewport of dream states."""

    def __init__(self) -> None:
        self._last_state: Dict[str, Any] | None = None
        self.total_drift: float = 0.0

    def _coords_for_dream(self, idx: int, dream: Dict[str, Any]) -> Tuple[float, float, float]:
        affect = float(dream.get("affect_delta", 0.0))
        theta = float(dream.get("theta_delta", idx))
        return float(idx), affect, theta

    def render_scene(self, dreams: List[Dict[str, Any]]) -> go.Figure:
        """Return a Plotly figure visualizing the dream sequence."""
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        texts: List[str] = []
        colors: List[float] = []

        prev = self._last_state
        for idx, dream in enumerate(dreams):
            x, y, z = self._coords_for_dream(idx, dream)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            texts.append(dream.get("description", f"Dream {idx}"))
            if prev is not None:
                drift = compute_drift_score(prev, dream)
                self.total_drift += drift
                colors.append(drift)
            else:
                colors.append(0.0)
            prev = dream

        self._last_state = prev

        scatter = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            marker=dict(size=5, color=colors, colorscale="Viridis"),
            text=texts,
        )

        fig = go.Figure(data=[scatter])
        fig.update_layout(
            title=f"Dream Memoryscape - Identity Drift {self.total_drift:.2f}",
            scene=dict(xaxis_title="time", yaxis_title="affect", zaxis_title="theta"),
        )
        return fig
