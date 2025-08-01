# LUKHAS Visual Drift Map Prototype - Conceptual Design

**ΛORIGIN_AGENT:** Jules-10
**ΛTASK_ID:** 233
**ΛCOMMIT_WINDOW:** symbolic-diagnostics-viz
**ΛAPPROVED_BY:** Human Overseer (GRDM)
**ΛTASK_TYPE:** Symbolic Metrics / Diagnostics Visualization

## 1. Purpose & Use Cases

### Purpose
The Visual Drift Map Prototype (VDMP) aims to provide an intuitive, visual representation of Symbolic Drift Density (SDD) and related drift metrics across the LUKHAS codebase and its operational timelines. It translates the conceptual data from the [Symbolic Drift Density Estimator (SDDE)](../../docs/diagnostics/DRIFT_DENSITY_ESTIMATOR.md) into actionable visual insights. This allows for rapid identification of drift hotspots (`#ΛDRIFT_ZONE`), potential `#ΛENTROPY_CLUSTER`s, temporal drift trends, and the effectiveness of recovery efforts (`#ΛANCHOR_REENTRY`).

### Use Cases
*   **Developer/Maintainer Anomaly Detection:** Quickly identify modules, files, or specific code regions exhibiting high or increasing drift density, guiding targeted review, refactoring, or debugging efforts.
*   **AGI System Health Monitoring:** Provide analysts and overseers with a high-level dashboard view of overall symbolic health, highlighting areas of degrading stability or emerging `#ΛOVERLOAD_REGION`s.
*   **Temporal Trend Analysis:** Understand how drift density evolves over time (e.g., across sprints, releases, or operational periods) and correlate drift spikes with specific system events or code changes.
*   **Impact Assessment of Changes:** Visualize the effect of new code commits or architectural changes on drift density in affected and related components.
*   **Effectiveness of Recovery Measures:** Visually confirm if `#ΛRECOVERY_POINT` implementations or other stabilization efforts lead to a reduction in drift density in targeted zones (`#ΛRE-CENTER_VIS`).
*   **Cross-Module Correlation:** Identify if high drift in one module correlates with increased drift or errors in dependent modules or at symbolic junctions.
*   **Input for Automated Systems (Future):** The structured data underlying the visuals could eventually feed into automated testing systems (triggering focused tests on high-drift zones) or even adaptive AGI components that can adjust behavior based on observed drift patterns.

## 2. Input Format Specification

The VDMP will primarily consume structured data, ideally in JSON format, as proposed by the `drift_density_report.json` output from the [Symbolic Drift Density Estimator (SDDE)](../../docs/diagnostics/DRIFT_DENSITY_ESTIMATOR.md).

**Key Expected Input Data Points (per entry/scope):**
*   `report_timestamp`: Timestamp of the data generation.
*   `scope_type`: (String) e.g., "module", "file", "directory", "temporal_window", "operational_phase", "junction_id".
*   `scope_identifier`: (String) Unique name or path for the scope (e.g., "memory.core_memory.fold_engine", "sprint_23_Q4_phase_learning").
*   `sdd_value`: (Float) The calculated Symbolic Drift Density score for the scope.
*   `sdd_category`: (String, Enum-like) Qualitative assessment, e.g., "LOW", "MEDIUM", "HIGH", "CRITICAL".
*   `status_tags`: (List of Strings) Applicable status tags like `#ΛDENSITY_ZONE_HIGH`, `#ΛDRIFT_CLUSTER_DETECTED`, `#ΛOVERLOAD_REGION`.
*   `contributing_drift_points_summary`:
    *   `count`: (Integer) Number of raw `#ΛDRIFT_POINT`s.
    *   `severity_distribution`: (Dict) e.g., `{"HIGH": 2, "MEDIUM": 5, "LOW": 10}`.
    *   `key_drift_point_examples`: (List of Dicts) Brief details of a few illustrative drift points (file, line, brief description).
*   `temporal_data` (for timeline views):
    *   `start_time`: ISO timestamp.
    *   `end_time`: ISO timestamp.
*   `entropy_score` (Optional, from `TRAIL_ENTROPY_ESTIMATOR.py`): (Float) If available, for overlay.
*   `recovery_events_count` (Optional, from `#ΛRECOVERY_POINT` data): (Integer) Number of recovery actions within the scope/time.
*   `related_trace_event_counts`: (Dict) Counts of relevant `#ΛTRACE` events (e.g., `{"ERROR": 5, "WARNING_DRIFT_RELATED": 12}`).

**Linkage to Other Systems:**
*   Input could also be enriched by a (conceptual) `DriftChainIntegrator.py` that correlates sequences of drift events or links drift to specific operational traces.
*   References to `GLYPH_VISUAL_DEBUG_LEGEND.md` would inform how glyphs are rendered if this file becomes available and defines them.

## 3. Visual Elements (`#ΛDIAGNOSTIC_VIZ`)

The VDMP will employ several interconnected visual paradigms:

1.  **Structural Heatmap (Treemap or Sunburst):**
    *   **Representation:** Hierarchical blocks representing the codebase structure (e.g., directories contain modules, modules contain files).
    *   **Size:** Area of each block can be proportional to Lines Of Code (LOC), number of functions/classes, or number of total symbolic tags.
    *   **Color:** Intensity of color (e.g., green-yellow-orange-red gradient) directly maps to the `sdd_value` or `sdd_category`.
    *   **Overlays/Icons:**
        *   Small icons or distinct border styles to indicate `#ΛDENSITY_ZONE` status (e.g., a hazard icon for `HIGH`).
        *   Specific markers for identified `#ΛDRIFT_CLUSTER`s.
        *   Markers for `#ΛOVERLOAD_REGION`s.
    *   **Interactivity (Conceptual):** Hovering shows detailed SDD stats for the block; clicking could drill down or link to source code/detailed reports.

2.  **Temporal Drift Timeline:**
    *   **X-axis:** Time (commits, dates, sprints, operational session IDs).
    *   **Y-axis:**
        *   Option 1: Overall system SDD score.
        *   Option 2: Multiple lines/bands, each representing SDD for a key module or component.
        *   Option 3: Stacked area chart showing contribution of different modules to overall drift.
    *   **Visuals:**
        *   Line charts for `sdd_value` evolution.
        *   Color-coded background indicating overall system symbolic health (derived from SDD).
        *   Vertical markers for significant events: deployments, major incidents, application of `#ΛRECOVERY_POINT`s, activation of `#ΛANCHOR_REENTRY` protocols.
    *   **Entropy Overlay:** If entropy data is available, plot as a secondary line on the Y2-axis to correlate with drift density.

3.  **Drift Point Detail View (Drill-down):**
    *   **Trigger:** Clicking on a high-density zone or cluster in the structural map or a peak in the timeline.
    *   **Content:** A list or table of individual `#ΛDRIFT_POINT`s contributing to the selected zone/event, showing:
        *   File, line number.
        *   Associated comment/description of the drift point.
        *   Severity (if available).
        *   Timestamp of introduction/last modification.
        *   Link to relevant `#ΛTRACE` logs (if correlated).

4.  **Color Overlays & Legends:**
    *   Consistent color scheme: e.g., Green (Low SDD), Yellow (Moderate), Orange (High), Red (Critical/Overload).
    *   Clear legends explaining color mappings, icon meanings, and status tags.

5.  **Drift Anchors & Recovery Visualization (`#ΛANCHOR_REENTRY`, `#ΛRE-CENTER_VIS`):**
    *   On timelines, successful `#ΛRECOVERY_POINT` applications or `#ΛANCHOR_REENTRY` events could be marked with a distinct visual symbol (e.g., a green checkmark, a downward arrow indicating successful re-centering).
    *   The impact could be shown by a subsequent, sustained decrease in the SDD for the affected scope.

## 4. Glyph or Tag Overlay Options (`#ΛGLYPH_OVERLAY`)

To enhance the diagnostic power of the visual map:

*   **Tag Density Overlay:** Allow users to toggle an overlay on the structural map showing the raw density of `#ΛDRIFT_POINT` tags (or other relevant tags like `#ΛCAUTION`).
*   **Glyph Integration (Conceptual - pending `GLYPH_VISUAL_DEBUG_LEGEND.md`):**
    *   If specific glyphs are defined for types of drift (e.g., semantic drift, state desynchronization, temporal anomaly), these could be used as icons within the structural map cells or on drift point detail views.
    *   Glyphs could also represent the status of a drift point (e.g., New, Investigating, Mitigated, Resolved - linked to `#ΛAUDIT_NOTE`s).
*   **Connectivity Overlay (for `#ΛDRIFT_CLUSTER`s):**
    *   When a `#ΛDRIFT_CLUSTER` is selected, optionally overlay lines on the structural map showing dependencies or communication paths between the clustered drift points or affected components. This could use Graphviz-like rendering in a detail panel.

## 5. Interpretation Guide

*   **Identifying Hotspots:** Look for persistently red/orange areas in the structural heatmap or sustained peaks in the timeline view. These are primary candidates for investigation.
*   **Understanding Trends:** Rising SDD over time in a component indicates degrading symbolic health. Sudden spikes may point to problematic recent changes. A decreasing trend after intervention signals successful mitigation.
*   **Correlating Drift:** Compare the SDD timeline with deployment markers, incident logs, or major feature releases to find potential causal links. Overlaying entropy scores can show if high drift correlates with high system disorder.
*   **Assessing Cluster Impact:** A dense `#ΛDRIFT_CLUSTER` in a critical module (e.g., `lukhas_id_reasoning_engine.py`) is higher risk than a sparse one in a less critical utility. The map should help prioritize based on location and density.
*   **Recognizing Overload:** If a `#ΛDENSITY_ZONE` is also flagged as an `#ΛOVERLOAD_REGION` (e.g., high error rates in associated `#ΛTRACE` logs), it indicates the component is actively failing to cope.
*   **Effectiveness of Fixes:** Look for `#ΛRE-CENTER_VIS` markers followed by a reduction in SDD in the relevant scope to confirm that recovery efforts were effective.

## 6. Next Steps / Implementation Ideas

*   **Phase 1 (Conceptual Refinement & Basic Mock-ups):**
    *   Finalize this `.md` specification.
    *   Create static mock-ups (e.g., using presentation software or drawing tools) of the key visual elements to get stakeholder feedback.
*   **Phase 2 (Data Ingestion & Basic Renderer):**
    *   Develop/finalize `tools/drift_map_data_collector.py` (or similar) to parse code for `#ΛDRIFT_POINT`s and aggregate `#ΛTRACE` logs into the specified JSON input format. This would coordinate with Jules-12's `symbolic_drift_tracker.py`.
    *   Implement a basic `tools/drift_map_renderer.py` using `matplotlib` or `seaborn` to generate static heatmap images and timeline plots from the JSON data.
*   **Phase 3 (Interactive Prototype):**
    *   Explore `Plotly Dash` or `Streamlit` for creating an interactive web-based dashboard.
    *   Implement features like hover-to-detail, drill-down, filtering by scope/time, and togglable overlays.
*   **Phase 4 (Advanced Features & Integration):**
    *   Integrate real-time data streams from `#ΛTRACE` logging.
    *   Develop more sophisticated clustering algorithms for `#ΛDRIFT_CLUSTER` identification.
    *   Integrate with version control to link drift patterns to specific commits/branches.
    *   Define and implement API for potential automated consumers (e.g., adaptive testing systems).
*   **Export Format:** Standardize `drift_density_report.json` for consumption by various tools, including this visualizer and potentially other diagnostic agents.

## 7. Appendix & Footer Metadata

*   **Related Documents:**
    *   [Symbolic Drift Density Estimator (SDDE)](./DRIFT_DENSITY_ESTIMATOR.md)
    *   [Symbolic Convergence Diagnostics README](./README_convergence.md)
    *   [JULES10_SYMBOLIC_CONVERGENCE.md](./JULES10_SYMBOLIC_CONVERGENCE.md)
    *   *DRIFT_RECOVERY_CONVERGENCE.md* (Awaited - Link to be added for Jules-12's work)
    *   *JULES04_RECOVERY_PATHWAYS.md* (Awaited - Link to be added for Jules-04's work)
    *   *TRAIL_ENTROPY_ESTIMATOR.py* (Awaited - Reference for entropy metrics)
    *   *DRIFTCHAININTEGRATOR.py* (Awaited - Reference for drift sequence analysis)
    *   *GLYPH_VISUAL_DEBUG_LEGEND.md* (Awaited - Reference for glyphs)
*   **Internal Tags Used in this Document:** `#ΛDRIFT_ZONE`, `#ΛENTROPY_CLUSTER`, `#ΛANCHOR_REENTRY`, `#ΛRE-CENTER_VIS`, `#ΛGLYPH_OVERLAY`, `#ΛDIAGNOSTIC_VIZ`, `#ΛTRACE`, `#ΛAUDIT_NOTE` (implicitly through SDDE linkage).

---
*   Last Updated: $(date -I) by Jules-10 (ΛTRACEWEAVER)
*   ΛVERSION: 0.1.0 (Initial Draft)
---
#ΛDIAGNOSTIC_VIZ #ΛDRIFT_ZONE #ΛENTROPY_CLUSTER #ΛTRACE
