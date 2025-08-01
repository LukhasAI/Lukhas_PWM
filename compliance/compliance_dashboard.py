

# ════════════════════════════════════════════════════════════════════════
# 📁 FILE: compliance_dashboard.py
# 🛡️ PURPOSE: Institutional compliance viewer for emergency logs and GDPR status
# 🎯 AUDIENCE: Governance reviewers (e.g. Sam Altman, auditors)
# ════════════════════════════════════════════════════════════════════════

import streamlit as st
import json
import os

LOG_PATH = "logs/emergency_log.jsonl"

st.set_page_config(page_title="LUCAS Institutional Compliance Viewer")
st.title("🛡️ LUCAS AGI – Compliance Audit Dashboard")

if not os.path.exists(LOG_PATH):
    st.warning("No emergency logs found.")
else:
    st.markdown("### 📜 Emergency Override Incidents")
    with open(LOG_PATH, "r") as f:
        logs = [json.loads(line) for line in f if line.strip()]

    for entry in reversed(logs[-25:]):
        st.markdown("---")
        st.markdown(f"**⏱️ Timestamp:** {entry.get('timestamp')}")
        st.markdown(f"**🔍 Reason:** {entry.get('reason')}")
        st.markdown(f"**🧑‍💼 User:** {entry.get('user')} (Tier {entry.get('tier')})")
        st.markdown("**🧩 Actions Taken:**")
        st.code(", ".join(entry.get("actions_taken", [])), language="bash")

        st.markdown("**📋 Compliance Tags:**")
        for tag, value in entry.get("institutional_compliance", {}).items():
            st.markdown(f"- `{tag}`: {'✅' if value else '❌'}")

st.caption("🔒 All emergency actions are traceable, tiered, and GDPR-aligned.")

# ────────────────────────────────────────────────────────────────────────────────
# Symbolic Trace Dashboard Viewer (Enhanced via trace_tools)
# ────────────────────────────────────────────────────────────────────────────────

import pandas as pd
from pathlib import Path
from core.tracing import trace_tools  # assuming trace_tools.py is importable

trace_path = Path("logs/symbolic_trace_dashboard.csv")
if trace_path.exists():
    st.markdown("### 🧠 Symbolic Trace Overview")

    try:
        df = pd.read_csv(trace_path)
        filter_cols = st.multiselect("Filter Columns", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[filter_cols] if filter_cols else df)

        # Optional Summary Tools
        st.markdown("### 📊 Symbolic Summary")
        summary = trace_tools.get_summary_stats(df)
        st.json(summary)

        if st.button("🧹 Filter by status = 'FAIL' or confidence < 0.6"):
            filtered = df[(df["status"] == "FAIL") | (df["confidence"] < 0.6)]
            st.dataframe(filtered)

    except Exception as e:
        st.error(f"Failed to load or process symbolic trace dashboard: {e}")
else:
    st.info("No symbolic trace data found.")