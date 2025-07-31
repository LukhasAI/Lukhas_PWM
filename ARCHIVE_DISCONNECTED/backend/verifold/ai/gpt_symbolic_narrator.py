# NOTE: VisualReplayer moved to tools/verifold_interface/
try:
    from tools.verifold_interface.visual_replayer import VisualReplayer
except ImportError:
    print("Warning: VisualReplayer not available - tools module not in path")
    VisualReplayer = None
import streamlit as st

from core.identity.identity_engine import ConsentScopeValidator


def main():
    # Initialize replayer (conditional based on availability)
    if VisualReplayer is not None:
        replayer = VisualReplayer()
    else:
        st.error("VisualReplayer not available - tools module not configured")
        return

    # Initialize consent validator
    consent_validator = ConsentScopeValidator()

    st.subheader("🔁 Replay Symbolic Memory")
    memory_hash = st.text_input("Enter Memory Hash")
    lukhas_id = st.text_input("Enter Lukhas_ID")

    if st.button("Generate Replay") and memory_hash and lukhas_id:
        frames = replayer.generate_narrative_replay(memory_hash, lukhas_id)

        # Consent Tier Check
        try:
            tier_level = consent_validator.get_tier_level(memory_hash, lukhas_id)
            if tier_level == 0:
                st.success("✅ Tier 0: Full replay access granted.")
            elif tier_level == 1:
                st.warning(
                    "⚠️ Tier 1: Limited access. Some symbolic metadata may be redacted."
                )
            elif tier_level == 2:
                st.error(
                    "🚫 Tier 2: Restricted replay. Viewer may not have full consent."
                )
            else:
                st.info("ℹ️ Unknown tier level.")
        except Exception as e:
            st.warning(f"Consent validation failed: {e}")

        show_consent_hash = st.checkbox("🛡️ Show Consent Hash Debug Info")

        for i, frame in enumerate(frames):
            if tier_level == 0:
                frame_color = "#e8f5e9"  # Light green
            elif tier_level == 1:
                frame_color = "#fff8e1"  # Light yellow
            else:
                frame_color = "#ffebee"  # Light red

            with st.container():
                st.markdown(
                    f"<div style='background-color:{frame_color}; padding:10px; border-radius:8px;'>"
                    f"<strong>Frame {i+1}</strong><br>"
                    f"🧠 Entropy: {frame.entropy_level:.2f}<br>"
                    f"💫 Emotions: {frame.emotional_state}<br>"
                    f"<strong>Narrative:</strong> {frame.narrative_text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if show_consent_hash:
                    st.code(
                        f"Consent Hash: {getattr(frame, 'consent_hash', 'N/A')}",
                        language="text",
                    )

                st.markdown("---")

        # Entropy Heatmap
        entropy_series = [frame.entropy_level for frame in frames]
        st.subheader("📊 Entropy Heatmap")
        st.line_chart(entropy_series)
