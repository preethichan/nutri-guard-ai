"""
Streamlit chat UI for the NutriGuard RAG assistant -- baseline and guardrailed
pipelines, side by side behind a single toggle.

Baseline (src.rag.chat.ChatSession) adds no guardrails of its own; it's the
"before" state measured in docs/eval_reports/baseline_v2_*. Guardrailed
(src.rag.chat_guarded.GuardedChatSession) adds the three layers described in
README.md ("Guardrails" section) and measured in docs/eval_reports/guarded_v1_*.
Switching the toggle starts a fresh conversation on the selected pipeline --
the two are not meant to share history, since they log to different files
and (for the guarded pipeline) redact before anything is stored.

Run with:
    PYTHONPATH=.deps:. streamlit run src/app/chat_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow `streamlit run src/app/chat_app.py` to resolve `src.*` imports
# regardless of the working directory Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.rag.chat import ChatSession  # noqa: E402
from src.rag.chat_guarded import GuardedChatSession  # noqa: E402

st.set_page_config(page_title="NutriGuard Assistant", page_icon="🩺")

PIPELINES = {
    "Baseline": ChatSession,
    "Guardrailed": GuardedChatSession,
}


def _get_session() -> ChatSession | GuardedChatSession:
    pipeline = st.session_state.pipeline
    if (
        "chat_session" not in st.session_state
        or st.session_state.get("active_pipeline") != pipeline
    ):
        st.session_state.chat_session = PIPELINES[pipeline]()
        st.session_state.messages = []  # [{"role", "content", "chunks"?, "meta"?}]
        st.session_state.active_pipeline = pipeline
    return st.session_state.chat_session


def _reset_session():
    pipeline = st.session_state.pipeline
    st.session_state.chat_session = PIPELINES[pipeline]()
    st.session_state.messages = []
    st.session_state.active_pipeline = pipeline


def _render_sources(chunks: list[dict]):
    with st.expander("Retrieved sources"):
        for c in chunks:
            m = c["metadata"]
            st.markdown(
                f"**[{c['score']:.3f}]** {m['source_org']} -- "
                f"{m['title']} ({m['heading']})  \n"
                f"[{m['url']}]({m['url']})"
            )


def _render_guardrail_badges(meta: dict):
    """Small inline indicators showing which guardrail layers fired on this turn."""
    badges = []
    if meta.get("pii_redacted"):
        entities = sorted({f["entity_type"] for f in meta.get("pii_redaction_findings", [])})
        badges.append(f"🔒 PII redacted ({', '.join(entities)})")
    if meta.get("low_confidence_retrieval"):
        badges.append("📉 Low-confidence retrieval -- nudge applied")
    if meta.get("guardrail_triggered"):
        badges.append(
            "🛡️ Output guardrail triggered -- regenerated"
            + (" (fallback used)" if meta.get("fallback_used") else "")
        )
    if badges:
        st.caption(" · ".join(badges))


def main():
    st.title("🩺 NutriGuard Assistant")
    st.caption("Triglyceride & nutrition guidance, grounded in WHO / NIH / AHA / USDA sources.")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = "Baseline"

    st.session_state.pipeline = st.segmented_control(
        "Pipeline",
        options=list(PIPELINES.keys()),
        default=st.session_state.pipeline,
        label_visibility="collapsed",
    ) or st.session_state.pipeline

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "ANTHROPIC_API_KEY is not set. Add it to the `.env` file in the project "
            "root, then restart the app."
        )
        st.stop()

    session = _get_session()

    if st.session_state.pipeline == "Baseline":
        st.warning(
            "**Baseline pipeline -- no guardrails.** No PII redaction, no scope "
            "filtering, no output/groundedness checks (the 'before' state measured "
            "in `docs/eval_reports/baseline_v2_*`). Conversation turns are logged "
            "**unredacted** to `data/processed/conversation_logs/`. Please don't "
            "share real personal or health information -- use fictional details if "
            "you want to test that behavior.",
            icon="⚠️",
        )
    else:
        st.success(
            "**Guardrailed pipeline active.** Input PII redaction, a retrieval-"
            "confidence nudge, and a combined output groundedness/scope/overclaim "
            "judge with bounded remediation (the 'after' state measured in "
            "`docs/eval_reports/guarded_v1_*`). Identifying PII (name/email/phone/"
            "location) is stripped before logging; clinically-relevant terms "
            "(medication/condition/lab value) are intentionally preserved. See "
            "`docs/guardrails_before_after.md`.",
            icon="🛡️",
        )

    with st.sidebar:
        st.subheader("Session")
        st.text(f"Pipeline: {st.session_state.pipeline}")
        st.text(f"Model: {session.model}")
        st.text(f"Session ID: {session.session_id[:8]}...")
        st.text(f"Log file:\n{session.log_path.relative_to(PROJECT_ROOT)}")
        st.divider()
        show_sources = st.toggle("Show retrieved sources", value=True)
        if st.button("🔄 New conversation", use_container_width=True):
            _reset_session()
            st.rerun()
        st.divider()
        if st.session_state.pipeline == "Guardrailed":
            st.caption(
                "Guardrails: input PII redaction · retrieval-confidence nudge · "
                "output groundedness/scope/overclaim judge + bounded remediation. "
                "See README.md 'Guardrails' section."
            )
        else:
            st.caption(
                "Un-guarded baseline. Switch to Guardrailed above for the same "
                "pipeline with PII redaction and safety checks active."
            )

    # Render prior turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if show_sources and msg.get("chunks"):
                    _render_sources(msg["chunks"])
                if msg.get("meta"):
                    _render_guardrail_badges(msg["meta"])

    user_message = st.chat_input("Ask a question about triglycerides, diet, or nutrition...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving sources and generating a response..."):
                result = session.send(user_message)
            st.markdown(result["answer"])
            if show_sources and result["retrieved_chunks"]:
                _render_sources(result["retrieved_chunks"])
            _render_guardrail_badges(result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "chunks": result["retrieved_chunks"],
                "meta": result,
            }
        )


if __name__ == "__main__":
    main()
