"""
Streamlit chat UI for the NutriGuard baseline (un-guarded) RAG assistant.

This is a thin UI layer over src.rag.chat.ChatSession -- it adds no guardrails
of its own. It exists so a human can interactively poke at the SAME baseline
pipeline that src/eval/run_eval.py measures, to build intuition before the
guardrailed version exists.

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

from src.rag.chat import ChatSession, DEFAULT_MODEL  # noqa: E402

st.set_page_config(page_title="NutriGuard Assistant (Baseline)", page_icon="🩺")


def _get_session() -> ChatSession:
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession()
        st.session_state.messages = []  # [{"role", "content", "chunks"?}]
    return st.session_state.chat_session


def _reset_session():
    st.session_state.chat_session = ChatSession()
    st.session_state.messages = []


def main():
    st.title("🩺 NutriGuard Assistant")
    st.caption("Triglyceride & nutrition guidance, grounded in WHO / NIH / AHA / USDA sources.")

    st.warning(
        "⚠️ **Baseline pipeline -- no guardrails.** This build has no PII redaction, "
        "no scope filtering, and no output/groundedness checks by design (it's the "
        "'before' state being measured in `docs/eval_reports/`). Conversation turns, "
        "including anything sensitive you type, are logged **unredacted** to "
        "`data/processed/conversation_logs/`. Please don't share real personal or "
        "health information -- use fictional details if you want to test that behavior.",
        icon="⚠️",
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "ANTHROPIC_API_KEY is not set. Add it to the `.env` file in the project "
            "root, then restart the app."
        )
        st.stop()

    session = _get_session()

    with st.sidebar:
        st.subheader("Session")
        st.text(f"Model: {session.model}")
        st.text(f"Session ID: {session.session_id[:8]}...")
        st.text(f"Log file:\n{session.log_path.relative_to(PROJECT_ROOT)}")
        st.divider()
        show_sources = st.toggle("Show retrieved sources", value=True)
        if st.button("🔄 New conversation", use_container_width=True):
            _reset_session()
            st.rerun()
        st.divider()
        st.caption(
            "This is the un-guarded baseline. Guardrailed input/output/retrieval "
            "layers are planned next -- see README.md project status."
        )

    # Render prior turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
                with st.expander("Retrieved sources"):
                    for c in msg["chunks"]:
                        m = c["metadata"]
                        st.markdown(
                            f"**[{c['score']:.3f}]** {m['source_org']} -- "
                            f"{m['title']} ({m['heading']})  \n"
                            f"[{m['url']}]({m['url']})"
                        )

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
                with st.expander("Retrieved sources"):
                    for c in result["retrieved_chunks"]:
                        m = c["metadata"]
                        st.markdown(
                            f"**[{c['score']:.3f}]** {m['source_org']} -- "
                            f"{m['title']} ({m['heading']})  \n"
                            f"[{m['url']}]({m['url']})"
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "chunks": result["retrieved_chunks"],
            }
        )


if __name__ == "__main__":
    main()
