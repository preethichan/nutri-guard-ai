"""
Baseline (un-guarded) RAG chat pipeline for the triglyceride nutrition assistant.

This is the "before guardrails" pipeline: system prompt + retrieval + Claude
generation, with NO input/output/retrieval validation layers. It exists so we
can measure (via src/eval/) how often it fails across the four failure modes
(hallucination, unintended use, information leakage, reputational/compliance
risk) before any guardrails are added -- the same "baseline" concept used in
the DeepLearning.AI guardrails course, but applied to a real, sourced
knowledge base instead of mock data.

Notably, like the baseline in that course, conversation history (including
anything sensitive a user types) is logged verbatim with NO redaction -- this
is intentional, to let us later demonstrate/measure the information-leakage
failure mode and the effect of adding log-redaction guardrails.

Usage:
    python -m src.rag.chat                  # interactive CLI chat
    python -m src.rag.chat --once "question" # single-turn, prints answer + retrieved chunks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from src.rag.query import retrieve
from src.rag.system_prompt import SYSTEM_PROMPT

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "processed" / "conversation_logs"
DEFAULT_MODEL = os.environ.get("NUTRIGUARD_MODEL", "claude-sonnet-4-5-20250929")
DEFAULT_N_RESULTS = 3


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a context block with source attribution."""
    if not chunks:
        return "(No relevant reference material was retrieved for this question.)"

    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        blocks.append(
            f"[Source {i}] {meta['source_org']} -- {meta['title']} "
            f"(section: {meta['heading']})\n"
            f"URL: {meta['url']}\n"
            f"Relevance score: {c['score']:.3f}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


@dataclass
class ChatSession:
    """A single conversation with the baseline assistant.

    Stores full turn-by-turn history (unredacted, by design -- see module
    docstring) and persists it to a JSONL log file on each turn, mirroring
    the "backend still has your PII" scenario from the course.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = DEFAULT_MODEL
    n_results: int = DEFAULT_N_RESULTS
    history: list[dict] = field(default_factory=list)  # [{"role", "content"}]
    client: Anthropic = field(default_factory=Anthropic)

    def __post_init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{self.session_id}.jsonl"

    def _log_turn(self, record: dict):
        record["session_id"] = self.session_id
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def send(self, user_message: str) -> dict:
        """Send a user message, return {"answer", "retrieved_chunks"}.

        No input validation, no scope checking, no PII redaction, no
        groundedness check on the output -- intentionally, for baseline
        measurement purposes.
        """
        retrieved_chunks = retrieve(user_message, n_results=self.n_results)
        context_block = _format_context(retrieved_chunks)

        user_turn_content = (
            f"Reference context:\n{context_block}\n\n"
            f"User question: {user_message}"
        )

        messages = self.history + [{"role": "user", "content": user_turn_content}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        )

        # Persist to in-memory history using the RAW user message (not the
        # context-stuffed version) so the conversation reads naturally and
        # any PII the user typed is preserved verbatim in history/logs --
        # again, intentionally unredacted for baseline measurement.
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        self._log_turn(
            {
                "user_message": user_message,
                "retrieved_sources": [c["metadata"]["source_file"] for c in retrieved_chunks],
                "retrieved_chunk_headings": [c["metadata"]["heading"] for c in retrieved_chunks],
                "assistant_response": answer,
            }
        )

        return {"answer": answer, "retrieved_chunks": retrieved_chunks}


def run_once(question: str):
    session = ChatSession()
    result = session.send(question)
    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print("-" * 60)
    print("Retrieved context:")
    for c in result["retrieved_chunks"]:
        m = c["metadata"]
        print(f"  - [{c['score']:.3f}] {m['source_file']} :: {m['heading']}")
    print(f"\nConversation log: {session.log_path}")


def run_interactive():
    session = ChatSession()
    print("NutriGuard Assistant (baseline, no guardrails) -- type 'exit' to quit")
    print(f"Session log: {session.log_path}\n")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break
        result = session.send(user_message)
        print(f"\nAssistant: {result['answer']}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline triglyceride nutrition RAG chat")
    parser.add_argument("--once", type=str, default=None, help="Single question, non-interactive")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Create a .env file in the project "
            "root with ANTHROPIC_API_KEY=sk-ant-... or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.once:
        run_once(args.once)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
