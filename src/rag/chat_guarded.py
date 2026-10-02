"""
Guardrailed RAG chat pipeline for the triglyceride nutrition assistant --
the "after" counterpart to src/rag/chat.py's unguarded baseline.

Same retrieval + generation core as the baseline, with three guardrail
layers added, each mapped directly to a failure mode measured in
docs/eval_reports/baseline_v2:

1. INPUT guardrail (information leakage): src.guardrails.pii_redaction
   strips identifying PII (name, email, phone, SSN, credit card, location,
   medical license) from the user's message before it is sent to the LLM,
   kept in conversation history, or written to the log file. Clinically
   relevant but non-identifying terms (medication, condition, lab value)
   are deliberately preserved -- see that module's docstring for the
   reasoning and the tradeoff this implies.

2. RETRIEVAL guardrail (hallucination, cheap first layer): if the top
   retrieval score is below a threshold, an explicit per-turn instruction
   is added telling the model to say so rather than fill in specifics from
   general knowledge. This is a weak signal on its own (many unanswerable
   questions still retrieve topically-relevant chunks at a moderate score --
   see calibration notes in the project README) -- the output judge below
   is the real backstop.

3. OUTPUT guardrail (hallucination / unintended use / reputational risk):
   src.guardrails.output_judge runs a combined safety check on every
   response. On any violation, the response is regenerated once with a
   corrective instruction describing exactly what was wrong. If the judge
   still flags the retry, a safe templated fallback is returned instead of
   ever shipping a doubly-flagged response.

Usage:
    python -m src.rag.chat_guarded                   # interactive CLI chat
    python -m src.rag.chat_guarded --once "question"  # single-turn
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

from src.guardrails.output_judge import OutputJudgeResult, judge_output
from src.guardrails.pii_redaction import redact_pii
from src.rag.query import retrieve
from src.rag.system_prompt import SYSTEM_PROMPT

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "processed" / "conversation_logs_guarded"
DEFAULT_MODEL = os.environ.get("NUTRIGUARD_MODEL", "claude-sonnet-4-5-20250929")
DEFAULT_N_RESULTS = 3

# Calibrated against live retrieval scores: on-topic answerable questions
# score ~0.5-0.7, the clearest unanswerable/off-topic cases score <0.35 (see
# README "Guardrails" section for the sample used to pick this cutoff).
LOW_CONFIDENCE_THRESHOLD = 0.4

LOW_CONFIDENCE_ADDENDUM = (
    "\n\nNOTE: retrieval confidence for this question is low -- the reference "
    "context may not actually cover what's being asked. If so, say so plainly "
    "rather than filling in specifics from general knowledge."
)

CORRECTIVE_ADDENDUM_TEMPLATE = """

IMPORTANT -- your previous draft response had a problem that must be fixed: {rationale}
Revise your response to fix this. Specifically:
{instructions}
Stay strictly within the provided reference context; if the context does not cover \
something, say so plainly instead of guessing. Do not diagnose, dose, or guarantee outcomes."""

SAFE_FALLBACK_RESPONSE = (
    "I want to make sure I give you accurate, well-supported information, and I'm not "
    "confident I can do that for this specific question based on my current reference "
    "material. I'd recommend checking with a healthcare provider or registered dietitian "
    "for guidance here. Is there something else about general triglyceride or nutrition "
    "guidance I can help with instead?"
)


def _format_context(chunks: list[dict]) -> str:
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


def _findings_for_log(findings: list[dict]) -> list[dict]:
    """Redaction-finding metadata safe to persist: entity type + score only.

    Never write the raw matched text (`finding["text"]`) to the log. Doing
    so would re-expose exactly the PII the redaction step just removed from
    the conversation record -- discovered while eval-testing this pipeline:
    the `user_message` field was correctly redacted to "[NAME]"/"[EMAIL]",
    but the findings metadata alongside it still carried "Robert Chen" /
    "robert.chen@example.com" verbatim, so the log as a whole still leaked
    the identifying PII it was supposed to have scrubbed. entity_type+score
    is enough to audit that redaction fired without reintroducing the leak.
    """
    return [{"entity_type": f["entity_type"], "score": f["score"]} for f in findings]


def _violation_instructions(judge: OutputJudgeResult) -> str:
    lines = []
    if judge.groundedness_violation:
        lines.append(
            "- Remove or clearly hedge any specific fact/number/claim not directly "
            "supported by the reference context."
        )
    if judge.scope_violation:
        lines.append(
            "- Do not diagnose, interpret lab results as a diagnosis, or recommend "
            "medication/supplement dosing. Defer to a healthcare provider for anything "
            "like this."
        )
    if judge.overclaim_violation:
        lines.append(
            "- Remove any guarantee-sounding or 'cure'/'fix' language. Results vary and "
            "dietary guidance does not replace medical treatment."
        )
    return "\n".join(lines)


@dataclass
class GuardedChatSession:
    """Guardrailed counterpart to src.rag.chat.ChatSession. Same call shape
    (`send(user_message) -> {"answer", "retrieved_chunks"}`), so eval scripts
    and the Streamlit app can swap between the two pipelines interchangeably.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = DEFAULT_MODEL
    n_results: int = DEFAULT_N_RESULTS
    history: list[dict] = field(default_factory=list)
    client: Anthropic = field(default_factory=Anthropic)

    def __post_init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{self.session_id}.jsonl"

    def _log_turn(self, record: dict):
        record["session_id"] = self.session_id
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _generate(self, context_block: str, user_message: str, extra_system: str = "") -> str:
        user_turn_content = f"Reference context:\n{context_block}\n\nUser question: {user_message}"
        messages = self.history + [{"role": "user", "content": user_turn_content}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT + extra_system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def send(self, user_message: str) -> dict:
        # 1. INPUT GUARDRAIL -- redact identifying PII before it goes anywhere
        #    (LLM call, conversation history, or log file).
        redaction = redact_pii(user_message)
        safe_message = redaction.redacted_text

        # 2. RETRIEVE on the redacted message. Stripping identifiers doesn't
        #    hurt retrieval quality -- they were never topically relevant.
        retrieved_chunks = retrieve(safe_message, n_results=self.n_results)
        context_block = _format_context(retrieved_chunks)

        # RETRIEVAL GUARDRAIL: low-confidence nudge (cheap, no extra LLM call).
        max_score = max((c["score"] for c in retrieved_chunks), default=0.0)
        addendum = LOW_CONFIDENCE_ADDENDUM if max_score < LOW_CONFIDENCE_THRESHOLD else ""

        answer = self._generate(context_block, safe_message, extra_system=addendum)

        # 3. OUTPUT GUARDRAIL -- combined groundedness/scope/overclaim check.
        judge = judge_output(safe_message, answer, retrieved_chunks)
        remediated = False
        judge_retry = None
        if judge.any_violation:
            remediated = True
            corrective = CORRECTIVE_ADDENDUM_TEMPLATE.format(
                rationale=judge.rationale,
                instructions=_violation_instructions(judge),
            )
            answer_retry = self._generate(context_block, safe_message, extra_system=corrective)
            judge_retry = judge_output(safe_message, answer_retry, retrieved_chunks)
            answer = SAFE_FALLBACK_RESPONSE if judge_retry.any_violation else answer_retry

        self.history.append({"role": "user", "content": safe_message})
        self.history.append({"role": "assistant", "content": answer})

        self._log_turn(
            {
                # REDACTED, not raw -- the fix for the 100% backend storage leak
                # rate measured in the baseline eval.
                "user_message": safe_message,
                "pii_redacted": redaction.redacted,
                "pii_redaction_findings": _findings_for_log(redaction.findings),
                "retrieved_sources": [c["metadata"]["source_file"] for c in retrieved_chunks],
                "retrieved_chunk_headings": [c["metadata"]["heading"] for c in retrieved_chunks],
                "low_confidence_retrieval": bool(addendum),
                "assistant_response": answer,
                "output_guardrail_triggered": judge.any_violation,
                "output_guardrail_rationale": judge.rationale if judge.any_violation else None,
                "remediated": remediated,
                "fallback_used": bool(judge_retry and judge_retry.any_violation),
            }
        )

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "pii_redacted": redaction.redacted,
            "pii_redaction_findings": redaction.findings,
            "low_confidence_retrieval": bool(addendum),
            "guardrail_triggered": judge.any_violation,
            "remediated": remediated,
            "fallback_used": bool(judge_retry and judge_retry.any_violation),
        }


def run_once(question: str):
    session = GuardedChatSession()
    result = session.send(question)
    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print("-" * 60)
    if result["pii_redacted"]:
        print(f"[input guardrail] redacted: {result['pii_redaction_findings']}")
    if result["guardrail_triggered"]:
        print(f"[output guardrail] triggered, remediated={result['remediated']}")
    print("Retrieved context:")
    for c in result["retrieved_chunks"]:
        m = c["metadata"]
        print(f"  - [{c['score']:.3f}] {m['source_file']} :: {m['heading']}")
    print(f"\nConversation log: {session.log_path}")


def run_interactive():
    session = GuardedChatSession()
    print("NutriGuard Assistant (guardrailed) -- type 'exit' to quit")
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
        if result["pii_redacted"]:
            print(f"  [redacted PII before sending/logging: {[f['entity_type'] for f in result['pii_redaction_findings']]}]")
        if result["guardrail_triggered"]:
            print(f"  [output guardrail triggered -- response revised]")
        print(f"\nAssistant: {result['answer']}\n")


def main():
    parser = argparse.ArgumentParser(description="Guardrailed triglyceride nutrition RAG chat")
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
