"""
Run the full eval harness against the baseline (or later, guardrailed) chat
pipeline:
  - golden set        -> retrieval + faithfulness (answerable-by-construction)
  - unanswerable set  -> hallucination-honesty (NOT answerable from the KB --
                         the actual hallucination stress-test)
  - synthetic set     -> scope adherence, compliance/overclaim, PII/PHI
                         leakage (both "echoed in response" and "stored
                         unredacted in backend logs")

Writes a JSON results file and a markdown summary report under
docs/eval_reports/.

Usage:
    python -m src.eval.run_eval --tag baseline
    python -m src.eval.run_eval --tag baseline --limit 5       # quick smoke test
    python -m src.eval.run_eval --tag baseline --skip-golden   # synthetic only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.eval import metrics as M
from src.rag.chat import ChatSession
from src.rag.chat_guarded import GuardedChatSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_qa.jsonl"
SYNTHETIC_PATH = PROJECT_ROOT / "tests" / "synthetic" / "synthetic_queries.jsonl"
UNANSWERABLE_PATH = PROJECT_ROOT / "tests" / "unanswerable" / "unanswerable_qa.jsonl"
REPORT_DIR = PROJECT_ROOT / "docs" / "eval_reports"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_golden_item(item: dict, skip_faithfulness: bool, session_factory=ChatSession) -> dict:
    session = session_factory()
    result = session.send(item["question"])
    answer = result["answer"]
    chunks = result["retrieved_chunks"]

    out = {
        "id": item["id"],
        "question": item["question"],
        "expected_source_file": item["expected_source_file"],
        "answer": answer,
        **M.retrieval_metrics(chunks, item["expected_source_file"]),
    }

    if not skip_faithfulness:
        out.update(M.faithfulness_and_relevancy(item["question"], answer, chunks))

    return out


def eval_unanswerable_item(item: dict, skip_faithfulness: bool, session_factory=ChatSession) -> dict:
    session = session_factory()
    result = session.send(item["question"])
    answer = result["answer"]
    chunks = result["retrieved_chunks"]

    honesty = M.hallucination_honesty(item["question"], answer, chunks)

    out = {
        "id": item["id"],
        "question": item["question"],
        "why_unanswerable": item.get("why_unanswerable", ""),
        "answer": answer,
        "fabricated_specifics": honesty.fabricated_specifics,
        "acknowledged_uncertainty": honesty.acknowledged_uncertainty,
        "honesty_rationale": honesty.rationale,
    }

    if not skip_faithfulness:
        out.update(M.faithfulness_and_relevancy(item["question"], answer, chunks))

    return out


def eval_synthetic_item(item: dict, run_faithfulness: bool, session_factory=ChatSession) -> dict:
    session = session_factory()
    result = session.send(item["question"])
    answer = result["answer"]
    chunks = result["retrieved_chunks"]

    scope = M.scope_adherence(item["question"], answer)
    compliance = M.compliance_overclaim(item["question"], answer)
    echo = M.pii_echo_scan(item["question"], answer)

    # Real "backend storage leak" check: does the persisted conversation log
    # actually contain the PII/PHI the user disclosed, unredacted? At
    # baseline this is trivially true whenever PII is present (chat.py does
    # zero redaction by design). Split by entity category so the guardrailed
    # pipeline's intentional tradeoff is visible: identifying entities
    # (name/email/phone/etc.) should drop to ~0% once redaction is added,
    # while clinical-context entities (medication/condition/lab value) are
    # deliberately preserved for answer quality and will still show up.
    # Only check the actual conversation-record fields (user_message,
    # assistant_response), not the raw file text. Other persisted metadata
    # (e.g. a redaction audit trail) may legitimately reference entity
    # types/scores without this check caring about it -- and historically
    # it *did* care, which masked a real bug: see chat_guarded.py's
    # _findings_for_log for the fix once this caught it for real.
    conversation_text = ""
    if session.log_path.exists():
        for line in session.log_path.read_text().splitlines():
            if not line.strip():
                continue
            turn = json.loads(line)
            conversation_text += " " + str(turn.get("user_message", ""))
            conversation_text += " " + str(turn.get("assistant_response", ""))
    conversation_text = conversation_text.lower()
    stored_unredacted = [
        f for f in echo["user_pii_findings"]
        if f["text"].strip().lower() in conversation_text
    ]
    stored_unredacted_identifying = [
        f for f in stored_unredacted if f["entity_type"] in M.IDENTIFYING_ENTITIES
    ]
    stored_unredacted_clinical = [
        f for f in stored_unredacted if f["entity_type"] in M.CLINICAL_ENTITIES
    ]

    out = {
        "id": item["id"],
        "category": item["category"],
        "failure_mode": item["failure_mode"],
        "question": item["question"],
        "answer": answer,
        "stayed_in_scope": scope.stayed_in_scope,
        "scope_violation_type": scope.violation_type,
        "scope_rationale": scope.rationale,
        "overclaim_detected": compliance.overclaim_detected,
        "compliance_rationale": compliance.rationale,
        "user_message_contains_pii": echo["user_message_contains_pii"],
        "user_pii_findings": echo["user_pii_findings"],
        "assistant_echoed_pii": echo["assistant_echoed_pii"],
        "assistant_echoed_pii_findings": echo["echoed_findings"],
        "backend_log_stored_pii_unredacted": len(stored_unredacted) > 0,
        "backend_log_stored_identifying_pii_unredacted": len(stored_unredacted_identifying) > 0,
        "backend_log_stored_clinical_phi_unredacted": len(stored_unredacted_clinical) > 0,
        # Guardrail-internal signals, only present when running GuardedChatSession.
        "pii_redacted_by_guardrail": result.get("pii_redacted"),
        "output_guardrail_triggered": result.get("guardrail_triggered"),
        "output_guardrail_remediated": result.get("remediated"),
    }

    if run_faithfulness:
        out.update(M.faithfulness_and_relevancy(item["question"], answer, chunks))

    return out


def run_with_errors_caught(fn, item, *args):
    try:
        return fn(item, *args)
    except Exception as e:
        return {
            "id": item.get("id"),
            "question": item.get("question"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def summarize(golden_results, unanswerable_results, synthetic_results) -> dict:
    def safe_mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    golden_ok = [r for r in golden_results if "error" not in r]
    unans_ok = [r for r in unanswerable_results if "error" not in r]
    synth_ok = [r for r in synthetic_results if "error" not in r]

    pii_present = [r for r in synth_ok if r.get("user_message_contains_pii")]

    summary = {
        "golden_total": len(golden_results),
        "golden_errors": len(golden_results) - len(golden_ok),
        "retrieval_hit_rate": safe_mean([1.0 if r.get("hit_at_k") else 0.0 for r in golden_ok]),
        "retrieval_mrr": safe_mean([r.get("reciprocal_rank") for r in golden_ok]),
        "golden_faithfulness_mean": safe_mean([r.get("faithfulness_score") for r in golden_ok]),
        "golden_answer_relevancy_mean": safe_mean(
            [r.get("answer_relevancy_score") for r in golden_ok]
        ),
        "unanswerable_total": len(unanswerable_results),
        "unanswerable_errors": len(unanswerable_results) - len(unans_ok),
        "fabrication_rate": safe_mean(
            [1.0 if r.get("fabricated_specifics") else 0.0 for r in unans_ok]
        ),
        "acknowledged_uncertainty_rate": safe_mean(
            [1.0 if r.get("acknowledged_uncertainty") else 0.0 for r in unans_ok]
        ),
        "unanswerable_faithfulness_mean": safe_mean(
            [r.get("faithfulness_score") for r in unans_ok]
        ),
        "synthetic_total": len(synthetic_results),
        "synthetic_errors": len(synthetic_results) - len(synth_ok),
        "scope_violation_rate": safe_mean(
            [0.0 if r.get("stayed_in_scope") else 1.0 for r in synth_ok]
        ),
        "overclaim_rate": safe_mean(
            [1.0 if r.get("overclaim_detected") else 0.0 for r in synth_ok]
        ),
        "user_pii_disclosure_rate": safe_mean(
            [1.0 if r.get("user_message_contains_pii") else 0.0 for r in synth_ok]
        ),
        "assistant_pii_echo_rate": safe_mean(
            [1.0 if r.get("assistant_echoed_pii") else 0.0 for r in synth_ok]
        ),
        "backend_storage_leak_rate_when_pii_present": safe_mean(
            [1.0 if r.get("backend_log_stored_pii_unredacted") else 0.0 for r in pii_present]
        ),
        "backend_storage_leak_rate_identifying_pii": safe_mean(
            [1.0 if r.get("backend_log_stored_identifying_pii_unredacted") else 0.0 for r in pii_present]
        ),
        "backend_storage_leak_rate_clinical_phi": safe_mean(
            [1.0 if r.get("backend_log_stored_clinical_phi_unredacted") else 0.0 for r in pii_present]
        ),
        "output_guardrail_trigger_rate": safe_mean(
            [1.0 if r.get("output_guardrail_triggered") else 0.0 for r in synth_ok]
        )
        if any(r.get("output_guardrail_triggered") is not None for r in synth_ok)
        else None,
        "by_category": {},
    }

    by_cat: dict[str, list[dict]] = {}
    for r in synth_ok:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, items in by_cat.items():
        cat_pii_present = [r for r in items if r.get("user_message_contains_pii")]
        summary["by_category"][cat] = {
            "count": len(items),
            "scope_violation_rate": safe_mean(
                [0.0 if r.get("stayed_in_scope") else 1.0 for r in items]
            ),
            "overclaim_rate": safe_mean(
                [1.0 if r.get("overclaim_detected") else 0.0 for r in items]
            ),
            "user_pii_disclosure_rate": safe_mean(
                [1.0 if r.get("user_message_contains_pii") else 0.0 for r in items]
            ),
            "assistant_pii_echo_rate": safe_mean(
                [1.0 if r.get("assistant_echoed_pii") else 0.0 for r in items]
            ),
            "backend_storage_leak_rate_when_pii_present": safe_mean(
                [1.0 if r.get("backend_log_stored_pii_unredacted") else 0.0 for r in cat_pii_present]
            ),
            "backend_storage_leak_rate_identifying_pii": safe_mean(
                [1.0 if r.get("backend_log_stored_identifying_pii_unredacted") else 0.0 for r in cat_pii_present]
            ),
            "backend_storage_leak_rate_clinical_phi": safe_mean(
                [1.0 if r.get("backend_log_stored_clinical_phi_unredacted") else 0.0 for r in cat_pii_present]
            ),
        }

    return summary


def write_markdown_report(path: Path, tag: str, summary: dict, pipeline: str = "baseline"):
    lines = [
        f"# Eval Report: {tag}",
        f"Pipeline: {pipeline}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    lines.append("## Golden set (retrieval + faithfulness; answerable by construction)")
    lines.append(f"- Total: {summary['golden_total']} (errors: {summary['golden_errors']})")
    lines.append(f"- Retrieval hit rate: {summary['retrieval_hit_rate']}")
    lines.append(f"- Retrieval MRR: {summary['retrieval_mrr']}")
    lines.append(f"- Faithfulness (mean): {summary['golden_faithfulness_mean']}")
    lines.append(f"- Answer relevancy (mean): {summary['golden_answer_relevancy_mean']}")
    lines.append("")

    lines.append("## Unanswerable set (real hallucination stress-test)")
    lines.append(f"- Total: {summary['unanswerable_total']} (errors: {summary['unanswerable_errors']})")
    lines.append(f"- **Fabrication rate: {summary['fabrication_rate']}**")
    lines.append(f"- Acknowledged uncertainty rate: {summary['acknowledged_uncertainty_rate']}")
    lines.append(f"- Faithfulness on unanswerable Qs (mean): {summary['unanswerable_faithfulness_mean']}")
    lines.append("")

    lines.append("## Synthetic set (scope, compliance, PII/PHI)")
    lines.append(f"- Total: {summary['synthetic_total']} (errors: {summary['synthetic_errors']})")
    lines.append(f"- Scope violation rate: {summary['scope_violation_rate']}")
    lines.append(f"- Overclaim rate: {summary['overclaim_rate']}")
    lines.append(f"- User PII/PHI disclosure rate: {summary['user_pii_disclosure_rate']}")
    lines.append(f"- Assistant PII echo rate (response repeats back user's PII): {summary['assistant_pii_echo_rate']}")
    lines.append(
        f"- **Backend storage leak rate, any PII/PHI (when present): "
        f"{summary['backend_storage_leak_rate_when_pii_present']}**"
    )
    lines.append(
        f"  - Identifying entities (name/email/phone/SSN/credit card/location/license): "
        f"{summary['backend_storage_leak_rate_identifying_pii']}"
    )
    lines.append(
        f"  - Clinical-context entities (medication/condition/lab value): "
        f"{summary['backend_storage_leak_rate_clinical_phi']}"
    )
    if summary.get("output_guardrail_trigger_rate") is not None:
        lines.append(
            f"- Output guardrail trigger rate (groundedness/scope/overclaim judge fired): "
            f"{summary['output_guardrail_trigger_rate']}"
        )
    lines.append("")
    lines.append("### By category")
    lines.append(
        "| Category | Count | Scope violation | Overclaim | PII/PHI disclosed | "
        "Echoed in response | Stored unredacted (any) | Stored unredacted (identifying) | Stored unredacted (clinical) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cat, s in summary["by_category"].items():
        lines.append(
            f"| {cat} | {s['count']} | {s['scope_violation_rate']} | "
            f"{s['overclaim_rate']} | {s['user_pii_disclosure_rate']} | "
            f"{s['assistant_pii_echo_rate']} | {s['backend_storage_leak_rate_when_pii_present']} | "
            f"{s['backend_storage_leak_rate_identifying_pii']} | {s['backend_storage_leak_rate_clinical_phi']} |"
        )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run the RAG eval harness")
    parser.add_argument("--tag", type=str, required=True, help="Run tag, e.g. 'baseline'")
    parser.add_argument(
        "--pipeline", type=str, default="baseline", choices=["baseline", "guarded"],
        help="Which chat pipeline to evaluate: 'baseline' (src.rag.chat.ChatSession, "
             "no guardrails) or 'guarded' (src.rag.chat_guarded.GuardedChatSession)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit items per dataset (smoke test)")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-unanswerable", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument(
        "--skip-faithfulness", action="store_true",
        help="Skip DeepEval faithfulness/relevancy scoring (faster/cheaper)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    session_factory = GuardedChatSession if args.pipeline == "guarded" else ChatSession

    golden_items = [] if args.skip_golden else load_jsonl(GOLDEN_PATH)
    unanswerable_items = [] if args.skip_unanswerable else load_jsonl(UNANSWERABLE_PATH)
    synthetic_items = [] if args.skip_synthetic else load_jsonl(SYNTHETIC_PATH)

    if args.limit:
        golden_items = golden_items[: args.limit]
        unanswerable_items = unanswerable_items[: args.limit]
        synthetic_items = synthetic_items[: args.limit]

    print(
        f"Running eval '{args.tag}' (pipeline={args.pipeline}): {len(golden_items)} golden, "
        f"{len(unanswerable_items)} unanswerable, {len(synthetic_items)} synthetic"
    )

    golden_results = []
    for i, item in enumerate(golden_items, start=1):
        print(f"  [golden {i}/{len(golden_items)}] {item['question'][:70]}")
        golden_results.append(
            run_with_errors_caught(eval_golden_item, item, args.skip_faithfulness, session_factory)
        )

    unanswerable_results = []
    for i, item in enumerate(unanswerable_items, start=1):
        print(f"  [unanswerable {i}/{len(unanswerable_items)}] {item['question'][:70]}")
        unanswerable_results.append(
            run_with_errors_caught(eval_unanswerable_item, item, args.skip_faithfulness, session_factory)
        )

    synthetic_results = []
    for i, item in enumerate(synthetic_items, start=1):
        print(f"  [synthetic {i}/{len(synthetic_items)}] [{item['category']}] {item['question'][:60]}")
        synthetic_results.append(
            run_with_errors_caught(eval_synthetic_item, item, False, session_factory)
        )

    summary = summarize(golden_results, unanswerable_results, synthetic_results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORT_DIR / f"{args.tag}_{timestamp}.json"
    md_path = REPORT_DIR / f"{args.tag}_{timestamp}.md"

    with open(json_path, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "pipeline": args.pipeline,
                "summary": summary,
                "golden_results": golden_results,
                "unanswerable_results": unanswerable_results,
                "synthetic_results": synthetic_results,
            },
            f,
            indent=2,
        )
    write_markdown_report(md_path, args.tag, summary, args.pipeline)

    print(f"\nWrote results to:\n  {json_path}\n  {md_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
