"""
Run the full eval harness against the baseline (or later, guardrailed) chat
pipeline: golden set (retrieval + faithfulness) and synthetic set (scope,
PII leakage, compliance/overclaim), writing a JSON results file and a
markdown summary report under docs/eval_reports/.

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_qa.jsonl"
SYNTHETIC_PATH = PROJECT_ROOT / "tests" / "synthetic" / "synthetic_queries.jsonl"
REPORT_DIR = PROJECT_ROOT / "docs" / "eval_reports"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_golden_item(item: dict, skip_faithfulness: bool) -> dict:
    session = ChatSession()
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


def eval_synthetic_item(item: dict, run_faithfulness: bool) -> dict:
    session = ChatSession()
    result = session.send(item["question"])
    answer = result["answer"]
    chunks = result["retrieved_chunks"]

    scope = M.scope_adherence(item["question"], answer)
    compliance = M.compliance_overclaim(item["question"], answer)
    user_pii = M.pii_leak_scan(item["question"])
    assistant_pii = M.pii_leak_scan(answer)

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
        "user_message_contains_pii": user_pii["pii_detected"],
        "user_pii_findings": user_pii["pii_findings"],
        "assistant_echoed_pii": assistant_pii["pii_detected"],
        "assistant_pii_findings": assistant_pii["pii_findings"],
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


def summarize(golden_results: list[dict], synthetic_results: list[dict]) -> dict:
    def safe_mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    golden_ok = [r for r in golden_results if "error" not in r]
    synth_ok = [r for r in synthetic_results if "error" not in r]

    summary = {
        "golden_total": len(golden_results),
        "golden_errors": len(golden_results) - len(golden_ok),
        "retrieval_hit_rate": safe_mean([1.0 if r.get("hit_at_k") else 0.0 for r in golden_ok]),
        "retrieval_mrr": safe_mean([r.get("reciprocal_rank") for r in golden_ok]),
        "golden_faithfulness_mean": safe_mean([r.get("faithfulness_score") for r in golden_ok]),
        "golden_answer_relevancy_mean": safe_mean(
            [r.get("answer_relevancy_score") for r in golden_ok]
        ),
        "synthetic_total": len(synthetic_results),
        "synthetic_errors": len(synthetic_results) - len(synth_ok),
        "scope_violation_rate": safe_mean(
            [0.0 if r.get("stayed_in_scope") else 1.0 for r in synth_ok]
        ),
        "overclaim_rate": safe_mean(
            [1.0 if r.get("overclaim_detected") else 0.0 for r in synth_ok]
        ),
        "assistant_pii_echo_rate": safe_mean(
            [1.0 if r.get("assistant_echoed_pii") else 0.0 for r in synth_ok]
        ),
        "by_category": {},
    }

    by_cat: dict[str, list[dict]] = {}
    for r in synth_ok:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, items in by_cat.items():
        summary["by_category"][cat] = {
            "count": len(items),
            "scope_violation_rate": safe_mean(
                [0.0 if r.get("stayed_in_scope") else 1.0 for r in items]
            ),
            "overclaim_rate": safe_mean(
                [1.0 if r.get("overclaim_detected") else 0.0 for r in items]
            ),
            "assistant_pii_echo_rate": safe_mean(
                [1.0 if r.get("assistant_echoed_pii") else 0.0 for r in items]
            ),
        }

    return summary


def write_markdown_report(path: Path, tag: str, summary: dict):
    lines = [f"# Eval Report: {tag}", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append("## Golden set (retrieval + faithfulness)")
    lines.append(f"- Total: {summary['golden_total']} (errors: {summary['golden_errors']})")
    lines.append(f"- Retrieval hit rate: {summary['retrieval_hit_rate']}")
    lines.append(f"- Retrieval MRR: {summary['retrieval_mrr']}")
    lines.append(f"- Faithfulness (mean): {summary['golden_faithfulness_mean']}")
    lines.append(f"- Answer relevancy (mean): {summary['golden_answer_relevancy_mean']}")
    lines.append("")
    lines.append("## Synthetic set (scope, compliance, PII)")
    lines.append(f"- Total: {summary['synthetic_total']} (errors: {summary['synthetic_errors']})")
    lines.append(f"- Scope violation rate: {summary['scope_violation_rate']}")
    lines.append(f"- Overclaim rate: {summary['overclaim_rate']}")
    lines.append(f"- Assistant PII echo rate: {summary['assistant_pii_echo_rate']}")
    lines.append("")
    lines.append("### By category")
    lines.append("| Category | Count | Scope violation rate | Overclaim rate | PII echo rate |")
    lines.append("|---|---|---|---|---|")
    for cat, s in summary["by_category"].items():
        lines.append(
            f"| {cat} | {s['count']} | {s['scope_violation_rate']} | "
            f"{s['overclaim_rate']} | {s['assistant_pii_echo_rate']} |"
        )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run the RAG eval harness")
    parser.add_argument("--tag", type=str, required=True, help="Run tag, e.g. 'baseline'")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per dataset (smoke test)")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument(
        "--skip-faithfulness", action="store_true",
        help="Skip DeepEval faithfulness/relevancy scoring (faster/cheaper)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    golden_items = [] if args.skip_golden else load_jsonl(GOLDEN_PATH)
    synthetic_items = [] if args.skip_synthetic else load_jsonl(SYNTHETIC_PATH)

    if args.limit:
        golden_items = golden_items[: args.limit]
        synthetic_items = synthetic_items[: args.limit]

    print(f"Running eval '{args.tag}': {len(golden_items)} golden, {len(synthetic_items)} synthetic")

    golden_results = []
    for i, item in enumerate(golden_items, start=1):
        print(f"  [golden {i}/{len(golden_items)}] {item['question'][:70]}")
        golden_results.append(
            run_with_errors_caught(eval_golden_item, item, args.skip_faithfulness)
        )

    synthetic_results = []
    for i, item in enumerate(synthetic_items, start=1):
        print(f"  [synthetic {i}/{len(synthetic_items)}] [{item['category']}] {item['question'][:60]}")
        synthetic_results.append(
            run_with_errors_caught(eval_synthetic_item, item, False)
        )

    summary = summarize(golden_results, synthetic_results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORT_DIR / f"{args.tag}_{timestamp}.json"
    md_path = REPORT_DIR / f"{args.tag}_{timestamp}.md"

    with open(json_path, "w") as f:
        json.dump(
            {"tag": args.tag, "summary": summary, "golden_results": golden_results,
             "synthetic_results": synthetic_results},
            f,
            indent=2,
        )
    write_markdown_report(md_path, args.tag, summary)

    print(f"\nWrote results to:\n  {json_path}\n  {md_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
