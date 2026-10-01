"""
Mechanically derive a golden Q&A regression set from data/raw/*.md.

For each knowledge-base chunk (same chunking logic used by src/rag/ingest.py),
an LLM is asked to write ONE natural user question whose answer is fully
contained within that chunk's text. This is NOT hand-crafted or adversarial --
it's a straightforward "write a question this text answers" task, run
uniformly across chunks, so the resulting test set isn't biased toward
failure the way hand-picked "gotcha" prompts would be.

Each golden item records the expected source file + heading, so later we can
measure retrieval quality (did the right chunk come back?) independently of
generation quality (did the model use it correctly?).

Usage:
    python -m src.eval.build_golden_set
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.claude_client import simple_generate
from src.rag.ingest import RAW_DATA_DIR, load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_qa.jsonl"

# Take up to this many chunks per source file, in document order, so the
# golden set has even coverage across all 8 sources (~3 x 8 = ~24 items)
# rather than being skewed toward whichever file happens to have more chunks.
MAX_CHUNKS_PER_FILE = 3

QUESTION_GEN_PROMPT = """You will be given a short excerpt from a nutrition/health reference document.

Write exactly ONE natural question that a curious, non-expert user might ask, such that this \
excerpt alone fully and directly answers it. The question should be answerable using ONLY the \
information in the excerpt below -- do not require outside knowledge.

Respond with ONLY the question text, no quotes, no preamble, no explanation.

Excerpt:
---
{chunk_text}
---

Question:"""


def build_golden_set() -> list[dict]:
    records = load_documents()

    # Group by source file, preserving order, then cap per file.
    by_file: dict[str, list[dict]] = {}
    for r in records:
        by_file.setdefault(r["metadata"]["source_file"], []).append(r)

    golden_items = []
    item_id = 0
    for source_file, chunks in sorted(by_file.items()):
        for chunk in chunks[:MAX_CHUNKS_PER_FILE]:
            prompt = QUESTION_GEN_PROMPT.format(chunk_text=chunk["text"])
            question = simple_generate(prompt).strip()

            golden_items.append(
                {
                    "id": f"golden_{item_id:03d}",
                    "question": question,
                    "expected_source_file": chunk["metadata"]["source_file"],
                    "expected_heading": chunk["metadata"]["heading"],
                    "expected_source_org": chunk["metadata"]["source_org"],
                    "reference_text": chunk["text"],
                }
            )
            item_id += 1
            print(f"[{item_id}] ({source_file}) {question}")

    return golden_items


def main():
    items = build_golden_set()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"\nWrote {len(items)} golden Q&A items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
