"""
Generate an "unanswerable question" set -- realistic, domain-relevant
questions whose specific factual answer is NOT contained anywhere in the
knowledge base (data/raw/*.md). This is the actual hallucination stress-test:
the golden set (build_golden_set.py) can't surface hallucination because
every question there is answerable by construction. This set inverts that --
an LLM is given the full corpus and asked to generate plausible-sounding
questions it CANNOT answer from that corpus, mirroring the course's "Veggie
Supreme recipe" trick but generated systematically rather than hand-picked.

Usage:
    python -m src.eval.build_unanswerable_set
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.eval.claude_client import simple_generate
from src.rag.ingest import RAW_DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "tests" / "unanswerable" / "unanswerable_qa.jsonl"

NUM_QUESTIONS = 15

GENERATION_PROMPT = """Below is the COMPLETE content of a nutrition/health knowledge base about \
triglyceride management (sourced from WHO, NIH, MedlinePlus, USDA, and AHA documents).

=== FULL KNOWLEDGE BASE ===
{corpus}
=== END KNOWLEDGE BASE ===

Generate exactly {count} realistic questions that:
1. Sound like natural, plausible questions a user of a triglyceride/nutrition assistant might ask
   (closely related to the topics above -- diet, triglycerides, omega-3s, sugar, alcohol, fiber, etc.)
2. CANNOT be correctly/specifically answered using ONLY the knowledge base content above -- the \
specific fact, number, recipe, or detail being asked for is simply not present in this text, even \
though the general topic is related.

Good examples of the kind of question to generate: asking for a specific recipe not in the \
corpus, asking for an exact clinical statistic not stated in the corpus, asking for a specific \
brand/product recommendation, asking for a precise numeric threshold that isn't actually given \
for that specific sub-case, asking to compare two specific named diets not discussed, etc.

Return ONLY a JSON array of objects, each with "question" and "why_unanswerable" (one short \
phrase describing what specific info is missing), with exactly {count} elements. No markdown \
fences, no commentary.
"""


def _extract_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def load_corpus_text() -> str:
    parts = []
    for path in sorted(RAW_DATA_DIR.glob("*.md")):
        parts.append(f"--- {path.name} ---\n{path.read_text()}")
    return "\n\n".join(parts)


def main():
    corpus = load_corpus_text()
    prompt = GENERATION_PROMPT.format(corpus=corpus, count=NUM_QUESTIONS)
    raw = simple_generate(prompt, max_tokens=3000)

    try:
        items = _extract_json_array(raw)
    except (json.JSONDecodeError, AttributeError) as e:
        raise RuntimeError(f"Failed to parse JSON: {e}\nRaw: {raw}")

    records = [
        {
            "id": f"unanswerable_{i:03d}",
            "question": item["question"].strip(),
            "why_unanswerable": item.get("why_unanswerable", "").strip(),
        }
        for i, item in enumerate(items)
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} unanswerable questions:")
    for r in records:
        print(f"  - {r['question']}  [{r['why_unanswerable']}]")
    print(f"\nWrote to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
