# nutri-guard-ai

A guardrailed RAG assistant project, inspired by DeepLearning.AI's *Safe and Reliable AI via Guardrails* course — rebuilt around a real, higher-stakes domain instead of the course's toy pizzeria example.

**Use case:** A dietary/nutrition assistant focused on **triglyceride management** — answering questions grounded in real guidance from WHO, NIH/NHLBI, NIH Office of Dietary Supplements, MedlinePlus, USDA/HHS Dietary Guidelines, and the American Heart Association.

**Why this domain:** Nutrition advice around a real health marker (triglycerides) is a great stress-test for AI reliability because it reproduces all four failure modes the course targets, with real consequences if guardrails fail:

| Failure mode | Risk in this domain |
|---|---|
| Hallucination | Fabricated lab-value thresholds, supplement dosages, or drug-nutrient interactions |
| Unintended use | Users asking for a diagnosis, medication advice, or treatment of symptoms (e.g., suspected pancreatitis) |
| Information leakage | Users sharing health conditions, medications, lab results, or other sensitive health data |
| Reputational/compliance risk | Giving medical-sounding advice a dietary assistant isn't licensed to give |

## Project status
- [x] Real, sourced knowledge base (`data/raw/`) — 8 documents, no mock data
- [x] Vector store ingestion pipeline (`src/rag/ingest.py`) using Chroma (local, offline embeddings)
- [x] Retrieval sanity-checked (`src/rag/query.py`)
- [ ] Baseline RAG chatbot (reproduce failure modes, "before" state)
- [ ] Input guardrails (scope, prompt-injection, PII detection)
- [ ] Retrieval guardrails (relevance threshold, source attribution)
- [ ] Output guardrails (groundedness/faithfulness, PII leak filter, medical-scope filter)
- [ ] Adversarial test suite + before/after evaluation

## Data sourcing
See [`data/SOURCES.md`](data/SOURCES.md) for the full manifest of real sources used (WHO, NIH/NHLBI, NIH ODS, MedlinePlus, USDA/HHS, AHA), each fetched live and attributed with URL + retrieval date in the document frontmatter. No synthetic/mock content was used.

## Project structure
```
data/
  raw/              # Sourced markdown knowledge base documents (frontmatter + content)
  processed/        # Generated Chroma vector store (gitignored, rebuildable)
  SOURCES.md         # Manifest of all sources used
src/
  rag/
    ingest.py        # Chunk + embed data/raw/*.md into the Chroma vector store
    query.py         # CLI tool to test retrieval against the vector store
  guardrails/         # (upcoming) input/output/retrieval guardrails
tests/                # (upcoming) adversarial test suite
docs/                 # (upcoming) design notes
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** In the sandboxed agent environment used to build this project, `python -m venv` and some pip install paths were blocked by filesystem sandboxing, so dependencies were installed to a local `.deps/` folder instead (`pip install --target=.deps -r requirements.txt`, then `PYTHONPATH=.deps`). On a normal machine, the standard venv workflow above works fine.

## Usage
```bash
# Build/rebuild the vector store from data/raw/
python -m src.rag.ingest

# Test retrieval
python -m src.rag.query "How does alcohol affect triglycerides?"
```

## Scope & disclaimer
This project is an educational exploration of AI reliability/guardrails techniques. The knowledge base and any assistant built on it are **not a substitute for professional medical or dietary advice**. Every source document includes explicit "scope note" sections marking where dietary guidance ends and clinical judgment begins — these are intentional seeds for the guardrail work still to come.
