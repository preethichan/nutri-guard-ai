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
- [x] Baseline RAG chatbot (reproduce failure modes, "before" state) — `src/rag/chat.py` (CLI) and `src/app/chat_app.py` (Streamlit UI)
- [x] Baseline evaluation harness (golden set, unanswerable set, synthetic adversarial set, DeepEval + Presidio metrics) — see `docs/eval_reports/`
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
    chat.py          # Baseline (no guardrails) chat pipeline, CLI entry point
    system_prompt.py # System prompt for the assistant
  app/
    chat_app.py      # Streamlit chat UI over the same baseline ChatSession
  eval/                # Eval harness: golden/synthetic/unanswerable set builders,
                       # metrics (retrieval, faithfulness, scope, PII/PHI, hallucination),
                       # and the eval runner (run_eval.py)
  guardrails/          # (upcoming) input/output/retrieval guardrails
tests/
  golden/              # Mechanically-derived Q&A pairs from the KB (retrieval/faithfulness)
  synthetic/           # LLM-generated adversarial/realistic query set (scope, PII, compliance)
  unanswerable/        # LLM-generated unanswerable questions (hallucination stress test)
docs/
  eval_reports/        # Baseline eval run reports (JSON + Markdown)
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

# Chat with the baseline (no guardrails) assistant -- CLI
python -m src.rag.chat                           # interactive loop
python -m src.rag.chat --once "question here"    # single turn

# Chat with the baseline assistant -- Streamlit web UI
PYTHONPATH=.deps:. python3 -m streamlit run src/app/chat_app.py
```

> **Note:** Use `python3 -m streamlit run ...`, not the bare `streamlit` command.
> Since dependencies here are installed to `.deps/` via `pip install --target`
> (sandbox workaround) rather than a real virtualenv, no `streamlit` executable
> gets placed on your `PATH` -- only the importable package. If you instead set
> up a normal `venv` and `pip install -r requirements.txt` there, the plain
> `streamlit run src/app/chat_app.py` command will work as usual.

> **Note:** The Streamlit app's `global.developmentMode` auto-detection gets
> confused when dependencies are installed to a non-standard path like `.deps/`
> (as in this sandboxed build) rather than a real `site-packages`. If you hit
> `RuntimeError: server.port does not work when global.developmentMode is true`
> or the browser URL points at a dead `:3000`, it's already worked around via
> `.streamlit/config.toml` (`global.developmentMode = false`). On a normal
> `pip install` (no `--target`), this wouldn't happen in the first place.

### Baseline evaluation
```bash
# Generate the golden/synthetic/unanswerable question sets (one-time, uses Claude)
python -m src.eval.build_golden_set
python -m src.eval.generate_synthetic_queries
python -m src.eval.build_unanswerable_set

# Run the full baseline eval (golden + unanswerable + synthetic), writes a report to docs/eval_reports/
python -m src.eval.run_eval --tag baseline
```

## Scope & disclaimer
This project is an educational exploration of AI reliability/guardrails techniques. The knowledge base and any assistant built on it are **not a substitute for professional medical or dietary advice**. Every source document includes explicit "scope note" sections marking where dietary guidance ends and clinical judgment begins — these are intentional seeds for the guardrail work still to come.
