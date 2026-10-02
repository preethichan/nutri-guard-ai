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
- [x] Baseline behavior evidence recorded (screenshots + transcripts) — see `docs/baseline_evidence/`
- [x] Input guardrail — PII/PHI redaction (`src/guardrails/pii_redaction.py`)
- [x] Retrieval guardrail — low-confidence nudge (`src/rag/chat_guarded.py`)
- [x] Output guardrail — combined groundedness/scope/overclaim judge + bounded remediation (`src/guardrails/output_judge.py`)
- [x] Guardrailed chat pipeline (`src/rag/chat_guarded.py`, `GuardedChatSession`) — "after" state, same `send()` interface as the baseline
- [x] Guardrailed pipeline run through the full eval harness, "after" report — see `docs/eval_reports/guarded_v1_*` and `docs/guardrails_before_after.md`
- [ ] Baseline/Guardrailed toggle in the Streamlit app for a live side-by-side demo
- [ ] "After" screenshots matching the recorded baseline evidence

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
    chat_guarded.py  # Guardrailed chat pipeline (GuardedChatSession), CLI entry point
    system_prompt.py # System prompt for the assistant
  app/
    chat_app.py      # Streamlit chat UI over the baseline ChatSession
  eval/                # Eval harness: golden/synthetic/unanswerable set builders,
                       # metrics (retrieval, faithfulness, scope, PII/PHI, hallucination),
                       # and the eval runner (run_eval.py, --pipeline baseline|guarded)
  guardrails/
    pii_redaction.py  # Input guardrail: redact identifying PII, preserve clinical context
    output_judge.py   # Output guardrail: combined groundedness/scope/overclaim judge
tests/
  golden/              # Mechanically-derived Q&A pairs from the KB (retrieval/faithfulness)
  synthetic/           # LLM-generated adversarial/realistic query set (scope, PII, compliance)
  unanswerable/        # LLM-generated unanswerable questions (hallucination stress test)
docs/
  eval_reports/        # Baseline + guardrailed eval run reports (JSON + Markdown)
  baseline_evidence/   # Screenshots + transcripts of baseline (pre-guardrail) failure modes
  guardrails_before_after.md  # Before/after comparison write-up (baseline_v2 vs guarded_v1)
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

# Chat with the guardrailed assistant -- CLI (input PII redaction, retrieval
# confidence nudge, output groundedness/scope/overclaim judge + remediation)
python -m src.rag.chat_guarded                           # interactive loop
python -m src.rag.chat_guarded --once "question here"    # single turn
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

### Evaluation
```bash
# Generate the golden/synthetic/unanswerable question sets (one-time, uses Claude)
python -m src.eval.build_golden_set
python -m src.eval.generate_synthetic_queries
python -m src.eval.build_unanswerable_set

# Run the full eval (golden + unanswerable + synthetic) against either pipeline,
# writes a report to docs/eval_reports/
python -m src.eval.run_eval --tag baseline   --pipeline baseline
python -m src.eval.run_eval --tag guarded_v2 --pipeline guarded

# Useful flags: --limit N (smoke test), --skip-faithfulness, --skip-golden,
# --skip-unanswerable, --skip-synthetic
```

## Guardrails

Three layers, each mapped directly to a failure mode measured in the baseline
eval (`docs/eval_reports/baseline_v2_*`). See `src/rag/chat_guarded.py` for
the full pipeline and `docs/guardrails_before_after.md` for the complete
before/after comparison.

1. **Input — PII/PHI redaction** (`src/guardrails/pii_redaction.py`): strips
   identifying entities (name, email, phone, SSN, credit card, location,
   medical license) before the message reaches the LLM, conversation
   history, or the log file. Clinically-relevant but non-identifying terms
   (medication, condition, lab value) are deliberately preserved for answer
   quality.
2. **Retrieval confidence** (cheap, no extra LLM call, in `chat_guarded.py`):
   if the top retrieval score is below a calibrated threshold (0.4), an
   explicit per-turn instruction tells the model to say so rather than fill
   in specifics from general knowledge.
3. **Output — combined safety judge** (`src/guardrails/output_judge.py`): a
   single LLM-judge call per response checks groundedness, scope, and
   overclaim together. On any violation the response is regenerated once
   with a corrective instruction; if still flagged, a safe templated
   fallback ships instead of a doubly-flagged response.

Headline results, same 134-item eval set, baseline vs. guarded:

| Metric | Baseline | Guarded |
|---|---|---|
| Fabrication rate (unanswerable set) | 13.3% | **6.7%** |
| Overclaim rate (synthetic set) | 2.1% | **0.0%** |
| Identifying-PII backend storage leak | 100% | **0.0%** |
| Clinical-PHI backend storage leak | ~100% (by construction) | 94.6% (preserved by design) |

Full breakdown, by-category tables, and two illustrative before/after
transcripts: [`docs/guardrails_before_after.md`](docs/guardrails_before_after.md).

## Scope & disclaimer
This project is an educational exploration of AI reliability/guardrails techniques. The knowledge base and any assistant built on it are **not a substitute for professional medical or dietary advice**. Every source document includes explicit "scope note" sections marking where dietary guidance ends and clinical judgment begins — the scope-violation output guardrail (`src/guardrails/output_judge.py`) is built directly on top of that boundary.
