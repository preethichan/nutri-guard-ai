"""
Ingest the triglyceride/nutrition knowledge base (data/raw/*.md) into a local
Chroma vector store (data/processed/chroma_db).

- Splits each markdown doc into chunks along "## " section headers so each
  chunk is a coherent, retrievable unit (and source-attributable).
- Preserves source metadata (org, title, url, retrieved_date, topic) on every
  chunk, so the RAG layer and later guardrails (source attribution,
  groundedness checks) can cite exactly where an answer came from.
- Uses Chroma's default local embedding function (all-MiniLM-L6-v2 via
  onnxruntime) so this runs fully offline with no API key required.

Usage:
    python -m src.rag.ingest
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
import frontmatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "triglyceride_nutrition_kb"

# Minimum characters for a chunk to be worth keeping on its own.
MIN_CHUNK_CHARS = 40


def split_into_sections(markdown_body: str) -> list[tuple[str, str]]:
    """Split a markdown document into (heading, content) chunks on '## ' headers.

    The top-level '# Title' line is treated as part of the first chunk's
    context rather than its own chunk.
    """
    lines = markdown_body.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            sections.append((current_heading, text))

    for line in lines:
        heading_match = re.match(r"^##\s+(.*)", line)
        if heading_match:
            flush()
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    return sections


def load_documents() -> list[dict]:
    """Load all markdown files from data/raw, returning chunk records."""
    records: list[dict] = []
    md_files = sorted(RAW_DATA_DIR.glob("*.md"))

    for path in md_files:
        post = frontmatter.load(path)
        metadata = dict(post.metadata)
        title = metadata.get("title", path.stem)
        source_org = metadata.get("source_org", "unknown")
        url = metadata.get("url", "")
        retrieved_date = str(metadata.get("retrieved_date", ""))
        topic = metadata.get("topic", "")

        sections = split_into_sections(post.content)
        for idx, (heading, text) in enumerate(sections):
            chunk_id = f"{path.stem}::chunk_{idx}"
            full_text = f"# {title}\n## {heading}\n\n{text}"
            records.append(
                {
                    "id": chunk_id,
                    "text": full_text,
                    "metadata": {
                        "source_file": path.name,
                        "title": title,
                        "heading": heading,
                        "source_org": source_org,
                        "url": url,
                        "retrieved_date": retrieved_date,
                        "topic": topic,
                    },
                }
            )
    return records


def build_vector_store(records: list[dict]) -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start fresh each time ingest is run, so the store always reflects the
    # current contents of data/raw/.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )
    return collection


def main():
    records = load_documents()
    if not records:
        raise SystemExit(
            f"No markdown documents found in {RAW_DATA_DIR}. "
            "Add source documents before running ingest."
        )

    collection = build_vector_store(records)
    print(f"Ingested {len(records)} chunks from {RAW_DATA_DIR} into "
          f"Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    print("\nChunks per source file:")
    counts: dict[str, int] = {}
    for r in records:
        counts[r["metadata"]["source_file"]] = counts.get(r["metadata"]["source_file"], 0) + 1
    for fname, count in sorted(counts.items()):
        print(f"  {fname}: {count} chunks")

    print(f"\nCollection count check: {collection.count()} documents stored.")


if __name__ == "__main__":
    main()
