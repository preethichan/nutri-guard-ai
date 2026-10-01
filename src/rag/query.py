"""
Quick test/query utility for the triglyceride/nutrition vector store.

Usage:
    python -m src.rag.query "How does alcohol affect triglycerides?"
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "triglyceride_nutrition_kb"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def query(question: str, n_results: int = 3):
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)

    print(f'\nQuery: "{question}"\n' + "-" * 60)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(f"\n[{i}] score={1 - dist:.3f}  source={meta['source_org']}  "
              f"file={meta['source_file']}  heading={meta['heading']}")
        print(f"    url: {meta['url']}")
        snippet = doc.strip().replace("\n", " ")
        print(f"    {snippet[:280]}{'...' if len(snippet) > 280 else ''}")


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        query(question)
    else:
        sample_questions = [
            "What triglyceride level is considered high?",
            "How does alcohol affect triglycerides?",
            "What foods are good sources of omega-3 fatty acids?",
            "How much added sugar is recommended per day?",
            "What is a safe diagnosis for my high triglycerides?",  # scope-probe
        ]
        for q in sample_questions:
            query(q)


if __name__ == "__main__":
    main()
