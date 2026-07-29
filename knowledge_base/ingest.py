"""
Embeds every file in knowledge_base/sample_docs into a persisted Chroma
collection. Run once before starting the API (or whenever docs change):

    python knowledge_base/ingest.py
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_store")
DOCS_DIR = Path(__file__).parent / "sample_docs"
COLLECTION_NAME = "ops_knowledge_base"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def main():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Uses Chroma's default local embedding function so ingestion works
    # without an API key. Swap for OpenAIEmbeddingFunction /
    # AnthropicEmbeddingFunction in production for higher quality.
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    ids, docs, metadatas = [], [], []
    for path in DOCS_DIR.glob("*.md"):
        text = path.read_text()
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metadatas.append({"source": path.name, "chunk": i})

    if not ids:
        print("No documents found in", DOCS_DIR)
        return

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Ingested {len(ids)} chunks from {len(list(DOCS_DIR.glob('*.md')))} docs "
          f"into collection '{COLLECTION_NAME}' at {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
