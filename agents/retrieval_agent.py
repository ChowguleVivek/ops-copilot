"""
Retrieval agent: thin wrapper around the persisted Chroma collection.
Given a query, returns the top-k relevant policy chunks so the reasoning
agents can ground their decisions in the actual support policy instead
of the LLM guessing at priority rules.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_store")
COLLECTION_NAME = "ops_knowledge_base"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )
    return _collection


def retrieve_policy_context(query: str, k: int = 3) -> str:
    """Return the top-k relevant policy chunks concatenated into one
    context string, ready to drop into a prompt."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "(no relevant policy found — run knowledge_base/ingest.py first)"
    return "\n---\n".join(docs)
