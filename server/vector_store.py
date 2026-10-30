from typing import Any, List

import chromadb
import requests

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    """Create and return a persistent Chroma client."""
    return chromadb.PersistentClient(path=Config.CHROMA_PATH)


def get_or_create_collection():
    """Get or create the Chroma collection for the knowledge assistant."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=Config.COLLECTION_NAME)


def get_embedding(text: str) -> list[float]:
    """Create an embedding for a piece of text using the local model service."""
    response = requests.post(
        f"{Config.OLLAMA_BASE_URL}/api/embed",
        json={
            "model": Config.EMBEDDING_MODEL,
            "input": text,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]
    if "embedding" in data:
        return data["embedding"]

    raise ValueError("Embedding response did not contain an embedding.")


def seed_vector_store(chunks: List[DocumentChunk]) -> int:
    """Add document chunks to the Chroma collection."""
    if not chunks:
        return 0

    collection = get_or_create_collection()

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in chunks:
        ids.append(chunk.id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
        )
        embeddings.append(get_embedding(chunk.text))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(ids)


def retrieve_relevant_chunks(
    question: str, top_k: int | None = None
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for a user question."""
    top_k = top_k or Config.TOP_K
    collection = get_or_create_collection()

    total = collection.count()
    if total == 0:
        return []

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        chunks.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", "Unknown Source"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return chunks
