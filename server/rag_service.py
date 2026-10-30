from typing import Any

import requests

from config import Config
from vector_store import retrieve_relevant_chunks


def answer_question(question: str) -> dict[str, Any]:
    """
    Run the RAG workflow for a user question.

    This function should:
    1. Retrieve relevant chunks.
    2. Build a prompt from the question and retrieved context.
    3. Send the prompt to the model service.
    4. Return the generated answer and supporting sources.
    """
    chunks = retrieve_relevant_chunks(question, top_k=Config.TOP_K)

    if not chunks:
        return {
            "answer": (
                "I could not find enough relevant information in the provided "
                "knowledge base to answer that question."
            ),
            "sources": [],
            "metadata": {
                "model": Config.GENERATION_MODEL,
                "retrieved_chunks": 0,
            },
        }

    prompt = build_prompt(question, chunks)
    answer = call_generation_model(prompt)

    return {
        "answer": answer,
        "sources": format_sources(chunks),
        "metadata": {
            "model": Config.GENERATION_MODEL,
            "retrieved_chunks": len(chunks),
        },
    }


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """
    Build a prompt that asks the model to answer using only retrieved context.
    """
    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown Source")
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")

        context_blocks.append(
            f"[Source {index}: {title} | {source}]\n{text}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a helpful internal knowledge assistant.

Use only the provided context to answer the user's question.
If the context does not contain enough information, say that you do not have enough information from the knowledge base.

Keep the answer clear and practical.
Do not invent policies, steps, or facts that are not supported by the context.

Context:
{context}

User question:
{question}

Answer:
""".strip()


def call_generation_model(prompt: str) -> str:
    """Send the final prompt to the configured generation model."""
    response = requests.post(
        f"{Config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": Config.GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": Config.TEMPERATURE,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()


def format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format retrieved chunks for the frontend."""
    sources = []

    for chunk in chunks:
        text = chunk.get("text", "")
        excerpt = text[:280] + "..." if len(text) > 280 else text

        sources.append(
            {
                "title": chunk.get("title", "Unknown Source"),
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index"),
                "excerpt": excerpt,
            }
        )

    return sources