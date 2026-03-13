from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from models.embeddings import embed_texts


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RAGIndex:
    chunks: list[Chunk]
    vectors: list[list[float]]  # normalized vectors


def chunk_text(
    text: str,
    *,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[str]:
    t = " ".join((text or "").split())
    if not t:
        return []
    if chunk_size <= 0:
        return [t]

    chunks: list[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunks.append(t[start:end])
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return chunks


def build_rag_index(
    *,
    docs: list[tuple[str, str, str]],  # (doc_id, source, text)
    embeddings_backend: str = "local",
    openai_api_key: str | None = None,
    openai_embeddings_model: str = "text-embedding-3-small",
    chunk_size: int = 900,
    overlap: int = 120,
) -> RAGIndex:
    chunks: list[Chunk] = []
    for doc_id, source, text in docs:
        for i, ct in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(Chunk(chunk_id=f"{doc_id}::c{i}", doc_id=doc_id, source=source, text=ct))

    if not chunks:
        return RAGIndex(chunks=[], vectors=[])

    embs = embed_texts(
        [c.text for c in chunks],
        backend=("openai" if embeddings_backend == "openai" else "local"),
        openai_api_key=openai_api_key,
        openai_model=openai_embeddings_model,
    )
    vectors = [_normalize(v) for v in embs.vectors]
    return RAGIndex(chunks=chunks, vectors=vectors)


def retrieve(
    *,
    index: RAGIndex,
    query: str,
    k: int = 4,
    embeddings_backend: str = "local",
    openai_api_key: str | None = None,
    openai_embeddings_model: str = "text-embedding-3-small",
) -> list[tuple[Chunk, float]]:
    if not index.chunks:
        return []

    q = embed_texts(
        [query],
        backend=("openai" if embeddings_backend == "openai" else "local"),
        openai_api_key=openai_api_key,
        openai_model=openai_embeddings_model,
        dim=len(index.vectors[0]) if index.vectors else 384,
    ).vectors[0]
    qn = _normalize(q)

    scored: list[tuple[int, float]] = []
    for i, v in enumerate(index.vectors):
        scored.append((i, _dot(qn, v)))
    scored.sort(key=lambda x: x[1], reverse=True)

    out: list[tuple[Chunk, float]] = []
    for i, s in scored[: max(1, k)]:
        out.append((index.chunks[i], s))
    return out


def format_context(chunks: list[tuple[Chunk, float]]) -> str:
    if not chunks:
        return ""
    parts: list[str] = ["Use the following local context if relevant:\n"]
    for c, score in chunks:
        parts.append(f"[source: {c.doc_id} | score: {score:.3f}]\n{c.text}\n")
    return "\n".join(parts).strip()


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(v: list[float]) -> list[float]:
    n = sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

