from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    dim: int


EmbeddingBackend = Literal["local", "openai", "deterministic"]


def embed_texts(
    texts: list[str],
    *,
    backend: EmbeddingBackend = "local",
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small",
    dim: int = 384,
) -> EmbeddingResult:
    """
    Embeddings entrypoint used by RAG.

    - backend="local": sentence-transformers (default, no API key)
    - backend="openai": OpenAI embeddings API
    - backend="deterministic": fallback hash embeddings (no dependencies)
    """
    backend = backend.strip().lower()  # type: ignore[assignment]
    if backend == "openai":
        return _embed_openai(texts, api_key=openai_api_key, model=openai_model)
    if backend == "local":
        try:
            return _embed_sentence_transformers(texts)
        except Exception:
            # If model download/install fails, still keep the app usable.
            return _embed_deterministic(texts, dim=dim)
    return _embed_deterministic(texts, dim=dim)


def _embed_sentence_transformers(texts: list[str]) -> EmbeddingResult:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode(texts, normalize_embeddings=True).tolist()
    dim = len(vecs[0]) if vecs else 0
    return EmbeddingResult(vectors=vecs, dim=dim)


def _embed_openai(texts: list[str], *, api_key: str | None, model: str) -> EmbeddingResult:
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY for embeddings.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    dim = len(vecs[0]) if vecs else 0
    return EmbeddingResult(vectors=vecs, dim=dim)


def _embed_deterministic(texts: list[str], *, dim: int) -> EmbeddingResult:
    vectors: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        buf = bytearray()
        while len(buf) < dim:
            buf.extend(hashlib.sha256(h + bytes([len(buf) % 256])).digest())
        v = [((b / 255.0) * 2.0 - 1.0) for b in buf[:dim]]
        vectors.append(v)
    return EmbeddingResult(vectors=vectors, dim=dim)

