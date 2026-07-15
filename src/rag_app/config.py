from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "documents"
    embedding_model: str = "Octen/Octen-Embedding-0.6B"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    generation_model: str | None = None
    trust_remote_code: bool = True
    enable_reranker: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_batch_size: int = 32
    top_k: int = 3
    candidate_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rank_constant: int = 10
    max_new_tokens: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", defaults.qdrant_url),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("RAG_COLLECTION", defaults.collection_name),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", defaults.embedding_model),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", defaults.reranker_model),
            generation_model=os.getenv("RAG_GENERATION_MODEL") or None,
            trust_remote_code=_as_bool("RAG_TRUST_REMOTE_CODE", defaults.trust_remote_code),
            enable_reranker=_as_bool("RAG_ENABLE_RERANKER", defaults.enable_reranker),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", defaults.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", defaults.chunk_overlap)),
            embedding_batch_size=int(
                os.getenv("RAG_EMBEDDING_BATCH_SIZE", defaults.embedding_batch_size)
            ),
            top_k=int(os.getenv("RAG_TOP_K", defaults.top_k)),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", defaults.candidate_k)),
            vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", defaults.vector_weight)),
            bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", defaults.bm25_weight)),
            rank_constant=int(os.getenv("RAG_RANK_CONSTANT", defaults.rank_constant)),
            max_new_tokens=int(os.getenv("RAG_MAX_NEW_TOKENS", defaults.max_new_tokens)),
        )

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP должен быть меньше RAG_CHUNK_SIZE")
        if self.top_k <= 0 or self.candidate_k < self.top_k:
            raise ValueError("RAG_CANDIDATE_K должен быть не меньше RAG_TOP_K > 0")
        if self.vector_weight < 0 or self.bm25_weight < 0:
            raise ValueError("Веса поиска не могут быть отрицательными")
        if self.vector_weight + self.bm25_weight == 0:
            raise ValueError("Хотя бы один вес поиска должен быть больше нуля")
