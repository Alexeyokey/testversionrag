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
    generation_model: str | None = "QuantTrio/Qwen3.6-27B-AWQ"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_timeout: float = 300.0
    trust_remote_code: bool = True
    temperature: float = 0.3
    thinking: bool = False
    enable_reranker: bool = True
    chunk_size: int = 1024
    chunk_overlap: int = 50
    docling_chunk_tokens: int = 1024
    embedding_batch_size: int = 32
    top_k: int = 3
    candidate_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rank_constant: int = 10
    max_new_tokens: int = 256
    ragas_judge_model: str | None = None
    ragas_threshold: float = 0.7
    ragas_max_tokens: int = 2048
    ragas_context_precision_concurrency: int = 5
    evaluation_artifact_cache_enabled: bool = True
    evaluation_artifact_cache_dir: str = "evaluation/artifact-cache"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", defaults.qdrant_url),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("RAG_COLLECTION", defaults.collection_name),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", defaults.embedding_model),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", defaults.reranker_model),
            generation_model=os.getenv("RAG_GENERATION_MODEL") or defaults.generation_model,
            vllm_base_url=os.getenv("RAG_VLLM_BASE_URL", defaults.vllm_base_url),
            vllm_api_key=os.getenv("RAG_VLLM_API_KEY") or None,
            vllm_timeout=float(os.getenv("RAG_VLLM_TIMEOUT", defaults.vllm_timeout)),
            trust_remote_code=_as_bool("RAG_TRUST_REMOTE_CODE", defaults.trust_remote_code),
            enable_reranker=_as_bool("RAG_ENABLE_RERANKER", defaults.enable_reranker),
            temperature=float(os.getenv("RAG_TEMPERATURE", defaults.temperature)),
            thinking=_as_bool("RAG_THINKING", defaults.thinking),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", defaults.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", defaults.chunk_overlap)),
            docling_chunk_tokens=int(
                os.getenv(
                    "RAG_DOCLING_CHUNK_TOKENS",
                    defaults.docling_chunk_tokens,
                )
            ),
            embedding_batch_size=int(
                os.getenv("RAG_EMBEDDING_BATCH_SIZE", defaults.embedding_batch_size)
            ),
            top_k=int(os.getenv("RAG_TOP_K", defaults.top_k)),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", defaults.candidate_k)),
            vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", defaults.vector_weight)),
            bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", defaults.bm25_weight)),
            rank_constant=int(os.getenv("RAG_RANK_CONSTANT", defaults.rank_constant)),
            max_new_tokens=int(os.getenv("RAG_MAX_NEW_TOKENS", defaults.max_new_tokens)),
            ragas_judge_model=os.getenv("RAGAS_JUDGE_MODEL") or None,
            ragas_threshold=float(
                os.getenv("RAGAS_THRESHOLD", defaults.ragas_threshold)
            ),
            ragas_max_tokens=int(
                os.getenv("RAGAS_MAX_TOKENS", defaults.ragas_max_tokens)
            ),
            ragas_context_precision_concurrency=int(
                os.getenv(
                    "RAGAS_CONTEXT_PRECISION_CONCURRENCY",
                    defaults.ragas_context_precision_concurrency,
                )
            ),
            evaluation_artifact_cache_enabled=_as_bool(
                "EVALUATION_ARTIFACT_CACHE_ENABLED",
                defaults.evaluation_artifact_cache_enabled,
            ),
            evaluation_artifact_cache_dir=(
                os.getenv("EVALUATION_ARTIFACT_CACHE_DIR")
                or defaults.evaluation_artifact_cache_dir
            ),
        )

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE должен быть больше нуля")
        if self.chunk_overlap < 0:
            raise ValueError("RAG_CHUNK_OVERLAP не может быть отрицательным")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP должен быть меньше RAG_CHUNK_SIZE")
        if self.docling_chunk_tokens <= 0:
            raise ValueError("RAG_DOCLING_CHUNK_TOKENS должен быть больше нуля")
        if self.embedding_batch_size <= 0:
            raise ValueError("RAG_EMBEDDING_BATCH_SIZE должен быть больше нуля")
        if self.top_k <= 0 or self.candidate_k < self.top_k:
            raise ValueError("RAG_CANDIDATE_K должен быть не меньше RAG_TOP_K > 0")
        if self.vector_weight < 0 or self.bm25_weight < 0:
            raise ValueError("Веса поиска не могут быть отрицательными")
        if self.vector_weight + self.bm25_weight == 0:
            raise ValueError("Хотя бы один вес поиска должен быть больше нуля")
        if self.rank_constant < 0:
            raise ValueError("RAG_RANK_CONSTANT не может быть отрицательным")
        if self.max_new_tokens <= 0:
            raise ValueError("RAG_MAX_NEW_TOKENS должен быть больше нуля")
        if self.vllm_timeout <= 0:
            raise ValueError("RAG_VLLM_TIMEOUT должен быть больше нуля")
        if not 0 <= self.ragas_threshold <= 1:
            raise ValueError("RAGAS_THRESHOLD должен находиться в диапазоне от 0 до 1")
        if self.ragas_max_tokens <= 0:
            raise ValueError("RAGAS_MAX_TOKENS должен быть больше нуля")
        if self.ragas_context_precision_concurrency <= 0:
            raise ValueError(
                "RAGAS_CONTEXT_PRECISION_CONCURRENCY должен быть больше нуля"
            )
        if not self.evaluation_artifact_cache_dir.strip():
            raise ValueError("EVALUATION_ARTIFACT_CACHE_DIR не может быть пустым")
