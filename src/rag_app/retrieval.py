from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from langchain_core.documents import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def document_key(document: Document) -> str:
    doc_id = document.metadata.get("doc_id")
    if not doc_id:
        raise ValueError("У документа отсутствует обязательный metadata['doc_id']")
    return str(doc_id)


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "Подходящий контекст не найден."
    parts = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        location = [f"Источник: {source}"]
        if sheet := document.metadata.get("sheet"):
            location.append(f"лист: {sheet}")
        page = document.metadata.get("page")
        if page is not None:
            location.append(f"страница: {page}")
        chunk_index = document.metadata.get("chunk_index")
        if chunk_index is not None:
            location.append(f"чанк: {chunk_index}")
        parts.append(
            f"[{index}] {'; '.join(location)}\n{document.page_content}"
        )
    return "\n\n---\n\n".join(parts)


class BM25Index:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.document_terms = [Counter(tokenize(item.page_content)) for item in documents]
        self.document_lengths = [sum(terms.values()) for terms in self.document_terms]
        self.average_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for terms in self.document_terms:
            document_frequency.update(terms.keys())
        document_count = len(documents)
        self.idf = {
            word: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for word, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[Document]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[float, int, Document]] = []
        for index, terms in enumerate(self.document_terms):
            length = self.document_lengths[index]
            score = 0.0
            for token in query_tokens:
                frequency = terms.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (
                    1 - b + b * length / (self.average_length or 1)
                )
                score += self.idf.get(token, 0.0) * frequency * (k1 + 1) / denominator
            if score > 0:
                original = self.documents[index]
                result = Document(
                    page_content=original.page_content,
                    metadata={**original.metadata, "_bm25_score": score},
                )
                scored.append((score, index, result))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [document for _, _, document in scored[:limit]]


class HybridRetriever:
    def __init__(
        self,
        documents: list[Document],
        vector_search: Callable[[str, int], list[Document]],
        reranker=None,
        *,
        top_k: int = 3,
        candidate_k: int = 8,
        rank_constant: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        self.vector_search = vector_search
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.rank_constant = rank_constant
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25Index(documents)

    def retrieve(self, query: str) -> list[Document]:
        vector_documents = self.vector_search(query, self.candidate_k)
        bm25_documents = self.bm25.search(query, self.candidate_k)
        candidates = self._merge(vector_documents, bm25_documents)[: self.candidate_k]
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, self.top_k)
        return candidates[: self.top_k]

    def _merge(
        self,
        vector_documents: list[Document],
        bm25_documents: list[Document],
    ) -> list[Document]:
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}

        for rank, document in enumerate(vector_documents, start=1):
            key = document_key(document)
            by_key[key] = document
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (
                self.rank_constant + rank
            )
        for rank, document in enumerate(bm25_documents, start=1):
            key = document_key(document)
            if key in by_key:
                by_key[key].metadata.update(document.metadata)
            else:
                by_key[key] = document
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (
                self.rank_constant + rank
            )

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [
            Document(
                page_content=by_key[key].page_content,
                metadata={**by_key[key].metadata, "_hybrid_score": scores[key]},
            )
            for key in ranked_keys
        ]
