from __future__ import annotations

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANK_INSTRUCTION = (
    "Given a user question for a Russian and English RAG knowledge base, "
    "retrieve passages that directly answer the question."
)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        max_length: int = 512,
        instruction: str = RERANK_INSTRUCTION,
    ) -> None:
        self.batch_size = batch_size
        self.instruction = instruction.strip()
        self.model = CrossEncoder(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=max_length,
        )

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            return []
        instructed_query = (
            f"{self.instruction}\n\nUser question: {query}"
            if self.instruction
            else query
        )
        pairs = [
            (instructed_query, document.page_content)
            for document in documents
        ]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked = [
            Document(
                page_content=document.page_content,
                metadata={**document.metadata, "_rerank_score": float(score)},
            )
            for document, score in zip(documents, scores, strict=True)
        ]
        ranked.sort(key=lambda item: item.metadata["_rerank_score"], reverse=True)
        return ranked[:top_k]
