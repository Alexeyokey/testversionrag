from __future__ import annotations

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        max_length: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=max_length,
        )

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            return []
        pairs = [(query, document.page_content) for document in documents]
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
