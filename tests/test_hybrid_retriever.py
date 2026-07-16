from langchain_core.documents import Document

from rag_app.retrieval import HybridRetriever, format_documents


def _document(doc_id: str, text: str, chunk_index: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "source": "sales.xlsx",
            "sheet": "Dashboard",
            "chunk_index": chunk_index,
        },
    )


def test_hybrid_retriever_merges_vector_and_bm25_by_doc_id() -> None:
    revenue = _document("revenue", "Общая выручка составляет 100 рублей", 1)
    customer = _document("customer", "Крупнейший клиент — Альфа", 2)

    def vector_search(_query: str, _limit: int) -> list[Document]:
        return [
            Document(
                page_content=customer.page_content,
                metadata={**customer.metadata, "vector_score": 0.9},
            ),
            Document(
                page_content=revenue.page_content,
                metadata={**revenue.metadata, "vector_score": 0.8},
            ),
        ]

    result = HybridRetriever(
        documents=[revenue, customer],
        vector_search=vector_search,
        top_k=2,
        candidate_k=2,
    ).retrieve("общая выручка")

    assert result[0].metadata["doc_id"] == "revenue"
    assert result[0].metadata["_bm25_score"] > 0
    assert result[0].metadata["vector_score"] == 0.8
    assert all("_hybrid_score" in document.metadata for document in result)


def test_format_documents_includes_source_location() -> None:
    context = format_documents([_document("revenue", "Текст чанка", 3)])

    assert "Источник: sales.xlsx" in context
    assert "лист: Dashboard" in context
    assert "чанк: 3" in context
    assert "Текст чанка" in context
