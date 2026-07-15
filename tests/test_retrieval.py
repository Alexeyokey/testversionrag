from langchain_core.documents import Document

from rag_app.retrieval import BM25Index, HybridRetriever


def _document(doc_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"doc_id": doc_id, "source": "test"})


def test_bm25_prefers_matching_document() -> None:
    documents = [_document("1", "красное яблоко"), _document("2", "синяя машина")]
    result = BM25Index(documents).search("яблоко", limit=2)
    assert [item.metadata["doc_id"] for item in result] == ["1"]


def test_hybrid_retriever_merges_by_doc_id() -> None:
    documents = [_document("1", "красное яблоко"), _document("2", "синяя машина")]

    def vector_search(_query: str, _limit: int) -> list[Document]:
        return [documents[1], documents[0]]

    result = HybridRetriever(
        documents,
        vector_search,
        top_k=2,
        candidate_k=2,
        vector_weight=0.5,
        bm25_weight=0.5,
    ).retrieve("яблоко")
    assert {item.metadata["doc_id"] for item in result} == {"1", "2"}
    assert all("_hybrid_score" in item.metadata for item in result)

