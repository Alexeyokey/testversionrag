from langchain_core.documents import Document

from rag_app.reranker import CrossEncoderReranker, RERANK_INSTRUCTION


class _Model:
    def __init__(self) -> None:
        self.pairs = []

    def predict(self, pairs, batch_size: int):
        self.pairs = pairs
        assert batch_size == 4
        return [0.2, 0.9]


def test_reranker_adds_instruction_to_query() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.batch_size = 4
    reranker.instruction = RERANK_INSTRUCTION
    reranker.model = _Model()
    documents = [
        Document(page_content="Первый фрагмент", metadata={"doc_id": "1"}),
        Document(page_content="Second passage", metadata={"doc_id": "2"}),
    ]

    result = reranker.rerank("Какова общая выручка?", documents, top_k=2)

    instructed_query, first_passage = reranker.model.pairs[0]
    assert instructed_query == (
        f"{RERANK_INSTRUCTION}\n\nUser question: Какова общая выручка?"
    )
    assert first_passage == "Первый фрагмент"
    assert [document.metadata["doc_id"] for document in result] == ["2", "1"]
    assert result[0].metadata["_rerank_score"] == 0.9
