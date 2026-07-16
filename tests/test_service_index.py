from langchain_core.documents import Document

from rag_app.config import Settings
from rag_app.service import RagService


class _EmbeddingModel:
    tokenizer = object()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["Тестовый чанк"]
        return [[0.1, 0.2, 0.3]]


class _Processor:
    received_tokenizer = None

    def __init__(self, **kwargs) -> None:
        type(self).received_tokenizer = kwargs["tokenizer"]

    def load(self, source: str) -> list[Document]:
        assert source == "/data/xlsx"
        return [
            Document(
                page_content="Тестовый чанк",
                metadata={
                    "doc_id": "chunk-1",
                    "source_id": "/data/xlsx/sales.xlsx",
                },
            )
        ]


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        self.calls.append(("ensure", vector_size, recreate))

    def delete_sources(self, source_ids: set[str]) -> None:
        self.calls.append(("delete", source_ids))

    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        self.calls.append(("upsert", documents, vectors))


def test_index_connects_processor_embeddings_and_vector_store(monkeypatch) -> None:
    embedding_model = _EmbeddingModel()
    store = _Store()
    service = RagService(Settings(enable_reranker=False))
    service._embedding_model = embedding_model
    service._store = store
    monkeypatch.setattr("rag_app.service.DocumentProcessor", _Processor)

    count = service.index("/data/xlsx", recreate=False)

    assert count == 1
    assert _Processor.received_tokenizer is embedding_model.tokenizer
    assert store.calls[0] == ("ensure", 3, False)
    assert store.calls[1] == ("delete", {"/data/xlsx/sales.xlsx"})
    assert store.calls[2][0] == "upsert"
    assert store.calls[2][2] == [[0.1, 0.2, 0.3]]
