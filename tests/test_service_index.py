from langchain_core.documents import Document

from rag_app.config import Settings
from rag_app.service import RagService


class _EmbeddingModel:
    tokenizer = object()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["Тестовый чанк"]
        return [[0.1, 0.2, 0.3]]


class _BatchEmbeddingModel:
    tokenizer = object()

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index), 0.2, 0.3] for index, _ in enumerate(texts)]


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


class _BatchProcessor:
    def __init__(self, **kwargs) -> None:
        pass

    def load(self, source: str) -> list[Document]:
        return [
            Document(
                page_content=f"Чанк {index}",
                metadata={
                    "doc_id": f"chunk-{index}",
                    "source_id": "/data/report.pdf",
                },
            )
            for index in range(5)
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


def test_index_embeds_and_upserts_in_bounded_batches(monkeypatch) -> None:
    embedding_model = _BatchEmbeddingModel()
    store = _Store()
    service = RagService(
        Settings(
            enable_reranker=False,
            embedding_batch_size=2,
        )
    )
    service._embedding_model = embedding_model
    service._store = store
    monkeypatch.setattr("rag_app.service.DocumentProcessor", _BatchProcessor)

    count = service.index("/data", recreate=False)

    assert count == 5
    assert [len(batch) for batch in embedding_model.calls] == [2, 2, 1]
    assert [call[0] for call in store.calls].count("ensure") == 1
    assert [call[0] for call in store.calls].count("delete") == 1
    upsert_calls = [call for call in store.calls if call[0] == "upsert"]
    assert [len(call[1]) for call in upsert_calls] == [2, 2, 1]
