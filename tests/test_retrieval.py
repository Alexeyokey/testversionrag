from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_app.config import Settings
from rag_app.documents import DocumentProcessor
from rag_app.embeddings import EmbeddingModel
from rag_app.vector_store import VectorStore


DEFAULT_XLSX_DIRECTORY = Path(
    r"C:\Users\alex_\Documents\Codex\2026-07-15\y\outputs\rag_test_dataset\xlsx"
)
EXPECTED_WORKBOOKS = {
    "07_sales_operations.xlsx",
    "08_project_portfolio.xlsx",
    "09_sensor_timeseries.xlsx",
}


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    model_name = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "Octen/Octen-Embedding-0.6B",
    )
    print(f"\n[Embedding] Модель: {model_name}")
    return EmbeddingModel(
        model_name=model_name,
        batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "8")),
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def xlsx_documents(embedding_model: EmbeddingModel) -> list[Document]:
    directory = Path(os.getenv("RAG_TEST_XLSX_DIR", DEFAULT_XLSX_DIRECTORY))
    if not directory.is_dir():
        pytest.fail(
            f"Каталог XLSX не найден: {directory}. "
            "Укажите его через RAG_TEST_XLSX_DIR."
        )

    paths = sorted(
        path
        for path in directory.glob("*.xlsx")
        if path.is_file() and not path.name.startswith("~$")
    )
    if not paths:
        pytest.fail(f"В каталоге {directory} не найдены XLSX-файлы")

    processor = DocumentProcessor(
        Settings(
            chunk_overlap=0,
            docling_chunk_tokens=int(os.getenv("RAG_TEST_CHUNK_TOKENS", "512")),
        ),
        tokenizer=embedding_model.tokenizer,
    )
    documents: list[Document] = []

    print("\n" + "=" * 100)
    print("DOCLING CHUNKS")
    print("=" * 100)

    for path in paths:
        file_documents = processor.load_file(path)
        documents.extend(file_documents)
        for document in file_documents:
            chunk_index = document.metadata["chunk_index"]
            chunk_key = f"{path.name}:{chunk_index}"
            print(f"\n[{chunk_key}]")
            print("-" * 100)
            print("INDEXED CHUNK:")
            print(document.page_content)

    if not documents:
        pytest.fail(
            "Docling не создал ни одного чанка "
            "из тестовых XLSX-файлов"
        )

    sources = {document.metadata["source"] for document in documents}
    assert EXPECTED_WORKBOOKS <= sources
    print(f"\n[Docling] Всего создано чанков: {len(documents)}")
    return documents


@pytest.fixture(scope="module")
def vector_store(
    xlsx_documents: list[Document],
    embedding_model: EmbeddingModel,
) -> VectorStore:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv(
        "RAG_TEST_COLLECTION",
        "test_retrieval_xlsx",
    )
    store = VectorStore(
        url=qdrant_url,
        collection_name=collection_name,
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    try:
        store.client.get_collections()
    except Exception as error:
        pytest.fail(f"Qdrant недоступен по адресу {qdrant_url}: {error}")

    print("\n" + "=" * 100)
    print("QDRANT INDEXING")
    print("=" * 100)
    print(f"[Qdrant] URL: {qdrant_url}")
    print(f"[Qdrant] Коллекция: {collection_name}")

    vectors = embedding_model.embed_documents(
        [document.page_content for document in xlsx_documents]
    )
    assert vectors
    store.ensure_collection(vector_size=len(vectors[0]), recreate=True)
    store.upsert(xlsx_documents, vectors)

    print(f"[Qdrant] Загружено точек: {len(xlsx_documents)}")
    return store


def test_chunks_are_loaded_into_qdrant(
    xlsx_documents: list[Document],
    vector_store: VectorStore,
) -> None:
    loaded_documents = vector_store.load_documents()

    expected_ids = {
        document.metadata["doc_id"] for document in xlsx_documents
    }
    loaded_ids = {
        document.metadata["doc_id"] for document in loaded_documents
    }

    assert loaded_ids == expected_ids
    assert len(loaded_documents) == len(xlsx_documents)


def test_vector_search_in_qdrant(
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> None:
    query = os.getenv(
        "RAG_TEST_QUERY",
        "Какова общая выручка за период?",
    )
    limit = int(os.getenv("RAG_TEST_SEARCH_LIMIT", "5"))

    query_vector = embedding_model.embed_query(query)
    result = vector_store.search(query_vector, limit=limit)

    print("\n" + "=" * 100)
    print("QDRANT VECTOR SEARCH")
    print("=" * 100)
    print(f"Запрос: {query}")

    for rank, document in enumerate(result, start=1):
        score = document.metadata.get("vector_score", 0.0)
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", "?")
        print(f"\n{rank}. score={score:.6f}; source={source}; chunk={chunk_index}")
        print("-" * 100)
        print(document.page_content)

    assert result
    assert any(
        document.metadata["source"] == "07_sales_operations.xlsx"
        for document in result
    )
