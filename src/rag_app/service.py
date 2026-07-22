from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from threading import RLock
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag_app.config import Settings
from rag_app.documents import DocumentProcessor
from rag_app.embeddings import EmbeddingModel
from rag_app.retrieval import HybridRetriever, format_documents
from rag_app.vector_store import VectorStore

if TYPE_CHECKING:
    from rag_app.generation import TextGenerator


class RagService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._embedding_model = embedding_model
        self._reranker = None
        self._retriever: HybridRetriever | None = None
        self._generator: TextGenerator | None = None
        # Защищает только ленивое создание компонентов. После инициализации
        # read-only inference разных API-запросов может выполняться параллельно.
        self._component_lock = RLock()
        self._store = VectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
            api_key=settings.qdrant_api_key,
        )

    @property
    def embedding_model(self) -> EmbeddingModel:
        # Загружаем embedding-модель только перед первой операцией, где она нужна.
        if self._embedding_model is None:
            with self._component_lock:
                if self._embedding_model is None:
                    self._embedding_model = EmbeddingModel(
                        self.settings.embedding_model,
                        batch_size=self.settings.embedding_batch_size,
                        trust_remote_code=self.settings.trust_remote_code,
                    )
        return self._embedding_model

    def index(self, source: str, recreate: bool = False) -> int:
        embedding_model = self.embedding_model
        processor = DocumentProcessor(
            self.settings,
            tokenizer=embedding_model.tokenizer,
        )

        documents = processor.load(source)
        if not documents:
            raise ValueError("В источнике не найдено текста для индексации")

        # Обрабатываем полный цикл небольшими порциями: после upsert векторы текущего
        # батча больше не удерживаются в памяти вместе с векторами всего корпуса.
        batch_size = self.settings.embedding_batch_size
        source_ids = {
            str(document.metadata["source_id"])
            for document in documents
        }
        collection_initialized = False

        for start in range(0, len(documents), batch_size):
            document_batch = documents[start : start + batch_size]
            vector_batch = embedding_model.embed_documents(
                [document.page_content for document in document_batch]
            )

            if len(vector_batch) != len(document_batch):
                raise RuntimeError(
                    "Embedding-модель вернула неожиданное количество векторов: "
                    f"{len(vector_batch)} вместо {len(document_batch)}"
                )

            if not collection_initialized:
                self._store.ensure_collection(
                    len(vector_batch[0]),
                    recreate=recreate,
                )
                if not recreate:
                    # Удаляем прежнюю версию источников только после успешного
                    # построения первого батча, чтобы ранняя ошибка сохранила индекс.
                    self._store.delete_sources(source_ids)
                collection_initialized = True

            self._store.upsert(document_batch, vector_batch)

        # При следующем поиске перечитать обновлённый индекс и перестроить BM25.
        with self._component_lock:
            self._retriever = None
        return len(documents)

    @property
    def generator(self) -> TextGenerator:
        """Create the lightweight vLLM client once and reuse it for the session."""
        if not self.settings.generation_model:
            raise ValueError("Для генерации задайте RAG_GENERATION_MODEL")
        if self._generator is None:
            with self._component_lock:
                if self._generator is None:
                    from rag_app.generation import TextGenerator

                    self._generator = TextGenerator(
                        self.settings.generation_model,
                        max_new_tokens=self.settings.max_new_tokens,
                        base_url=self.settings.vllm_base_url,
                        api_key=self.settings.vllm_api_key,
                        temperature=self.settings.temperature,
                        thinking=self.settings.thinking,
                        timeout=self.settings.vllm_timeout,
                    )
        return self._generator

    def search(self, query: str) -> list[Document]:
        if self._retriever is None:
            with self._component_lock:
                if self._retriever is None:
                    # BM25 строится по снимку коллекции, vector search
                    # обращается в тот же Qdrant.
                    documents = self._store.load_documents()
                    if not documents:
                        return []

                    def vector_search(text: str, limit: int) -> list[Document]:
                        vector = self.embedding_model.embed_query(text)
                        return self._store.search(vector, limit)

                    if self.settings.enable_reranker and self._reranker is None:
                        from rag_app.reranker import CrossEncoderReranker

                        self._reranker = CrossEncoderReranker(
                            self.settings.reranker_model
                        )

                    self._retriever = HybridRetriever(
                        documents=documents,
                        vector_search=vector_search,
                        reranker=self._reranker,
                        top_k=self.settings.top_k,
                        candidate_k=self.settings.candidate_k,
                        rank_constant=self.settings.rank_constant,
                        vector_weight=self.settings.vector_weight,
                        bm25_weight=self.settings.bm25_weight,
                    )
        return self._retriever.retrieve(query)

    def set_retrieval_weights(
        self,
        *,
        vector_weight: float,
        bm25_weight: float,
    ) -> None:
        """Изменить только веса уже загруженного hybrid retrieval."""
        updated_settings = replace(
            self.settings,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        updated_settings.validate()
        with self._component_lock:
            self.settings = updated_settings
            if self._retriever is not None:
                # Меняем лёгкую обёртку, не пересоздавая BM25 и reranker.
                self._retriever = self._retriever.with_weights(
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                )

    def ask(
        self,
        question: str,
        *,
        chat_history: str = "Предыдущий разговор отсутствует.",
    ) -> tuple[str, list[Document]]:
        documents = self.search(question)
        answer = self.answer_from_documents(
            question,
            documents,
            chat_history=chat_history,
        )
        return answer, documents

    def answer_from_documents(
        self,
        question: str,
        documents: list[Document],
        *,
        chat_history: str = "Предыдущий разговор отсутствует.",
    ) -> str:
        """Сгенерировать ответ по уже извлечённому контексту без повторного поиска."""
        return self.generator.answer(
            question,
            format_documents(documents),
            chat_history=chat_history,
        )

    def ask_stream(
        self,
        question: str,
        *,
        chat_history: str = "Предыдущий разговор отсутствует.",
    ) -> tuple[Iterator[str], list[Document]]:
        """Return a token stream while keeping retrieval models in this service."""
        documents = self.search(question)
        chunks = self.generator.stream_answer(
            question,
            format_documents(documents),
            chat_history=chat_history,
        )
        return chunks, documents

    def warmup(self) -> int:
        """Load retrieval models before the first real question."""
        return len(self.search("Проверка готовности поисковой модели"))

    def stats(self) -> dict[str, object]:
        """Report which reusable components are already initialized."""
        return {
            "collection": self.settings.collection_name,
            "embedding_model": self.settings.embedding_model,
            "generation_model": self.settings.generation_model,
            "top_k": self.settings.top_k,
            "embedding_loaded": self._embedding_model is not None,
            "reranker_loaded": self._reranker is not None,
            "retriever_loaded": self._retriever is not None,
            "generator_client_created": self._generator is not None,
        }

