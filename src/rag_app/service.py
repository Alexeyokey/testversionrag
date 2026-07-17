from __future__ import annotations

from langchain_core.documents import Document

from rag_app.config import Settings
from rag_app.documents import DocumentProcessor
from rag_app.embeddings import EmbeddingModel
from rag_app.retrieval import HybridRetriever, format_documents
from rag_app.vector_store import VectorStore


class RagService:
    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self._embedding_model: EmbeddingModel | None = None
        self._store = VectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
            api_key=settings.qdrant_api_key,
        )

    @property
    def embedding_model(self) -> EmbeddingModel:
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
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            docling_chunk_tokens=self.settings.docling_chunk_tokens,
            embedding_model=self.settings.embedding_model,
            trust_remote_code=self.settings.trust_remote_code,
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

        return len(documents)

    def search(self, query: str) -> list[Document]:
        documents = self._store.load_documents()
        if not documents:
            return []

        def vector_search(text: str, limit: int) -> list[Document]:
            vector = self.embedding_model.embed_query(text)
            return self._store.search(vector, limit)

        reranker = None
        if self.settings.enable_reranker:
            from rag_app.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(self.settings.reranker_model)

        retriever = HybridRetriever(
            documents=documents,
            vector_search=vector_search,
            reranker=reranker,
            top_k=self.settings.top_k,
            candidate_k=self.settings.candidate_k,
            rank_constant=self.settings.rank_constant,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
        )
        return retriever.retrieve(query)

    def ask(self, question: str) -> tuple[str, list[Document]]:
        if not self.settings.generation_model:
            raise ValueError("Для команды ask задайте RAG_GENERATION_MODEL")
        documents = self.search(question)
        from rag_app.generation import TextGenerator

        generator = TextGenerator(
            self.settings.generation_model,
            max_new_tokens=self.settings.max_new_tokens,
            trust_remote_code=self.settings.trust_remote_code,
            base_url=self.settings.vllm_base_url,
            api_key=self.settings.vllm_api_key,
            temperature=self.settings.temperature,
            thinking=self.settings.thinking,
            timeout=self.settings.vllm_timeout,
        )
        return generator.answer(question, format_documents(documents)), documents

