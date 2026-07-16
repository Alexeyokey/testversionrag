from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


class VectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if exists:
            collection = self.client.get_collection(self.collection_name)
            vectors_config = collection.config.params.vectors
            if (
                isinstance(vectors_config, models.VectorParams)
                and vectors_config.size != vector_size
            ):
                raise ValueError(
                    "Размер векторов существующей коллекции "
                    f"({vectors_config.size}) не совпадает с моделью "
                    f"({vector_size}). Повторите index с --recreate."
                )
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, documents: list[Document], vectors: list[list[float]]) -> None:
        if len(documents) != len(vectors):
            raise ValueError("Количество документов и векторов не совпадает")
        missing_ids = [
            index
            for index, document in enumerate(documents)
            if not document.metadata.get("doc_id")
        ]
        if missing_ids:
            raise ValueError(
                "У документов отсутствует metadata['doc_id']: "
                + ", ".join(map(str, missing_ids))
            )
        points = [
            models.PointStruct(
                id=document.metadata["doc_id"],
                vector=vector,
                payload={"text": document.page_content, **document.metadata},
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + 128],
                wait=True,
            )

    def delete_sources(self, source_ids: set[str]) -> None:
        """Удалить старые чанки переиндексируемых источников."""
        if not source_ids or not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                        for source_id in sorted(source_ids)
                    ]
                )
            ),
            wait=True,
        )

    def search(self, query_vector: list[float], limit: int) -> list[Document]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        documents: list[Document] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            payload["vector_score"] = float(point.score)
            payload.setdefault("doc_id", str(point.id))
            documents.append(Document(page_content=text, metadata=payload))
        return documents

    def load_documents(self) -> list[Document]:
        if not self.client.collection_exists(self.collection_name):
            return []
        documents: list[Document] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                text = str(payload.pop("text", ""))
                payload.setdefault("doc_id", str(point.id))
                documents.append(Document(page_content=text, metadata=payload))
            if offset is None:
                break
        documents.sort(key=lambda item: (str(item.metadata.get("source", "")), item.metadata.get("chunk_index", 0)))
        return documents
