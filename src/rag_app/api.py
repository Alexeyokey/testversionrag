from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import Condition, Lock
from typing import TYPE_CHECKING, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_app.chat import ChatSession

if TYPE_CHECKING:
    from rag_app.service import RagService


_service: RagService | None = None
_service_lock = Lock()


class _SessionState:
    """Одна история диалога и блокировка порядка сообщений внутри неё."""

    def __init__(self, session: ChatSession) -> None:
        self.session = session
        self.lock = Lock()


class _RagAccessGate:
    """Ограничить параллельные вопросы и дать индексации эксклюзивный доступ."""

    def __init__(self, max_concurrent_queries: int) -> None:
        if max_concurrent_queries <= 0:
            raise ValueError("max_concurrent_queries must be greater than zero")
        self.max_concurrent_queries = max_concurrent_queries
        self._condition = Condition()
        self._active_queries = 0
        self._ingest_pending = False

    @contextmanager
    def query(self) -> Iterator[None]:
        with self._condition:
            while (
                self._ingest_pending
                or self._active_queries >= self.max_concurrent_queries
            ):
                self._condition.wait()
            self._active_queries += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_queries -= 1
                self._condition.notify_all()

    @contextmanager
    def ingest(self) -> Iterator[None]:
        with self._condition:
            # Как только индексация встала в очередь, новые вопросы не начинаются.
            while self._ingest_pending:
                self._condition.wait()
            self._ingest_pending = True
            while self._active_queries:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._ingest_pending = False
                self._condition.notify_all()

    def stats(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "api_max_concurrent_queries": self.max_concurrent_queries,
                "api_active_queries": self._active_queries,
                "api_ingest_pending": self._ingest_pending,
            }


_access_gate = _RagAccessGate(max_concurrent_queries=2)
_sessions: dict[str, _SessionState] = {}
_sessions_lock = Lock()
_data_root = Path("data").resolve()
_max_history_turns = 6
_warmup_on_start = True


def _get_service() -> RagService:
    """Создать RagService один раз и затем переиспользовать загруженные модели."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from rag_app.config import Settings
                from rag_app.service import RagService

                _service = RagService(Settings.from_env())
    return _service


def _warmup_service() -> None:
    service = _get_service()
    print("[Warmup] Загрузка embedding-модели и reranker ...")
    found = service.warmup()
    print(f"[Warmup] Поисковые модели готовы; найдено фрагментов: {found}.")


def _get_session(session_id: str) -> _SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
        if state is None:
            state = _SessionState(
                ChatSession(
                    _get_service(),
                    max_history_turns=_max_history_turns,
                )
            )
            _sessions[session_id] = state
        return state


def _resolve_data_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        # Клиенту достаточно передать "npatest", а не полный путь /data/npatest.
        candidate = _data_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_data_root)
    except ValueError as error:
        raise ValueError(
            f"Only paths inside '{_data_root}' can be ingested."
        ) from error
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return resolved


class IngestRequest(BaseModel):
    path: str = "documents"
    reset: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "default"


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    session_id: str


class IngestResponse(BaseModel):
    indexed_chunks: int
    stats: dict[str, Any]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Прогрев вынесен из import-time: тесты и `rag --help` не загружают модели.
    if _warmup_on_start:
        await asyncio.to_thread(_warmup_service)
    yield


app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, Any]:
    result = dict(_get_service().stats())
    with _sessions_lock:
        result["api_sessions"] = len(_sessions)
    result.update(_access_gate.stats())
    return result


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    path = request.path.strip()
    if not path:
        raise HTTPException(
            status_code=400,
            detail="Field 'path' must be a non-empty string.",
        )
    try:
        target = _resolve_data_path(path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # Индексация меняет Qdrant и пересобирает retriever, поэтому она ждёт все
    # активные вопросы и временно не допускает новые.
    with _access_gate.ingest():
        count = _get_service().index(str(target), recreate=request.reset)
        if request.reset:
            with _sessions_lock:
                _sessions.clear()

    return IngestResponse(
        indexed_chunks=count,
        stats=dict(_get_service().stats()),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Field 'question' must be a non-empty string.",
        )
    session_id = request.session_id.strip() or "default"

    # Разные сессии могут выполняться одновременно. Внутренний lock конкретной
    # сессии не даёт двум её сообщениям поменять местами историю диалога.
    with _access_gate.query():
        state = _get_session(session_id)
        with state.lock:
            response = state.session.ask(question)
    return QueryResponse(
        question=question,
        answer=response.answer,
        sources=list(response.sources),
        session_id=session_id,
    )


def serve_api(
    service: RagService | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    data_root: str | Path = "data",
    max_history_turns: int = 6,
    max_concurrent_queries: int = 2,
    warmup: bool = True,
) -> None:
    global _service, _access_gate, _data_root, _max_history_turns, _warmup_on_start
    if not 1 <= port <= 65535:
        raise ValueError("API port must be between 1 and 65535.")
    if max_history_turns < 0:
        raise ValueError("max_history_turns cannot be negative")
    if max_concurrent_queries <= 0:
        raise ValueError("max_concurrent_queries must be greater than zero")

    if service is not None:
        _service = service
    _data_root = Path(data_root).resolve()
    _max_history_turns = max_history_turns
    _access_gate = _RagAccessGate(max_concurrent_queries)
    _warmup_on_start = warmup
    with _sessions_lock:
        _sessions.clear()

    # Один worker принципиален: дополнительные процессы загрузили бы отдельную
    # копию embedding-модели и reranker в память GPU.
    uvicorn.run(app, host=host, port=port, workers=1)


def main() -> None:
    load_dotenv()
    serve_api(
        host=os.getenv("RAG_API_HOST", "0.0.0.0"),
        port=int(os.getenv("RAG_API_PORT", "8080")),
        data_root=os.getenv("RAG_API_DATA_ROOT", "data"),
        max_history_turns=int(os.getenv("RAG_API_HISTORY_TURNS", "6")),
        max_concurrent_queries=int(
            os.getenv("RAG_API_MAX_CONCURRENT_QUERIES", "2")
        ),
        warmup=os.getenv("RAG_API_WARMUP", "true").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
