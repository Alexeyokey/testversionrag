from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, BrokenBarrierError, Event, Lock

import pytest
from fastapi import HTTPException

from rag_app import api


@dataclass
class _Document:
    metadata: dict[str, str]


class _Service:
    def __init__(self) -> None:
        self.histories: list[str] = []
        self.index_calls: list[tuple[str, bool]] = []

    def ask(self, question: str, *, chat_history: str):
        self.histories.append(chat_history)
        return f"Ответ на {question}", [_Document({"source": "law.rtf"})]

    def index(self, source: str, recreate: bool = False) -> int:
        self.index_calls.append((source, recreate))
        return 7

    def stats(self):
        return {"collection": "documents", "embedding_loaded": True}


class _ConcurrentService(_Service):
    def __init__(self, expected_concurrency: int) -> None:
        super().__init__()
        self.barrier = Barrier(expected_concurrency)
        self.counter_lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def ask(self, question: str, *, chat_history: str):
        with self.counter_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            self.barrier.wait(timeout=2)
            return super().ask(question, chat_history=chat_history)
        except BrokenBarrierError as error:
            raise AssertionError("RAG requests did not execute concurrently") from error
        finally:
            with self.counter_lock:
                self.active_calls -= 1


class _BlockingService(_Service):
    def __init__(self) -> None:
        super().__init__()
        self.first_call_started = Event()
        self.release_first_call = Event()
        self.second_call_started = Event()
        self.index_started = Event()
        self._call_lock = Lock()
        self._call_count = 0

    def ask(self, question: str, *, chat_history: str):
        with self._call_lock:
            self._call_count += 1
            call_number = self._call_count
        if call_number == 1:
            self.first_call_started.set()
            if not self.release_first_call.wait(timeout=2):
                raise AssertionError("Timed out while waiting to finish first request")
        else:
            self.second_call_started.set()
        return super().ask(question, chat_history=chat_history)

    def index(self, source: str, recreate: bool = False) -> int:
        self.index_started.set()
        return super().index(source, recreate=recreate)


@pytest.fixture
def api_service(monkeypatch, tmp_path):
    service = _Service()
    monkeypatch.setattr(api, "_service", service)
    monkeypatch.setattr(api, "_data_root", tmp_path.resolve())
    monkeypatch.setattr(api, "_max_history_turns", 2)
    monkeypatch.setattr(api, "_access_gate", api._RagAccessGate(2))
    with api._sessions_lock:
        api._sessions.clear()
    yield service
    with api._sessions_lock:
        api._sessions.clear()


def test_query_reuses_history_only_inside_the_same_session(api_service) -> None:
    first = api.query(api.QueryRequest(question="Первый", session_id="alice"))
    api.query(api.QueryRequest(question="Второй", session_id="alice"))
    api.query(api.QueryRequest(question="Другой", session_id="bob"))

    assert first.question == "Первый"
    assert first.answer == "Ответ на Первый"
    assert first.sources == ["law.rtf"]
    assert first.session_id == "alice"
    assert "Первый" in api_service.histories[1]
    assert "Первый" not in api_service.histories[2]
    assert api.stats()["api_sessions"] == 2


def test_different_sessions_execute_concurrently(monkeypatch, tmp_path) -> None:
    service = _ConcurrentService(expected_concurrency=2)
    monkeypatch.setattr(api, "_service", service)
    monkeypatch.setattr(api, "_data_root", tmp_path.resolve())
    monkeypatch.setattr(api, "_access_gate", api._RagAccessGate(2))
    with api._sessions_lock:
        api._sessions.clear()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            api.query,
            api.QueryRequest(question="Первый", session_id="alice"),
        )
        second = executor.submit(
            api.query,
            api.QueryRequest(question="Второй", session_id="bob"),
        )
        assert first.result(timeout=3).session_id == "alice"
        assert second.result(timeout=3).session_id == "bob"

    assert service.max_active_calls == 2


def test_same_session_preserves_request_order(monkeypatch, tmp_path) -> None:
    service = _BlockingService()
    monkeypatch.setattr(api, "_service", service)
    monkeypatch.setattr(api, "_data_root", tmp_path.resolve())
    monkeypatch.setattr(api, "_access_gate", api._RagAccessGate(2))
    with api._sessions_lock:
        api._sessions.clear()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            api.query,
            api.QueryRequest(question="Первый", session_id="alice"),
        )
        assert service.first_call_started.wait(timeout=1)
        second = executor.submit(
            api.query,
            api.QueryRequest(question="Второй", session_id="alice"),
        )
        assert not service.second_call_started.wait(timeout=0.1)
        service.release_first_call.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert service.second_call_started.is_set()
    assert "Первый" in service.histories[1]


def test_ingest_waits_for_active_queries(monkeypatch, tmp_path) -> None:
    service = _BlockingService()
    target = tmp_path / "npa"
    target.mkdir()
    monkeypatch.setattr(api, "_service", service)
    monkeypatch.setattr(api, "_data_root", tmp_path.resolve())
    monkeypatch.setattr(api, "_access_gate", api._RagAccessGate(2))
    with api._sessions_lock:
        api._sessions.clear()

    with ThreadPoolExecutor(max_workers=2) as executor:
        question = executor.submit(
            api.query,
            api.QueryRequest(question="Вопрос", session_id="alice"),
        )
        assert service.first_call_started.wait(timeout=1)
        indexing = executor.submit(
            api.ingest,
            api.IngestRequest(path="npa"),
        )
        assert not service.index_started.wait(timeout=0.1)
        service.release_first_call.set()
        question.result(timeout=2)
        assert indexing.result(timeout=2).indexed_chunks == 7

    assert service.index_started.is_set()


def test_ingest_accepts_only_paths_inside_data_root(api_service, tmp_path) -> None:
    target = tmp_path / "npa"
    target.mkdir()

    result = api.ingest(api.IngestRequest(path="npa", reset=True))

    assert result.indexed_chunks == 7
    assert api_service.index_calls == [(str(target.resolve()), True)]

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    with pytest.raises(HTTPException) as error:
        api.ingest(api.IngestRequest(path=str(outside)))
    assert error.value.status_code == 400
    assert "Only paths inside" in str(error.value.detail)


def test_ingest_returns_not_found_for_missing_path(api_service) -> None:
    with pytest.raises(HTTPException) as error:
        api.ingest(api.IngestRequest(path="missing"))

    assert error.value.status_code == 404


def test_ingest_rejects_whitespace_only_path(api_service) -> None:
    with pytest.raises(HTTPException) as error:
        api.ingest(api.IngestRequest(path="   "))

    assert error.value.status_code == 400


def test_query_rejects_whitespace_only_question(api_service) -> None:
    with pytest.raises(HTTPException) as error:
        api.query(api.QueryRequest(question="   "))

    assert error.value.status_code == 400


def test_health_does_not_initialize_models(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_get_service",
        lambda: pytest.fail("health endpoint must not initialize RagService"),
    )

    assert api.health() == {"status": "ok"}
