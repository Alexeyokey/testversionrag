from __future__ import annotations

from typing import Any

import pytest

from rag_app.generation import TextGenerator, check_vllm_server


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "  Готовый ответ [1].  "}}]}


class _StreamResponse:
    closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, *, decode_unicode: bool):
        assert decode_unicode is True
        yield 'data: {"choices":[{"delta":{"content":"Потоковый "}}]}'
        yield 'data: {"choices":[{"delta":{"content":"ответ"}}]}'
        yield "data: [DONE]"

    def close(self) -> None:
        self.closed = True


class _JsonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_generator_uses_vllm_chat_api(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _Response()

    monkeypatch.setattr("rag_app.generation.requests.post", fake_post)
    generator = TextGenerator(
        "example/model",
        max_new_tokens=42,
        base_url="http://vllm:8000/v1/",
        api_key="secret",
        timeout=7,
    )

    answer = generator.answer("Вопрос?", "[1] Контекст")

    assert answer == "Готовый ответ [1]."
    assert captured["url"] == "http://vllm:8000/v1/chat/completions"
    assert captured["json"]["model"] == "example/model"
    assert captured["json"]["max_tokens"] == 42
    assert captured["json"]["temperature"] == 0
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 7


def test_generator_streams_vllm_chat_api(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = _StreamResponse()

    def fake_post(url, *, json, headers, timeout, stream):
        captured.update(
            url=url,
            json=json,
            headers=headers,
            timeout=timeout,
            stream=stream,
        )
        return response

    monkeypatch.setattr("rag_app.generation.requests.post", fake_post)
    generator = TextGenerator("example/model")

    answer = "".join(generator.stream_answer("Вопрос?", "[1] Контекст"))

    assert answer == "Потоковый ответ"
    assert captured["json"]["stream"] is True
    assert captured["stream"] is True
    assert response.closed is True


def test_check_vllm_server_checks_version_model_and_generation(monkeypatch) -> None:
    requested_urls: list[str] = []
    captured_post: dict[str, Any] = {}

    def fake_get(url, *, headers, timeout):
        requested_urls.append(url)
        assert headers["Authorization"] == "Bearer secret"
        assert timeout == 9
        if url.endswith("/version"):
            return _JsonResponse({"version": "0.19.0"})
        return _JsonResponse({"data": [{"id": "QuantTrio/Qwen3.6-27B-AWQ"}]})

    def fake_post(url, *, json, headers, timeout):
        captured_post.update(url=url, json=json, headers=headers, timeout=timeout)
        return _JsonResponse({"choices": [{"message": {"content": "ГОТОВО"}}]})

    monkeypatch.setattr("rag_app.generation.requests.get", fake_get)
    monkeypatch.setattr("rag_app.generation.requests.post", fake_post)

    result = check_vllm_server(
        "QuantTrio/Qwen3.6-27B-AWQ",
        base_url="http://vllm:8000/v1/",
        api_key="secret",
        timeout=9,
    )

    assert requested_urls == [
        "http://vllm:8000/version",
        "http://vllm:8000/v1/models",
    ]
    assert captured_post["url"] == "http://vllm:8000/v1/chat/completions"
    assert captured_post["json"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert result["vllm_version"] == "0.19.0"
    assert result["response"] == "ГОТОВО"


def test_check_vllm_server_rejects_another_served_model(monkeypatch) -> None:
    def fake_get(url, *, headers, timeout):
        del headers, timeout
        if url.endswith("/version"):
            return _JsonResponse({"version": "0.19.0"})
        return _JsonResponse({"data": [{"id": "another/model"}]})

    monkeypatch.setattr("rag_app.generation.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="не обслуживает модель"):
        check_vllm_server("QuantTrio/Qwen3.6-27B-AWQ")
