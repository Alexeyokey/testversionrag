from __future__ import annotations

from typing import Any

from rag_app.generation import TextGenerator


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "  Готовый ответ [1].  "}}]}


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
