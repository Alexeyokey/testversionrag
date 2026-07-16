from __future__ import annotations

from typing import Any

import requests

SYSTEM_PROMPT = """Ты полезный многоязычный ассистент. Отвечай только на основании предоставленного контекста.
Если ответа нет в контексте, ответь: «Ответ отсутствует в предоставленных документах».
Отвечай на языке вопроса. Не показывай ход рассуждений. При ссылке на фрагмент указывай
его номер в формате [1], [2]. Давай краткий, законченный ответ."""

HUMAN_TEMPLATE = """\
Conversation history:
{chat_history}

Context:
{context}

Answer language:
{answer_language}

Question: {question}

/no_think
Answer directly. Do not include a thinking process."""


class TextGenerator:
    """Generate answers through a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        trust_remote_code: bool = True,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.3,
        thinking: bool = False,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        # trust_remote_code remains in the signature for backwards compatibility.
        # Model-loading options now belong to the vLLM server process.
        del trust_remote_code
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.temperature = temperature,
        self.thinking = thinking
        self.api_key = api_key
        self.timeout = timeout

    def answer(self, question: str, context: str, chat_history: str = "Предыдущий разговор отсутствует.", answer_language: str ="Русский") -> str:
        human_content = HUMAN_TEMPLATE.format(
            context=context,
            question=question,
            answer_language=answer_language,
            chat_history=chat_history,
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": human_content},
            ],
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": {
        "enable_thinking": self.thinking},
            "temperature": self.temperature}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if not response.ok:
                raise RuntimeError(
                    f"vLLM вернул HTTP {response.status_code}: {response.text}"
                )

            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"Не удалось получить ответ от vLLM по адресу {self.endpoint}: {error}"
            ) from error

        if not isinstance(content, str):
            raise RuntimeError("vLLM вернул ответ в неожиданном формате")
        return content.strip()
