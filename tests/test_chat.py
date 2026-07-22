from __future__ import annotations

from dataclasses import dataclass, field

from rag_app.chat import ChatSession, normalize_command, run_interactive


@dataclass
class _Document:
    metadata: dict[str, str]


@dataclass
class _Service:
    histories: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    def ask(self, question: str, *, chat_history: str):
        self.questions.append(question)
        self.histories.append(chat_history)
        return f"Ответ на {question}", [
            _Document({"source": "contract.docx"}),
            _Document({"source": "contract.docx"}),
        ]

    def ask_stream(self, question: str, *, chat_history: str):
        self.questions.append(question)
        self.histories.append(chat_history)
        return iter(("Потоковый ", "ответ")), [_Document({"source": "report.pdf"})]

    def stats(self):
        return {"embedding_loaded": True}


def test_chat_reuses_service_and_passes_previous_turn_as_history() -> None:
    service = _Service()
    session = ChatSession(service, max_history_turns=2)

    first = session.ask("Первый вопрос")
    second = session.ask("Второй вопрос")

    assert first.sources == ("contract.docx",)
    assert service.histories[0] == "Предыдущий разговор отсутствует."
    assert "Первый вопрос" in service.histories[1]
    assert first.answer in service.histories[1]
    assert second.answer == "Ответ на Второй вопрос"


def test_streaming_answer_is_recorded_after_consumption() -> None:
    service = _Service()
    session = ChatSession(service)

    reply = session.ask_stream("Вопрос")
    assert session.stats()["turns_in_history"] == 0

    assert "".join(reply.chunks) == "Потоковый ответ"
    assert reply.sources == ("report.pdf",)
    assert session.stats()["turns_in_history"] == 1


def test_clear_removes_history_but_keeps_processed_counter() -> None:
    session = ChatSession(_Service())
    session.ask("Вопрос")

    session.clear_history()

    assert session.stats()["turns_in_history"] == 0
    assert session.stats()["questions_processed"] == 1


def test_normalize_command_removes_invisible_characters() -> None:
    assert normalize_command("\ufeff\u200b/CLEAR extra") == "/clear"


def test_interactive_multiline_paste_is_processed_as_one_question() -> None:
    service = _Service()
    pasted_traceback = (
        "Traceback (most recent call last):\n"
        '  File "/app/example.py", line 10, in <module>\n'
        "RuntimeError: example failure"
    )
    inputs = iter((pasted_traceback, "/quit"))

    run_interactive(service, read_input=lambda _message: next(inputs))

    assert service.questions == [pasted_traceback]


def test_stream_chat_prints_retrieval_and_generation_time(capsys) -> None:
    service = _Service()
    inputs = iter(("Вопрос", "/quit"))

    run_interactive(
        service,
        stream=True,
        read_input=lambda _message: next(inputs),
    )

    output = capsys.readouterr().out
    assert "[Время stream] контекст:" in output
    assert "первый фрагмент генерации:" in output
    assert "генерация:" in output
    assert "всего:" in output
