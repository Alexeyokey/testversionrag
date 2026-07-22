from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from rag_app.service import RagService


@dataclass(frozen=True)
class ChatReply:
    answer: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class StreamingChatReply:
    chunks: Iterator[str]
    sources: tuple[str, ...]


class ChatSession:
    """Keep conversation history around one long-lived RagService instance."""

    def __init__(self, service: RagService, *, max_history_turns: int = 6) -> None:
        if max_history_turns < 0:
            raise ValueError("--history-turns не может быть меньше нуля")
        self.service = service
        self.max_history_turns = max_history_turns
        self._history: list[tuple[str, str]] = []
        self._question_count = 0

    def _history_text(self) -> str:
        if not self._history or self.max_history_turns == 0:
            return "Предыдущий разговор отсутствует."
        turns = self._history[-self.max_history_turns :]
        return "\n\n".join(
            f"Пользователь: {question}\nАссистент: {answer}"
            for question, answer in turns
        )

    @staticmethod
    def _sources(documents: list[Document]) -> tuple[str, ...]:
        # dict preserves order and removes duplicate source names.
        return tuple(
            dict.fromkeys(
                str(document.metadata.get("source", "unknown"))
                for document in documents
            )
        )

    def _record(self, question: str, answer: str) -> None:
        self._question_count += 1
        if self.max_history_turns == 0:
            return
        self._history.append((question, answer))
        # Оставляем последние N реплик, иначе prompt и prefill будут постоянно расти.
        del self._history[: -self.max_history_turns]

    def ask(self, question: str) -> ChatReply:
        answer, documents = self.service.ask(
            question,
            chat_history=self._history_text(),
        )
        self._record(question, answer)
        return ChatReply(answer=answer, sources=self._sources(documents))

    def ask_stream(self, question: str) -> StreamingChatReply:
        chunks, documents = self.service.ask_stream(
            question,
            chat_history=self._history_text(),
        )

        def record_answer() -> Iterator[str]:
            # Полный ответ нужен для истории, хотя пользователю он уходит по частям.
            answer_parts: list[str] = []
            completed = False
            try:
                for chunk in chunks:
                    answer_parts.append(chunk)
                    yield chunk
                completed = True
            finally:
                # Do not add a partial answer to history when streaming failed.
                if completed:
                    self._record(question, "".join(answer_parts).strip())

        return StreamingChatReply(
            chunks=record_answer(),
            sources=self._sources(documents),
        )

    def clear_history(self) -> None:
        self._history.clear()

    def stats(self) -> dict[str, object]:
        return {
            "questions_processed": self._question_count,
            "turns_in_history": len(self._history),
            "max_history_turns_in_prompt": self.max_history_turns,
            "service": self.service.stats(),
        }


def normalize_command(value: str) -> str:
    cleaned = (
        str(value)
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .strip()
        .lower()
    )
    if cleaned.startswith("/"):
        return cleaned.split(maxsplit=1)[0]
    return cleaned


def run_interactive(
    service: RagService,
    *,
    stream: bool = False,
    max_history_turns: int = 6,
    read_input: Callable[[str], str] | None = None,
) -> None:
    session = ChatSession(service, max_history_turns=max_history_turns)
    if read_input is None:
        # PromptSession enables bracketed paste. A pasted traceback containing
        # newlines stays in one editable buffer instead of becoming a queue of
        # independent input() calls.
        from prompt_toolkit import PromptSession

        prompt_session: PromptSession[str] = PromptSession(multiline=False)
        read_input = prompt_session.prompt

    print(
        "\n================================================\n"
        "  Интерактивный RAG-чат\n"
        "================================================\n"
        "Команды: /stats  /clear  /quit\n"
    )

    while True:
        try:
            question = read_input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nЧат завершён.")
            break

        if not question:
            continue

        command = normalize_command(question)
        if command in {"/quit", "/exit", "quit", "exit"}:
            print("\nЧат завершён.")
            break
        if command == "/clear":
            session.clear_history()
            print("\nКонтекст диалога очищен. Загруженные модели сохранены.\n")
            continue
        if command == "/stats":
            print(json.dumps(session.stats(), ensure_ascii=False, indent=2))
            continue

        print("\nАссистент: ", end="", flush=True)
        try:
            if stream:
                # Первый фрагмент включает поиск и prefill — эту паузу видит
                # пользователь.
                stream_started_at = perf_counter()
                first_chunk_seconds: float | None = None
                reply = session.ask_stream(question)
                for chunk in reply.chunks:
                    if first_chunk_seconds is None:
                        first_chunk_seconds = perf_counter() - stream_started_at
                    print(chunk, end="", flush=True)
                total_seconds = perf_counter() - stream_started_at
                print()
                first_chunk_label = (
                    f"{first_chunk_seconds:.3f} с"
                    if first_chunk_seconds is not None
                    else "нет данных"
                )
                print(
                    "[Время stream] первый фрагмент: "
                    f"{first_chunk_label}; полный ответ: {total_seconds:.3f} с"
                )
            else:
                reply = session.ask(question)
                print(reply.answer)
        except Exception as error:
            print(f"Не удалось обработать запрос: {error}")
            print("Можно очистить историю командой /clear и повторить вопрос.")
            continue

        if reply.sources:
            print(f"\n[Источники] {', '.join(reply.sources)}")
        print()
