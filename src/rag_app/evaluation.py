from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_app.service import RagService


# Обязательные метрики определяют passed и основной mean_score. Необязательные
# остаются диагностическими и не должны отклонять корректный ответ сами по себе.
REQUIRED_JUDGE_METRICS = (
    "faithfulness",
    "context_recall",
    "answer_accuracy",
)
OPTIONAL_JUDGE_METRICS = (
    "context_precision",
    "answer_relevancy",
)
JUDGE_METRICS = REQUIRED_JUDGE_METRICS + OPTIONAL_JUDGE_METRICS


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Один вопрос и простые условия, которым должен соответствовать результат."""

    question: str
    reference: str = ""
    answer_terms: tuple[str, ...] = ()
    context_terms: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Подробный результат одной проверки."""

    question: str
    reference: str
    answer: str
    sources: tuple[str, ...]
    answer_score: float | None
    context_score: float | None
    source_score: float | None
    missing_answer_terms: tuple[str, ...]
    missing_context_terms: tuple[str, ...]
    missing_sources: tuple[str, ...]
    passed: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RagEvaluationSample:
    """Один зафиксированный прогон RAG для последующей оценки разными инструментами."""

    question: str
    reference: str
    response: str
    retrieved_contexts: tuple[str, ...]
    sources: tuple[str, ...]
    error: str | None = None


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """Загрузить тесты из JSONL: один JSON-объект на строку."""
    testset_path = Path(path)
    cases: list[EvaluationCase] = []

    with testset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{testset_path}:{line_number}: некорректный JSON"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(
                    f"{testset_path}:{line_number}: ожидается JSON-объект"
                )

            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(
                    f"{testset_path}:{line_number}: отсутствует question"
                )

            reference = str(item.get("reference", "")).strip()
            answer_terms = _read_terms(item, "answer_terms", testset_path, line_number)
            context_terms = _read_terms(item, "context_terms", testset_path, line_number)
            expected_sources = _read_terms(
                item,
                "expected_sources",
                testset_path,
                line_number,
            )

            # Короткий reference можно использовать как единственную проверку ответа.
            # Для длинного эталона лучше явно перечислить устойчивые answer_terms.
            if not answer_terms and reference:
                answer_terms = (reference,)
            if not answer_terms and not context_terms and not expected_sources:
                raise ValueError(
                    f"{testset_path}:{line_number}: задайте reference или хотя бы "
                    "одно из полей answer_terms, context_terms, expected_sources"
                )

            cases.append(
                EvaluationCase(
                    question=question,
                    reference=reference,
                    answer_terms=answer_terms,
                    context_terms=context_terms,
                    expected_sources=expected_sources,
                )
            )

    if not cases:
        raise ValueError(f"Набор проверок пуст: {testset_path}")
    return cases


def evaluate(service: RagService, cases: list[EvaluationCase]) -> list[EvaluationResult]:
    """Задать все вопросы сервису и проверить ответ, контекст и источники."""
    results: list[EvaluationResult] = []
    for case in cases:
        try:
            answer, documents = service.ask(case.question)
            context = "\n".join(document.page_content for document in documents)
            sources = tuple(
                dict.fromkeys(
                    str(document.metadata.get("source", "unknown"))
                    for document in documents
                )
            )

            missing_answer = _missing_terms(answer, case.answer_terms)
            missing_context = _missing_terms(context, case.context_terms)
            missing_sources = _missing_sources(sources, case.expected_sources)

            results.append(
                EvaluationResult(
                    question=case.question,
                    reference=case.reference,
                    answer=answer,
                    sources=sources,
                    answer_score=_score(case.answer_terms, missing_answer),
                    context_score=_score(case.context_terms, missing_context),
                    source_score=_score(case.expected_sources, missing_sources),
                    missing_answer_terms=missing_answer,
                    missing_context_terms=missing_context,
                    missing_sources=missing_sources,
                    passed=not (missing_answer or missing_context or missing_sources),
                )
            )
        except Exception as error:
            # Ошибка одного вопроса записывается в отчёт и не останавливает остальные.
            results.append(
                EvaluationResult(
                    question=case.question,
                    reference=case.reference,
                    answer="",
                    sources=(),
                    answer_score=0.0 if case.answer_terms else None,
                    context_score=0.0 if case.context_terms else None,
                    source_score=0.0 if case.expected_sources else None,
                    missing_answer_terms=case.answer_terms,
                    missing_context_terms=case.context_terms,
                    missing_sources=case.expected_sources,
                    passed=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return results


def collect_rag_samples(
    service: RagService,
    cases: list[EvaluationCase],
) -> list[RagEvaluationSample]:
    """Один раз получить ответы и контексты, чтобы evaluator-ы сравнивали одинаковые данные."""
    samples: list[RagEvaluationSample] = []
    for case in cases:
        if not case.reference:
            samples.append(
                RagEvaluationSample(
                    question=case.question,
                    reference="",
                    response="",
                    retrieved_contexts=(),
                    sources=(),
                    error="Для LLM-оценки поле reference обязательно",
                )
            )
            continue

        try:
            response, documents = service.ask(case.question)
            samples.append(
                RagEvaluationSample(
                    question=case.question,
                    reference=case.reference,
                    response=response,
                    retrieved_contexts=tuple(
                        document.page_content for document in documents
                    ),
                    sources=tuple(
                        dict.fromkeys(
                            str(document.metadata.get("source", "unknown"))
                            for document in documents
                        )
                    ),
                )
            )
        except Exception as error:
            samples.append(
                RagEvaluationSample(
                    question=case.question,
                    reference=case.reference,
                    response="",
                    retrieved_contexts=(),
                    sources=(),
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return samples


def summarize(results: list[EvaluationResult]) -> dict[str, int | float | None]:
    """Посчитать долю пройденных тестов и средние оценки доступных проверок."""
    passed = sum(result.passed for result in results)
    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "answer_score": _average(results, "answer_score"),
        "context_score": _average(results, "context_score"),
        "source_score": _average(results, "source_score"),
    }


def write_report(
    path: str | Path,
    results: list[EvaluationResult],
) -> Path:
    """Сохранить человекочитаемый JSON с итогом и деталями каждого вопроса."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _read_terms(
    item: dict[str, Any],
    field_name: str,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    raw_value = item.get(field_name, [])
    if not isinstance(raw_value, list):
        raise ValueError(f"{path}:{line_number}: {field_name} должен быть списком")
    return tuple(str(value).strip() for value in raw_value if str(value).strip())


def _normalize(value: str) -> str:
    return " ".join(value.replace("\x00", " ").replace("\u00a0", " ").casefold().split())


def _contains(text: str, expected: str) -> bool:
    normalized_text = _normalize(text)
    normalized_expected = _normalize(expected)
    if normalized_expected in normalized_text:
        return True

    # Числа считаются одинаковыми с пробелами и без них: 8 269 600 == 8269600.
    if normalized_expected and all(
        character.isdigit() or character in " .,"
        for character in normalized_expected
    ):
        expected_digits = "".join(filter(str.isdigit, normalized_expected))
        text_digits = "".join(filter(str.isdigit, normalized_text))
        return bool(expected_digits and expected_digits in text_digits)
    return False


def _missing_terms(text: str, expected_terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in expected_terms if not _contains(text, term))


def _missing_sources(
    actual_sources: tuple[str, ...],
    expected_sources: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_actual = {_normalize(source) for source in actual_sources}
    return tuple(
        source
        for source in expected_sources
        if _normalize(source) not in normalized_actual
    )


def _score(expected: tuple[str, ...], missing: tuple[str, ...]) -> float | None:
    if not expected:
        return None
    return (len(expected) - len(missing)) / len(expected)


def _average(
    results: list[EvaluationResult],
    field_name: str,
) -> float | None:
    values = [
        float(value)
        for result in results
        if (value := getattr(result, field_name)) is not None
    ]
    return sum(values) / len(values) if values else None
