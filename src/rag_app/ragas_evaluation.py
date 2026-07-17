from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_app.evaluation import EvaluationCase

if TYPE_CHECKING:
    from rag_app.config import Settings
    from rag_app.service import RagService


METRIC_NAMES = (
    "faithfulness",
    "factual_correctness",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True, slots=True)
class RagasEvaluationResult:
    """Ответ RAG и оценки judge-модели для одного вопроса."""

    question: str
    reference: str
    response: str
    retrieved_contexts: tuple[str, ...]
    sources: tuple[str, ...]
    scores: dict[str, float]
    mean_score: float | None
    passed: bool
    metric_errors: dict[str, str]
    error: str | None = None


def build_ragas_scorers(settings: Settings) -> dict[str, Any]:
    """Создать четыре RAGAS-метрики поверх существующего vLLM OpenAI API."""
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )
    except ImportError as error:
        raise RuntimeError(
            "Для RAGAS-оценки установите зависимости проекта заново: "
            "python -m pip install -e ."
        ) from error

    judge_model = settings.ragas_judge_model or settings.generation_model
    if not judge_model:
        raise ValueError(
            "Задайте RAGAS_JUDGE_MODEL или RAG_GENERATION_MODEL для judge-модели"
        )

    # vLLM реализует OpenAI-compatible API, поэтому RAGAS использует обычный
    # OpenAI-клиент с тем же URL, что и генеративная модель приложения.
    client = OpenAI(
        api_key=settings.vllm_api_key or "local-vllm-key",
        base_url=settings.vllm_base_url,
        timeout=settings.vllm_timeout,
        max_retries=2,
    )
    judge = llm_factory(
        judge_model,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=settings.ragas_max_tokens,
        system_prompt=(
            "You are an impartial RAG evaluator. Follow the requested structured "
            "output schema exactly and judge only the supplied text."
        ),
    )

    return {
        "faithfulness": Faithfulness(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


def evaluate_with_ragas(
    service: RagService,
    cases: list[EvaluationCase],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
) -> list[RagasEvaluationResult]:
    """Получить ответы RAG и последовательно оценить их четырьмя метриками."""
    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    if not 0 <= resolved_threshold <= 1:
        raise ValueError("Порог RAGAS должен находиться в диапазоне от 0 до 1")
    active_scorers = scorers or build_ragas_scorers(settings)
    missing_metrics = set(METRIC_NAMES) - set(active_scorers)
    if missing_metrics:
        raise ValueError(
            "Не созданы обязательные RAGAS-метрики: " + ", ".join(sorted(missing_metrics))
        )

    results: list[RagasEvaluationResult] = []
    for case in cases:
        if not case.reference:
            results.append(
                _failed_result(
                    case,
                    "Для RAGAS-оценки поле reference обязательно",
                )
            )
            continue

        try:
            response, documents = service.ask(case.question)
        except Exception as error:
            results.append(
                _failed_result(case, f"{type(error).__name__}: {error}")
            )
            continue

        contexts = tuple(document.page_content for document in documents)
        sources = tuple(
            dict.fromkeys(
                str(document.metadata.get("source", "unknown"))
                for document in documents
            )
        )
        metric_arguments = {
            "faithfulness": {
                "user_input": case.question,
                "response": response,
                "retrieved_contexts": list(contexts),
            },
            "factual_correctness": {
                "response": response,
                "reference": case.reference,
            },
            "context_precision": {
                "user_input": case.question,
                "reference": case.reference,
                "retrieved_contexts": list(contexts),
            },
            "context_recall": {
                "user_input": case.question,
                "reference": case.reference,
                "retrieved_contexts": list(contexts),
            },
        }

        scores: dict[str, float] = {}
        metric_errors: dict[str, str] = {}
        for metric_name in METRIC_NAMES:
            try:
                metric_result = active_scorers[metric_name].score(
                    **metric_arguments[metric_name]
                )
                value = getattr(metric_result, "value", metric_result)
                scores[metric_name] = float(value)
            except Exception as error:
                # Ошибка одной метрики не скрывает оценки, полученные от остальных.
                metric_errors[metric_name] = f"{type(error).__name__}: {error}"

        mean_score = sum(scores.values()) / len(scores) if scores else None
        passed = (
            not metric_errors
            and len(scores) == len(METRIC_NAMES)
            and all(score >= resolved_threshold for score in scores.values())
        )
        results.append(
            RagasEvaluationResult(
                question=case.question,
                reference=case.reference,
                response=response,
                retrieved_contexts=contexts,
                sources=sources,
                scores=scores,
                mean_score=mean_score,
                passed=passed,
                metric_errors=metric_errors,
            )
        )

    return results


def summarize_ragas(
    results: list[RagasEvaluationResult],
    threshold: float,
) -> dict[str, Any]:
    """Посчитать средний балл каждой метрики и долю пройденных примеров."""
    passed = sum(result.passed for result in results)
    metric_averages: dict[str, float | None] = {}
    for metric_name in METRIC_NAMES:
        values = [
            result.scores[metric_name]
            for result in results
            if metric_name in result.scores
        ]
        metric_averages[metric_name] = sum(values) / len(values) if values else None

    mean_values = [
        result.mean_score
        for result in results
        if result.mean_score is not None
    ]
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "threshold": threshold,
        "mean_score": sum(mean_values) / len(mean_values) if mean_values else None,
        "metrics": metric_averages,
    }


def write_ragas_report(
    path: str | Path,
    results: list[RagasEvaluationResult],
    settings: Settings,
    threshold: float,
) -> Path:
    """Сохранить настройки judge, сводку и результаты в JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge": {
            "model": settings.ragas_judge_model or settings.generation_model,
            "base_url": settings.vllm_base_url,
        },
        "summary": summarize_ragas(results, threshold),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _failed_result(case: EvaluationCase, error: str) -> RagasEvaluationResult:
    return RagasEvaluationResult(
        question=case.question,
        reference=case.reference,
        response="",
        retrieved_contexts=(),
        sources=(),
        scores={},
        mean_score=None,
        passed=False,
        metric_errors={},
        error=error,
    )
