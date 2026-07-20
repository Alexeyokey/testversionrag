from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_app.evaluation import (
    EvaluationCase,
    JUDGE_METRICS,
    OPTIONAL_JUDGE_METRICS,
    RagEvaluationSample,
    REQUIRED_JUDGE_METRICS,
    collect_rag_samples,
)

if TYPE_CHECKING:
    from rag_app.config import Settings
    from rag_app.embeddings import EmbeddingModel
    from rag_app.service import RagService


METRIC_NAMES = JUDGE_METRICS


@dataclass(frozen=True, slots=True)
class RagasEvaluationResult:
    """Ответ RAG и оценки judge-модели для одного вопроса."""

    question: str
    reference: str
    response: str
    retrieved_contexts: tuple[str, ...]
    sources: tuple[str, ...]
    scores: dict[str, float | None]
    mean_score: float | None
    passed: bool
    metric_errors: dict[str, str]
    metric_reasons: dict[str, str]
    skipped_metrics: tuple[str, ...]
    error: str | None = None


def build_ragas_scorers(
    settings: Settings,
    embedding_model: EmbeddingModel | None = None,
    *,
    include_answer_relevancy: bool = True,
) -> dict[str, Any]:
    """Создать обязательные и выбранные диагностические RAGAS-метрики."""
    try:
        from ragas.embeddings.base import BaseRagasEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerAccuracy,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
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

    # Collections-метрики RAGAS вызывают llm.agenerate(), поэтому judge должен
    # использовать AsyncOpenAI даже при синхронном запуске команды CLI.
    client = _build_vllm_async_client(settings)
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

    scorers: dict[str, Any] = {
        "faithfulness": Faithfulness(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "answer_accuracy": AnswerAccuracy(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
    }
    if include_answer_relevancy:
        if embedding_model is None:
            from rag_app.embeddings import EmbeddingModel

            embedding_model = EmbeddingModel(
                settings.embedding_model,
                batch_size=settings.embedding_batch_size,
                trust_remote_code=settings.trust_remote_code,
            )

        class ExistingEmbeddingAdapter(BaseRagasEmbedding):
            """Адаптировать embedding-модель проекта к интерфейсу RAGAS."""

            def __init__(self, model: EmbeddingModel) -> None:
                super().__init__()
                self.model = model

            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                del kwargs
                return self.model.embed_query(text)

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                import asyncio

                del kwargs
                return await asyncio.to_thread(self.model.embed_query, text)

            def embed_texts(
                self,
                texts: list[str],
                **kwargs: Any,
            ) -> list[list[float]]:
                del kwargs
                return self.model.embed_documents(texts)

            async def aembed_texts(
                self,
                texts: list[str],
                **kwargs: Any,
            ) -> list[list[float]]:
                import asyncio

                del kwargs
                return await asyncio.to_thread(self.model.embed_documents, texts)

        scorers["answer_relevancy"] = AnswerRelevancy(
            llm=judge,
            embeddings=ExistingEmbeddingAdapter(embedding_model),
        )
    return scorers


def _build_vllm_async_client(settings: Settings) -> Any:
    """Создать асинхронный OpenAI-compatible клиент для RAGAS judge."""
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise RuntimeError(
            "Для RAGAS-оценки установите зависимости проекта заново: "
            "python -m pip install -e ."
        ) from error

    return AsyncOpenAI(
        api_key=settings.vllm_api_key or "local-vllm-key",
        base_url=settings.vllm_base_url,
        timeout=settings.vllm_timeout,
        max_retries=2,
    )


def evaluate_with_ragas(
    service: RagService,
    cases: list[EvaluationCase],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    include_answer_relevancy: bool = True,
) -> list[RagasEvaluationResult]:
    """Получить ответы RAG и оценить обязательными и диагностическими метриками."""
    samples = collect_rag_samples(service, cases)
    return evaluate_samples_with_ragas(
        samples,
        settings,
        threshold=threshold,
        scorers=scorers,
        embedding_model=service.embedding_model if scorers is None else None,
        include_answer_relevancy=include_answer_relevancy,
    )


def evaluate_samples_with_ragas(
    samples: list[RagEvaluationSample],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    embedding_model: EmbeddingModel | None = None,
    include_answer_relevancy: bool = True,
) -> list[RagasEvaluationResult]:
    """Оценить заранее сохранённые ответы и контексты, не вызывая RAG повторно."""
    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    if not 0 <= resolved_threshold <= 1:
        raise ValueError("Порог RAGAS должен находиться в диапазоне от 0 до 1")
    active_scorers = scorers or build_ragas_scorers(
        settings,
        embedding_model,
        include_answer_relevancy=include_answer_relevancy,
    )
    missing_metrics = set(REQUIRED_JUDGE_METRICS) - set(active_scorers)
    if missing_metrics:
        raise ValueError(
            "Не созданы обязательные RAGAS-метрики: " + ", ".join(sorted(missing_metrics))
        )
    enabled_metric_names = tuple(
        metric_name
        for metric_name in METRIC_NAMES
        if metric_name in active_scorers
        and (metric_name != "answer_relevancy" or include_answer_relevancy)
    )
    skipped_metric_names = tuple(
        metric_name for metric_name in METRIC_NAMES if metric_name not in enabled_metric_names
    )

    results: list[RagasEvaluationResult] = []
    for sample in samples:
        if sample.error:
            results.append(_failed_result(sample, sample.error))
            continue

        metric_arguments = {
            "faithfulness": {
                "user_input": sample.question,
                "response": sample.response,
                "retrieved_contexts": list(sample.retrieved_contexts),
            },
            "context_precision": {
                "user_input": sample.question,
                "reference": sample.reference,
                "retrieved_contexts": list(sample.retrieved_contexts),
            },
            "context_recall": {
                "user_input": sample.question,
                "reference": sample.reference,
                "retrieved_contexts": list(sample.retrieved_contexts),
            },
            "answer_accuracy": {
                "user_input": sample.question,
                "response": sample.response,
                "reference": sample.reference,
            },
            "answer_relevancy": {
                "user_input": sample.question,
                "response": sample.response,
            },
        }

        scores: dict[str, float | None] = {
            metric_name: None for metric_name in METRIC_NAMES
        }
        metric_errors: dict[str, str] = {}
        metric_reasons: dict[str, str] = {}
        for metric_name in enabled_metric_names:
            try:
                metric_result = active_scorers[metric_name].score(
                    **metric_arguments[metric_name]
                )
                value = getattr(metric_result, "value", metric_result)
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(f"RAGAS вернул нечисловой балл {numeric_value!r}")
                scores[metric_name] = numeric_value
                reason = getattr(metric_result, "reason", None)
                if reason:
                    metric_reasons[metric_name] = str(reason)
            except Exception as error:
                # Ошибка одной метрики не скрывает оценки, полученные от остальных.
                metric_errors[metric_name] = f"{type(error).__name__}: {error}"

        required_scores = [
            scores[metric_name]
            for metric_name in REQUIRED_JUDGE_METRICS
            if scores[metric_name] is not None
        ]
        mean_score = (
            sum(float(score) for score in required_scores) / len(required_scores)
            if required_scores
            else None
        )
        passed = (
            not any(name in metric_errors for name in REQUIRED_JUDGE_METRICS)
            and all(
                scores[name] is not None
                and float(scores[name]) >= resolved_threshold
                for name in REQUIRED_JUDGE_METRICS
            )
        )
        results.append(
            RagasEvaluationResult(
                question=sample.question,
                reference=sample.reference,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
                sources=sample.sources,
                scores=scores,
                mean_score=mean_score,
                passed=passed,
                metric_errors=metric_errors,
                metric_reasons=metric_reasons,
                skipped_metrics=skipped_metric_names,
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
            if result.scores.get(metric_name) is not None
        ]
        metric_averages[metric_name] = (
            sum(float(value) for value in values) / len(values) if values else None
        )

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
        "required_metrics": list(REQUIRED_JUDGE_METRICS),
        "optional_metrics": list(OPTIONAL_JUDGE_METRICS),
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
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _failed_result(
    sample: EvaluationCase | RagEvaluationSample,
    error: str,
) -> RagasEvaluationResult:
    return RagasEvaluationResult(
        question=sample.question,
        reference=sample.reference,
        response="",
        retrieved_contexts=(),
        sources=(),
        scores={metric_name: None for metric_name in METRIC_NAMES},
        mean_score=None,
        passed=False,
        metric_errors={},
        metric_reasons={},
        skipped_metrics=METRIC_NAMES,
        error=error,
    )
