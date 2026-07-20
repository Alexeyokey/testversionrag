from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from rag_app.evaluation import RagEvaluationSample

if TYPE_CHECKING:
    from rag_app.config import Settings


METRIC_NAMES = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
)


@dataclass(frozen=True, slots=True)
class DeepEvalEvaluationResult:
    """Оценки DeepEval для одного заранее зафиксированного ответа RAG."""

    question: str
    reference: str
    response: str
    retrieved_contexts: tuple[str, ...]
    sources: tuple[str, ...]
    scores: dict[str, float]
    mean_score: float | None
    passed: bool
    metric_errors: dict[str, str]
    metric_reasons: dict[str, str]
    error: str | None = None


def build_vllm_deepeval_judge(settings: Settings) -> Any:
    """Создать DeepEvalBaseLLM, использующий JSON schema через тот же vLLM API."""
    try:
        from deepeval.models import DeepEvalBaseLLM
        from openai import AsyncOpenAI, OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Для DeepEval установите зависимости проекта заново: "
            "python -m pip install -e ."
        ) from error

    model_name = settings.ragas_judge_model or settings.generation_model
    if not model_name:
        raise ValueError(
            "Задайте RAGAS_JUDGE_MODEL или RAG_GENERATION_MODEL для judge-модели"
        )

    class VllmDeepEvalJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            api_key = settings.vllm_api_key or "local-vllm-key"
            self.model_name = model_name
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings.vllm_base_url,
                timeout=settings.vllm_timeout,
                max_retries=2,
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.vllm_base_url,
                timeout=settings.vllm_timeout,
                max_retries=2,
            )

        def load_model(self) -> OpenAI:
            return self.client

        def get_model_name(self) -> str:
            return f"vLLM:{self.model_name}"

        def _request_arguments(self, prompt: str, schema: Any | None) -> dict[str, Any]:
            arguments: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an impartial RAG evaluator. Judge only the supplied "
                            "text and follow the requested output schema exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": settings.ragas_max_tokens,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
            if schema is not None:
                schema_type = schema if isinstance(schema, type) else type(schema)
                arguments["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_type.__name__,
                        "schema": schema_type.model_json_schema(),
                        "strict": True,
                    },
                }
            return arguments

        @staticmethod
        def _parse_content(content: str, schema: Any | None) -> Any:
            if schema is None:
                return content
            schema_type = schema if isinstance(schema, type) else type(schema)
            return schema_type.model_validate_json(content)

        def generate(self, prompt: str, schema: Any | None = None) -> Any:
            completion = self.client.chat.completions.create(
                **self._request_arguments(prompt, schema)
            )
            content = completion.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("vLLM вернул пустой structured output для DeepEval")
            return self._parse_content(content, schema)

        async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
            completion = await self.async_client.chat.completions.create(
                **self._request_arguments(prompt, schema)
            )
            content = completion.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("vLLM вернул пустой structured output для DeepEval")
            return self._parse_content(content, schema)

    return VllmDeepEvalJudge()


def build_deepeval_scorers(
    settings: Settings,
    *,
    threshold: float | None = None,
    judge: Any | None = None,
) -> dict[str, Any]:
    """Создать четыре нативные RAG-метрики DeepEval с общей judge-моделью."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
    except ImportError as error:
        raise RuntimeError(
            "Для DeepEval установите зависимости проекта заново: "
            "python -m pip install -e ."
        ) from error

    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    evaluator = judge or build_vllm_deepeval_judge(settings)
    common = {
        "model": evaluator,
        "threshold": resolved_threshold,
        "include_reason": True,
        "async_mode": False,
    }
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "context_precision": ContextualPrecisionMetric(**common),
        "context_recall": ContextualRecallMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
    }


def evaluate_samples_with_deepeval(
    samples: list[RagEvaluationSample],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    test_case_factory: Callable[..., Any] | None = None,
) -> list[DeepEvalEvaluationResult]:
    """Оценить те же ответы и контексты четырьмя нативными метриками DeepEval."""
    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    if not 0 <= resolved_threshold <= 1:
        raise ValueError("Порог DeepEval должен находиться в диапазоне от 0 до 1")

    active_scorers = scorers or build_deepeval_scorers(
        settings,
        threshold=resolved_threshold,
    )
    missing_metrics = set(METRIC_NAMES) - set(active_scorers)
    if missing_metrics:
        raise ValueError(
            "Не созданы обязательные DeepEval-метрики: "
            + ", ".join(sorted(missing_metrics))
        )

    if test_case_factory is None:
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as error:
            raise RuntimeError(
                "Для DeepEval установите зависимости проекта заново: "
                "python -m pip install -e ."
            ) from error
        test_case_factory = LLMTestCase

    results: list[DeepEvalEvaluationResult] = []
    for sample in samples:
        if sample.error:
            results.append(_failed_result(sample, sample.error))
            continue

        test_case = test_case_factory(
            input=sample.question,
            actual_output=sample.response,
            expected_output=sample.reference,
            retrieval_context=list(sample.retrieved_contexts),
        )
        scores: dict[str, float] = {}
        metric_errors: dict[str, str] = {}
        metric_reasons: dict[str, str] = {}

        for metric_name in METRIC_NAMES:
            metric = active_scorers[metric_name]
            try:
                measured = metric.measure(test_case)
                value = getattr(metric, "score", measured)
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"DeepEval вернул нечисловой балл {numeric_value!r}"
                    )
                scores[metric_name] = numeric_value
                reason = getattr(metric, "reason", None)
                if reason:
                    metric_reasons[metric_name] = str(reason)
            except Exception as error:
                metric_errors[metric_name] = f"{type(error).__name__}: {error}"

        mean_score = sum(scores.values()) / len(scores) if scores else None
        passed = (
            not metric_errors
            and len(scores) == len(METRIC_NAMES)
            and all(score >= resolved_threshold for score in scores.values())
        )
        results.append(
            DeepEvalEvaluationResult(
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
            )
        )

    return results


def summarize_deepeval(
    results: list[DeepEvalEvaluationResult],
    threshold: float,
) -> dict[str, Any]:
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


def _failed_result(
    sample: RagEvaluationSample,
    error: str,
) -> DeepEvalEvaluationResult:
    return DeepEvalEvaluationResult(
        question=sample.question,
        reference=sample.reference,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
        sources=sample.sources,
        scores={},
        mean_score=None,
        passed=False,
        metric_errors={},
        metric_reasons={},
        error=error,
    )
