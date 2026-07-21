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
from rag_app.artifact_cache import (
    ArtifactCache,
    _to_json_value,
    evaluator_artifact_config,
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
    artifacts: dict[str, Any]
    skipped_metrics: tuple[str, ...]
    cached_artifacts: tuple[str, ...]
    error: str | None = None


class _CachedRagasPrompt:
    """Кэшировать результат одного генеративного шага внутри RAGAS-метрики."""

    def __init__(
        self,
        prompt: Any,
        cache: ArtifactCache,
        *,
        artifact_name: str,
        evaluator_config: dict[str, Any],
        refresh: bool,
    ) -> None:
        self._prompt = prompt
        self._cache = cache
        self.artifact_name = artifact_name
        self._evaluator_config = evaluator_config
        self._refresh = refresh
        self.cache_hits = 0
        self.last_value: Any | None = None
        self._session: dict[str, dict[str, Any]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prompt, name)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("generate", args, kwargs)

    async def generate_multiple(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("generate_multiple", args, kwargs)

    def begin_sample(self) -> None:
        """Начать одну оценку и не путать три question-generation вызова с тремя тестами."""
        self._session = {}
        self.last_value = None

    def finish_sample(self, *, save: bool) -> None:
        """Атомарно сохранить все результаты генерации только после успешной метрики."""
        session = self._session
        self._session = None
        if not save or session is None:
            return
        for entry in session.values():
            generated = entry["generated"]
            if not generated:
                continue
            value = generated[0] if len(generated) == 1 else generated
            try:
                self._cache.put(
                    evaluator="ragas",
                    artifact_name=self.artifact_name,
                    evaluator_config=self._evaluator_config,
                    inputs=entry["inputs"],
                    value=value,
                )
            except OSError:
                # Невозможность записать оптимизационный кэш не отменяет оценивание.
                pass

    async def _call(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        inputs = _prompt_cache_inputs(method_name, args, kwargs)
        if self._session is not None:
            return await self._call_in_session(method_name, args, kwargs, inputs)

        if not self._refresh:
            cached_value = self._cache.get(
                evaluator="ragas",
                artifact_name=self.artifact_name,
                evaluator_config=self._evaluator_config,
                inputs=inputs,
            )
            if cached_value is not None:
                self.cache_hits += 1
                self.last_value = cached_value
                return self._restore(
                    cached_value,
                    multiple=_expects_multiple(method_name, kwargs),
                )

        value = await getattr(self._prompt, method_name)(*args, **kwargs)
        serialized_value = _to_json_value(value)
        self.last_value = serialized_value
        try:
            self._cache.put(
                evaluator="ragas",
                artifact_name=self.artifact_name,
                evaluator_config=self._evaluator_config,
                inputs=inputs,
                value=serialized_value,
            )
        except OSError:
            # Невозможность записать оптимизационный кэш не отменяет оценивание.
            pass
        return value

    async def _call_in_session(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        inputs: dict[str, Any],
    ) -> Any:
        assert self._session is not None
        session_key = json.dumps(
            inputs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        entry = self._session.get(session_key)
        if entry is None:
            cached_value = None
            if not self._refresh:
                cached_value = self._cache.get(
                    evaluator="ragas",
                    artifact_name=self.artifact_name,
                    evaluator_config=self._evaluator_config,
                    inputs=inputs,
                )
            entry = {
                "inputs": inputs,
                "cached": cached_value,
                "position": 0,
                "generated": [],
            }
            self._session[session_key] = entry
            if cached_value is not None:
                self.cache_hits += 1
                self.last_value = cached_value

        cached_value = entry["cached"]
        if cached_value is not None:
            if _expects_multiple(method_name, kwargs):
                return self._restore(cached_value, multiple=True)
            if isinstance(cached_value, list):
                position = int(entry["position"])
                if position >= len(cached_value):
                    raise RuntimeError(
                        f"Кэш {self.artifact_name} содержит недостаточно результатов"
                    )
                entry["position"] = position + 1
                return self._restore(cached_value[position], multiple=False)
            entry["position"] = int(entry["position"]) + 1
            return self._restore(cached_value, multiple=False)

        value = await getattr(self._prompt, method_name)(*args, **kwargs)
        serialized_value = _to_json_value(value)
        entry["generated"].append(serialized_value)
        generated = entry["generated"]
        self.last_value = generated[0] if len(generated) == 1 else list(generated)
        return value

    def _restore(self, value: Any, *, multiple: bool) -> Any:
        output_model = getattr(self._prompt, "output_model", None)
        if output_model is None:
            raise RuntimeError(
                f"RAGAS prompt {type(self._prompt).__name__} не объявляет output_model"
            )
        if multiple:
            return [_validate_model(output_model, item) for item in value]
        return _validate_model(output_model, value)


def _validate_model(model: Any, value: Any) -> Any:
    if hasattr(model, "model_validate"):
        return model.model_validate(value)
    return model.parse_obj(value)


def _expects_multiple(method_name: str, kwargs: dict[str, Any]) -> bool:
    if method_name == "generate_multiple":
        return True
    try:
        return int(kwargs.get("n", 1)) > 1
    except (TypeError, ValueError):
        return False


def _prompt_cache_inputs(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    data = kwargs.get("data")
    if data is None:
        # У PydanticPrompt первые позиционные параметры — llm и data.
        if len(args) >= 2:
            data = args[1]
        elif args:
            data = args[0]
    return {
        "method": method_name,
        "data": _to_json_value(data),
        "n": kwargs.get("n", 1),
        "temperature": kwargs.get("temperature"),
        "stop": _to_json_value(kwargs.get("stop")),
    }


def _cache_prompt(
    scorer: Any,
    attribute: str,
    cache: ArtifactCache | None,
    *,
    artifact_name: str,
    evaluator_config: dict[str, Any],
    refresh: bool,
) -> None:
    if cache is None:
        return
    prompt = getattr(scorer, attribute, None)
    if prompt is None:
        raise RuntimeError(
            f"Версия RAGAS несовместима с artifact cache: отсутствует {attribute}"
        )
    setattr(
        scorer,
        attribute,
        _CachedRagasPrompt(
            prompt,
            cache,
            artifact_name=artifact_name,
            evaluator_config=evaluator_config,
            refresh=refresh,
        ),
    )


def build_ragas_scorers(
    settings: Settings,
    embedding_model: EmbeddingModel | None = None,
    *,
    include_answer_relevancy: bool = True,
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
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

    faithfulness = Faithfulness(llm=judge)
    cache_config = evaluator_artifact_config(settings, "ragas")
    _cache_prompt(
        faithfulness,
        "statement_generator_prompt",
        artifact_cache,
        artifact_name="faithfulness_statements",
        evaluator_config=cache_config,
        refresh=refresh_artifact_cache,
    )
    scorers: dict[str, Any] = {
        "faithfulness": faithfulness,
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

        answer_relevancy = AnswerRelevancy(
            llm=judge,
            embeddings=ExistingEmbeddingAdapter(embedding_model),
        )
        _cache_prompt(
            answer_relevancy,
            "question_generation",
            artifact_cache,
            artifact_name="answer_relevancy_questions",
            evaluator_config={
                **cache_config,
                "strictness": getattr(answer_relevancy, "strictness", 3),
            },
            refresh=refresh_artifact_cache,
        )
        scorers["answer_relevancy"] = answer_relevancy
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
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
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
        artifact_cache=artifact_cache,
        refresh_artifact_cache=refresh_artifact_cache,
    )


def evaluate_samples_with_ragas(
    samples: list[RagEvaluationSample],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    embedding_model: EmbeddingModel | None = None,
    include_answer_relevancy: bool = True,
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
) -> list[RagasEvaluationResult]:
    """Оценить заранее сохранённые ответы и контексты, не вызывая RAG повторно."""
    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    if not 0 <= resolved_threshold <= 1:
        raise ValueError("Порог RAGAS должен находиться в диапазоне от 0 до 1")
    active_scorers = scorers or build_ragas_scorers(
        settings,
        embedding_model,
        include_answer_relevancy=include_answer_relevancy,
        artifact_cache=artifact_cache,
        refresh_artifact_cache=refresh_artifact_cache,
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
        artifacts: dict[str, Any] = {}
        cached_artifacts: list[str] = []
        for metric_name in enabled_metric_names:
            scorer = active_scorers[metric_name]
            artifact_prompt = _artifact_prompt(scorer)
            cache_hits_before = artifact_prompt.cache_hits if artifact_prompt else 0
            metric_succeeded = False
            if artifact_prompt is not None:
                artifact_prompt.begin_sample()
            try:
                metric_result = scorer.score(
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
                metric_succeeded = True
            except Exception as error:
                # Ошибка одной метрики не скрывает оценки, полученные от остальных.
                metric_errors[metric_name] = f"{type(error).__name__}: {error}"
            finally:
                if artifact_prompt is not None:
                    artifact_prompt.finish_sample(save=metric_succeeded)
                    if artifact_prompt.last_value is not None:
                        artifacts[artifact_prompt.artifact_name] = artifact_prompt.last_value
                        if artifact_prompt.cache_hits > cache_hits_before:
                            cached_artifacts.append(artifact_prompt.artifact_name)

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
                artifacts=artifacts,
                skipped_metrics=skipped_metric_names,
                cached_artifacts=tuple(cached_artifacts),
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
        "artifact_cache_hits": sum(
            len(result.cached_artifacts) for result in results
        ),
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
        artifacts={},
        skipped_metrics=METRIC_NAMES,
        cached_artifacts=(),
        error=error,
    )


def _artifact_prompt(scorer: Any) -> _CachedRagasPrompt | None:
    for attribute in ("statement_generator_prompt", "question_generation"):
        prompt = getattr(scorer, attribute, None)
        if isinstance(prompt, _CachedRagasPrompt):
            return prompt
    return None
