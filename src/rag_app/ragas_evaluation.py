from __future__ import annotations

import asyncio
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from rag_app.evaluation import (
    EvaluationCase,
    JUDGE_METRICS,
    RagEvaluationSample,
    REQUIRED_JUDGE_METRICS,
    collect_rag_samples,
    judge_outcome,
    select_judge_metrics,
    summarize_judge_results,
)
from rag_app.artifact_cache import (
    ArtifactCache,
    _to_json_value,
    ragas_artifact_config,
)

if TYPE_CHECKING:
    from rag_app.config import Settings
    from rag_app.embeddings import EmbeddingModel
    from rag_app.service import RagService


METRIC_NAMES = JUDGE_METRICS
_METRIC_RETRY_DELAYS_SECONDS = (1.0, 2.0)


class _ManagedRagasScorers(dict[str, Any]):
    """Набор метрик, владеющий общим асинхронным клиентом judge-модели."""

    def __init__(self, *args: Any, async_client: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.async_client = async_client
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.async_client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


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
    retrieval_seconds: float | None = None
    generation_seconds: float | None = None
    error: str | None = None


class _CachedRagasArtifact:
    """Cache selected structured LLM outputs, never final metric scores."""

    def __init__(
        self,
        cache: ArtifactCache,
        *,
        artifact_name: str,
        evaluator_config: dict[str, Any],
        refresh: bool,
    ) -> None:
        self._cache = cache
        self.artifact_name = artifact_name
        self._evaluator_config = evaluator_config
        self._refresh = refresh
        self.cache_hits = 0
        self.last_value: Any | None = None
        self.last_error: str | None = None
        self._session: dict[str, dict[str, Any]] | None = None

    def begin_sample(self) -> None:
        """Start one metric evaluation and group its repeated LLM calls."""
        self._session = {}
        self.last_value = None
        self.last_error = None

    def finish_sample(self, *, save: bool) -> str | None:
        """Persist generated outputs only after the metric finished successfully."""
        session = self._session
        self._session = None
        if not save or session is None:
            return None
        errors: list[str] = []
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
            except (OSError, TypeError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")
        if errors:
            self.last_error = "; ".join(errors)
        return self.last_error

    async def resolve(
        self,
        invoke: Callable[[], Awaitable[Any]],
        *,
        inputs: dict[str, Any],
        output_model: Any,
    ) -> Any:
        if self._session is not None:
            return await self._resolve_in_session(
                invoke,
                inputs=inputs,
                output_model=output_model,
            )

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
                return _validate_model(output_model, cached_value)

        value = await invoke()
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
        except (OSError, TypeError, ValueError) as error:
            self.last_error = f"{type(error).__name__}: {error}"
        return value

    async def _resolve_in_session(
        self,
        invoke: Callable[[], Awaitable[Any]],
        *,
        inputs: dict[str, Any],
        output_model: Any,
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
            if isinstance(cached_value, list):
                position = int(entry["position"])
                if position >= len(cached_value):
                    raise RuntimeError(
                        f"Артефакт {self.artifact_name} содержит недостаточно результатов"
                    )
                entry["position"] = position + 1
                return _validate_model(output_model, cached_value[position])
            entry["position"] = int(entry["position"]) + 1
            return _validate_model(output_model, cached_value)

        value = await invoke()
        serialized_value = _to_json_value(value)
        entry["generated"].append(serialized_value)
        generated = entry["generated"]
        self.last_value = generated[0] if len(generated) == 1 else list(generated)
        return value


class _ArtifactCachingLLM:
    """Intercept only one RAGAS structured-output type and delegate every other call."""

    def __init__(
        self,
        llm: Any,
        artifact: _CachedRagasArtifact,
        *,
        output_model_name: str,
    ) -> None:
        self._llm = llm
        self._artifact = artifact
        self._output_model_name = output_model_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    async def agenerate(self, *args: Any, **kwargs: Any) -> Any:
        output_model = _llm_output_model(args, kwargs)
        if getattr(output_model, "__name__", None) != self._output_model_name:
            return await self._llm.agenerate(*args, **kwargs)

        inputs = _llm_artifact_inputs(args, kwargs, output_model)
        return await self._artifact.resolve(
            lambda: self._llm.agenerate(*args, **kwargs),
            inputs=inputs,
            output_model=output_model,
        )


@dataclass(frozen=True, slots=True)
class _MetricValue:
    """Минимальный совместимый с RAGAS результат пользовательской метрики."""

    value: float
    reason: str | None = None


class _ParallelContextPrecision:
    """Параллельно получить независимый verdict для каждого retrieved context."""

    def __init__(self, scorer: Any, *, concurrency: int) -> None:
        if concurrency <= 0:
            raise ValueError("Context Precision concurrency должна быть больше нуля")
        self._scorer = scorer
        self._concurrency = concurrency

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scorer, name)

    async def ascore(
        self,
        *,
        user_input: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> _MetricValue:
        # RAGAS оценивает каждый чанк независимо. Запуск исходной метрики с одним
        # чанком сохраняет её официальный prompt и structured-output контракт.
        if not retrieved_contexts:
            result = await self._scorer.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=[],
            )
            return _MetricValue(value=float(result.value))

        semaphore = asyncio.Semaphore(self._concurrency)

        async def evaluate_context(context: str) -> int:
            async with semaphore:
                result = await self._scorer.ascore(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=[context],
                )
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError(
                    f"RAGAS вернул нечисловой verdict Context Precision {value!r}"
                )
            # Для одного чанка Average Precision равна 1 (relevant) либо 0.
            return int(value >= 0.5)

        verdicts = await asyncio.gather(
            *(evaluate_context(context) for context in retrieved_contexts)
        )
        return _MetricValue(value=self._calculate_average_precision(verdicts))

    @staticmethod
    def _calculate_average_precision(verdicts: list[int]) -> float:
        relevant_so_far = 0
        weighted_precision = 0.0
        for rank, verdict in enumerate(verdicts, start=1):
            relevant_so_far += verdict
            if verdict:
                weighted_precision += relevant_so_far / rank
        return weighted_precision / (relevant_so_far + 1e-10)


def _validate_model(model: Any, value: Any) -> Any:
    if hasattr(model, "model_validate"):
        return model.model_validate(value)
    return model.parse_obj(value)


def _llm_output_model(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any | None:
    if len(args) >= 2:
        return args[1]
    return kwargs.get("response_model") or kwargs.get("output_model")


def _llm_artifact_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_model: Any,
) -> dict[str, Any]:
    prompt = args[0] if args else kwargs.get("prompt")
    return {
        "prompt": _to_json_value(prompt),
        "output_model": (
            f"{getattr(output_model, '__module__', '')}."
            f"{getattr(output_model, '__qualname__', repr(output_model))}"
        ),
    }


def _cache_llm_artifact(
    scorer: Any,
    cache: ArtifactCache | None,
    *,
    artifact_name: str,
    output_model_name: str,
    evaluator_config: dict[str, Any],
    refresh: bool,
) -> None:
    if cache is None:
        return
    llm = getattr(scorer, "llm", None)
    if llm is None:
        raise RuntimeError(
            f"Версия RAGAS несовместима с artifact cache: "
            f"у {type(scorer).__name__} отсутствует llm"
        )
    artifact = _CachedRagasArtifact(
        cache,
        artifact_name=artifact_name,
        evaluator_config=evaluator_config,
        refresh=refresh,
    )
    scorer.llm = _ArtifactCachingLLM(
        llm,
        artifact,
        output_model_name=output_model_name,
    )
    scorer._artifact_cache_controller = artifact


def build_ragas_scorers(
    settings: Settings,
    embedding_model: EmbeddingModel | None = None,
    *,
    include_answer_relevancy: bool = True,
    include_context_precision: bool = True,
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
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        system_prompt=(
            "You are an impartial RAG evaluator. Follow the requested structured "
            "output schema exactly and judge only the supplied text."
        ),
    )

    faithfulness = Faithfulness(llm=judge)
    cache_config = ragas_artifact_config(settings)
    _cache_llm_artifact(
        faithfulness,
        artifact_cache,
        artifact_name="faithfulness_statements",
        output_model_name="StatementGeneratorOutput",
        evaluator_config=cache_config,
        refresh=refresh_artifact_cache,
    )
    scorers: dict[str, Any] = _ManagedRagasScorers(
        async_client=client,
    )
    scorers.update({
        "faithfulness": faithfulness,
        "context_recall": ContextRecall(llm=judge),
        "answer_accuracy": AnswerAccuracy(llm=judge),
    })
    if include_context_precision:
        scorers["context_precision"] = _ParallelContextPrecision(
            ContextPrecision(llm=judge),
            concurrency=settings.ragas_context_precision_concurrency,
        )
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
                del kwargs
                return await asyncio.to_thread(self.model.embed_documents, texts)

        answer_relevancy = AnswerRelevancy(
            llm=judge,
            embeddings=ExistingEmbeddingAdapter(embedding_model),
        )
        _cache_llm_artifact(
            answer_relevancy,
            artifact_cache,
            artifact_name="answer_relevancy_questions",
            output_model_name="AnswerRelevanceOutput",
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


async def _invoke_metric_scorer(
    scorer: Any,
    arguments: dict[str, Any],
) -> Any:
    """Предпочесть нативный async API RAGAS, сохранив совместимость с тестовыми scorer."""
    ascore = getattr(scorer, "ascore", None)
    if callable(ascore):
        return await ascore(**arguments)

    score = getattr(scorer, "score", None)
    if not callable(score):
        raise TypeError(f"У scorer {type(scorer).__name__} отсутствуют ascore() и score()")
    # Синхронные scorer используются тестовыми двойниками и сторонними расширениями.
    # Отдельный поток не блокирует единый event loop асинхронного judge-клиента.
    return await asyncio.to_thread(score, **arguments)


async def _score_metric_with_retries(
    scorer: Any,
    arguments: dict[str, Any],
    *,
    metric_name: str,
    artifact: _CachedRagasArtifact | None,
    progress: Callable[[str], None] | None,
    progress_prefix: str,
) -> tuple[float, str | None]:
    """Повторить только временный сетевой сбой и вернуть проверенный балл метрики."""
    delays = _METRIC_RETRY_DELAYS_SECONDS
    total_attempts = len(delays) + 1
    for attempt_index in range(total_attempts):
        try:
            metric_result = await _invoke_metric_scorer(scorer, arguments)
            value = getattr(metric_result, "value", metric_result)
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"RAGAS вернул нечисловой балл {numeric_value!r}")
            reason = getattr(metric_result, "reason", None)
            return numeric_value, str(reason) if reason else None
        except Exception as error:
            has_next_attempt = attempt_index < len(delays)
            if not has_next_attempt or not _is_transient_metric_error(error):
                raise
            # Не переносим частично сгенерированные данные неудачной попытки
            # в следующую и никогда не сохраняем незавершённый артефакт.
            if artifact is not None:
                artifact.finish_sample(save=False)
                artifact.begin_sample()
            delay = delays[attempt_index]
            if progress:
                progress(
                    f"{progress_prefix} Повтор метрики {metric_name}: "
                    f"попытка {attempt_index + 2}/{total_attempts} через {delay:g} с"
                )
            await asyncio.sleep(delay)

    raise RuntimeError("Недостижимое состояние повтора RAGAS-метрики")


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_transient_metric_error(error: BaseException) -> bool:
    transient_types = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
    }
    transient_messages = (
        "connection error",
        "connection reset",
        "server disconnected",
        "temporarily unavailable",
        "temporary failure",
        "timed out",
        "timeout",
    )
    for item in _exception_chain(error):
        if isinstance(item, (ConnectionError, TimeoutError)):
            return True
        if type(item).__name__ in transient_types:
            return True
        message = str(item).casefold()
        if any(marker in message for marker in transient_messages):
            return True
    return False


def _format_metric_error(error: BaseException) -> str:
    """Не потерять исходную причину, скрытую Instructor/OpenAI-обёртками."""
    parts: list[str] = []
    for item in _exception_chain(error):
        label = f"{type(item).__name__}: {item}"
        if label not in parts:
            parts.append(label)
    return " <- ".join(parts)


def evaluate_with_ragas(
    service: RagService,
    cases: list[EvaluationCase],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    include_answer_relevancy: bool = True,
    include_context_precision: bool = True,
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[RagasEvaluationResult]:
    """Получить ответы RAG и оценить обязательными и диагностическими метриками."""
    samples = collect_rag_samples(service, cases, progress=progress)
    return evaluate_samples_with_ragas(
        samples,
        settings,
        threshold=threshold,
        scorers=scorers,
        embedding_model=service.embedding_model if scorers is None else None,
        include_answer_relevancy=include_answer_relevancy,
        include_context_precision=include_context_precision,
        artifact_cache=artifact_cache,
        refresh_artifact_cache=refresh_artifact_cache,
        progress=progress,
    )


def evaluate_samples_with_ragas(
    samples: list[RagEvaluationSample],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    embedding_model: EmbeddingModel | None = None,
    include_answer_relevancy: bool = True,
    include_context_precision: bool = True,
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[RagasEvaluationResult]:
    """Синхронно запустить всю RAGAS-оценку внутри одного event loop."""
    return asyncio.run(
        _evaluate_samples_with_ragas_async(
            samples,
            settings,
            threshold=threshold,
            scorers=scorers,
            embedding_model=embedding_model,
            include_answer_relevancy=include_answer_relevancy,
            include_context_precision=include_context_precision,
            artifact_cache=artifact_cache,
            refresh_artifact_cache=refresh_artifact_cache,
            progress=progress,
        )
    )


async def _evaluate_samples_with_ragas_async(
    samples: list[RagEvaluationSample],
    settings: Settings,
    *,
    threshold: float | None = None,
    scorers: dict[str, Any] | None = None,
    embedding_model: EmbeddingModel | None = None,
    include_answer_relevancy: bool = True,
    include_context_precision: bool = True,
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[RagasEvaluationResult]:
    """Оценить сохранённые ответы через один общий асинхронный клиент judge."""
    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    if not 0 <= resolved_threshold <= 1:
        raise ValueError("Порог RAGAS должен находиться в диапазоне от 0 до 1")
    active_scorers = scorers or build_ragas_scorers(
        settings,
        embedding_model,
        include_answer_relevancy=include_answer_relevancy,
        include_context_precision=include_context_precision,
        artifact_cache=artifact_cache,
        refresh_artifact_cache=refresh_artifact_cache,
    )
    try:
        return await _evaluate_samples_with_active_scorers(
            samples,
            active_scorers,
            threshold=resolved_threshold,
            include_answer_relevancy=include_answer_relevancy,
            include_context_precision=include_context_precision,
            progress=progress,
        )
    finally:
        if isinstance(active_scorers, _ManagedRagasScorers):
            await active_scorers.aclose()


async def _evaluate_samples_with_active_scorers(
    samples: list[RagEvaluationSample],
    active_scorers: dict[str, Any],
    *,
    threshold: float,
    include_answer_relevancy: bool,
    include_context_precision: bool,
    progress: Callable[[str], None] | None,
) -> list[RagasEvaluationResult]:
    """Выполнить метрики; жизненным циклом клиента управляет вызывающий код."""
    missing_metrics = set(REQUIRED_JUDGE_METRICS) - set(active_scorers)
    if missing_metrics:
        raise ValueError(
            "Не созданы обязательные RAGAS-метрики: " + ", ".join(sorted(missing_metrics))
        )
    enabled_metric_names, skipped_metric_names = select_judge_metrics(
        active_scorers,
        include_answer_relevancy=include_answer_relevancy,
        include_context_precision=include_context_precision,
    )
    results: list[RagasEvaluationResult] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        if progress:
            progress(f"[RAGAS {index}/{total}] Вопрос: {sample.question}")
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
            if progress:
                progress(f"[RAGAS {index}/{total}] Метрика: {metric_name}")
            scorer = active_scorers[metric_name]
            artifact = _artifact_controller(scorer)
            cache_hits_before = artifact.cache_hits if artifact else 0
            metric_succeeded = False
            if artifact is not None:
                artifact.begin_sample()
            try:
                numeric_value, reason = await _score_metric_with_retries(
                    scorer,
                    metric_arguments[metric_name],
                    metric_name=metric_name,
                    artifact=artifact,
                    progress=progress,
                    progress_prefix=f"[RAGAS {index}/{total}]",
                )
                scores[metric_name] = numeric_value
                if reason:
                    metric_reasons[metric_name] = reason
                metric_succeeded = True
            except Exception as error:
                # Ошибка одной метрики не скрывает оценки, полученные от остальных.
                metric_errors[metric_name] = _format_metric_error(error)
            finally:
                if artifact is not None:
                    write_error = artifact.finish_sample(save=metric_succeeded)
                    if write_error:
                        metric_errors[f"{metric_name}_artifact_cache"] = write_error
                    if artifact.last_value is not None:
                        artifacts[artifact.artifact_name] = artifact.last_value
                        if artifact.cache_hits > cache_hits_before:
                            cached_artifacts.append(artifact.artifact_name)

        mean_score, passed = judge_outcome(
            scores,
            metric_errors,
            threshold=threshold,
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
                retrieval_seconds=sample.retrieval_seconds,
                generation_seconds=sample.generation_seconds,
            )
        )

    return results


def summarize_ragas(
    results: list[RagasEvaluationResult],
    threshold: float,
) -> dict[str, Any]:
    """Посчитать средний балл каждой метрики и долю пройденных примеров."""
    summary = summarize_judge_results(results, threshold=threshold)
    summary["artifact_cache_hits"] = sum(
        len(result.cached_artifacts) for result in results
    )
    return summary


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
            "max_tokens": settings.ragas_max_tokens,
            "thinking": False,
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
        retrieval_seconds=getattr(sample, "retrieval_seconds", None),
        generation_seconds=getattr(sample, "generation_seconds", None),
        error=error,
    )


def _artifact_controller(scorer: Any) -> _CachedRagasArtifact | None:
    artifact = getattr(scorer, "_artifact_cache_controller", None)
    return artifact if isinstance(artifact, _CachedRagasArtifact) else None
