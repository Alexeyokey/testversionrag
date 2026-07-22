from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rag_app.artifact_cache import ArtifactCache
from rag_app.deepeval_evaluation import (
    DeepEvalEvaluationResult,
    build_deepeval_scorers,
    evaluate_samples_with_deepeval,
    summarize_deepeval,
)
from rag_app.evaluation import (
    JUDGE_METRICS,
    EvaluationCase,
    RagEvaluationSample,
    collect_rag_samples,
)
from rag_app.ragas_evaluation import (
    RagasEvaluationResult,
    evaluate_samples_with_ragas,
    summarize_ragas,
)

if TYPE_CHECKING:
    from rag_app.config import Settings
    from rag_app.service import RagService


METRIC_NAMES = JUDGE_METRICS
TUNING_METRICS = ("context_recall", "context_precision", "answer_accuracy")
DEFAULT_VECTOR_WEIGHTS = (0.2, 0.4, 0.5, 0.6, 0.8)


@dataclass(frozen=True, slots=True)
class RetrievalWeightConfiguration:
    name: str
    description: str
    vector_weight: float
    bm25_weight: float


def build_weight_configurations(
    vector_weights: tuple[float, ...] = DEFAULT_VECTOR_WEIGHTS,
) -> tuple[RetrievalWeightConfiguration, ...]:
    """Построить hybrid-конфигурации; BM25 получает вес, дополняющий vector до 1."""
    if len(vector_weights) < 4:
        raise ValueError("Для подбора весов требуется минимум 4 значения")
    if len(set(vector_weights)) != len(vector_weights):
        raise ValueError("Веса vector search не должны повторяться")

    configurations: list[RetrievalWeightConfiguration] = []
    for vector_weight in vector_weights:
        if not 0 < vector_weight < 1:
            raise ValueError(
                "Каждый вес vector search должен быть строго между 0 и 1: "
                "отдельные vector-only и BM25-only режимы не используются"
            )
        bm25_weight = round(1.0 - vector_weight, 10)
        vector_percent = round(vector_weight * 100)
        bm25_percent = round(bm25_weight * 100)
        configurations.append(
            RetrievalWeightConfiguration(
                name=f"hybrid_v{vector_percent}_b{bm25_percent}",
                description=(
                    f"Hybrid weighted RRF: vector {vector_weight:.2f}, "
                    f"BM25 {bm25_weight:.2f}"
                ),
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
        )
    return tuple(configurations)


DEFAULT_WEIGHT_CONFIGURATIONS = build_weight_configurations()


@dataclass(frozen=True, slots=True)
class RetrievalTuningResult:
    configuration: RetrievalWeightConfiguration
    samples: tuple[RagEvaluationSample, ...]
    ragas_results: tuple[RagasEvaluationResult, ...]
    deepeval_results: tuple[DeepEvalEvaluationResult, ...]
    ragas_summary: dict[str, Any]
    deepeval_summary: dict[str, Any]


ServiceFactory = Callable[["Settings"], "RagService"]


def run_retrieval_tuning(
    settings: Settings,
    cases: list[EvaluationCase],
    *,
    threshold: float | None = None,
    configurations: tuple[
        RetrievalWeightConfiguration, ...
    ] = DEFAULT_WEIGHT_CONFIGURATIONS,
    service_factory: ServiceFactory | None = None,
    ragas_scorers: dict[str, Any] | None = None,
    deepeval_scorers: dict[str, Any] | None = None,
    test_case_factory: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
    include_answer_relevancy: bool = True,
    include_context_precision: bool = True,
    artifact_cache: ArtifactCache | None = None,
    refresh_artifact_cache: bool = False,
) -> list[RetrievalTuningResult]:
    """Подобрать баланс vector/BM25, не меняя остальные параметры RAG."""
    if len(configurations) < 4:
        raise ValueError("Для подбора весов требуется минимум 4 конфигурации")
    if not cases:
        raise ValueError("Синтетический тестовый набор пуст")

    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    shared_service = None
    if service_factory is None:
        from rag_app.service import RagService

        shared_service = RagService(settings)

    # Production RAGAS scorers are built inside each evaluation event loop.
    # Injected test scorers remain reusable because they do not own AsyncOpenAI.
    active_ragas_scorers = ragas_scorers
    active_deepeval_scorers = deepeval_scorers or build_deepeval_scorers(
        settings,
        threshold=resolved_threshold,
        include_answer_relevancy=include_answer_relevancy,
        include_context_precision=include_context_precision,
    )

    tuning_results: list[RetrievalTuningResult] = []
    for configuration in configurations:
        question_progress: Callable[[str], None] | None = None
        if progress:
            def report_question_progress(
                message: str,
                name: str = configuration.name,
            ) -> None:
                progress(f"[{name}] {message}")

            question_progress = report_question_progress
        if progress:
            progress(f"[{configuration.name}] генерация RAG-ответов...")
        config_settings = replace(
            settings,
            vector_weight=configuration.vector_weight,
            bm25_weight=configuration.bm25_weight,
        )
        if shared_service is not None:
            shared_service.set_retrieval_weights(
                vector_weight=configuration.vector_weight,
                bm25_weight=configuration.bm25_weight,
            )
            service = shared_service
        else:
            assert service_factory is not None
            service = service_factory(config_settings)
        samples = collect_rag_samples(
            service,
            cases,
            progress=question_progress,
        )
        if progress:
            progress(f"[{configuration.name}] оценка RAGAS...")
        ragas_results = evaluate_samples_with_ragas(
            samples,
            settings,
            threshold=resolved_threshold,
            scorers=active_ragas_scorers,
            embedding_model=getattr(service, "embedding_model", None),
            include_answer_relevancy=include_answer_relevancy,
            include_context_precision=include_context_precision,
            artifact_cache=artifact_cache,
            refresh_artifact_cache=refresh_artifact_cache,
            progress=question_progress,
        )
        if progress:
            progress(f"[{configuration.name}] оценка DeepEval...")
        deepeval_results = evaluate_samples_with_deepeval(
            samples,
            settings,
            threshold=resolved_threshold,
            scorers=active_deepeval_scorers,
            test_case_factory=test_case_factory,
            include_answer_relevancy=include_answer_relevancy,
            include_context_precision=include_context_precision,
            progress=question_progress,
        )
        tuning_results.append(
            RetrievalTuningResult(
                configuration=configuration,
                samples=tuple(samples),
                ragas_results=tuple(ragas_results),
                deepeval_results=tuple(deepeval_results),
                ragas_summary=summarize_ragas(ragas_results, resolved_threshold),
                deepeval_summary=summarize_deepeval(
                    deepeval_results,
                    resolved_threshold,
                ),
            )
        )
        if progress:
            progress(f"[{configuration.name}] готово")

    return tuning_results


def comparison_rows(
    results: list[RetrievalTuningResult],
) -> list[dict[str, Any]]:
    """Преобразовать результаты в строки «веса hybrid retrieval × метрики»."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "configuration": result.configuration.name,
            "description": result.configuration.description,
            "vector_weight": result.configuration.vector_weight,
            "bm25_weight": result.configuration.bm25_weight,
        }
        retrieval_values = [
            sample.retrieval_seconds
            for sample in result.samples
            if sample.retrieval_seconds is not None
        ]
        generation_values = [
            sample.generation_seconds
            for sample in result.samples
            if sample.generation_seconds is not None
        ]
        row["retrieval_mean_seconds"] = (
            sum(retrieval_values) / len(retrieval_values)
            if retrieval_values
            else None
        )
        row["generation_mean_seconds"] = (
            sum(generation_values) / len(generation_values)
            if generation_values
            else None
        )
        for tool_name, summary in (
            ("ragas", result.ragas_summary),
            ("deepeval", result.deepeval_summary),
        ):
            for metric_name in METRIC_NAMES:
                row[f"{tool_name}_{metric_name}"] = summary["metrics"].get(
                    metric_name
                )
            row[f"{tool_name}_mean"] = summary.get("mean_score")

        tuning_scores = [
            float(row[f"{tool_name}_{metric_name}"])
            for tool_name in ("ragas", "deepeval")
            for metric_name in TUNING_METRICS
            if row[f"{tool_name}_{metric_name}"] is not None
        ]
        row["tuning_score"] = (
            sum(tuning_scores) / len(tuning_scores) if tuning_scores else None
        )
        rows.append(row)
    return rows


def write_retrieval_tuning_reports(
    output_directory: str | Path,
    results: list[RetrievalTuningResult],
    settings: Settings,
) -> dict[str, Path]:
    """Сохранить подробный JSON, плоский CSV и человекочитаемый Markdown."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(results)

    json_path = output_dir / "retrieval-tuning-details.json"
    json_path.write_text(
        json.dumps(
            {
                "judge": {
                    "model": settings.ragas_judge_model or settings.generation_model,
                    "base_url": settings.vllm_base_url,
                },
                "comparison": rows,
                "configurations": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "retrieval-weights.csv"
    fieldnames = [
        "configuration",
        "description",
        "vector_weight",
        "bm25_weight",
        "retrieval_mean_seconds",
        "generation_mean_seconds",
    ] + [
        f"{tool_name}_{metric_name}"
        for tool_name in ("ragas", "deepeval")
        for metric_name in METRIC_NAMES
    ] + ["ragas_mean", "deepeval_mean", "tuning_score"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "retrieval-tuning-report.md"
    markdown_path.write_text(_markdown_report(rows), encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def _markdown_report(rows: list[dict[str, Any]]) -> str:
    metric_titles = {
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "answer_accuracy": "Answer Accuracy",
        "answer_relevancy": "Answer Relevancy",
    }
    headers = [
        "Конфигурация",
        "Vector",
        "BM25",
        "Retrieval, с",
        "Generation, с",
    ] + [
        f"{tool} {metric_titles[metric]}"
        for tool in ("RAGAS", "DeepEval")
        for metric in METRIC_NAMES
    ] + ["Tuning score"]

    lines = [
        "# Подбор весов гибридного поиска",
        "",
        "Во всех конфигурациях используются одновременно vector search и BM25. "
        "Reranker, top-k и остальные параметры остаются неизменными; меняются только "
        "веса источников в weighted RRF.",
        "",
        "## Метрики",
        "",
        (
            "- **Faithfulness** — какая доля утверждений ответа подтверждается "
            "найденным контекстом; важна для контроля галлюцинаций генератора."
        ),
        (
            "- **Context Precision** — насколько высоко ранжируются полезные чанки; "
            "важна при настройке top-k и reranker."
        ),
        (
            "- **Context Recall** — хватает ли найденного контекста для полного "
            "эталонного ответа; важна для диагностики пропусков retrieval."
        ),
        (
            "- **Answer Accuracy** — совпадает ли ответ с эталоном по фактам и "
            "полноте; обязательна для прохождения проверки."
        ),
        (
            "- **Context Precision** и **Answer Relevancy** — диагностические "
            "метрики: отображаются в отчёте, но не определяют passed и основное "
            "среднее."
        ),
        (
            "- **Tuning score** — среднее Context Recall, Context Precision и "
            "Answer Accuracy из доступных оценок RAGAS и DeepEval; используется "
            "только для выбора весов retrieval."
        ),
        "",
        "## Сводная таблица",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        values = [
            row["configuration"],
            f"{float(row['vector_weight']):.2f}",
            f"{float(row['bm25_weight']):.2f}",
            _format_seconds(row["retrieval_mean_seconds"]),
            _format_seconds(row["generation_mean_seconds"]),
        ]
        values.extend(
            _format_score(row[f"{tool}_{metric}"])
            for tool in ("ragas", "deepeval")
            for metric in METRIC_NAMES
        )
        values.append(_format_score(row["tuning_score"]))
        lines.append("| " + " | ".join(values) + " |")

    valid_rows = [row for row in rows if row["tuning_score"] is not None]
    lines.extend(["", "## Выводы", ""])
    if valid_rows:
        best = max(valid_rows, key=lambda row: float(row["tuning_score"]))
        lines.append(
            f"Лучший баланс — **vector {float(best['vector_weight']):.2f} / "
            f"BM25 {float(best['bm25_weight']):.2f}** "
            f"(tuning score {float(best['tuning_score']):.3f})."
        )
    else:
        lines.append(
            "Победитель не определён: judge-метрики не вернули числовых результатов. "
            "Проверьте ошибки в retrieval-tuning-details.json."
        )
    lines.extend(
        [
            "",
            (
                "- **RAGAS** удобнее для исследовательских экспериментов, работы с "
                "датасетами и компактного расчёта классических RAG-метрик."
            ),
            (
                "- **DeepEval** удобнее для pytest/CI, пороговых regression-тестов "
                "и диагностики: его нативные метрики сохраняют текстовые причины "
                "judge-модели."
            ),
            (
                "- После выбора результата перенесите его vector/BM25 weights в "
                "RAG_VECTOR_WEIGHT и RAG_BM25_WEIGHT, затем перепроверьте на "
                "отложенном наборе вопросов."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _format_seconds(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"
