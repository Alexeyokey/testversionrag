from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rag_app.deepeval_evaluation import (
    DeepEvalEvaluationResult,
    build_deepeval_scorers,
    evaluate_samples_with_deepeval,
    summarize_deepeval,
)
from rag_app.evaluation import (
    JUDGE_METRICS,
    REQUIRED_JUDGE_METRICS,
    EvaluationCase,
    RagEvaluationSample,
    collect_rag_samples,
)
from rag_app.ragas_evaluation import (
    RagasEvaluationResult,
    build_ragas_scorers,
    evaluate_samples_with_ragas,
    summarize_ragas,
)

if TYPE_CHECKING:
    from rag_app.config import Settings
    from rag_app.service import RagService


METRIC_NAMES = JUDGE_METRICS


@dataclass(frozen=True, slots=True)
class RagConfiguration:
    name: str
    description: str
    vector_weight: float
    bm25_weight: float
    enable_reranker: bool
    top_k: int = 3
    candidate_k: int = 8


DEFAULT_CONFIGURATIONS = (
    RagConfiguration(
        name="vector_only",
        description="Только dense vector search",
        vector_weight=1.0,
        bm25_weight=0.0,
        enable_reranker=False,
    ),
    RagConfiguration(
        name="bm25_only",
        description="Только лексический BM25",
        vector_weight=0.0,
        bm25_weight=1.0,
        enable_reranker=False,
    ),
    RagConfiguration(
        name="hybrid",
        description="Vector + BM25 через weighted RRF",
        vector_weight=0.6,
        bm25_weight=0.4,
        enable_reranker=False,
    ),
    RagConfiguration(
        name="hybrid_reranker",
        description="Vector + BM25 + CrossEncoder reranker",
        vector_weight=0.6,
        bm25_weight=0.4,
        enable_reranker=True,
    ),
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfigurationResult:
    configuration: RagConfiguration
    samples: tuple[RagEvaluationSample, ...]
    ragas_results: tuple[RagasEvaluationResult, ...]
    deepeval_results: tuple[DeepEvalEvaluationResult, ...]
    ragas_summary: dict[str, Any]
    deepeval_summary: dict[str, Any]


ServiceFactory = Callable[["Settings"], "RagService"]


def run_benchmark(
    settings: Settings,
    cases: list[EvaluationCase],
    *,
    threshold: float | None = None,
    configurations: tuple[RagConfiguration, ...] = DEFAULT_CONFIGURATIONS,
    service_factory: ServiceFactory | None = None,
    ragas_scorers: dict[str, Any] | None = None,
    deepeval_scorers: dict[str, Any] | None = None,
    test_case_factory: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
    include_answer_relevancy: bool = True,
) -> list[BenchmarkConfigurationResult]:
    """Прогнать четыре RAG-конфигурации и оценить одни ответы двумя инструментами."""
    if len(configurations) < 4:
        raise ValueError("Для сравнительного эксперимента требуется минимум 4 конфигурации")
    if not cases:
        raise ValueError("Синтетический тестовый набор пуст")

    resolved_threshold = settings.ragas_threshold if threshold is None else threshold
    shared_embedding_model = None
    if service_factory is None:
        from rag_app.embeddings import EmbeddingModel
        from rag_app.service import RagService

        shared_embedding_model = EmbeddingModel(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            trust_remote_code=settings.trust_remote_code,
        )

        def service_factory(config_settings: Settings) -> RagService:
            return RagService(
                config_settings,
                embedding_model=shared_embedding_model,
            )

    active_ragas_scorers = ragas_scorers or build_ragas_scorers(
        settings,
        shared_embedding_model,
        include_answer_relevancy=include_answer_relevancy,
    )
    active_deepeval_scorers = deepeval_scorers or build_deepeval_scorers(
        settings,
        threshold=resolved_threshold,
        include_answer_relevancy=include_answer_relevancy,
    )

    benchmark_results: list[BenchmarkConfigurationResult] = []
    for configuration in configurations:
        if progress:
            progress(f"[{configuration.name}] генерация RAG-ответов...")
        config_settings = replace(
            settings,
            vector_weight=configuration.vector_weight,
            bm25_weight=configuration.bm25_weight,
            enable_reranker=configuration.enable_reranker,
            top_k=configuration.top_k,
            candidate_k=configuration.candidate_k,
        )
        service = service_factory(config_settings)
        samples = collect_rag_samples(service, cases)
        if progress:
            progress(f"[{configuration.name}] оценка RAGAS...")
        ragas_results = evaluate_samples_with_ragas(
            samples,
            settings,
            threshold=resolved_threshold,
            scorers=active_ragas_scorers,
            include_answer_relevancy=include_answer_relevancy,
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
        )
        benchmark_results.append(
            BenchmarkConfigurationResult(
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

    return benchmark_results


def comparison_rows(
    results: list[BenchmarkConfigurationResult],
) -> list[dict[str, Any]]:
    """Преобразовать результаты в плоские строки «конфигурации × метрики»."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "configuration": result.configuration.name,
            "description": result.configuration.description,
        }
        for tool_name, summary in (
            ("ragas", result.ragas_summary),
            ("deepeval", result.deepeval_summary),
        ):
            for metric_name in METRIC_NAMES:
                row[f"{tool_name}_{metric_name}"] = summary["metrics"].get(
                    metric_name
                )
            row[f"{tool_name}_mean"] = summary.get("mean_score")

        available_scores = [
            float(row[f"{tool_name}_{metric_name}"])
            for tool_name in ("ragas", "deepeval")
            for metric_name in REQUIRED_JUDGE_METRICS
            if row[f"{tool_name}_{metric_name}"] is not None
        ]
        row["combined_mean"] = (
            sum(available_scores) / len(available_scores)
            if available_scores
            else None
        )
        rows.append(row)
    return rows


def write_benchmark_reports(
    output_directory: str | Path,
    results: list[BenchmarkConfigurationResult],
    settings: Settings,
) -> dict[str, Path]:
    """Сохранить подробный JSON, плоский CSV и человекочитаемый Markdown."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(results)

    json_path = output_dir / "benchmark-details.json"
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

    csv_path = output_dir / "benchmark-comparison.csv"
    fieldnames = ["configuration", "description"] + [
        f"{tool_name}_{metric_name}"
        for tool_name in ("ragas", "deepeval")
        for metric_name in METRIC_NAMES
    ] + ["ragas_mean", "deepeval_mean", "combined_mean"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "benchmark-report.md"
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
    headers = ["Конфигурация"] + [
        f"{tool} {metric_titles[metric]}"
        for tool in ("RAGAS", "DeepEval")
        for metric in METRIC_NAMES
    ] + ["Среднее"]

    lines = [
        "# Сравнение RAGAS и DeepEval",
        "",
        "Обе библиотеки оценивали одинаковые ответы и одинаковый retrieval context. "
        "Абсолютные значения между библиотеками не тождественны: сравнивать следует "
        "общий тренд и ранжирование конфигураций.",
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
        "",
        "## Сводная таблица",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        values = [row["configuration"]]
        values.extend(
            _format_score(row[f"{tool}_{metric}"])
            for tool in ("ragas", "deepeval")
            for metric in METRIC_NAMES
        )
        values.append(_format_score(row["combined_mean"]))
        lines.append("| " + " | ".join(values) + " |")

    valid_rows = [row for row in rows if row["combined_mean"] is not None]
    lines.extend(["", "## Выводы", ""])
    if valid_rows:
        best = max(valid_rows, key=lambda row: float(row["combined_mean"]))
        lines.append(
            f"Лучшая конфигурация по среднему обязательных оценок — "
            f"**{best['configuration']}** ({float(best['combined_mean']):.3f})."
        )
    else:
        lines.append(
            "Победитель не определён: judge-метрики не вернули числовых результатов. "
            "Проверьте ошибки в benchmark-details.json."
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
                "- Для выбора RAG-конфигурации ориентируйтесь на согласованный рост "
                "метрик обеих библиотек, а расхождения проверяйте по причинам и "
                "подробным результатам каждого вопроса."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
