from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from rag_app.config import Settings


def _add_artifact_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-cache-dir",
        dest="artifact_cache_dir",
        type=Path,
        default=None,
        help="Каталог кэша фактов Faithfulness и вопросов Answer Relevancy",
    )
    parser.add_argument(
        "--no-artifact-cache",
        dest="no_artifact_cache",
        action="store_true",
        help="Не читать и не сохранять промежуточную разметку RAGAS",
    )
    parser.add_argument(
        "--refresh-artifact-cache",
        dest="refresh_artifact_cache",
        action="store_true",
        help="Заново сгенерировать факты и вопросы, заменив совпавшие записи",
    )


def _artifact_cache_from_args(args, settings: Settings):
    if args.no_artifact_cache or not settings.evaluation_artifact_cache_enabled:
        return None
    from rag_app.artifact_cache import ArtifactCache

    directory = args.artifact_cache_dir or Path(settings.evaluation_artifact_cache_dir)
    return ArtifactCache(directory)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Гибридный RAG: Qdrant + BM25 + reranker")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check-vllm",
        help="Проверить версию, модель и тестовую генерацию vLLM",
    )
    check_parser.add_argument(
        "--prompt",
        default="Ответь одним словом: ГОТОВО.",
        help="Короткий запрос для smoke-теста генерации",
    )

    index_parser = subparsers.add_parser("index", help="Проиндексировать файл или папку")
    index_parser.add_argument("source")
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Удалить существующую коллекцию перед индексацией",
    )

    search_parser = subparsers.add_parser("search", help="Найти релевантные фрагменты")
    search_parser.add_argument("query")
    search_parser.add_argument("--json", action="store_true", dest="as_json")

    ask_parser = subparsers.add_parser("ask", help="Сформировать ответ по найденному контексту")
    ask_parser.add_argument("question")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Запустить интерактивный чат без перезагрузки моделей между вопросами",
    )
    chat_parser.add_argument(
        "--stream",
        action="store_true",
        help="Печатать ответ по мере поступления токенов от vLLM",
    )
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Переопределить количество фрагментов в контексте",
    )
    chat_parser.add_argument(
        "--history-turns",
        type=int,
        default=6,
        help="Количество последних пар вопрос/ответ, передаваемых модели (по умолчанию: 6)",
    )
    chat_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Не загружать поисковые модели до первого вопроса",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Запустить простой набор проверок RAG",
    )
    evaluate_parser.add_argument("testset", type=Path, help="JSONL-файл с тестами")
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Куда сохранить JSON-отчёт",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Проверить только первые N вопросов",
    )

    ragas_parser = subparsers.add_parser(
        "evaluate-ragas",
        help="Оценить RAG через judge-модель и RAGAS",
    )
    ragas_parser.add_argument("testset", type=Path, help="JSONL-файл с reference")
    ragas_parser.add_argument(
        "--output",
        type=Path,
        default=Path("ragas_results.json"),
        help="Куда сохранить JSON-отчёт RAGAS",
    )
    ragas_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Оценить только первые N вопросов",
    )
    ragas_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Минимальный балл каждой обязательной метрики от 0 до 1",
    )
    ragas_parser.add_argument(
        "--skip-answer-relevancy",
        action="store_true",
        help="Не вычислять диагностическую Answer Relevancy; записать null в отчёт",
    )
    ragas_parser.add_argument(
        "--skip-context-precision",
        action="store_true",
        help="Не вычислять диагностическую Context Precision; записать null в отчёт",
    )
    _add_artifact_cache_arguments(ragas_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Сравнить 4 RAG-конфигурации через RAGAS и DeepEval",
    )
    benchmark_parser.add_argument(
        "testset",
        type=Path,
        help="Синтетический JSONL-набор с question и reference",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/benchmark-results"),
        help="Каталог для JSON, CSV и Markdown-отчётов",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Использовать только первые N вопросов",
    )
    benchmark_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Минимальный балл обязательной метрики от 0 до 1",
    )
    benchmark_parser.add_argument(
        "--skip-answer-relevancy",
        action="store_true",
        help="Не вычислять Answer Relevancy в RAGAS и DeepEval",
    )
    benchmark_parser.add_argument(
        "--skip-context-precision",
        action="store_true",
        help="Не вычислять Context Precision в RAGAS и DeepEval",
    )
    _add_artifact_cache_arguments(benchmark_parser)
    return parser


def _document_payload(document) -> dict:
    return {"text": document.page_content, "metadata": document.metadata}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    if args.command == "check-vllm":
        if not settings.generation_model:
            parser.error("RAG_GENERATION_MODEL не задана")
        from rag_app.generation import check_vllm_server

        try:
            result = check_vllm_server(
                settings.generation_model,
                base_url=settings.vllm_base_url,
                api_key=settings.vllm_api_key,
                timeout=settings.vllm_timeout,
                prompt=args.prompt,
            )
        except RuntimeError as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "chat" and args.top_k is not None:
        if args.top_k <= 0:
            parser.error("--top-k должен быть больше нуля")
        settings = replace(
            settings,
            top_k=args.top_k,
            candidate_k=max(settings.candidate_k, args.top_k),
        )
    # Heavy retrieval dependencies are imported only after CLI arguments are valid.
    from rag_app.service import RagService

    service = RagService(settings)
    try:
        if args.command == "index":
            count = service.index(args.source, recreate=args.recreate)
            print(f"Проиндексировано фрагментов: {count}")
        elif args.command == "search":
            documents = service.search(args.query)
            if args.as_json:
                payload = [_document_payload(item) for item in documents]
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                for index, document in enumerate(documents, start=1):
                    print(f"[{index}] {document.metadata.get('source', 'unknown')}")
                    print(document.page_content)
                    print()
        elif args.command == "ask":
            answer, _ = service.ask(args.question)
            print(answer)
        elif args.command == "chat":
            from rag_app.chat import run_interactive

            if not args.no_warmup:
                print("[Warmup] Загрузка embedding-модели и reranker ...")
                found = service.warmup()
                print(f"[Warmup] Поисковые модели готовы; найдено фрагментов: {found}.")
            run_interactive(
                service,
                stream=args.stream,
                max_history_turns=args.history_turns,
            )
        elif args.command == "evaluate":
            from rag_app.evaluation import evaluate, load_cases, summarize, write_report
            cases = load_cases(args.testset)
            if args.limit is not None:
                if args.limit <= 0:
                    raise ValueError("--limit должен быть больше нуля")
                cases = cases[: args.limit]
            results = evaluate(service, cases, progress=print)
            summary = summarize(results)
            report_path = write_report(args.output, results)
            print(
                "\nПроверки: "
                f"{summary['passed']}/{summary['total']} пройдено "
                f"({summary['pass_rate']:.1%})"
            )
            print(f"Отчёт: {report_path}")
            if summary["failed"]:
                raise SystemExit(1)
        elif args.command == "evaluate-ragas":
            from rag_app.evaluation import load_cases
            from rag_app.ragas_evaluation import (
                evaluate_with_ragas,
                summarize_ragas,
                write_ragas_report,
            )
            cases = load_cases(args.testset)
            if args.limit is not None:
                if args.limit <= 0:
                    raise ValueError("--limit должен быть больше нуля")
                cases = cases[: args.limit]
            threshold = (
                settings.ragas_threshold
                if args.threshold is None
                else args.threshold
            )
            artifact_cache = _artifact_cache_from_args(args, settings)
            results = evaluate_with_ragas(
                service,
                cases,
                settings,
                threshold=threshold,
                include_answer_relevancy=not args.skip_answer_relevancy,
                include_context_precision=not args.skip_context_precision,
                artifact_cache=artifact_cache,
                refresh_artifact_cache=args.refresh_artifact_cache,
                progress=print,
            )
            summary = summarize_ragas(results, threshold)
            report_path = write_ragas_report(
                args.output,
                results,
                settings,
                threshold,
            )
            mean_score = summary["mean_score"]
            mean_label = "n/a" if mean_score is None else f"{mean_score:.3f}"
            print(
                "\nRAGAS: "
                f"{summary['passed']}/{summary['total']} пройдено; "
                f"средний балл: {mean_label}"
            )
            print(f"Отчёт: {report_path}")
            if artifact_cache is not None:
                print(
                    f"Кэш артефактов: {artifact_cache.directory}; "
                    f"создано: {artifact_cache.writes}; "
                    f"повторно использовано: {summary['artifact_cache_hits']}"
                )
                for cache_error in artifact_cache.write_errors:
                    print(f"Ошибка записи артефакта: {cache_error}")
            if args.skip_answer_relevancy:
                print("Answer Relevancy: не измерялась (null)")
            if args.skip_context_precision:
                print("Context Precision: не измерялась (null)")
            if summary["failed"]:
                raise SystemExit(1)
        elif args.command == "benchmark":
            from rag_app.benchmark import (
                comparison_rows,
                run_benchmark,
                write_benchmark_reports,
            )
            from rag_app.evaluation import load_cases

            cases = load_cases(args.testset)
            if args.limit is not None:
                if args.limit <= 0:
                    raise ValueError("--limit должен быть больше нуля")
                cases = cases[: args.limit]
            threshold = (
                settings.ragas_threshold
                if args.threshold is None
                else args.threshold
            )
            artifact_cache = _artifact_cache_from_args(args, settings)
            results = run_benchmark(
                settings,
                cases,
                threshold=threshold,
                include_answer_relevancy=not args.skip_answer_relevancy,
                include_context_precision=not args.skip_context_precision,
                progress=print,
                artifact_cache=artifact_cache,
                refresh_artifact_cache=args.refresh_artifact_cache,
            )
            report_paths = write_benchmark_reports(
                args.output_dir,
                results,
                settings,
            )
            rows = comparison_rows(results)
            valid_rows = [row for row in rows if row["combined_mean"] is not None]
            if valid_rows:
                best = max(valid_rows, key=lambda row: float(row["combined_mean"]))
                print(
                    "Лучшая конфигурация: "
                    f"{best['configuration']} ({float(best['combined_mean']):.3f})"
                )
            for report_name, report_path in report_paths.items():
                print(f"{report_name}: {report_path}")
            if artifact_cache is not None:
                cache_hits = sum(
                    result.ragas_summary["artifact_cache_hits"]
                    for result in results
                )
                print(
                    f"Кэш артефактов: {artifact_cache.directory}; "
                    f"создано: {artifact_cache.writes}; "
                    f"повторно использовано: {cache_hits}"
                )
                for cache_error in artifact_cache.write_errors:
                    print(f"Ошибка записи артефакта: {cache_error}")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
