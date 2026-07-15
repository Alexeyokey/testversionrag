from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from rag_app.config import Settings
from rag_app.service import RagService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Гибридный RAG: Qdrant + BM25 + reranker")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Проиндексировать файл, папку или URL")
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

    service = RagService(Settings.from_env())
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
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
