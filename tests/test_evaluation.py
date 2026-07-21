import json
from dataclasses import dataclass
from pathlib import Path

from rag_app.evaluation import EvaluationCase, evaluate, load_cases, summarize, write_report


@dataclass
class _Document:
    page_content: str
    metadata: dict


class _Service:
    def ask(self, question: str) -> tuple[str, list[_Document]]:
        if question == "Ошибка":
            raise RuntimeError("vLLM недоступен")
        return (
            "Договор заключён 15 марта 2025 года. Выручка составила 8 269 600 рублей.",
            [
                _Document(
                    page_content="Дата договора: 15 марта 2025. Выручка: 8269600.",
                    metadata={"source": "contract.pdf"},
                )
            ],
        )


def test_load_cases_uses_reference_as_default_answer_check(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Когда заключён договор?",
                "reference": "15 марта 2025",
                "context_terms": ["Дата договора"],
                "expected_sources": ["contract.pdf"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].answer_terms == ("15 марта 2025",)
    assert cases[0].expected_sources == ("contract.pdf",)


def test_evaluate_separates_answer_context_and_source_checks(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        {
            "question": "Когда заключён договор?",
            "answer_terms": ["15 марта 2025", "8 269 600"],
            "context_terms": ["Дата договора", "8269600"],
            "expected_sources": ["contract.pdf"],
        },
        {
            "question": "Ошибка",
            "answer_terms": ["ответ"],
        },
    ]
    cases_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    results = evaluate(_Service(), load_cases(cases_path))
    summary = summarize(results)
    report_path = write_report(tmp_path / "report.json", results)

    assert results[0].passed is True
    assert results[0].answer_score == 1.0
    assert results[0].context_score == 1.0
    assert results[0].source_score == 1.0
    assert results[1].passed is False
    assert "vLLM недоступен" in str(results[1].error)
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["summary"]["total"] == 2


def test_evaluate_reports_current_question() -> None:
    messages: list[str] = []
    cases = [EvaluationCase(question="Какой сейчас вопрос?")]

    evaluate(_Service(), cases[:1], progress=messages.append)

    assert messages == ["[RAG 1/1] Вопрос: Какой сейчас вопрос?"]
