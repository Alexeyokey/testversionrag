from dataclasses import dataclass
from types import SimpleNamespace

from rag_app.config import Settings
from rag_app.evaluation import EvaluationCase
from rag_app.ragas_evaluation import evaluate_with_ragas, summarize_ragas


@dataclass
class _Document:
    page_content: str
    metadata: dict


class _Service:
    def ask(self, question: str):
        return (
            "Договор заключён 15 марта 2025 года.",
            [
                _Document(
                    page_content="Дата договора: 15 марта 2025 года.",
                    metadata={"source": "contract.pdf"},
                )
            ],
        )


class _Scorer:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict] = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def test_ragas_scores_answer_and_context_with_injected_judges() -> None:
    scorers = {
        "faithfulness": _Scorer(0.9),
        "context_precision": _Scorer(0.75),
        "context_recall": _Scorer(0.6),
        "answer_relevancy": _Scorer(0.8),
    }
    cases = [
        EvaluationCase(
            question="Когда заключён договор?",
            reference="Договор заключён 15 марта 2025 года.",
        )
    ]

    results = evaluate_with_ragas(
        _Service(),
        cases,
        Settings(enable_reranker=False),
        threshold=0.7,
        scorers=scorers,
    )
    summary = summarize_ragas(results, threshold=0.7)

    assert results[0].scores["faithfulness"] == 0.9
    assert results[0].passed is False
    assert results[0].sources == ("contract.pdf",)
    assert summary["metrics"]["context_recall"] == 0.6
    assert scorers["faithfulness"].calls[0]["retrieved_contexts"] == [
        "Дата договора: 15 марта 2025 года."
    ]


def test_ragas_requires_reference() -> None:
    scorers = {name: _Scorer(1.0) for name in (
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    )}

    results = evaluate_with_ragas(
        _Service(),
        [EvaluationCase(question="Вопрос", answer_terms=("ответ",))],
        Settings(enable_reranker=False),
        scorers=scorers,
    )

    assert results[0].passed is False
    assert "reference обязательно" in str(results[0].error)
