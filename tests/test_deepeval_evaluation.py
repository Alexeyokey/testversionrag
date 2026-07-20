from types import SimpleNamespace

from rag_app.config import Settings
from rag_app.deepeval_evaluation import (
    evaluate_samples_with_deepeval,
    summarize_deepeval,
)
from rag_app.evaluation import RagEvaluationSample


class _Metric:
    def __init__(self, score: float, reason: str = "ok") -> None:
        self.next_score = score
        self.score = None
        self.reason = None
        self.cases = []
        self.next_reason = reason

    def measure(self, test_case) -> float:
        self.cases.append(test_case)
        self.score = self.next_score
        self.reason = self.next_reason
        return self.next_score


def _case_factory(**kwargs):
    return SimpleNamespace(**kwargs)


def test_deepeval_scores_precomputed_sample_without_calling_rag() -> None:
    scorers = {
        "faithfulness": _Metric(0.9),
        "context_precision": _Metric(0.8),
        "context_recall": _Metric(0.7),
        "answer_relevancy": _Metric(0.95),
    }
    sample = RagEvaluationSample(
        question="Когда заключён договор?",
        reference="15 марта 2025 года",
        response="Договор заключён 15 марта 2025 года.",
        retrieved_contexts=("Дата договора — 15 марта 2025 года.",),
        sources=("contract.md",),
    )

    results = evaluate_samples_with_deepeval(
        [sample],
        Settings(enable_reranker=False),
        threshold=0.7,
        scorers=scorers,
        test_case_factory=_case_factory,
    )
    summary = summarize_deepeval(results, threshold=0.7)

    assert results[0].passed is True
    assert results[0].metric_reasons["faithfulness"] == "ok"
    assert summary["metrics"]["answer_relevancy"] == 0.95
    assert scorers["context_recall"].cases[0].expected_output == "15 марта 2025 года"
    assert scorers["context_recall"].cases[0].retrieval_context == [
        "Дата договора — 15 марта 2025 года."
    ]


def test_deepeval_preserves_rag_error() -> None:
    scorers = {
        name: _Metric(1.0)
        for name in (
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
        )
    }
    sample = RagEvaluationSample(
        question="Вопрос",
        reference="Ответ",
        response="",
        retrieved_contexts=(),
        sources=(),
        error="RAG failed",
    )

    result = evaluate_samples_with_deepeval(
        [sample],
        Settings(enable_reranker=False),
        scorers=scorers,
        test_case_factory=_case_factory,
    )[0]

    assert result.passed is False
    assert result.error == "RAG failed"
    assert not scorers["faithfulness"].cases
