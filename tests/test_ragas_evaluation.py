from dataclasses import dataclass
from types import SimpleNamespace

from rag_app.config import Settings
from rag_app.evaluation import EvaluationCase
from rag_app.ragas_evaluation import (
    _build_vllm_async_client,
    evaluate_with_ragas,
    summarize_ragas,
)


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
        "context_recall": _Scorer(0.6),
        "answer_accuracy": _Scorer(0.95),
        "context_precision": _Scorer(0.75),
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
    assert scorers["answer_accuracy"].calls[0]["reference"] == (
        "Договор заключён 15 марта 2025 года."
    )


def test_ragas_optional_metrics_do_not_fail_case() -> None:
    scorers = {
        "faithfulness": _Scorer(0.9),
        "context_recall": _Scorer(0.8),
        "answer_accuracy": _Scorer(1.0),
        "context_precision": _Scorer(0.1),
        "answer_relevancy": _Scorer(0.2),
    }

    result = evaluate_with_ragas(
        _Service(),
        [
            EvaluationCase(
                question="Когда заключён договор?",
                reference="Договор заключён 15 марта 2025 года.",
            )
        ],
        Settings(enable_reranker=False),
        threshold=0.7,
        scorers=scorers,
    )[0]

    assert result.passed is True
    assert result.mean_score == 0.9


def test_ragas_can_skip_answer_relevancy() -> None:
    scorers = {
        "faithfulness": _Scorer(0.9),
        "context_recall": _Scorer(0.8),
        "answer_accuracy": _Scorer(1.0),
        "context_precision": _Scorer(0.75),
    }

    results = evaluate_with_ragas(
        _Service(),
        [
            EvaluationCase(
                question="Когда заключён договор?",
                reference="Договор заключён 15 марта 2025 года.",
            )
        ],
        Settings(enable_reranker=False),
        threshold=0.7,
        scorers=scorers,
        include_answer_relevancy=False,
    )
    summary = summarize_ragas(results, threshold=0.7)

    assert results[0].passed is True
    assert results[0].scores["answer_relevancy"] is None
    assert results[0].skipped_metrics == ("answer_relevancy",)
    assert summary["metrics"]["answer_relevancy"] is None


def test_ragas_requires_reference() -> None:
    scorers = {name: _Scorer(1.0) for name in (
        "faithfulness",
        "context_recall",
        "answer_accuracy",
        "context_precision",
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


def test_ragas_judge_uses_async_openai_client(monkeypatch) -> None:
    captured = {}

    class _AsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("openai.AsyncOpenAI", _AsyncOpenAI)
    settings = Settings(
        enable_reranker=False,
        vllm_base_url="http://vllm:8000/v1",
        vllm_api_key="test-key",
        vllm_timeout=42,
    )

    client = _build_vllm_async_client(settings)

    assert isinstance(client, _AsyncOpenAI)
    assert captured == {
        "api_key": "test-key",
        "base_url": "http://vllm:8000/v1",
        "timeout": 42,
        "max_retries": 2,
    }
