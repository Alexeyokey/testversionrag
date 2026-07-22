import json
from dataclasses import replace
from types import SimpleNamespace

from rag_app.retrieval_tuning import (
    build_weight_configurations,
    comparison_rows,
    run_retrieval_tuning,
    write_retrieval_tuning_reports,
)
from rag_app.config import Settings
from rag_app.evaluation import EvaluationCase


class _Service:
    def __init__(
        self,
        settings: Settings,
        calls: list[tuple[float, float, bool, str]],
    ) -> None:
        self.settings = settings
        self.calls = calls

    def search(self, question: str):
        self.calls.append(
            (
                self.settings.vector_weight,
                self.settings.bm25_weight,
                self.settings.enable_reranker,
                question,
            )
        )
        document = SimpleNamespace(
            page_content="Эталонный контекст",
            metadata={"source": "synthetic.md"},
        )
        return [document]

    def answer_from_documents(self, question: str, documents) -> str:
        assert question == "Вопрос"
        assert len(documents) == 1
        return "Эталонный ответ"

    def ask(self, question: str):
        raise AssertionError(f"Повторный retrieval через ask(): {question}")


class _RagasMetric:
    def score(self, **kwargs):
        return SimpleNamespace(value=0.8, reason="ok")


class _DeepEvalMetric:
    score = 0.8
    reason = "ok"

    def measure(self, test_case):
        return self.score


def test_retrieval_tuning_runs_rag_once_per_weight_and_question(tmp_path) -> None:
    rag_calls: list[tuple[float, float, bool, str]] = []
    progress_messages: list[str] = []
    metric_names = (
        "faithfulness",
        "context_recall",
        "answer_accuracy",
        "context_precision",
        "answer_relevancy",
    )
    results = run_retrieval_tuning(
        Settings(enable_reranker=True),
        [EvaluationCase(question="Вопрос", reference="Эталон")],
        service_factory=lambda settings: _Service(settings, rag_calls),
        ragas_scorers={name: _RagasMetric() for name in metric_names},
        deepeval_scorers={name: _DeepEvalMetric() for name in metric_names},
        test_case_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        progress=progress_messages.append,
    )
    rows = comparison_rows(results)

    assert len(results) == 5
    assert len(rag_calls) == 5
    assert [row["configuration"] for row in rows] == [
        "hybrid_v20_b80",
        "hybrid_v40_b60",
        "hybrid_v50_b50",
        "hybrid_v60_b40",
        "hybrid_v80_b20",
    ]
    assert all(0 < vector < 1 and 0 < bm25 < 1 for vector, bm25, _, _ in rag_calls)
    assert all(reranker is True for _, _, reranker, _ in rag_calls)
    assert rows[0]["ragas_faithfulness"] == 0.8
    assert rows[0]["ragas_answer_accuracy"] == 0.8
    assert rows[0]["deepeval_answer_relevancy"] == 0.8
    assert abs(rows[0]["tuning_score"] - 0.8) < 1e-12
    assert rows[0]["retrieval_mean_seconds"] is not None
    assert rows[0]["generation_mean_seconds"] is not None
    assert results[4].samples[0].retrieval_seconds is not None
    assert results[4].samples[0].generation_seconds is not None
    assert "[hybrid_v20_b80] [RAG 1/1] Вопрос: Вопрос" in progress_messages
    assert any(
        message.startswith("[hybrid_v20_b80] [RAG 1/1] Retrieval:")
        for message in progress_messages
    )
    assert any(
        message.startswith("[hybrid_v20_b80] [RAG 1/1] Generation:")
        for message in progress_messages
    )
    assert "[hybrid_v20_b80] [RAGAS 1/1] Вопрос: Вопрос" in progress_messages
    assert "[hybrid_v20_b80] [DeepEval 1/1] Вопрос: Вопрос" in progress_messages

    reports = write_retrieval_tuning_reports(
        tmp_path,
        results,
        Settings(enable_reranker=False),
    )
    assert set(reports) == {"json", "csv", "markdown"}
    assert "hybrid_v80_b20" in reports["csv"].read_text(encoding="utf-8-sig")
    details = json.loads(reports["json"].read_text(encoding="utf-8"))
    sample = details["configurations"][0]["samples"][0]
    assert sample["retrieval_seconds"] is not None
    assert sample["generation_seconds"] is not None
    assert "Лучший баланс" in reports["markdown"].read_text(encoding="utf-8")


def test_default_retrieval_tuning_reuses_one_service(monkeypatch) -> None:
    instances: list[_Service] = []
    calls: list[tuple[float, float, bool, str]] = []

    class _ReusableService(_Service):
        embedding_model = None

        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, calls)
            instances.append(self)

        def set_retrieval_weights(
            self,
            *,
            vector_weight: float,
            bm25_weight: float,
        ) -> None:
            self.settings = replace(
                self.settings,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )

    monkeypatch.setattr("rag_app.service.RagService", _ReusableService)
    metric_names = (
        "faithfulness",
        "context_recall",
        "answer_accuracy",
        "context_precision",
        "answer_relevancy",
    )

    run_retrieval_tuning(
        Settings(enable_reranker=True),
        [EvaluationCase(question="Вопрос", reference="Эталон")],
        ragas_scorers={name: _RagasMetric() for name in metric_names},
        deepeval_scorers={name: _DeepEvalMetric() for name in metric_names},
        test_case_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert len(instances) == 1
    assert len(calls) == 5


def test_weight_configurations_never_create_standalone_search() -> None:
    configurations = build_weight_configurations((0.1, 0.3, 0.7, 0.9))

    assert all(
        0 < configuration.vector_weight < 1
        and 0 < configuration.bm25_weight < 1
        for configuration in configurations
    )


def test_weight_configurations_reject_standalone_search() -> None:
    for vector_weights in (
        (0.0, 0.3, 0.6, 0.9),
        (0.1, 0.4, 0.7, 1.0),
    ):
        try:
            build_weight_configurations(vector_weights)
        except ValueError as error:
            assert "vector-only и BM25-only" in str(error)
        else:
            raise AssertionError("Standalone retrieval configuration was accepted")
