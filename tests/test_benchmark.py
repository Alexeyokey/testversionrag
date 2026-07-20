from types import SimpleNamespace

from rag_app.benchmark import comparison_rows, run_benchmark, write_benchmark_reports
from rag_app.config import Settings
from rag_app.evaluation import EvaluationCase


class _Service:
    def __init__(self, settings: Settings, calls: list[tuple[str, str]]) -> None:
        self.settings = settings
        self.calls = calls

    def ask(self, question: str):
        self.calls.append((str(self.settings.vector_weight), question))
        document = SimpleNamespace(
            page_content="Эталонный контекст",
            metadata={"source": "synthetic.md"},
        )
        return "Эталонный ответ", [document]


class _RagasMetric:
    def score(self, **kwargs):
        return SimpleNamespace(value=0.8, reason="ok")


class _DeepEvalMetric:
    score = 0.8
    reason = "ok"

    def measure(self, test_case):
        return self.score


def test_benchmark_runs_rag_once_per_configuration_and_question(tmp_path) -> None:
    rag_calls: list[tuple[str, str]] = []
    metric_names = (
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    )

    results = run_benchmark(
        Settings(enable_reranker=False),
        [EvaluationCase(question="Вопрос", reference="Эталон")],
        service_factory=lambda settings: _Service(settings, rag_calls),
        ragas_scorers={name: _RagasMetric() for name in metric_names},
        deepeval_scorers={name: _DeepEvalMetric() for name in metric_names},
        test_case_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    rows = comparison_rows(results)

    assert len(results) == 4
    assert len(rag_calls) == 4
    assert [row["configuration"] for row in rows] == [
        "vector_only",
        "bm25_only",
        "hybrid",
        "hybrid_reranker",
    ]
    assert rows[0]["ragas_faithfulness"] == 0.8
    assert rows[0]["deepeval_answer_relevancy"] == 0.8
    assert rows[0]["combined_mean"] == 0.8

    reports = write_benchmark_reports(
        tmp_path,
        results,
        Settings(enable_reranker=False),
    )
    assert set(reports) == {"json", "csv", "markdown"}
    assert "hybrid_reranker" in reports["csv"].read_text(encoding="utf-8-sig")
    assert "Лучшая конфигурация" in reports["markdown"].read_text(encoding="utf-8")
