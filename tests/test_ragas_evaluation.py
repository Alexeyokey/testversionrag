import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

from rag_app.config import Settings
from rag_app.evaluation import EvaluationCase
from rag_app.ragas_evaluation import (
    _ArtifactCachingLLM,
    _CachedRagasArtifact,
    _ParallelContextPrecision,
    _build_vllm_async_client,
    evaluate_with_ragas,
    summarize_ragas,
)
from rag_app.artifact_cache import ArtifactCache


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


def test_ragas_reports_current_question_and_metric() -> None:
    messages: list[str] = []
    scorers = {
        name: _Scorer(1.0)
        for name in (
            "faithfulness",
            "context_recall",
            "answer_accuracy",
            "context_precision",
            "answer_relevancy",
        )
    }

    evaluate_with_ragas(
        _Service(),
        [EvaluationCase(question="Когда договор?", reference="15 марта")],
        Settings(enable_reranker=False),
        scorers=scorers,
        progress=messages.append,
    )

    assert messages[0] == "[RAG 1/1] Вопрос: Когда договор?"
    assert messages[1] == "[RAGAS 1/1] Вопрос: Когда договор?"
    assert "[RAGAS 1/1] Метрика: faithfulness" in messages


class _AsyncContextPrecisionScorer:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    async def ascore(self, **kwargs):
        context = kwargs["retrieved_contexts"][0]
        self.calls.append(context)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
        finally:
            self.active -= 1
        value = 0.9999999999 if context.startswith("relevant") else 0.0
        return SimpleNamespace(value=value)


@dataclass
class _StatementOutput:
    statements: list[str]

    @classmethod
    def model_validate(cls, value):
        return cls(statements=list(value["statements"]))


@dataclass
class _VerdictOutput:
    verdict: int


class _Prompt:
    def __init__(self) -> None:
        self.calls = 0

    async def agenerate(self, prompt, output_model):
        del prompt
        self.calls += 1
        if output_model is _StatementOutput:
            return _StatementOutput(["Факт из ответа."])
        assert output_model is _VerdictOutput
        return _VerdictOutput(verdict=1)


@dataclass
class _QuestionOutput:
    question: str
    noncommittal: int = 0

    @classmethod
    def model_validate(cls, value):
        return cls(
            question=str(value["question"]),
            noncommittal=int(value.get("noncommittal", 0)),
        )


class _QuestionLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def agenerate(self, prompt, output_model):
        del prompt
        self.calls += 1
        assert output_model is _QuestionOutput
        return _QuestionOutput(question=f"Вопрос {self.calls}?")


class _ArtifactScorer(_Scorer):
    def __init__(
        self,
        value: float,
        llm: _ArtifactCachingLLM,
        artifact: _CachedRagasArtifact,
    ) -> None:
        super().__init__(value)
        self.llm = llm
        self._artifact_cache_controller = artifact

    def score(self, **kwargs):
        self.calls.append(kwargs)
        asyncio.run(
            self.llm.agenerate(
                f"{kwargs['user_input']}\n{kwargs['response']}",
                _StatementOutput,
            )
        )
        return SimpleNamespace(value=self.value)


def test_ragas_llm_cache_reuses_statement_artifact(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    original_llm = _Prompt()
    facts = _CachedRagasArtifact(
        cache,
        artifact_name="faithfulness_statements",
        evaluator_config={"judge_model": "local-model"},
        refresh=False,
    )
    llm = _ArtifactCachingLLM(
        original_llm,
        facts,
        output_model_name="_StatementOutput",
    )

    facts.begin_sample()
    first = asyncio.run(llm.agenerate("Что?\nОтвет.", _StatementOutput))
    facts.finish_sample(save=True)
    facts.begin_sample()
    second = asyncio.run(llm.agenerate("Что?\nОтвет.", _StatementOutput))
    facts.finish_sample(save=True)

    # NLI verdict calls use another output model and must always reach the judge.
    asyncio.run(llm.agenerate("Контекст 1", _VerdictOutput))
    asyncio.run(llm.agenerate("Контекст 2", _VerdictOutput))

    assert first == second
    assert original_llm.calls == 3
    assert facts.cache_hits == 1
    assert facts.last_value == {"statements": ["Факт из ответа."]}
    assert cache.writes == 1


def test_answer_relevancy_cache_preserves_three_sequential_questions(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    original_llm = _QuestionLLM()
    questions = _CachedRagasArtifact(
        cache,
        artifact_name="answer_relevancy_questions",
        evaluator_config={"judge_model": "local-model", "strictness": 3},
        refresh=False,
    )
    llm = _ArtifactCachingLLM(
        original_llm,
        questions,
        output_model_name="_QuestionOutput",
    )

    questions.begin_sample()
    generated = [
        asyncio.run(llm.agenerate("Ответ.", _QuestionOutput))
        for _ in range(3)
    ]
    questions.finish_sample(save=True)

    assert [item.question for item in generated] == [
        "Вопрос 1?",
        "Вопрос 2?",
        "Вопрос 3?",
    ]

    questions.begin_sample()
    cached = [
        asyncio.run(llm.agenerate("Ответ.", _QuestionOutput))
        for _ in range(3)
    ]
    questions.finish_sample(save=True)

    assert [item.question for item in cached] == [
        "Вопрос 1?",
        "Вопрос 2?",
        "Вопрос 3?",
    ]
    assert original_llm.calls == 3
    assert questions.cache_hits == 1
    assert cache.writes == 1


def test_ragas_report_marks_cached_artifact_while_recomputing_score(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    original_llm = _Prompt()
    artifact = _CachedRagasArtifact(
        cache,
        artifact_name="faithfulness_statements",
        evaluator_config={"judge_model": "local-model"},
        refresh=False,
    )
    cached_llm = _ArtifactCachingLLM(
        original_llm,
        artifact,
        output_model_name="_StatementOutput",
    )
    faithfulness = _ArtifactScorer(0.9, cached_llm, artifact)
    scorers = {
        "faithfulness": faithfulness,
        "context_recall": _Scorer(0.8),
        "answer_accuracy": _Scorer(1.0),
        "context_precision": _Scorer(0.75),
        "answer_relevancy": _Scorer(0.85),
    }
    cases = [
        EvaluationCase(
            question="Когда заключён договор?",
            reference="Договор заключён 15 марта 2025 года.",
        )
    ]

    first = evaluate_with_ragas(
        _Service(),
        cases,
        Settings(enable_reranker=False),
        scorers=scorers,
    )[0]
    second = evaluate_with_ragas(
        _Service(),
        cases,
        Settings(enable_reranker=False),
        scorers=scorers,
    )[0]

    assert first.artifacts == {
        "faithfulness_statements": {"statements": ["Факт из ответа."]}
    }
    assert first.cached_artifacts == ()
    assert second.cached_artifacts == ("faithfulness_statements",)
    assert len(faithfulness.calls) == 2
    assert original_llm.calls == 1


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


def test_ragas_can_skip_context_precision() -> None:
    context_precision = _Scorer(0.75)
    scorers = {
        "faithfulness": _Scorer(0.9),
        "context_recall": _Scorer(0.8),
        "answer_accuracy": _Scorer(1.0),
        "context_precision": context_precision,
        "answer_relevancy": _Scorer(0.85),
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
        scorers=scorers,
        include_context_precision=False,
    )[0]

    assert result.scores["context_precision"] is None
    assert result.skipped_metrics == ("context_precision",)
    assert context_precision.calls == []


def test_context_precision_evaluates_contexts_in_parallel_and_preserves_rank() -> None:
    original = _AsyncContextPrecisionScorer()
    scorer = _ParallelContextPrecision(original, concurrency=2)

    result = scorer.score(
        user_input="Вопрос",
        reference="Эталон",
        retrieved_contexts=["relevant-1", "noise", "relevant-2", "noise-2"],
    )

    expected = (1.0 + 2 / 3) / 2
    assert abs(result.value - expected) < 1e-9
    assert original.max_active == 2
    assert original.calls == ["relevant-1", "noise", "relevant-2", "noise-2"]


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

    openai_module = ModuleType("openai")
    openai_module.AsyncOpenAI = _AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
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


def test_ragas_recomputes_every_metric_score_when_artifact_cache_is_enabled(
    tmp_path,
) -> None:
    scorers = {
        "faithfulness": _Scorer(0.9),
        "context_recall": _Scorer(0.8),
        "answer_accuracy": _Scorer(1.0),
        "context_precision": _Scorer(0.75),
        "answer_relevancy": _Scorer(0.85),
    }
    cases = [
        EvaluationCase(
            question="Когда заключён договор?",
            reference="Договор заключён 15 марта 2025 года.",
        )
    ]
    settings = Settings(enable_reranker=False)
    cache = ArtifactCache(tmp_path)

    first = evaluate_with_ragas(
        _Service(),
        cases,
        settings,
        scorers=scorers,
        artifact_cache=cache,
    )[0]
    second = evaluate_with_ragas(
        _Service(),
        cases,
        settings,
        scorers=scorers,
        artifact_cache=cache,
    )[0]
    assert first.cached_artifacts == ()
    assert second.cached_artifacts == ()
    assert all(len(scorer.calls) == 2 for scorer in scorers.values())
    assert list(tmp_path.rglob("*.json")) == []
