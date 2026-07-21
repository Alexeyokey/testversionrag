import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from rag_app.config import Settings
from rag_app.evaluation import EvaluationCase
from rag_app.ragas_evaluation import (
    _CachedRagasPrompt,
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


@dataclass
class _PromptOutput:
    statements: list[str]

    @classmethod
    def model_validate(cls, value):
        return cls(statements=list(value["statements"]))


class _Prompt:
    output_model = _PromptOutput

    def __init__(self) -> None:
        self.generate_calls = 0
        self.multiple_calls = 0

    async def generate(self, **kwargs):
        del kwargs
        self.generate_calls += 1
        return _PromptOutput(["Факт из ответа."])

    async def generate_multiple(self, **kwargs):
        self.multiple_calls += 1
        return [
            _PromptOutput([f"Вопрос {index}?"])
            for index in range(kwargs["n"])
        ]


class _SequentialPrompt(_Prompt):
    async def generate(self, **kwargs):
        del kwargs
        self.generate_calls += 1
        return _PromptOutput([f"Вопрос {self.generate_calls}?"])


class _ArtifactScorer(_Scorer):
    def __init__(self, value: float, prompt: _CachedRagasPrompt) -> None:
        super().__init__(value)
        self.statement_generator_prompt = prompt

    def score(self, **kwargs):
        self.calls.append(kwargs)
        asyncio.run(
            self.statement_generator_prompt.generate(
                llm=object(),
                data={
                    "question": kwargs["user_input"],
                    "answer": kwargs["response"],
                },
            )
        )
        return SimpleNamespace(value=self.value)


def test_ragas_prompt_cache_reuses_artifacts_but_not_scores(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    prompt = _Prompt()
    facts = _CachedRagasPrompt(
        prompt,
        cache,
        artifact_name="faithfulness_statements",
        evaluator_config={"judge_model": "local-model"},
        refresh=False,
    )

    first = asyncio.run(
        facts.generate(llm=object(), data={"question": "Что?", "answer": "Ответ."})
    )
    second = asyncio.run(
        facts.generate(llm=object(), data={"question": "Что?", "answer": "Ответ."})
    )

    assert first == second
    assert prompt.generate_calls == 1
    assert facts.cache_hits == 1
    assert facts.last_value == {"statements": ["Факт из ответа."]}

    questions = _CachedRagasPrompt(
        prompt,
        cache,
        artifact_name="answer_relevancy_questions",
        evaluator_config={"judge_model": "local-model"},
        refresh=False,
    )
    first_questions = asyncio.run(
        questions.generate_multiple(llm=object(), data={"response": "Ответ."}, n=3)
    )
    second_questions = asyncio.run(
        questions.generate_multiple(llm=object(), data={"response": "Ответ."}, n=3)
    )

    assert first_questions == second_questions
    assert prompt.multiple_calls == 1
    assert questions.cache_hits == 1
    assert len(questions.last_value) == 3


def test_answer_relevancy_cache_preserves_three_sequential_questions(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    prompt = _SequentialPrompt()
    questions = _CachedRagasPrompt(
        prompt,
        cache,
        artifact_name="answer_relevancy_questions",
        evaluator_config={"judge_model": "local-model", "strictness": 3},
        refresh=False,
    )
    call = {"llm": object(), "data": {"response": "Ответ."}}

    questions.begin_sample()
    generated = [asyncio.run(questions.generate(**call)) for _ in range(3)]
    questions.finish_sample(save=True)

    assert [item.statements[0] for item in generated] == [
        "Вопрос 1?",
        "Вопрос 2?",
        "Вопрос 3?",
    ]

    questions.begin_sample()
    cached = [asyncio.run(questions.generate(**call)) for _ in range(3)]
    questions.finish_sample(save=True)

    assert [item.statements[0] for item in cached] == [
        "Вопрос 1?",
        "Вопрос 2?",
        "Вопрос 3?",
    ]
    assert prompt.generate_calls == 3
    assert questions.cache_hits == 1


def test_ragas_report_marks_cached_artifact_while_recomputing_score(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    cached_prompt = _CachedRagasPrompt(
        _Prompt(),
        cache,
        artifact_name="faithfulness_statements",
        evaluator_config={"judge_model": "local-model"},
        refresh=False,
    )
    faithfulness = _ArtifactScorer(0.9, cached_prompt)
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
