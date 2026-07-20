from rag_app.metric_cache import MetricScoreCache


def test_metric_cache_reuses_only_identical_inputs(tmp_path) -> None:
    cache = MetricScoreCache(tmp_path)
    evaluator_config = {"judge_model": "local-model", "package_version": "1.0"}
    inputs = {"question": "Когда?", "response": "Сегодня."}

    cache.put(
        evaluator="ragas",
        metric_name="answer_relevancy",
        evaluator_config=evaluator_config,
        inputs=inputs,
        value=0.75,
        reason="релевантно",
    )

    cached = cache.get(
        evaluator="ragas",
        metric_name="answer_relevancy",
        evaluator_config=evaluator_config,
        inputs=inputs,
    )
    changed = cache.get(
        evaluator="ragas",
        metric_name="answer_relevancy",
        evaluator_config=evaluator_config,
        inputs={"question": "Когда?", "response": "Завтра."},
    )

    assert cached is not None
    assert cached.value == 0.75
    assert cached.reason == "релевантно"
    assert changed is None


def test_metric_cache_separates_ragas_and_deepeval(tmp_path) -> None:
    cache = MetricScoreCache(tmp_path)
    config = {"judge_model": "local-model"}
    inputs = {"question": "Вопрос", "response": "Ответ"}
    cache.put(
        evaluator="ragas",
        metric_name="answer_relevancy",
        evaluator_config=config,
        inputs=inputs,
        value=1.0,
    )

    assert cache.get(
        evaluator="deepeval",
        metric_name="answer_relevancy",
        evaluator_config=config,
        inputs=inputs,
    ) is None
