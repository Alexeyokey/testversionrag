from rag_app.artifact_cache import ArtifactCache


def test_artifact_cache_reuses_only_identical_inputs(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    evaluator_config = {"judge_model": "local-model", "package_version": "1.0"}
    inputs = {
        "data": {
            "question": "Когда?",
            "answer": "Договор заключён сегодня.",
        }
    }
    facts = {"statements": ["Договор заключён сегодня."]}

    cache.put(
        evaluator="ragas",
        artifact_name="faithfulness_statements",
        evaluator_config=evaluator_config,
        inputs=inputs,
        value=facts,
    )

    cached = cache.get(
        evaluator="ragas",
        artifact_name="faithfulness_statements",
        evaluator_config=evaluator_config,
        inputs=inputs,
    )
    changed = cache.get(
        evaluator="ragas",
        artifact_name="faithfulness_statements",
        evaluator_config=evaluator_config,
        inputs={"data": {"question": "Когда?", "answer": "Завтра."}},
    )

    assert cached == facts
    assert changed is None
    assert cache.hits == 1
    assert cache.writes == 1


def test_artifact_cache_does_not_contain_metric_score_fields(tmp_path) -> None:
    cache = ArtifactCache(tmp_path)
    cache.put(
        evaluator="ragas",
        artifact_name="answer_relevancy_questions",
        evaluator_config={"judge_model": "local-model"},
        inputs={"data": {"response": "Ответ"}, "n": 3},
        value=[{"question": "Вопрос 1?"}, {"question": "Вопрос 2?"}],
    )

    cache_file = next(tmp_path.rglob("*.json"))
    payload = cache_file.read_text(encoding="utf-8")

    assert '"artifact": "answer_relevancy_questions"' in payload
    assert '"score"' not in payload
    assert '"reason"' not in payload
    assert '"passed"' not in payload


def test_artifact_cache_reports_write_errors(tmp_path) -> None:
    blocked_directory = tmp_path / "not-a-directory"
    blocked_directory.write_text("file", encoding="utf-8")
    cache = ArtifactCache(blocked_directory)

    try:
        cache.put(
            evaluator="ragas",
            artifact_name="faithfulness_statements",
            evaluator_config={"judge_model": "local-model"},
            inputs={"prompt": "test"},
            value={"statements": ["Факт"]},
        )
    except OSError:
        pass

    assert cache.writes == 0
    assert cache.write_errors
