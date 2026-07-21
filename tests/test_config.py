from rag_app.config import Settings


def test_default_generation_model_is_qwen36(monkeypatch) -> None:
    monkeypatch.delenv("RAG_GENERATION_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.generation_model == "QuantTrio/Qwen3.6-27B-AWQ"


def test_default_text_chunk_size_is_1024(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)

    settings = Settings.from_env()

    assert settings.chunk_size == 1024


def test_default_docling_chunk_size_is_1024(monkeypatch) -> None:
    monkeypatch.delenv("RAG_DOCLING_CHUNK_TOKENS", raising=False)

    settings = Settings.from_env()

    assert settings.docling_chunk_tokens == 1024


def test_legacy_metric_cache_environment_variables_are_ignored(monkeypatch) -> None:
    monkeypatch.delenv("EVALUATION_ARTIFACT_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("EVALUATION_ARTIFACT_CACHE_DIR", raising=False)
    monkeypatch.setenv("EVALUATION_METRIC_CACHE_ENABLED", "false")
    monkeypatch.setenv("EVALUATION_METRIC_CACHE_DIR", "evaluation/legacy-cache")

    settings = Settings.from_env()

    assert settings.evaluation_artifact_cache_enabled is True
    assert settings.evaluation_artifact_cache_dir == "evaluation/artifact-cache"


def test_context_precision_concurrency_is_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAGAS_CONTEXT_PRECISION_CONCURRENCY", "3")

    settings = Settings.from_env()

    assert settings.ragas_context_precision_concurrency == 3
    settings.validate()
