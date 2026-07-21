from rag_app.config import Settings


def test_legacy_metric_cache_environment_variables_are_ignored(monkeypatch) -> None:
    monkeypatch.delenv("EVALUATION_ARTIFACT_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("EVALUATION_ARTIFACT_CACHE_DIR", raising=False)
    monkeypatch.setenv("EVALUATION_METRIC_CACHE_ENABLED", "false")
    monkeypatch.setenv("EVALUATION_METRIC_CACHE_DIR", "evaluation/legacy-cache")

    settings = Settings.from_env()

    assert settings.evaluation_artifact_cache_enabled is True
    assert settings.evaluation_artifact_cache_dir == "evaluation/artifact-cache"
