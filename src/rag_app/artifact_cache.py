from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

if TYPE_CHECKING:
    from rag_app.config import Settings


# Версия схемы относится только к промежуточным артефактам разметки.
CACHE_SCHEMA_VERSION = 2
EVALUATOR_PROMPT_REVISION = 1


class ArtifactCache:
    """Хранить только промежуточные данные разметки, но не итоговые оценки метрик."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.hits = 0
        self.writes = 0
        self.write_errors: list[str] = []

    def get(
        self,
        *,
        evaluator: str,
        artifact_name: str,
        evaluator_config: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> Any | None:
        identity = _cache_identity(
            evaluator=evaluator,
            artifact_name=artifact_name,
            evaluator_config=evaluator_config,
            inputs=inputs,
        )
        cache_path = self._entry_path(evaluator, artifact_name, identity)
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != CACHE_SCHEMA_VERSION
                or payload.get("identity") != identity
                or payload.get("artifact") != artifact_name
            ):
                return None
            self.hits += 1
            return payload["value"]
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # Повреждённый или устаревший файл считается промахом и будет перезаписан.
            return None

    def put(
        self,
        *,
        evaluator: str,
        artifact_name: str,
        evaluator_config: Mapping[str, Any],
        inputs: Mapping[str, Any],
        value: Any,
    ) -> Path:
        identity = _cache_identity(
            evaluator=evaluator,
            artifact_name=artifact_name,
            evaluator_config=evaluator_config,
            inputs=inputs,
        )
        cache_path = self._entry_path(evaluator, artifact_name, identity)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "identity": identity,
                "evaluator": evaluator,
                "artifact": artifact_name,
                "evaluator_config": dict(evaluator_config),
                "inputs": _to_json_value(dict(inputs)),
                "value": _to_json_value(value),
            }
            temporary_path = cache_path.with_suffix(
                f".{os.getpid()}.{uuid4().hex}.tmp"
            )
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            # Запись через временный файл не оставит битый JSON, если процесс прервётся.
            temporary_path.replace(cache_path)
        except (OSError, TypeError, ValueError) as error:
            self.write_errors.append(f"{type(error).__name__}: {error}")
            raise
        self.writes += 1
        return cache_path

    def _entry_path(self, evaluator: str, artifact_name: str, identity: str) -> Path:
        return self.directory / evaluator / "artifacts" / artifact_name / f"{identity}.json"


def ragas_artifact_config(settings: Settings) -> dict[str, Any]:
    """Описать RAGAS-настройки, от которых зависит артефакт."""
    try:
        package_version = version("ragas")
    except PackageNotFoundError:
        package_version = "unknown"

    return {
        "judge_model": settings.ragas_judge_model or settings.generation_model,
        "judge_temperature": 0.0,
        "judge_max_tokens": settings.ragas_max_tokens,
        "package_version": package_version,
        "prompt_revision": EVALUATOR_PROMPT_REVISION,
    }


def _cache_identity(
    *,
    evaluator: str,
    artifact_name: str,
    evaluator_config: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> str:
    # Смена модели, версии RAGAS или prompt должна инвалидировать старый артефакт.
    raw_identity = json.dumps(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "evaluator": evaluator,
            "artifact": artifact_name,
            "evaluator_config": dict(evaluator_config),
            "inputs": _to_json_value(dict(inputs)),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw_identity).hexdigest()


def _to_json_value(value: Any) -> Any:
    """Преобразовать Pydantic/dataclass-контейнер в стабильное JSON-значение."""
    if hasattr(value, "model_dump"):
        return _to_json_value(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
