from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

if TYPE_CHECKING:
    from rag_app.config import Settings


CACHE_SCHEMA_VERSION = 1
EVALUATOR_PROMPT_REVISION = 1


@dataclass(frozen=True, slots=True)
class CachedMetricScore:
    """Успешная оценка одной метрики, прочитанная из постоянного кэша."""

    value: float
    reason: str | None = None


class MetricScoreCache:
    """Хранить каждую оценку отдельным JSON-файлом с ключом по её входам."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def get(
        self,
        *,
        evaluator: str,
        metric_name: str,
        evaluator_config: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> CachedMetricScore | None:
        identity = _cache_identity(
            evaluator=evaluator,
            metric_name=metric_name,
            evaluator_config=evaluator_config,
            inputs=inputs,
        )
        cache_path = self._entry_path(evaluator, metric_name, identity)
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != CACHE_SCHEMA_VERSION
                or payload.get("identity") != identity
            ):
                return None
            value = float(payload["value"])
            if not math.isfinite(value):
                return None
            reason = payload.get("reason")
            return CachedMetricScore(
                value=value,
                reason=str(reason) if reason else None,
            )
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # Повреждённая или устаревшая запись считается промахом: метрика
            # пересчитается и затем перезапишет корректный файл.
            return None

    def put(
        self,
        *,
        evaluator: str,
        metric_name: str,
        evaluator_config: Mapping[str, Any],
        inputs: Mapping[str, Any],
        value: float,
        reason: str | None = None,
    ) -> Path:
        identity = _cache_identity(
            evaluator=evaluator,
            metric_name=metric_name,
            evaluator_config=evaluator_config,
            inputs=inputs,
        )
        cache_path = self._entry_path(evaluator, metric_name, identity)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity": identity,
            "evaluator": evaluator,
            "metric": metric_name,
            "evaluator_config": dict(evaluator_config),
            "value": value,
            "reason": reason,
        }
        # replace() не оставляет частично записанный JSON, если процесс прервётся
        # в момент сохранения оценки.
        temporary_path = cache_path.with_suffix(
            f".{os.getpid()}.{uuid4().hex}.tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
        return cache_path

    def _entry_path(self, evaluator: str, metric_name: str, identity: str) -> Path:
        return self.directory / evaluator / metric_name / f"{identity}.json"


def evaluator_cache_config(settings: Settings, evaluator: str) -> dict[str, Any]:
    """Описать всё, что способно изменить оценку при тех же текстах."""
    package_name = "ragas" if evaluator == "ragas" else "deepeval"
    try:
        package_version = version(package_name)
    except PackageNotFoundError:
        package_version = "unknown"

    config: dict[str, Any] = {
        "judge_model": settings.ragas_judge_model or settings.generation_model,
        "judge_temperature": 0.0,
        "judge_max_tokens": settings.ragas_max_tokens,
        "package_version": package_version,
        "prompt_revision": EVALUATOR_PROMPT_REVISION,
    }
    if evaluator == "ragas":
        # Answer Relevancy в RAGAS использует embeddings, поэтому смена модели
        # должна инвалидировать старую оценку.
        config["embedding_model"] = settings.embedding_model
    return config


def metric_inputs(
    metric_name: str,
    *,
    question: str,
    reference: str,
    response: str,
    retrieved_contexts: tuple[str, ...],
) -> dict[str, Any]:
    """Оставить в ключе только те поля, которые реально использует метрика."""
    if metric_name == "faithfulness":
        return {
            "question": question,
            "response": response,
            "retrieved_contexts": list(retrieved_contexts),
        }
    if metric_name in {"context_precision", "context_recall"}:
        return {
            "question": question,
            "reference": reference,
            "retrieved_contexts": list(retrieved_contexts),
        }
    if metric_name == "answer_accuracy":
        return {
            "question": question,
            "reference": reference,
            "response": response,
        }
    if metric_name == "answer_relevancy":
        return {
            "question": question,
            "response": response,
        }
    raise ValueError(f"Неизвестная judge-метрика: {metric_name}")


def _cache_identity(
    *,
    evaluator: str,
    metric_name: str,
    evaluator_config: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> str:
    raw_identity = json.dumps(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "evaluator": evaluator,
            "metric": metric_name,
            "evaluator_config": dict(evaluator_config),
            "inputs": dict(inputs),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw_identity).hexdigest()
