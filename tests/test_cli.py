from __future__ import annotations

import pytest

from rag_app.cli import _build_parser


def test_check_vllm_command_accepts_custom_prompt() -> None:
    parser = _build_parser()

    args = parser.parse_args(["check-vllm", "--prompt", "ping"])

    assert args.command == "check-vllm"
    assert args.prompt == "ping"


def test_evaluation_accepts_only_artifact_cache_flags() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "evaluate-ragas",
            "testset.jsonl",
            "--artifact-cache-dir",
            "artifacts",
            "--refresh-artifact-cache",
        ]
    )

    assert str(args.artifact_cache_dir) == "artifacts"
    assert args.refresh_artifact_cache is True


def test_evaluation_accepts_context_precision_skip_flag() -> None:
    parser = _build_parser()

    ragas_args = parser.parse_args(
        ["evaluate-ragas", "testset.jsonl", "--skip-context-precision"]
    )
    tuning_args = parser.parse_args(
        ["tune-retrieval", "testset.jsonl", "--skip-context-precision"]
    )

    assert ragas_args.skip_context_precision is True
    assert tuning_args.skip_context_precision is True


def test_retrieval_tuning_accepts_custom_vector_weights() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "tune-retrieval",
            "testset.jsonl",
            "--vector-weights",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
        ]
    )

    assert args.command == "tune-retrieval"
    assert args.vector_weights == [0.2, 0.4, 0.6, 0.8]


@pytest.mark.parametrize(
    "legacy_flag",
    ["--metric-cache-dir", "--no-metric-cache", "--refresh-metric-cache"],
)
def test_legacy_metric_cache_flags_are_rejected(legacy_flag: str) -> None:
    parser = _build_parser()
    arguments = ["evaluate-ragas", "testset.jsonl", legacy_flag]
    if legacy_flag == "--metric-cache-dir":
        arguments.append("legacy-cache")

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)
