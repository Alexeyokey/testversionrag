from __future__ import annotations

import pytest

from rag_app.cli import _build_parser


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
