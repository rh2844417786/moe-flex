from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexmoe.bench.report import (
    RunEvidence,
    classify_support,
    latest_complete_run,
    validate_run_directory,
)


def _complete_evidence() -> RunEvidence:
    return RunEvidence(
        output_tokens_per_second=100.0,
        mapped_bytes=4096,
        h2d_bytes=2048,
        decompressed_bytes=2048,
        output_tokens_match=True,
        router_topk_match=True,
        weights_bit_exact=True,
    )


def test_missing_mechanism_evidence_is_inconclusive() -> None:
    evidence = RunEvidence(
        output_tokens_per_second=100.0,
        mapped_bytes=0,
        h2d_bytes=0,
        decompressed_bytes=0,
        output_tokens_match=True,
        router_topk_match=True,
        weights_bit_exact=True,
    )

    classification = classify_support(evidence)

    assert classification.status == "INCONCLUSIVE"
    assert "mapped_bytes" in classification.reasons


@pytest.mark.parametrize(
    "field",
    ["output_tokens_match", "router_topk_match", "weights_bit_exact"],
)
def test_correctness_mismatch_is_inconclusive(field: str) -> None:
    values = _complete_evidence().__dict__.copy()
    values[field] = False

    classification = classify_support(RunEvidence(**values), stressed_delta=0.2)

    assert classification.status == "INCONCLUSIVE"
    assert field in classification.reasons


def test_improvement_or_delayed_oom_supports_trend() -> None:
    evidence = _complete_evidence()

    assert classify_support(evidence, stressed_delta=0.1).status == "SUPPORTED"
    assert classify_support(evidence, delayed_oom=True).status == "SUPPORTED"


def test_flat_or_opposite_complete_trend_is_classified() -> None:
    evidence = _complete_evidence()

    assert classify_support(evidence, stressed_delta=0.0).status == "MIXED"
    assert classify_support(evidence, stressed_delta=-0.1).status == (
        "NOT_SUPPORTED"
    )


def test_invalid_throughput_is_inconclusive() -> None:
    values = _complete_evidence().__dict__.copy()
    values["output_tokens_per_second"] = 0.0

    assert classify_support(RunEvidence(**values), stressed_delta=0.1).status == (
        "INCONCLUSIVE"
    )


def test_validate_run_directory_requires_complete_strict_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "20260901T120000Z-aaaaaaaaaaaa"
    run_dir.mkdir()
    for filename in ("config.json", "environment.json"):
        (run_dir / filename).write_text("{}\n", encoding="utf-8")
    (run_dir / "events.jsonl").write_text('{"kind":"mapping"}\n')
    (run_dir / "state.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "repetitions": [
                    {"output_tokens_per_second": 100.0},
                    {"output_tokens_per_second": 110.0},
                    {"output_tokens_per_second": 90.0},
                ],
                "mechanism_counters": {
                    "mapped_bytes": 4096,
                    "h2d_bytes": 2048,
                    "decompressed_bytes": 2048,
                },
                "correctness": {
                    "output_tokens_match": True,
                    "router_topk_match": True,
                    "weights_bit_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_run_directory(run_dir)

    assert evidence.output_tokens_per_second == 100.0
    assert latest_complete_run(tmp_path) == run_dir


def test_validate_run_directory_rejects_unvalidated_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps({"status": "MEASURED_UNVALIDATED"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not COMPLETE"):
        validate_run_directory(run_dir)
