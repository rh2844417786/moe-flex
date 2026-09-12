from __future__ import annotations

from copy import deepcopy

import pytest

from flexmoe.analysis.decode_counts import (
    summarize_activations,
    validate_activation_row,
)


def activation_rows() -> list[dict[str, object]]:
    return [
        {
            "step": step,
            "layer": layer,
            "phase": "decode",
            "actual_batch": 2,
            "histogram": [2, 1, 1, 0] if step == 0 else [0, 1, 1, 2],
        }
        for step in (0, 1)
        for layer in (0, 1)
    ]


def test_validate_activation_row_returns_a_detached_normalized_row() -> None:
    source: dict[str, object] = {
        "step": 0,
        "layer": 1,
        "phase": "decode",
        "actual_batch": 2,
        "histogram": [2, 1, 1, 0],
    }

    normalized = validate_activation_row(
        source, layers=2, experts=4, top_k=2
    )

    assert normalized == source
    assert normalized is not source
    assert normalized["histogram"] is not source["histogram"]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("step", True, "step"),
        ("step", -1, "step"),
        ("layer", 2, "layer"),
        ("phase", "generation", "phase"),
        ("actual_batch", False, "actual_batch"),
        ("actual_batch", 0, "actual_batch"),
        ("histogram", [True, 1, 1, 1], "histogram"),
        ("histogram", [2, 1, -1, 2], "histogram"),
        ("histogram", [2, 1, float("nan"), 1], "histogram"),
        ("histogram", [2, 1, 1], "length"),
        ("histogram", [1, 1, 1, 0], "sum"),
    ],
)
def test_validate_activation_row_rejects_malformed_values(
    field: str, replacement: object, message: str
) -> None:
    row = activation_rows()[0]
    row[field] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        validate_activation_row(row, layers=2, experts=4, top_k=2)


def test_validate_activation_row_rejects_more_selections_than_batch_per_expert(
) -> None:
    row = {
        "step": 0,
        "layer": 0,
        "phase": "decode",
        "actual_batch": 2,
        "histogram": [4, 0, 0, 0],
    }

    with pytest.raises(ValueError, match="cannot exceed actual_batch"):
        validate_activation_row(row, layers=2, experts=4, top_k=2)


def test_validate_activation_row_rejects_top_k_greater_than_experts() -> None:
    row = {
        "step": 0,
        "layer": 0,
        "phase": "decode",
        "actual_batch": 1,
        "histogram": [2],
    }

    with pytest.raises(ValueError, match="top_k cannot exceed experts"):
        validate_activation_row(row, layers=1, experts=1, top_k=2)


def test_summarize_counts_token_expert_edges_and_per_step_coverage() -> None:
    result = summarize_activations(
        activation_rows(),
        target_batch=2,
        layers=2,
        experts=4,
        top_k=2,
        min_steps=2,
    )

    assert result["status"] == "complete"
    assert result["accepted_steps"] == 2
    assert result["complete_steps"] == 2
    assert result["phase_step_counts"] == {"decode": 2}
    assert result["decode_batch_step_counts"] == {"2": 2}
    assert result["excluded_steps"] == {
        "non_decode_phase": 0,
        "non_target_batch": 0,
    }
    assert result["per_layer"] == [
        {
            "layer": layer,
            "histogram_total": [2, 2, 2, 2],
            "selection_probability": [0.25, 0.25, 0.25, 0.25],
            "coverage_mean": 3.0,
            "coverage_p50": 3.0,
            "coverage_p95": 3.0,
            "coverage_max": 3,
            "token_count": 4,
            "selection_count": 8,
        }
        for layer in (0, 1)
    ]


def test_summarize_preserves_phase_and_batch_exclusion_denominators() -> None:
    rows = activation_rows()
    rows.extend(
        {
            "step": 2,
            "layer": layer,
            "phase": "prefill",
            "actual_batch": 2,
            "histogram": [1, 1, 1, 1],
        }
        for layer in (0, 1)
    )
    rows.extend(
        {
            "step": 3,
            "layer": layer,
            "phase": "mixed",
            "actual_batch": 2,
            "histogram": [1, 1, 1, 1],
        }
        for layer in (0, 1)
    )
    rows.extend(
        {
            "step": 4,
            "layer": layer,
            "phase": "decode",
            "actual_batch": 3,
            "histogram": [2, 2, 1, 1],
        }
        for layer in (0, 1)
    )

    result = summarize_activations(
        rows,
        target_batch=2,
        layers=2,
        experts=4,
        top_k=2,
        min_steps=2,
    )

    assert result["accepted_steps"] == 2
    assert result["complete_steps"] == 5
    assert result["observed_rows"] == 10
    assert result["phase_row_counts"] == {
        "decode": 6,
        "mixed": 2,
        "prefill": 2,
    }
    assert result["phase_step_counts"] == {
        "decode": 3,
        "mixed": 1,
        "prefill": 1,
    }
    assert result["decode_batch_row_counts"] == {"2": 4, "3": 2}
    assert result["decode_batch_step_counts"] == {"2": 2, "3": 1}
    assert result["excluded_steps"] == {
        "non_decode_phase": 2,
        "non_target_batch": 1,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate layer"),
        ("missing", "missing layer"),
        ("phase", "phase disagreement"),
        ("batch", "actual_batch disagreement"),
    ],
)
def test_summarize_rejects_non_rectangular_or_disagreeing_steps(
    mutation: str, message: str
) -> None:
    rows = activation_rows()
    if mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop(0)
    elif mutation == "phase":
        rows[1]["phase"] = "mixed"
    else:
        rows[1]["actual_batch"] = 3
        rows[1]["histogram"] = [2, 2, 1, 1]

    with pytest.raises(ValueError, match=message):
        summarize_activations(
            rows,
            target_batch=2,
            layers=2,
            experts=4,
            top_k=2,
            min_steps=2,
        )


def test_summarize_with_no_matching_steps_is_unreached() -> None:
    rows = [
        {
            **row,
            "phase": "prefill",
        }
        for row in activation_rows()
    ]

    result = summarize_activations(
        rows,
        target_batch=2,
        layers=2,
        experts=4,
        top_k=2,
        min_steps=2,
    )

    assert result["status"] == "unreached"
    assert result["accepted_steps"] == 0
    assert result["per_layer"] == []


def test_summarize_with_short_matching_window_has_no_layer_statistics() -> None:
    result = summarize_activations(
        activation_rows()[:2],
        target_batch=2,
        layers=2,
        experts=4,
        top_k=2,
        min_steps=2,
    )

    assert result["status"] == "insufficient-window"
    assert result["accepted_steps"] == 1
    assert result["per_layer"] == []
