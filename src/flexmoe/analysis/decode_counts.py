"""Standard-library contracts for validating and summarizing decode counts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from math import ceil
from typing import cast

_PHASES = frozenset({"decode", "mixed", "prefill"})


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def validate_activation_row(
    row: Mapping[str, object],
    *,
    layers: int,
    experts: int,
    top_k: int,
) -> dict[str, object]:
    """Validate one all-token expert histogram and return a detached row."""

    layer_count = _positive_int(layers, "layers")
    expert_count = _positive_int(experts, "experts")
    selections_per_token = _positive_int(top_k, "top_k")

    try:
        step_value = row["step"]
        layer_value = row["layer"]
        phase_value = row["phase"]
        batch_value = row["actual_batch"]
        histogram_value = row["histogram"]
    except KeyError as error:
        raise ValueError(f"activation row is missing {error.args[0]}") from error

    if type(step_value) is not int or step_value < 0:
        raise ValueError("step must be a nonnegative int")
    if type(layer_value) is not int or not 0 <= layer_value < layer_count:
        raise ValueError(f"layer must be an int in [0, {layer_count})")
    if not isinstance(phase_value, str) or phase_value not in _PHASES:
        raise ValueError(f"phase must be one of {sorted(_PHASES)}")
    actual_batch = _positive_int(batch_value, "actual_batch")
    if not isinstance(histogram_value, list):
        raise TypeError("histogram must be a list of nonnegative ints")
    if len(histogram_value) != expert_count:
        raise ValueError(
            f"histogram length must be {expert_count}, got {len(histogram_value)}"
        )
    if any(type(count) is not int or count < 0 for count in histogram_value):
        raise ValueError("histogram counts must be nonnegative ints")
    histogram = cast(list[int], histogram_value)
    expected = actual_batch * selections_per_token
    if sum(histogram) != expected:
        raise ValueError(f"histogram sum must be {expected}")

    return {
        "step": step_value,
        "layer": layer_value,
        "phase": phase_value,
        "actual_batch": actual_batch,
        "histogram": list(histogram),
    }


def _nearest_rank(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def summarize_activations(
    rows: Iterable[Mapping[str, object]],
    *,
    target_batch: int,
    layers: int,
    experts: int,
    top_k: int,
    min_steps: int = 64,
) -> dict[str, object]:
    """Summarize complete, pure-decode steps at exactly ``target_batch``."""

    target = _positive_int(target_batch, "target_batch")
    layer_count = _positive_int(layers, "layers")
    expert_count = _positive_int(experts, "experts")
    selections_per_token = _positive_int(top_k, "top_k")
    minimum = _positive_int(min_steps, "min_steps")

    normalized = [
        validate_activation_row(
            row,
            layers=layer_count,
            experts=expert_count,
            top_k=selections_per_token,
        )
        for row in rows
    ]
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    phase_rows: Counter[str] = Counter()
    decode_batch_rows: Counter[int] = Counter()
    for row in normalized:
        step = cast(int, row["step"])
        phase = cast(str, row["phase"])
        grouped[step].append(row)
        phase_rows[phase] += 1
        if phase == "decode":
            decode_batch_rows[cast(int, row["actual_batch"])] += 1

    phase_steps: Counter[str] = Counter()
    decode_batch_steps: Counter[int] = Counter()
    accepted: list[list[dict[str, object]]] = []
    non_decode = 0
    non_target = 0
    expected_layers = set(range(layer_count))
    for step, step_rows in sorted(grouped.items()):
        row_layers = [cast(int, row["layer"]) for row in step_rows]
        if len(set(row_layers)) != len(row_layers):
            raise ValueError(f"step {step} has a duplicate layer")
        missing = expected_layers - set(row_layers)
        if missing:
            raise ValueError(f"step {step} has missing layer(s): {sorted(missing)}")
        phases = {cast(str, row["phase"]) for row in step_rows}
        if len(phases) != 1:
            raise ValueError(f"step {step} has phase disagreement")
        batches = {cast(int, row["actual_batch"]) for row in step_rows}
        if len(batches) != 1:
            raise ValueError(f"step {step} has actual_batch disagreement")

        phase = next(iter(phases))
        actual_batch = next(iter(batches))
        phase_steps[phase] += 1
        if phase != "decode":
            non_decode += 1
            continue
        decode_batch_steps[actual_batch] += 1
        if actual_batch != target:
            non_target += 1
            continue
        accepted.append(step_rows)

    accepted_count = len(accepted)
    if accepted_count == 0:
        status = "unreached"
    elif accepted_count < minimum:
        status = "insufficient-window"
    else:
        status = "complete"

    per_layer: list[dict[str, object]] = []
    if status == "complete":
        by_layer = {
            layer: [
                next(row for row in step_rows if row["layer"] == layer)
                for step_rows in accepted
            ]
            for layer in range(layer_count)
        }
        token_count = accepted_count * target
        selection_count = token_count * selections_per_token
        for layer, layer_rows in by_layer.items():
            histogram_total = [
                sum(cast(list[int], row["histogram"])[expert] for row in layer_rows)
                for expert in range(expert_count)
            ]
            coverage = [
                sum(count > 0 for count in cast(list[int], row["histogram"]))
                for row in layer_rows
            ]
            per_layer.append(
                {
                    "layer": layer,
                    "histogram_total": histogram_total,
                    "selection_probability": [
                        count / selection_count for count in histogram_total
                    ],
                    "coverage_mean": sum(coverage) / len(coverage),
                    "coverage_p50": _nearest_rank(coverage, 0.50),
                    "coverage_p95": _nearest_rank(coverage, 0.95),
                    "coverage_max": max(coverage),
                    "token_count": token_count,
                    "selection_count": selection_count,
                }
            )

    return {
        "status": status,
        "accepted_steps": accepted_count,
        "min_steps": minimum,
        "target_batch": target,
        "observed_rows": len(normalized),
        "complete_steps": len(grouped),
        "phase_row_counts": dict(sorted(phase_rows.items())),
        "phase_step_counts": dict(sorted(phase_steps.items())),
        "decode_batch_row_counts": {
            str(batch): count for batch, count in sorted(decode_batch_rows.items())
        },
        "decode_batch_step_counts": {
            str(batch): count for batch, count in sorted(decode_batch_steps.items())
        },
        "excluded_steps": {
            "non_decode_phase": non_decode,
            "non_target_batch": non_target,
        },
        "per_layer": per_layer,
    }


__all__ = ["summarize_activations", "validate_activation_row"]
