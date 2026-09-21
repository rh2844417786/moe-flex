import gzip
import json

import pytest


def identity():
    return {
        "commit": "a" * 40,
        "model_identity_sha256": "b" * 64,
        "profile_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "geometry": {"total_layers": 6, "num_experts": 16, "top_k": 2},
        "seed": 20260912,
        "context_length": 4096,
        "output_length": 256,
        "actual_batch": 16,
    }


def rows_for_two_steps():
    routes = {
        0: (0, 1),
        1: (1, 2),
        2: (2, 3),
        3: (3, 6),
        4: (4, 7),
        5: (1, 9),
    }
    return [
        {
            "step_id": step,
            "layer_id": layer,
            "actual_batch": 16,
            "actual_expert_ids": list(routes[layer]),
        }
        for step in (10, 11)
        for layer in range(6)
    ]


def test_oracle_trace_returns_only_horizon_one_and_two():
    from flexmoe.runtime.oracle_trace import OracleTrace

    trace = OracleTrace.from_rows(rows_for_two_steps(), identity())

    assert trace.future(step=10, layer=3, horizon=1) == (4, 7)
    assert trace.future(step=10, layer=3, horizon=2) == (1, 9)
    assert trace.future(step=10, layer=5, horizon=1) is None
    with pytest.raises(ValueError, match="horizon"):
        trace.future(step=10, layer=3, horizon=3)


def test_route_mismatch_is_explicit_and_never_changes_actual_ids():
    from flexmoe.runtime.oracle_trace import OracleTrace

    trace = OracleTrace.from_rows(rows_for_two_steps(), identity())
    match = trace.validate_actual(step=10, layer=4, ids=(4, 8))

    assert match.status == "route-mismatch"
    assert match.predicted_ids == (4, 7)
    assert match.actual_ids == (4, 8)
    assert match.missing_ids == (8,)
    assert match.unused_ids == (7,)


def test_forced_omission_changes_prediction_once_but_preserves_reference_route():
    from flexmoe.runtime.oracle_trace import OracleTrace

    original = OracleTrace.from_rows(rows_for_two_steps(), identity())
    omitted = original.with_forced_omission(step=10, layer=4, expert=7)

    assert original.future(step=10, layer=3, horizon=1) == (4, 7)
    assert omitted.future(step=10, layer=3, horizon=1) == (4,)
    assert omitted.validate_actual(step=10, layer=4, ids=(4, 7)).status == "match"
    assert omitted.forced_omission == (10, 4, 7)
    with pytest.raises(ValueError, match="present"):
        original.with_forced_omission(step=10, layer=4, expert=8)


def test_rank0_profile_loader_requires_exact_identity_and_complete_layers(tmp_path):
    from flexmoe.runtime.oracle_trace import OracleTrace

    artifact = {
        "artifact_kind": "decode-logical-cache-trace",
        "contract": identity(),
        "rows": rows_for_two_steps(),
        "logical_sha256": "e" * 64,
    }
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(artifact, stream)

    trace = OracleTrace.from_rank0_profile(path, identity(), minimum_steps=2)
    assert trace.step_range == (10, 11)
    assert trace.logical_sha256 == "e" * 64

    wrong = identity()
    wrong["input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        OracleTrace.from_rank0_profile(path, wrong, minimum_steps=2)

    artifact["rows"].pop()
    with gzip.open(path, "wt") as stream:
        json.dump(artifact, stream)
    with pytest.raises(ValueError, match="complete"):
        OracleTrace.from_rank0_profile(path, identity(), minimum_steps=2)
