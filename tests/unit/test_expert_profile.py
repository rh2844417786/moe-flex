from __future__ import annotations

import copy
import math
from typing import cast

import pytest

from flexmoe.runtime.expert_profile import (
    ExpertProfile,
    select_resident_ids,
)

CONFIG_HASH = "a" * 64
IDENTITY_HASH = "b" * 64
INPUT_HASHES = ("c" * 64, "d" * 64)


def make_profile() -> ExpertProfile:
    return ExpertProfile(
        schema_version=1,
        geometry={
            "total_layers": 2,
            "num_experts": 4,
            "hidden_size": 16,
            "intermediate_size": 32,
        },
        model_config_sha256=CONFIG_HASH,
        model_identity_sha256=IDENTITY_HASH,
        tensor_parallel_size=4,
        calibration_input_hashes=INPUT_HASHES,
        counts=((9.0, 3.0, 2.0, 1.0), (8.0, 2.0, 1.0, 0.0)),
        forward_counts=(9, 8),
    )


def test_profile_round_trip_has_exact_public_json_schema() -> None:
    profile = make_profile()

    encoded = profile.to_dict()

    assert set(encoded) == {
        "schema_version",
        "geometry",
        "model_config_sha256",
        "model_identity_sha256",
        "tensor_parallel_size",
        "calibration_input_hashes",
        "counts",
        "forward_counts",
    }
    assert encoded["counts"] == [[9.0, 3.0, 2.0, 1.0], [8.0, 2.0, 1.0, 0.0]]
    assert ExpertProfile.from_dict(encoded) == profile


def test_validated_profile_geometry_cannot_be_mutated_through_public_field() -> None:
    profile = make_profile()

    with pytest.raises(TypeError):
        cast(dict[str, int], profile.geometry)["num_experts"] = 99

    assert profile.num_experts == 4
    assert ExpertProfile.from_dict(profile.to_dict()) == profile


def test_profile_validates_all_consumer_identity_fields() -> None:
    profile = make_profile()

    profile.validate(
        expected_geometry={
            "total_layers": 2,
            "num_experts": 4,
            "hidden_size": 16,
            "intermediate_size": 32,
        },
        expected_model_config_sha256=CONFIG_HASH,
        expected_model_identity_sha256=IDENTITY_HASH,
        expected_tensor_parallel_size=4,
        expected_calibration_input_hashes=INPUT_HASHES,
    )


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"expected_geometry": {"total_layers": 2, "num_experts": 4}}, "geometry"),
        ({"expected_model_config_sha256": "e" * 64}, "model config"),
        ({"expected_model_identity_sha256": "e" * 64}, "model identity"),
        ({"expected_tensor_parallel_size": 2}, "tensor parallel"),
        ({"expected_calibration_input_hashes": ("e" * 64,)}, "calibration"),
    ],
)
def test_profile_consumer_mismatch_fails_closed(
    expected: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_profile().validate(**expected)  # type: ignore[arg-type]


def test_resident_selection_is_deterministic_for_ties_and_boundaries() -> None:
    counts = [[4, 4, 1, 0], [0, 2, 2, 2]]

    assert select_resident_ids(counts, 0.0) == ((), ())
    assert select_resident_ids(counts, 0.5) == ((0, 1), (1, 2))
    assert select_resident_ids(counts, 1.0) == ((0, 1, 2, 3), (1, 2, 3, 0))


@pytest.mark.parametrize("ratio", [-0.01, 1.01, math.nan, math.inf])
def test_resident_selection_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        select_resident_ids([[1, 0]], ratio)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data.update(schema_version=True), "schema_version"),
        (lambda data: data.update(model_config_sha256="not-a-hash"), "SHA-256"),
        (lambda data: data.update(tensor_parallel_size=0), "tensor_parallel_size"),
        (lambda data: data.update(counts=[[1.0]]), "shape"),
        (
            lambda data: data.update(
                counts=[[math.nan, 3, 2, 1], [8, 2, 1, 0]]
            ),
            "finite",
        ),
        (lambda data: data.update(forward_counts=[9]), "forward_counts"),
        (lambda data: data.update(forward_counts=[2, 8]), "exceeds"),
        (
            lambda data: data.update(
                calibration_input_hashes=[INPUT_HASHES[0], INPUT_HASHES[0]]
            ),
            "unique",
        ),
        (lambda data: data.update(unknown_field=1), "fields"),
    ],
)
def test_corrupt_profile_payload_fails_closed(mutation: object, message: str) -> None:
    data = copy.deepcopy(make_profile().to_dict())
    mutation(data)  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError), match=message):
        ExpertProfile.from_dict(data)


def test_profile_rejects_malformed_geometry_and_counts_at_construction() -> None:
    data = make_profile().to_dict()
    data["geometry"] = {"total_layers": 2, "num_experts": 4, "hidden_size": 0}
    with pytest.raises(ValueError, match="geometry"):
        ExpertProfile.from_dict(data)

    with pytest.raises(ValueError, match="non-negative"):
        select_resident_ids([[1, -1]], 0.5)
