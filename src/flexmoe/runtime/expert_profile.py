"""Typed, privacy-preserving calibration profiles for hot-expert selection."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "geometry",
        "model_config_sha256",
        "model_identity_sha256",
        "tensor_parallel_size",
        "calibration_input_hashes",
        "counts",
        "forward_counts",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _sequence(value: object, name: str) -> Sequence[object]:
    if not _is_sequence(value):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _normalise_geometry(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("geometry must be a mapping")
    geometry: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("geometry keys must be non-empty strings")
        geometry[key] = _positive_int(item, f"geometry[{key!r}]")
    for required in ("total_layers", "num_experts"):
        if required not in geometry:
            raise ValueError(f"geometry must contain {required}")
    return dict(sorted(geometry.items()))


def _normalise_counts(
    value: object,
    *,
    total_layers: int | None = None,
    num_experts: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for layer, raw_row in enumerate(_sequence(value, "counts")):
        row_values = _sequence(raw_row, f"counts row {layer}")
        row: list[float] = []
        for expert, raw_count in enumerate(row_values):
            if isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
                raise TypeError(
                    f"counts[{layer}][{expert}] must be a real number"
                )
            count = float(raw_count)
            if not math.isfinite(count):
                raise ValueError("counts must contain only finite values")
            if count < 0.0:
                raise ValueError("counts must be non-negative")
            row.append(count)
        rows.append(tuple(row))
    if not rows or not rows[0]:
        raise ValueError("counts shape must have at least one layer and expert")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("counts shape must be rectangular")
    if total_layers is not None and len(rows) != total_layers:
        raise ValueError("counts shape does not match geometry total_layers")
    if num_experts is not None and width != num_experts:
        raise ValueError("counts shape does not match geometry num_experts")
    return tuple(rows)


def _normalise_forward_counts(
    value: object, *, total_layers: int
) -> tuple[int, ...]:
    result: list[int] = []
    for count in _sequence(value, "forward_counts"):
        if type(count) is not int or count < 0:
            raise ValueError("forward_counts must contain non-negative integers")
        result.append(count)
    if len(result) != total_layers:
        raise ValueError("forward_counts shape does not match geometry total_layers")
    return tuple(result)


def _normalise_hashes(value: object) -> tuple[str, ...]:
    hashes = tuple(
        _sha256(item, "calibration input hash")
        for item in _sequence(value, "calibration_input_hashes")
    )
    if not hashes:
        raise ValueError("calibration_input_hashes must not be empty")
    if len(set(hashes)) != len(hashes):
        raise ValueError("calibration_input_hashes must be unique")
    return hashes


def _normalise_ratio(ratio: float) -> float:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TypeError("ratio must be a real number")
    value = float(ratio)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("ratio must be finite and between 0 and 1")
    return value


def select_resident_ids(
    counts: Sequence[Sequence[float]], ratio: float
) -> tuple[tuple[int, ...], ...]:
    """Select each layer's hottest experts with stable expert-id tie breaks."""

    rows = _normalise_counts(counts)
    selected_count = math.floor(len(rows[0]) * _normalise_ratio(ratio))
    return tuple(
        tuple(
            sorted(range(len(row)), key=lambda expert: (-row[expert], expert))[
                :selected_count
            ]
        )
        for row in rows
    )


@dataclass(frozen=True)
class ExpertProfile:
    """Versioned calibration data safe to serialize without source prompts."""

    schema_version: int
    geometry: Mapping[str, int]
    model_config_sha256: str
    model_identity_sha256: str
    tensor_parallel_size: int
    calibration_input_hashes: Sequence[str]
    counts: Sequence[Sequence[float]]
    forward_counts: Sequence[int]

    def __post_init__(self) -> None:
        geometry = _normalise_geometry(self.geometry)
        counts = _normalise_counts(
            self.counts,
            total_layers=geometry["total_layers"],
            num_experts=geometry["num_experts"],
        )
        forward_counts = _normalise_forward_counts(
            self.forward_counts, total_layers=geometry["total_layers"]
        )
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        config_hash = _sha256(self.model_config_sha256, "model config")
        identity_hash = _sha256(self.model_identity_sha256, "model identity")
        tp_size = _positive_int(self.tensor_parallel_size, "tensor_parallel_size")
        hashes = _normalise_hashes(self.calibration_input_hashes)
        for layer, row in enumerate(counts):
            if any(count > forward_counts[layer] for count in row):
                raise ValueError(
                    f"counts for layer {layer} exceeds its forward_counts value"
                )

        object.__setattr__(self, "geometry", MappingProxyType(geometry))
        object.__setattr__(self, "model_config_sha256", config_hash)
        object.__setattr__(self, "model_identity_sha256", identity_hash)
        object.__setattr__(self, "tensor_parallel_size", tp_size)
        object.__setattr__(self, "calibration_input_hashes", hashes)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "forward_counts", forward_counts)

    @property
    def total_layers(self) -> int:
        return self.geometry["total_layers"]

    @property
    def num_experts(self) -> int:
        return self.geometry["num_experts"]

    def validate(
        self,
        *,
        expected_geometry: Mapping[str, int] | None = None,
        expected_model_config_sha256: str | None = None,
        expected_model_identity_sha256: str | None = None,
        expected_tensor_parallel_size: int | None = None,
        expected_calibration_input_hashes: Sequence[str] | None = None,
    ) -> None:
        """Fail closed when consumer-provided model or input identity differs."""

        if expected_geometry is not None and self.geometry != _normalise_geometry(
            expected_geometry
        ):
            raise ValueError("profile geometry does not match the consumer")
        if (
            expected_model_config_sha256 is not None
            and self.model_config_sha256
            != _sha256(expected_model_config_sha256, "expected model config")
        ):
            raise ValueError("profile model config hash does not match the consumer")
        if (
            expected_model_identity_sha256 is not None
            and self.model_identity_sha256
            != _sha256(expected_model_identity_sha256, "expected model identity")
        ):
            raise ValueError("profile model identity hash does not match the consumer")
        if (
            expected_tensor_parallel_size is not None
            and self.tensor_parallel_size
            != _positive_int(
                expected_tensor_parallel_size, "expected tensor_parallel_size"
            )
        ):
            raise ValueError("profile tensor parallel size does not match the consumer")
        if (
            expected_calibration_input_hashes is not None
            and tuple(self.calibration_input_hashes)
            != _normalise_hashes(expected_calibration_input_hashes)
        ):
            raise ValueError("profile calibration input hashes do not match the consumer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "geometry": dict(self.geometry),
            "model_config_sha256": self.model_config_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "tensor_parallel_size": self.tensor_parallel_size,
            "calibration_input_hashes": list(self.calibration_input_hashes),
            "counts": [list(row) for row in self.counts],
            "forward_counts": list(self.forward_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExpertProfile:
        if not isinstance(data, Mapping):
            raise TypeError("profile must be a mapping")
        if set(data) != _REQUIRED_FIELDS:
            missing = sorted(_REQUIRED_FIELDS - set(data))
            extra = sorted(set(data) - _REQUIRED_FIELDS)
            raise ValueError(f"profile fields mismatch: missing={missing}, extra={extra}")
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            geometry=data["geometry"],  # type: ignore[arg-type]
            model_config_sha256=data["model_config_sha256"],  # type: ignore[arg-type]
            model_identity_sha256=data["model_identity_sha256"],  # type: ignore[arg-type]
            tensor_parallel_size=data["tensor_parallel_size"],  # type: ignore[arg-type]
            calibration_input_hashes=data["calibration_input_hashes"],  # type: ignore[arg-type]
            counts=data["counts"],  # type: ignore[arg-type]
            forward_counts=data["forward_counts"],  # type: ignore[arg-type]
        )


__all__ = ["ExpertProfile", "select_resident_ids"]
