"""Immutable logical route traces for constrained expert-prefetch oracles."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ids(value: object, *, experts: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    ids = tuple(value)
    if (
        not ids
        or any(type(expert) is not int or not 0 <= expert < experts for expert in ids)
        or tuple(sorted(set(ids))) != ids
    ):
        raise ValueError(f"{name} must be sorted unique in-range expert IDs")
    return ids


@dataclass(frozen=True)
class RouteMatch:
    status: str
    predicted_ids: tuple[int, ...]
    actual_ids: tuple[int, ...]
    missing_ids: tuple[int, ...]
    unused_ids: tuple[int, ...]


@dataclass(frozen=True)
class OracleTrace:
    identity: dict[str, Any]
    _actual: dict[tuple[int, int], tuple[int, ...]]
    _predictions: dict[tuple[int, int], tuple[int, ...]]
    step_range: tuple[int, int]
    logical_sha256: str
    forced_omission: tuple[int, int, int] | None = None

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        identity: dict[str, Any],
        *,
        logical_sha256: str | None = None,
    ) -> OracleTrace:
        geometry = identity.get("geometry")
        if not isinstance(geometry, dict):
            raise TypeError("oracle identity geometry is required")
        layers, experts = geometry.get("total_layers"), geometry.get("num_experts")
        batch = identity.get("actual_batch")
        if (
            type(layers) is not int
            or layers <= 0
            or type(experts) is not int
            or experts <= 0
            or type(batch) is not int
            or batch <= 0
        ):
            raise ValueError("oracle identity geometry or actual batch is invalid")
        if not rows or len(rows) % layers:
            raise ValueError("oracle trace does not contain complete layers")
        first = rows[0].get("step_id")
        if type(first) is not int or first < 0:
            raise ValueError("oracle trace start step is invalid")
        actual: dict[tuple[int, int], tuple[int, ...]] = {}
        canonical = []
        for index, row in enumerate(rows):
            step, layer = row.get("step_id"), row.get("layer_id")
            expected_step, expected_layer = first + index // layers, index % layers
            if (
                step != expected_step
                or layer != expected_layer
                or row.get("actual_batch") != batch
            ):
                raise ValueError(
                    "oracle trace is not a complete contiguous layer sequence"
                )
            expert_ids = _ids(
                row.get("actual_expert_ids"), experts=experts, name="actual_expert_ids"
            )
            actual[(step, layer)] = expert_ids
            canonical.append(
                {
                    "step_id": step,
                    "layer_id": layer,
                    "actual_batch": batch,
                    "actual_expert_ids": list(expert_ids),
                }
            )
        last = first + len(rows) // layers - 1
        digest = logical_sha256 or _digest(canonical)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("logical trace SHA-256 is invalid")
        return cls(
            identity=dict(identity),
            _actual=actual,
            _predictions=dict(actual),
            step_range=(first, last),
            logical_sha256=digest,
        )

    @classmethod
    def from_rank0_profile(
        cls,
        path: Path,
        identity: dict[str, Any],
        *,
        minimum_steps: int = 64,
    ) -> OracleTrace:
        if type(minimum_steps) is not int or minimum_steps <= 0:
            raise ValueError("minimum_steps must be a positive integer")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            artifact = json.load(stream)
        if artifact.get("artifact_kind") != "decode-logical-cache-trace":
            raise ValueError("oracle trace artifact kind differs")
        contract = artifact.get("oracle_identity", artifact.get("contract"))
        if not isinstance(contract, dict) or any(
            contract.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("oracle trace identity differs")
        rows = artifact.get("rows")
        if not isinstance(rows, list):
            raise TypeError("oracle trace rows are unavailable")
        trace = cls.from_rows(
            rows, identity, logical_sha256=artifact.get("logical_sha256")
        )
        if trace.step_range[1] - trace.step_range[0] + 1 < minimum_steps:
            raise ValueError("oracle trace lacks the required complete step window")
        return trace

    def future(self, *, step: int, layer: int, horizon: int) -> tuple[int, ...] | None:
        if horizon not in (1, 2):
            raise ValueError("oracle prefetch horizon must be 1 or 2")
        layers = self.identity["geometry"]["total_layers"]
        target = layer + horizon
        if target >= layers:
            return None
        return self._predictions.get((step, target))

    def validate_actual(
        self, *, step: int, layer: int, ids: tuple[int, ...]
    ) -> RouteMatch:
        experts = self.identity["geometry"]["num_experts"]
        actual = _ids(ids, experts=experts, name="runtime actual IDs")
        predicted = self._actual.get((step, layer))
        if predicted is None:
            return RouteMatch(
                status="trace-unavailable",
                predicted_ids=(),
                actual_ids=actual,
                missing_ids=actual,
                unused_ids=(),
            )
        missing = tuple(sorted(set(actual) - set(predicted)))
        unused = tuple(sorted(set(predicted) - set(actual)))
        return RouteMatch(
            status="match" if not missing and not unused else "route-mismatch",
            predicted_ids=predicted,
            actual_ids=actual,
            missing_ids=missing,
            unused_ids=unused,
        )

    def with_forced_omission(
        self, *, step: int, layer: int, expert: int
    ) -> OracleTrace:
        if self.forced_omission is not None:
            raise ValueError("oracle trace already has a forced omission")
        key = (step, layer)
        predicted = self._predictions.get(key)
        if predicted is None or expert not in predicted:
            raise ValueError(
                "forced omission expert is not present in the target route"
            )
        predictions = dict(self._predictions)
        predictions[key] = tuple(value for value in predicted if value != expert)
        return replace(
            self,
            _predictions=predictions,
            forced_omission=(step, layer, expert),
        )

    def to_dict(self) -> dict[str, Any]:
        rows = [
            {
                "step_id": step,
                "layer_id": layer,
                "actual_batch": self.identity["actual_batch"],
                "actual_expert_ids": list(ids),
                "predicted_expert_ids": list(self._predictions[(step, layer)]),
            }
            for (step, layer), ids in sorted(self._actual.items())
        ]
        return {
            "identity": self.identity,
            "step_range": list(self.step_range),
            "logical_sha256": self.logical_sha256,
            "forced_omission": list(self.forced_omission)
            if self.forced_omission is not None
            else None,
            "rows": rows,
        }


__all__ = ["OracleTrace", "RouteMatch"]
