"""Explicit calibration capture and the shared producer/consumer identity."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import torch


class ModelProfileIdentity(TypedDict):
    geometry: dict[str, int]
    model_config_sha256: str
    model_identity_sha256: str
    tensor_parallel_size: int


def model_profile_identity(
    model_path: str | Path, tensor_parallel_size: int
) -> ModelProfileIdentity:
    """Bind raw config/index bytes, resolved local path, and exact TP geometry.

    This is an artifact identity, not a hash of every checkpoint weight byte.
    Both calibration producer and serving consumer MUST call this helper.
    """
    root = Path(model_path).resolve(strict=True)
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") != "qwen3_next":
        raise ValueError("expert-cache currently supports qwen3_next only")
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    geometry = {
        "total_layers": config["num_hidden_layers"],
        "num_experts": config["num_experts"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["moe_intermediate_size"],
    }
    if any(type(value) is not int or value <= 0 for value in geometry.values()):
        raise ValueError("invalid Qwen3 MoE geometry")
    if geometry["intermediate_size"] % tensor_parallel_size:
        raise ValueError("expert intermediate size must be TP divisible")
    geometry["intermediate_size"] //= tensor_parallel_size
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    config_hash = sha256(config_bytes).hexdigest()
    identity = json.dumps(
        {
            "resolved_model_path": str(root),
            "model_config_sha256": config_hash,
            "model_index_sha256": sha256(index_bytes).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "geometry": geometry,
        "model_config_sha256": config_hash,
        "model_identity_sha256": sha256(identity).hexdigest(),
        "tensor_parallel_size": tensor_parallel_size,
    }


class ExpertCalibration:
    def __init__(self, total_layers: int, num_experts: int) -> None:
        self.total_layers, self.num_experts = total_layers, num_experts
        self.active = False
        self.counts = [[0] * num_experts for _ in range(total_layers)]
        self.forwards = [0] * total_layers

    def start(self) -> dict[str, object]:
        if self.active:
            raise RuntimeError("calibration is already active")
        self.counts = [[0] * self.num_experts for _ in range(self.total_layers)]
        self.forwards = [0] * self.total_layers
        self.active = True
        return {"active": True}

    def record(self, layer: int, topk_ids: torch.Tensor) -> None:
        if not self.active:
            return
        if not 0 <= layer < self.total_layers or topk_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("invalid calibration layer/IDs")
        ids = set(topk_ids.detach().to(device="cpu").flatten().tolist())
        if any(not 0 <= expert < self.num_experts for expert in ids):
            raise ValueError("invalid calibration expert ID")
        for expert in ids:
            self.counts[layer][expert] += 1
        self.forwards[layer] += 1

    def stop(self) -> dict[str, object]:
        self.active = False
        return {
            "active": False,
            "counts": [list(row) for row in self.counts],
            "forward_counts": list(self.forwards),
        }


_CAPTURE: ExpertCalibration | None = None


def calibration_rpc(action: str, tp_size: int, rank: int) -> dict[str, object]:
    global _CAPTURE
    if os.environ.get("FLUXMOE_EXPERT_CALIBRATION") != "1":
        raise ValueError("FLUXMOE_EXPERT_CALIBRATION=1 is required")
    if action not in ("start", "stop"):
        raise ValueError("calibration action must be start or stop")
    identity = model_profile_identity(os.environ["FLUXMOE_MODEL_PATH"], tp_size)
    if _CAPTURE is None:
        geometry = identity["geometry"]
        _CAPTURE = ExpertCalibration(geometry["total_layers"], geometry["num_experts"])
    result = _CAPTURE.start() if action == "start" else _CAPTURE.stop()
    return {"schema_version": 1, "rank": rank, **identity, **result}


def record_calibration(layer_name: str, topk_ids: torch.Tensor) -> None:
    if _CAPTURE is not None:
        from flexmoe.vllm.partial import _layer_index

        _CAPTURE.record(_layer_index(layer_name), topk_ids)


def reset_calibration() -> None:
    global _CAPTURE
    _CAPTURE = None
