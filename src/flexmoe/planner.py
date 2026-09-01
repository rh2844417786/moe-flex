"""Budget-aware expert residency controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from flexmoe.config import PlannerConfig
from flexmoe.errors import ConfigurationError

PlannerAction = Literal["increase", "hold", "decrease"]


@dataclass(frozen=True)
class PlannerObservation:
    """One controller observation at a decode iteration."""

    iteration: int
    compute_reference_s: float
    load_s: float
    gpu_capacity_bytes: int
    kv_bytes: int
    fixed_bytes: int
    expert_gpu_bytes: int
    step_bytes: int


@dataclass(frozen=True)
class PlannerDecision:
    """Capacity-safe expert residency decision."""

    action: PlannerAction
    expert_gpu_bytes: int
    ratio: float
    reason: str


class ResidencyPlanner:
    """Negative-feedback controller with an asymmetric dead zone."""

    def __init__(self, config: PlannerConfig) -> None:
        if not (
            0 < config.io_bound_threshold <= config.compute_bound_threshold
        ):
            raise ConfigurationError("planner thresholds are inconsistent")
        if config.decision_interval <= 0:
            raise ConfigurationError("decision_interval must be positive")
        self.config = config

    @staticmethod
    def _validate(observation: PlannerObservation) -> None:
        if observation.iteration < 0:
            raise ConfigurationError("iteration must be non-negative")
        if observation.compute_reference_s <= 0 or observation.load_s <= 0:
            raise ConfigurationError("compute and load times must be positive")
        byte_values = (
            observation.gpu_capacity_bytes,
            observation.kv_bytes,
            observation.fixed_bytes,
            observation.expert_gpu_bytes,
        )
        if any(value < 0 for value in byte_values):
            raise ConfigurationError("memory byte counts must be non-negative")
        if observation.step_bytes <= 0:
            raise ConfigurationError("step_bytes must be positive")

    @staticmethod
    def _action(current: int, target: int) -> PlannerAction:
        if target > current:
            return "increase"
        if target < current:
            return "decrease"
        return "hold"

    def decide(self, observation: PlannerObservation) -> PlannerDecision:
        """Return the next expert GPU byte budget without violating capacity."""

        self._validate(observation)
        ratio = observation.compute_reference_s / observation.load_s
        maximum = max(
            0,
            observation.gpu_capacity_bytes
            - observation.kv_bytes
            - observation.fixed_bytes,
        )
        current = observation.expert_gpu_bytes

        if current > maximum:
            return PlannerDecision(
                action="decrease",
                expert_gpu_bytes=maximum,
                ratio=ratio,
                reason="capacity",
            )
        if observation.iteration % self.config.decision_interval != 0:
            return PlannerDecision("hold", current, ratio, "interval")

        if ratio > self.config.compute_bound_threshold:
            requested = current - observation.step_bytes
            reason = "compute_bound"
        elif ratio < self.config.io_bound_threshold:
            requested = current + observation.step_bytes
            reason = "io_bound"
        else:
            requested = current
            reason = "dead_zone"

        target = min(max(0, requested), maximum)
        action = self._action(current, target)
        if target != requested:
            reason = "capacity"
        return PlannerDecision(action, target, ratio, reason)
