import pytest

from flexmoe.config import PlannerConfig
from flexmoe.errors import ConfigurationError
from flexmoe.planner import PlannerObservation, ResidencyPlanner


def observation(
    load_s: float,
    *,
    iteration: int = 300,
    expert_bytes: int = 400,
) -> PlannerObservation:
    return PlannerObservation(
        iteration=iteration,
        compute_reference_s=1.0,
        load_s=load_s,
        gpu_capacity_bytes=1_000,
        kv_bytes=200,
        fixed_bytes=200,
        expert_gpu_bytes=expert_bytes,
        step_bytes=100,
    )


def test_planner_decreases_holds_and_increases_residency() -> None:
    planner = ResidencyPlanner(PlannerConfig())

    assert planner.decide(observation(0.5)).action == "decrease"
    assert planner.decide(observation(1.05)).action == "hold"
    assert planner.decide(observation(1.2)).action == "increase"


def test_capacity_precedes_io_direction() -> None:
    planner = ResidencyPlanner(PlannerConfig())

    decision = planner.decide(observation(1.2, expert_bytes=700))

    assert decision.action == "decrease"
    assert decision.expert_gpu_bytes == 600
    assert decision.reason == "capacity"


def test_planner_holds_between_decision_intervals() -> None:
    planner = ResidencyPlanner(PlannerConfig(decision_interval=300))

    decision = planner.decide(observation(0.5, iteration=301))

    assert decision.action == "hold"
    assert decision.expert_gpu_bytes == 400
    assert decision.reason == "interval"


def test_dead_zone_remains_stable_across_repeated_decisions() -> None:
    planner = ResidencyPlanner(PlannerConfig())
    current = 400

    for iteration in range(300, 2401, 300):
        decision = planner.decide(
            observation(1.05, iteration=iteration, expert_bytes=current)
        )
        assert decision.action == "hold"
        current = decision.expert_gpu_bytes

    assert current == 400


@pytest.mark.parametrize(
    "invalid",
    [
        PlannerObservation(300, 0.0, 1.0, 1_000, 0, 0, 100, 10),
        PlannerObservation(300, 1.0, 0.0, 1_000, 0, 0, 100, 10),
        PlannerObservation(300, 1.0, 1.0, 1_000, -1, 0, 100, 10),
        PlannerObservation(300, 1.0, 1.0, 1_000, 0, 0, 100, 0),
    ],
)
def test_planner_rejects_invalid_observations(invalid: PlannerObservation) -> None:
    planner = ResidencyPlanner(PlannerConfig())

    with pytest.raises(ConfigurationError):
        planner.decide(invalid)
