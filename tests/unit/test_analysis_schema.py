from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from textwrap import dedent

import pytest

HASH = "a" * 64
OTHER_HASH = "b" * 64
COMMIT = "c" * 40


def contract(*, rank_prompt: str = HASH) -> dict[str, object]:
    return {
        "model_identity_sha256": HASH,
        "model_config_sha256": OTHER_HASH,
        "dataset_sha256": "d" * 64,
        "input_sha256": "e" * 64,
        "commit": COMMIT,
        "tensor_parallel_size": 4,
        "batch_size": 2,
        "context_length": 8,
        "output_length": 2,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 16,
        "gpu_memory_utilization": 0.9,
        "engine_policy_sha256": "f" * 64,
        "versions": {
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "1" * 40,
        },
        "hardware_sha256": "2" * 64,
        "prompt_hashes": [rank_prompt, "3" * 64],
        "seed": 7,
    }


def transport_contract() -> dict[str, object]:
    return {
        "commit": COMMIT,
        "tensor_parallel_size": 4,
        "versions": {
            "torch": "2.8.0",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "1" * 40,
        },
        "hardware_sha256": "2" * 64,
        "measurement_id": "transport-001",
        "benchmark_policy_sha256": "4" * 64,
    }


def events():
    from flexmoe.analysis.schema import TraceEvent

    return (
        TraceEvent(0, 0, 2, 2, "prefill", (0, 1)),
        TraceEvent(0, 1, 2, 2, "prefill", (1,)),
        TraceEvent(1, 0, 2, 2, "decode", (1, 2)),
        TraceEvent(1, 1, 2, 2, "decode", (0, 2)),
    )


def trace(**changes):
    from flexmoe.analysis.schema import DemandTrace

    values = {
        "trace_id": "trace-r0",
        "rank": 0,
        "contract": contract(),
        "total_layers": 2,
        "num_experts": 4,
        "top_k": 1,
        "expert_bytes": 100,
        "observed_steps": 2,
        "captured_steps": 2,
        "full_workload": True,
        "generated_tokens": 4,
        "events": events(),
    }
    values.update(changes)
    return DemandTrace(**values)


def test_trace_roundtrip_preserves_full_literal_fixture():
    from flexmoe.analysis.schema import DemandTrace

    row = trace()
    assert DemandTrace.from_dict(row.to_dict()) == row
    assert row.events[2].experts == (1, 2)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(1),
        lambda rows: rows.__setitem__(1, rows[0]),
        lambda rows: rows.__setitem__(2, rows[2] | {"step": 2}),
        lambda rows: rows.__setitem__(0, rows[0] | {"experts": [1, 1]}),
        lambda rows: rows.__setitem__(0, rows[0] | {"experts": [4]}),
        lambda rows: rows.__setitem__(0, rows[0] | {"experts": [0, 1, 2]}),
    ],
)
def test_trace_rejects_incomplete_or_impossible_event_grid(mutate):
    from flexmoe.analysis.schema import DemandTrace

    rows = [event.to_dict() for event in events()]
    mutate(rows)
    with pytest.raises((TypeError, ValueError)):
        DemandTrace.from_dict(trace().to_dict() | {"events": rows})


@pytest.mark.parametrize(
    "changes",
    [
        {"rank": 4},
        {"observed_steps": 3},
        {"generated_tokens": 0},
        {"top_k": True},
        {"expert_bytes": -1},
    ],
)
def test_trace_rejects_wrong_rank_false_coverage_and_invalid_numbers(changes):
    with pytest.raises((TypeError, ValueError)):
        trace(**changes)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tensor_parallel_size", 2),
        ("batch_size", True),
        ("gpu_memory_utilization", float("nan")),
        ("model_identity_sha256", "not-a-hash"),
        ("commit", HASH),
        ("unexpected_private_field", 1),
    ],
)
def test_trace_contract_fails_closed(key, value):
    raw = contract()
    raw[key] = value
    with pytest.raises((TypeError, ValueError)):
        trace(contract=raw)


def test_non_trace_contract_may_explicitly_record_unavailable_hardware():
    from flexmoe.analysis.schema import TimingPoint

    raw = contract()
    raw["hardware_sha256"] = None
    raw["hardware_unavailable_reason"] = "uuid-query-unavailable"
    point = TimingPoint(
        "baseline",
        raw,
        4,
        (1.0,),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
        "native-measured",
        "native",
    )
    assert TimingPoint.from_dict(point.to_dict()) == point


def test_trace_may_record_explicitly_unavailable_hardware():
    raw = contract()
    raw["hardware_sha256"] = None
    raw["hardware_unavailable_reason"] = "uuid-query-unavailable"
    row = trace(contract=raw)
    assert row.contract["hardware_sha256"] is None


def test_null_hardware_without_explicit_reason_is_rejected():
    raw = contract()
    raw["hardware_sha256"] = None
    with pytest.raises(ValueError):
        trace(contract=raw)


def test_truncated_trace_keeps_window_but_is_not_full_workload():
    row = trace(
        observed_steps=3,
        captured_steps=2,
        full_workload=False,
        generated_tokens=4,
    )
    assert row.captured_steps == 2
    assert row.full_workload is False


def test_transfer_and_replay_config_roundtrip_and_validate_exact_payload():
    from flexmoe.analysis.schema import ReplayConfig, TransferSample

    config_row = ReplayConfig(((0,), (1,)), 2, 2, "lru", 25)
    assert ReplayConfig.from_dict(config_row.to_dict()) == config_row
    sample = TransferSample(
        0,
        transport_contract(),
        2,
        100,
        "gather",
        "isolated",
        0,
        200,
        0.2,
        0.1,
        0.05,
        None,
    )
    assert TransferSample.from_dict(sample.to_dict()) == sample
    with pytest.raises(ValueError):
        TransferSample.from_dict(sample.to_dict() | {"payload_bytes": 201})
    with pytest.raises(ValueError):
        TransferSample.from_dict(sample.to_dict() | {"wall_s": 0.05})


def test_analysis_schema_imports_under_python_s_without_torch():
    env = deepcopy(os.environ)
    env["PYTHONPATH"] = "src"
    script = dedent(
        """\
        import importlib.util, pathlib, sys
        package_dir = pathlib.Path("src/flexmoe/analysis")
        spec = importlib.util.spec_from_file_location(
            "analysis_core",
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        try:
            module.replay_trace(None, None)
        except TypeError:
            pass
        else:
            raise AssertionError("replay fail-closed path was not called")
        try:
            module.analyze_feasibility((), (), (), None, None)
        except ValueError:
            pass
        else:
            raise AssertionError("cost fail-closed path was not called")
        assert "torch" not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_saved_artifact_uses_one_fixed_diagnostic_root_envelope():
    from flexmoe.analysis.schema import diagnostic_artifact, parse_diagnostic_artifact

    saved = diagnostic_artifact("demand-trace", trace().to_dict())
    assert saved["schema_version"] == 1
    assert saved["artifact_kind"] == "demand-trace"
    assert saved["diagnostic_only"] is True
    assert saved["formal_offload_gain"] is False
    assert saved["deployment_gain_proven"] is False
    payload = parse_diagnostic_artifact(saved, "demand-trace")
    assert payload == trace().to_dict()
    with pytest.raises(ValueError):
        parse_diagnostic_artifact(
            saved | {"deployment_gain_proven": True}, "demand-trace"
        )
