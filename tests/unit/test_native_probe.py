from types import SimpleNamespace

import pytest
import torch


def test_root_probe_counts_only_measured_calls_and_live_requests():
    from flexmoe.vllm.native_probe import NativeProbe

    model = torch.nn.Linear(2, 2)
    runner = SimpleNamespace(model=model, input_batch=SimpleNamespace(num_reqs=2))
    probe = NativeProbe(runner, 0)
    model(torch.ones(1, 2))
    probe.start()
    with pytest.raises(RuntimeError):
        probe.start()
    model(torch.ones(1, 2))
    runner.input_batch.num_reqs = 4
    model(torch.ones(1, 2))
    row = probe.stop()
    model(torch.ones(1, 2))
    assert row["forward_calls"] == 2
    assert row["request_observations"] == 2
    assert row["request_sum"] == 6
    assert row["request_peak"] == 4
    assert row["request_mean"] == 3
    assert row["all_parameters_cuda"] is False
    assert probe.snapshot() == row
    assert len(model._forward_pre_hooks) == 0
    probe.start()
    model(torch.ones(1, 2))
    assert probe.stop()["forward_calls"] == 1


def test_probe_unavailable_requests_are_null_not_submitted_batch_size():
    from flexmoe.vllm.native_probe import NativeProbe

    runner = SimpleNamespace(model=torch.nn.Linear(2, 2))
    probe = NativeProbe(runner, 3)
    probe.start()
    runner.model(torch.ones(1, 2))
    row = probe.stop()
    assert row["forward_calls"] == 1
    assert row["request_telemetry_available"] is False
    assert row["request_mean"] is None
    assert row["request_peak"] is None
    assert row["kv_occupancy"] is None
    assert row["preemptions"] is None
