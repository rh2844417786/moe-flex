from types import SimpleNamespace

import pytest
import torch


def collector(**kwargs):
    from flexmoe.vllm.analysis_trace import AnalysisTraceCollector

    model = torch.nn.Identity()
    runner = SimpleNamespace(
        model=model,
        input_batch=SimpleNamespace(
            num_reqs=2, num_prompt_tokens=[3, 3], num_computed_tokens_cpu=[0, 3]
        ),
        query_start_loc=SimpleNamespace(np=[0, 1, 3]),
    )
    return AnalysisTraceCollector(
        runner, total_layers=2, num_experts=4, top_k=2, device="cpu", **kwargs
    ), runner


def test_capture_trims_padding_tracks_cpu_phase_and_stops_outside_window():
    cap, runner = collector(max_trace_steps=2)
    cap.start()
    runner.model(torch.zeros(4))
    for layer in range(2):
        cap.record(layer, torch.tensor([[0, 0], [1, 2], [2, 2], [3, 3]]))
    result = cap.stop()
    assert result["events"][0] == {
        "step": 0,
        "layer": 0,
        "token_rows": 3,
        "requests": 2,
        "phase": "mixed",
        "experts": [0, 1, 2],
    }
    assert result["full_workload"] is True
    assert not runner.model._forward_pre_hooks
    runner.model(torch.zeros(4))
    cap.record(0, torch.tensor([[3, 3]]))
    assert cap.status()["observed_steps"] == 1


def test_overflow_and_missing_layers_keep_only_complete_prefix():
    cap, runner = collector(max_trace_steps=2)
    cap.start()
    for step in range(3):
        runner.model(torch.zeros(4))
        cap.record(0, torch.tensor([[0, 1]] * 3))
        if step != 1:
            cap.record(1, torch.tensor([[2, 3]] * 3))
    result = cap.stop()
    assert result["observed_steps"] == 3
    assert result["captured_steps"] == 1
    assert result["truncated"] and not result["full_workload"]
    assert result["missing_layer_events"] == 1
    cap.start()
    cap.start()
    assert len(runner.model._forward_pre_hooks) == 1
    cap.stop()


def test_budget_checked_before_allocation_and_shape_errors_are_visible():
    with pytest.raises(ValueError, match="budget"):
        collector(max_trace_steps=3, trace_budget_bytes=23)[0].start()
    cap, runner = collector()
    cap.start()
    runner.model(torch.zeros(4))
    with pytest.raises(ValueError, match="dtype|shape"):
        cap.record(0, torch.ones(3, 2))
    cap.stop()


def test_cpu_metadata_missing_is_unknown():
    cap, runner = collector()
    runner.input_batch = None
    cap.start()
    runner.model(torch.zeros(1))
    for layer in range(2):
        cap.record(layer, torch.tensor([[1, 2]]))
    assert cap.stop()["events"][0]["requests"] is None


def test_explicit_gateway_preserves_calibration(monkeypatch):
    from flexmoe.vllm import analysis_trace, bridge, expert_calibration

    old, new = [], []
    monkeypatch.setattr(
        expert_calibration, "record_calibration", lambda *x: old.append(x)
    )
    monkeypatch.setattr(analysis_trace, "record_trace", lambda *x: new.append(x))
    monkeypatch.delenv("FLUXMOE_ANALYSIS_TRACE", raising=False)
    route = torch.tensor([[0, 1]])
    bridge.record_calibration("model.layers.0.mlp", route)
    monkeypatch.setenv("FLUXMOE_ANALYSIS_TRACE", "1")
    bridge.record_calibration("model.layers.0.mlp", route)
    assert len(old) == len(new) == 1


def test_cuda_reserve_fails_before_any_gpu_allocation(monkeypatch):
    from flexmoe.vllm.analysis_trace import AnalysisTraceCollector

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (2_000_000_001, 80_000_000_000)
    )
    cap = AnalysisTraceCollector(
        SimpleNamespace(model=torch.nn.Identity()),
        total_layers=2,
        num_experts=4,
        top_k=2,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="reserve"):
        cap.start()
    assert cap.buffer is None
