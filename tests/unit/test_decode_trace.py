from types import SimpleNamespace

import pytest
import torch


class PreparedRunner:
    """Pinned CPU preparation boundary; execution does not call a model hook."""

    def _prepare_inputs(self, batch=2, phase="decode", rows=None):
        self.input_batch = SimpleNamespace(
            num_reqs=batch,
            num_prompt_tokens=[2] * batch,
            num_computed_tokens_cpu=[0 if phase == "prefill" else 2] * batch,
        )
        self.query_start_loc = SimpleNamespace(np=list(range(batch)) + [rows or batch])
        return "prepared"

    def execute_model(self, **kwargs):
        return self._prepare_inputs(**kwargs)


def collector(runner, **kwargs):
    from flexmoe.vllm.decode_trace import DecodeTraceCollector

    return DecodeTraceCollector(
        runner,
        total_layers=2,
        num_experts=4,
        top_k=2,
        device="cpu",
        capture_steps=2,
        min_capture_steps=1,
        max_batch=4,
        **kwargs,
    )


def test_counts_multiplicity_masks_padding_and_excludes_before_after_window():
    runner = PreparedRunner()
    capture = collector(runner, profile=True, target_batch=2)
    ids = torch.tensor([[0, 1], [0, 2], [3, 1]])
    runner.execute_model()
    capture.record(0, ids)
    original = runner._prepare_inputs
    capture.start()
    assert runner.execute_model() == "prepared"
    for layer in range(2):
        capture.record(layer, ids)
    result = capture.stop()
    assert runner._prepare_inputs == original
    runner.execute_model()
    capture.record(0, ids)
    assert result["activation_rows"] == [
        {
            "step": 0,
            "layer": layer,
            "phase": "decode",
            "actual_batch": 2,
            "histogram": [2, 1, 1, 0],
        }
        for layer in range(2)
    ]
    assert result["coverage_status"] == "complete"
    assert result["observed_steps"] == 1
    assert capture.stop() == result


def test_selected_steps_bounded_but_graph_safe_observation_continues():
    runner = PreparedRunner()
    capture = collector(runner, profile=True, target_batch=2)
    capture.start()
    for batch, phase in [(2, "prefill"), (1, "decode"), (2, "decode")] * 3:
        runner.execute_model(batch=batch, phase=phase)
        for layer in range(2):
            capture.record(layer, torch.tensor([[0, 1]] * batch))
    result = capture.stop()
    assert result["observed_steps"] == 9
    assert result["captured_steps"] == 2
    assert [r["step"] for r in result["activation_rows"]] == [2, 2, 5, 5]
    assert result["decode_batch_step_counts"] == {"1": 3, "2": 3}
    assert result["skipped_steps"] == {
        "non_decode": 3,
        "non_target_batch": 3,
        "capacity": 1,
    }


def test_missing_layers_unreached_and_non_one_token_decode_fail_closed():
    runner = PreparedRunner()
    capture = collector(runner, profile=True, target_batch=2)
    capture.start()
    runner.execute_model(rows=3)
    capture.record(0, torch.tensor([[0, 1]] * 3))
    runner.execute_model()
    capture.record(0, torch.tensor([[0, 1]] * 2))
    result = capture.stop()
    assert result["missing_layer_events"] == 1
    assert result["coverage_status"] == "incomplete-layers"
    other = collector(runner, profile=True, target_batch=3)
    other.start()
    runner.execute_model()
    assert other.stop()["coverage_status"] == "unreached"


def test_native_observes_without_histogram_allocation_and_restores_on_failure():
    runner = PreparedRunner()
    capture = collector(runner, profile=False)
    capture.start()
    assert capture.buffer is None
    runner.execute_model(batch=3)
    result = capture.stop()
    assert result["decode_batch_step_counts"] == {"3": 1}
    assert result["activation_rows"] == []
    assert result["coverage_status"] == "not-requested"


def test_budget_includes_int32_histograms_and_ones_before_allocation():
    with pytest.raises(ValueError, match="budget"):
        collector(PreparedRunner(), profile=True, trace_budget_bytes=63).start()


def test_fixed_route_index_scratch_is_counted_in_budget():
    # 64 histogram + 32 ones + 64 int64 routing scratch, all preallocated.
    with pytest.raises(ValueError, match="budget"):
        collector(PreparedRunner(), profile=True, trace_budget_bytes=128).start()


def test_target_coverage_can_be_checked_without_profile():
    runner = PreparedRunner()
    capture = collector(runner, profile=False, target_batch=3)
    capture.start()
    runner.execute_model(batch=2)
    assert capture.stop()["coverage_status"] == "unreached"


def test_non_unit_query_lengths_are_not_pure_decode_even_when_sum_matches():
    class Unequal(PreparedRunner):
        def _prepare_inputs(self, **kwargs):
            result = super()._prepare_inputs(**kwargs)
            self.query_start_loc.np = [0, 0, 2]
            return result

    runner = Unequal()
    capture = collector(runner, profile=True)
    capture.start()
    runner.execute_model()
    assert capture.stop()["captured_steps"] == 0
    assert capture.stop()["phase_step_counts"] == {"unknown": 1}


def test_scheduler_observer_forwards_pinned_logger_and_restores():
    from flexmoe.vllm.decode_trace import SchedulerObserver

    class Logger:
        def __init__(self):
            self.values = []

        def record(self, scheduler_stats, iteration_stats, engine_idx=0):
            self.values.append((scheduler_stats.kv_cache_usage, engine_idx))
            return "forwarded"

    original = Logger()
    owner = SimpleNamespace(stat_logger=original)
    observer = SchedulerObserver(owner)
    observer.start()
    for usage, running, waiting, preempt in [(0.2, 2, 3, 1), (0.8, 4, 0, 2)]:
        assert (
            owner.stat_logger.record(
                SimpleNamespace(
                    kv_cache_usage=usage,
                    num_running_reqs=running,
                    num_waiting_reqs=waiting,
                    step_counter=2,
                ),
                SimpleNamespace(num_preempted_reqs=preempt),
                engine_idx=0,
            )
            == "forwarded"
        )
    stats = observer.stop()
    assert owner.stat_logger is original
    assert original.values == [(0.2, 0), (0.8, 0)]
    assert stats["kv_cache_usage"] == {
        "status": "measured",
        "samples": 2,
        "mean": 0.5,
        "peak": 0.8,
    }
    assert stats["preemptions"] == {"status": "measured", "samples": 2, "total": 3}


def test_scheduler_unavailable_and_invalid_values_not_occupancy():
    from flexmoe.vllm.decode_trace import SchedulerObserver

    observer = SchedulerObserver(SimpleNamespace(stat_logger=None))
    observer.start()
    assert observer.stop()["kv_cache_usage"]["status"] == "unavailable"


def test_scheduler_invalid_samples_preserve_partial_and_null_unavailable_counts():
    from flexmoe.vllm.decode_trace import SchedulerObserver

    original = SimpleNamespace(record=lambda **kwargs: None)
    owner = SimpleNamespace(stat_logger=original)
    observer = SchedulerObserver(owner)
    observer.start()
    for value in (0.5, None, float("nan"), -0.1, 1.1, True, "invalid"):
        owner.stat_logger.record(
            SimpleNamespace(kv_cache_usage=value), SimpleNamespace()
        )
    stats = observer.stop()
    assert owner.stat_logger is original
    assert stats["kv_cache_usage"] == {
        "status": "partial",
        "samples": 1,
        "mean": 0.5,
        "peak": 0.5,
    }
    assert stats["invalid_samples"]["kv_cache_usage"] == 6
    for metric in ("running_requests", "waiting_requests", "preemptions"):
        assert stats[metric]["status"] == "unavailable"
        assert stats[metric]["samples"] == 0
        assert stats["invalid_samples"][metric] == 7
        assert stats[metric]["total" if metric == "preemptions" else "mean"] is None


def test_worker_rpc_gateway_attaches_real_pool_and_restores(monkeypatch):
    from test_decode_pool_timing import observed_pool
    from test_expert_cache_runtime import invoke

    from flexmoe.vllm import decode_trace, expert_cache
    from flexmoe.vllm.bridge import FluxMoEWorkerExtension, record_calibration

    pool, sources = observed_pool()
    monkeypatch.setattr(expert_cache, "_REGISTRY", SimpleNamespace(pool=pool))
    monkeypatch.setattr(decode_trace, "_COLLECTOR", None)
    monkeypatch.setenv("FLUXMOE_DECODE_MECHANISM", "1")
    monkeypatch.setenv("FLUXMOE_EXPERT_CALIBRATION", "1")
    monkeypatch.setenv("FLUXMOE_ENABLE", "1")
    monkeypatch.setenv("FLUXMOE_STORAGE_MODE", "expert-cache")
    worker = FluxMoEWorkerExtension()
    worker.rank = 2
    worker.model_runner = PreparedRunner()
    worker.model_runner.model_config = SimpleNamespace(enforce_eager=True)
    worker.fluxmoe_decode_mechanism(
        "start",
        total_layers=2,
        num_experts=4,
        top_k=2,
        device="cpu",
        profile=True,
        capture_steps=2,
        min_capture_steps=1,
        max_batch=4,
    )
    worker.model_runner.execute_model()
    for layer in range(2):
        record_calibration(
            f"model.layers.{layer}.mlp.experts", torch.tensor([[0, 1], [0, 2]])
        )
        invoke(pool, sources, layer, [0, 1, 2])
    # A status-like repeated start must not reset an in-flight pool window.
    worker.fluxmoe_decode_mechanism("start")
    result = worker.fluxmoe_decode_mechanism("stop")
    assert result["rank"] == 2
    assert len(result["pool_profile"]["rows"]) == 2
    assert result["activation_rows"][0]["histogram"] == [2, 1, 1, 0]
    assert pool.profile_observer is None


def test_worker_rpc_profile_rejects_graph_mode(monkeypatch):
    from flexmoe.vllm import decode_trace

    monkeypatch.setenv("FLUXMOE_DECODE_MECHANISM", "1")
    monkeypatch.setattr(decode_trace, "_COLLECTOR", None)
    runner = PreparedRunner()
    runner.model_config = SimpleNamespace(enforce_eager=False)
    with pytest.raises(ValueError, match="eager"):
        decode_trace.decode_rpc(
            "start",
            0,
            runner,
            total_layers=2,
            num_experts=4,
            top_k=2,
            profile=True,
            device="cpu",
        )


def test_worker_stop_restores_wrapper_even_if_cuda_synchronization_fails(monkeypatch):
    from flexmoe.vllm import decode_trace

    monkeypatch.setenv("FLUXMOE_DECODE_MECHANISM", "1")
    runner = PreparedRunner()
    original = runner._prepare_inputs
    capture = collector(runner, profile=False)
    capture.start()
    capture.device = torch.device("cuda", 0)
    monkeypatch.setattr(decode_trace, "_COLLECTOR", capture)

    def fail(*args):
        raise RuntimeError("cuda failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="cuda failure"):
        decode_trace.decode_rpc("stop", 0, runner)
    assert runner._prepare_inputs == original
    assert not capture.active
