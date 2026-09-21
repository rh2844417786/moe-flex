from types import SimpleNamespace

import pytest


def test_phase_timeline_separates_prefill_and_decode_without_rank_summing():
    from flexmoe.vllm.decode_phase import PhaseTimeline

    clock = iter((1_000, 2_000, 5_000, 7_000, 9_000, 13_000))
    timeline = PhaseTimeline(clock_ns=lambda: next(clock))
    timeline.start()
    timeline.record(step=0, phase="prefill", actual_batch=2)
    timeline.finish(step=0)
    timeline.record(step=1, phase="decode", actual_batch=2)
    timeline.finish(step=1)
    result = timeline.stop()

    assert result["prefill_wall_time_s"] == 6e-6
    assert result["decode_wall_time_s"] == 2e-6
    assert result["decode_step_ms_p50"] == pytest.approx(0.002)
    assert result["decode_step_ms_p95"] == pytest.approx(0.002)
    assert result["phase_sequence"] == ["prefill", "decode"]
    assert result["boundaries"] == {
        "measurement_start_ns": 1_000,
        "first_prefill_start_ns": 2_000,
        "first_prefill_end_ns": 5_000,
        "first_decode_start_ns": 7_000,
        "last_decode_end_ns": 9_000,
        "synchronized_measurement_end_ns": 13_000,
    }
    assert result["scope"] == (
        "per-rank monotonic worker wall; never summed across TP ranks"
    )


def test_missing_decode_keeps_phase_metrics_unavailable():
    from flexmoe.vllm.decode_phase import PhaseTimeline

    timeline = PhaseTimeline(clock_ns=iter((10, 20, 30, 40)).__next__)
    timeline.start()
    timeline.record(step=0, phase="prefill", actual_batch=1)
    timeline.finish(step=0)
    result = timeline.stop()

    assert result["prefill_wall_time_s"] is None
    assert result["decode_wall_time_s"] is None
    assert result["status"] == "incomplete-phase-boundaries"


def _output(arrival: float, first: float, finished: float) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4, 5])],
        metrics=SimpleNamespace(
            arrival_time=arrival,
            first_token_time=first,
            finished_time=finished,
        ),
    )


def test_request_metrics_report_ttft_latency_and_derived_tpot_percentiles():
    from flexmoe.bench.partial_runner import summarize_request_metrics

    result = summarize_request_metrics(
        [_output(0.0, 1.0, 5.0), _output(0.0, 2.0, 8.0)],
        output_length=5,
    )

    assert result["ttft_s"]["p50"] == 1.5
    assert result["request_latency_s"]["p95"] == pytest.approx(7.85)
    assert result["derived_tpot_s"]["values"] == [1.0, 1.5]
    assert result["itl_s"] == {
        "status": "unavailable",
        "reason": "per-token timestamps absent",
    }


def test_request_metrics_do_not_invent_tpot_when_timestamps_are_missing():
    from flexmoe.bench.partial_runner import summarize_request_metrics

    output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])], metrics=None)
    result = summarize_request_metrics([output], output_length=2)

    assert result["ttft_s"]["status"] == "unavailable"
    assert result["request_latency_s"]["status"] == "unavailable"
    assert result["derived_tpot_s"]["status"] == "unavailable"
