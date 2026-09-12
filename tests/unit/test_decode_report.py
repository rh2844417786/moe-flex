import gzip
import json

from test_decode_suite import evidence

from flexmoe.analysis import decode_report as report


def write(path, value):
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as stream:
            json.dump(value, stream)
    else:
        path.write_text(json.dumps(value))


def capture():
    row = evidence(profile=True)
    return dict(
        {
            k: row[k]
            for k in (
                "schema_version",
                "mode",
                "profile",
                "evidence_kind",
                "timing_eligible",
                "diagnostic_only",
                "formal_offload_gain",
                "deployment_gain_proven",
                "comparison_backend",
                "contract",
                "geometry",
            )
        },
        artifact_kind="decode-profile",
        repetition=0,
        rank=0,
        generation_status="complete",
        observation={
            "rank": 0,
            "profile": True,
            "coverage_status": "complete",
            "captured_steps": 64,
            "observed_steps": 64,
            "phase_step_counts": {"decode": 64},
            "decode_batch_step_counts": {"2": 64},
            "activation_rows": [
                {
                    "step": s,
                    "layer": l,
                    "phase": "decode",
                    "actual_batch": 2,
                    "histogram": [2, 1, 1, 0],
                }
                for s in range(64)
                for l in range(2)
            ],
            "model_step_spans": [{"step": 0, "status": "measured", "cuda_s": 0.006}],
            "pool_profile": {
                "capacity": 128,
                "dropped_rows": 0,
                "unmeasured_rows": 0,
                "rows": [
                    {
                        "step": 0,
                        "layer": 0,
                        "actual_batch": 2,
                        "phase": "decode",
                        "status": "complete",
                        "resident_hits": 1,
                        "cache_hits": 0,
                        "unique_misses": 2,
                        "unique_demands": 3,
                        "first_loads": 1,
                        "reloads": 1,
                        "loaded_bytes": 384,
                        "metadata_bytes": 12,
                        "promotion_bytes": 192,
                        "evictions": 1,
                        "bypasses": 0,
                        "cuda_timing": {
                            "status": "measured",
                            "load_s": 0.002,
                            "payload_s": 0.0015,
                            "metadata_s": 0.0005,
                            "compute_s": 0.003,
                            "promotion_s": 0.001,
                            "span_s": 0.006,
                        },
                        "cpu_timing": {
                            "status": "measured",
                            "route_d2h_s": 0.004,
                            "host_reuse_wait_s": 0.007,
                        },
                    }
                ],
            },
        },
    )


def test_profile_counts_are_logically_replicated_and_intervals_remain_local(tmp_path):
    write(tmp_path / "summary.json", evidence(profile=True))
    for rank in range(4):
        row = capture()
        row["rank"] = rank
        row["observation"]["rank"] = rank
        write(tmp_path / f"decode-rep-000-rank-{rank}.json.gz", row)
    result = report.summarize_directory(tmp_path)
    assert len(result["captures"]) == 4
    first = result["captures"][0]
    layer = first["activation_summaries"][0]["per_layer"][0]
    assert layer["token_count"] == 128
    assert layer["selection_count"] == 256
    assert layer["histogram_total"] == [128, 64, 64, 0]
    assert layer["selection_probability"] == [0.5, 0.25, 0.25, 0]
    assert layer["coverage_mean"] == 3
    assert first["pool"]["load_s"] == 0.002
    assert first["pool"]["host_reuse_wait_s"] == 0.007
    assert first["pool"]["failed_rows"] == 0
    assert "net_loss_s" not in first["pool"]
    assert result["validation"]["timing_eligible"] is False


def test_malformed_summary_preserves_smoke_numeric_rejection_and_profile_failures(
    tmp_path,
):
    (tmp_path / "summary.json").write_text("{bad")
    sample = evidence()["repetitions"][0]
    sample.update(
        measurement_status="rejected",
        capture_status="failed",
        observer_finalization={
            "status": "failed",
            "errors": [{"stage": "capture_stop", "error_type": "ValueError"}],
        },
    )
    write(tmp_path / "failed-rep-000.json", sample)
    write(tmp_path / "smoke.json", evidence()["smoke"])
    row = capture()
    row["observation"]["pool_profile"]["rows"][0]["status"] = "failed"
    row["observation"]["activation_rows"][0]["histogram"] = [3, 1, 0, 0]
    write(tmp_path / "decode-rep-000-rank-0.json.gz", row)
    result = report.summarize_directory(tmp_path)
    assert "malformed-json" in result["validation"]["reasons"]
    assert result["smoke"]["generated_tokens"] == 1
    assert result["repetitions"][0]["generated_tokens"] == 1000
    assert result["repetitions"][0]["elapsed_s"] == 10
    assert result["repetitions"][0]["measurement_status"] == "rejected"
    assert result["repetitions"][0]["observer_errors"] == ["capture_stop:ValueError"]
    assert result["captures"][0]["pool"]["failed_rows"] == 1
    assert result["captures"][0]["activation_status"] == "invalid-histogram"


def test_typed_export_leaks_no_text_paths_ids_or_injected_strings_and_no_fake_charts(
    tmp_path,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    row = evidence()
    secret = "PRIVATE_PROMPT_/home/person_UUID_deadbeef"
    row.update(prompt=secret, run_id=secret, logs=secret, error=secret)
    row["repetitions"][0]["scheduler"] = {
        "kv_cache_usage": {
            "status": "unavailable",
            "unavailable_reason": secret,
            "mean": secret,
        },
        "source": secret,
    }
    row["repetitions"][0]["observer_finalization"]["errors"] = [
        {"stage": secret, "error_type": secret}
    ]
    row["status"] = secret
    write(raw / "summary.json", row)
    out = tmp_path / "public"
    report.export_report(raw, out)
    files = list(out.iterdir())
    assert {p.name for p in files} == {"report.json", "report.csv", "report.md"}
    text = "".join(p.read_text() for p in files)
    assert secret not in text
    parsed = json.loads((out / "report.json").read_text())
    assert (
        parsed["repetitions"][0]["scheduler"]["kv_cache_usage"]["status"]
        == "unavailable"
    )
    assert (
        parsed["repetitions"][0]["scheduler"]["kv_cache_usage"]["unavailable_reason"]
        == "unavailable-at-source"
    )
    assert "output_sha256" not in text


def test_eligible_chart_contains_actual_kv_and_measured_tps(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "summary.json", evidence())
    out = tmp_path / "out"
    report.export_report(raw, out)
    svg = (out / "kv-throughput.svg").read_text()
    assert 'data-x="1000"' in svg
    assert 'data-y="100"' in svg
    assert len(list(out.glob("*.svg"))) == 1
    assert not list(out.glob("*coverage*"))


def test_export_comparison_retains_measured_ratios_and_launcher_failure(tmp_path):
    sources = []
    for name, row in [
        ("a", evidence()),
        ("b", evidence("offload", 12)),
        ("c", evidence("offload", 9, 1400)),
        ("native", evidence("native", 8)),
    ]:
        source = tmp_path / name
        source.mkdir()
        write(source / "summary.json", row)
        sources.append(source)
    write(sources[0] / "launcher.json", {"status": "complete", "exit_code": 0})
    out = tmp_path / "public"
    report.export_report(sources[0], out, comparisons=sources[1:])
    saved = json.loads((out / "report.json").read_text())
    assert saved["launcher"] == {"status": "complete", "exit_code": 0}
    assert saved["comparison"]["offload_overhead_s"] == 2
    assert saved["comparison"]["kv_recovery_s"] == 3
    assert "matched_net_ratio" in (out / "report.csv").read_text()
    assert 'data-x="1400"' in (out / "kv-throughput.svg").read_text()


def test_report_exposes_actual_pool_memory_with_no_inferred_rounding(tmp_path):
    raw = evidence("offload")
    raw["expert_cache_stats"][0].update(
        resident_slots=23,
        gpu_resident_bytes=4416,
        gpu_ingress_bytes=768,
        gpu_cache_bytes=192,
    )
    write(tmp_path / "summary.json", raw)
    result = report.summarize_directory(tmp_path)
    assert result["expert_cache_stats"][0]["resident_slots"] == 23
    assert result["expert_cache_stats"][0]["gpu_resident_bytes"] == 4416
