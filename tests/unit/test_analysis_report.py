import json
import subprocess

import pytest
from test_analysis_cli import ROOT, invoke, native, save
from test_analysis_cost import samples, timing, trace_replays

from flexmoe.analysis.cost import analyze_feasibility
from flexmoe.analysis.schema import diagnostic_artifact, parse_diagnostic_artifact


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("diagnostic_only", 1),
        ("formal_offload_gain", 0),
        ("deployment_gain_proven", 0.0),
    ],
)
def test_exact_envelope_scalar_types(key, value):
    raw = diagnostic_artifact("demand-trace", {})
    raw[key] = value
    with pytest.raises(ValueError):
        parse_diagnostic_artifact(raw, "demand-trace")


def test_proxy_preserves_service_without_double_counting_prediction():
    from dataclasses import replace

    traces, replays = trace_replays()
    rows = [
        replace(row, contention="gemm-nccl-proxy", compute_s=0.01) for row in samples()
    ]
    result = analyze_feasibility(
        traces, replays, rows, timing("r", 10, kv=100), timing("k", 8, kv=200)
    )
    assert result["predictions"] is None
    assert isinstance(result["missing_evidence"], list)
    assert "incremental-transfer-calibration" in result["missing_evidence"]
    transport = result["transport"]
    assert isinstance(transport, dict)
    assert transport["serial_layer_barrier_service_s"] > 0


def test_export_retains_independent_smoke_failed_peaks_and_scrubs_nested_strings(
    tmp_path,
):
    source = tmp_path / "run"
    source.mkdir()
    (source / "summary.json").write_text("UNREADABLE PRIVATE PROMPT")
    smoke = native()["smoke"]
    smoke["output_sha256"] = "a" * 64
    smoke["private"] = "/private/user/prompt"
    save(source / "smoke.json", smoke)
    save(
        source / "launcher.json",
        diagnostic_artifact(
            "launcher-failure",
            {
                "status": "failed",
                "exit_code": 124,
                "command": "/private/token",
                "memory": [{"rank": 0, "torch_peak_reserved_bytes": 456}],
            },
        ),
    )
    output = tmp_path / "public"
    result = invoke("export", "--source", source, "--output", output)
    assert result.returncode == 0, result.stderr
    raw = json.loads((output / "report.json").read_text())
    assert raw["diagnostic_only"] is True
    text = "\n".join(p.read_text() for p in output.iterdir())
    assert "456" in text and "smoke" in text
    assert "PRIVATE" not in text and "/private" not in text
    assert (source / "summary.json").read_text() == "UNREADABLE PRIVATE PROMPT"


def test_server_clean_guard_real_git_and_host_refusal(tmp_path):
    script = ROOT / "scripts/server/run_offload_analysis.sh"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; analysis_clean_code "$2"',
            "test",
            str(script),
            str(tmp_path),
        ],
        check=False,
    )
    assert result.returncode == 0
    generated = tmp_path / "docs/results/offload-analysis-test"
    generated.mkdir(parents=True)
    (generated / "report.json").write_text("{}")
    assert (
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; analysis_clean_code "$2"',
                "test",
                str(script),
                str(tmp_path),
            ],
            check=False,
        ).returncode
        == 0
    )
    (tmp_path / "source.py").write_text("changed")
    assert (
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; analysis_clean_code "$2"',
                "test",
                str(script),
                str(tmp_path),
            ],
            check=False,
        ).returncode
        != 0
    )
    assert (
        subprocess.run(
            ["bash", str(script), "resident", "--run-id", "test"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )


def test_formal_consumers_reject_analysis_marker_before_sanitizing():
    from flexmoe.bench.partial_suite import analyze_triplet, public_run

    # A missing/forged diagnostic boolean must not let known analysis kinds in.
    for kind in (
        "native-summary",
        "analysis-smoke",
        "analysis-repetition",
        "feasibility-analysis",
        "offload-analysis-report",
    ):
        row = {
            "artifact_kind": kind,
            "run_id": "x",
            "status": "complete",
            "arm": "resident",
        }
        with pytest.raises(ValueError, match="analysis"):
            public_run(row)
        assert analyze_triplet(row, row, row)["status"] == "invalid-comparison"


def test_container_seam_argv_has_inside_timeout_and_literal_custom_path(tmp_path):
    runner = tmp_path / "scripts/server/run_container.sh"
    runner.parent.mkdir(parents=True)
    # External Docker boundary is replaced by a shell argument recorder only.
    runner.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    script = ROOT / "scripts/server/run_offload_analysis.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; analysis_container "$2" 123 python3 -m flexmoe.bench.analysis_runner --model-path "$3"',
            "test",
            str(script),
            str(tmp_path),
            "/mnt/public_data/model with spaces",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv.index("timeout") < argv.index("python3")
    assert argv[argv.index("--kill-after=30") + 1] == "123"
    assert argv[-1] == "/mnt/public_data/model with spaces"
    assert "HF_HUB_OFFLINE=1" in argv


def test_public_whitelist_never_copies_nested_commands_versions_or_trace_events():
    from flexmoe.analysis.report import public_artifact

    raw = diagnostic_artifact(
        "analysis-suite",
        {
            "analyses": [
                {
                    "status": "incomplete",
                    "command": "SECRET",
                    "assumptions": ["SECRET", "synthetic-workload"],
                    "baseline": {"point_id": "SECRET", "engine_mode": "SECRET"},
                    "contract": {
                        "versions": {"torch": "/private/SECRET"},
                        "input_sha256": "d" * 64,
                    },
                    "trace": {"events": [{"experts": [1]}]},
                }
            ]
        },
    )
    exported = public_artifact(raw)
    text = json.dumps(exported)
    assert "SECRET" not in text and "events" not in text
    assert "d" * 64 in text


def test_wrapper_private_public_namespaces_and_symlink_escape(tmp_path):
    script = ROOT / "scripts/server/run_offload_analysis.sh"

    def check(mode, path):
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; analysis_output_path "$2" "$3" "$4"',
                "test",
                str(script),
                str(tmp_path),
                mode,
                str(path),
            ],
            capture_output=True,
            check=False,
        ).returncode

    assert check("plan", "runs/offload-analysis/plan.json") == 0
    assert check("analyze", "docs/results/offload-analysis-private.json") != 0
    assert check("export", "docs/results/offload-analysis-public") == 0
    assert check("export", "runs/offload-analysis/public") != 0
    outside = tmp_path / "outside"
    outside.mkdir()
    namespace = tmp_path / "runs/offload-analysis"
    namespace.mkdir(parents=True)
    (namespace / "escape").symlink_to(outside, target_is_directory=True)
    assert check("replay", "runs/offload-analysis/escape/private.json") != 0


def test_transport_model_flag_only_reaches_preflight(tmp_path):
    script = ROOT / "scripts/server/run_offload_analysis.sh"
    for mode in ("resident", "trace", "transport"):
        result = subprocess.run(
            [
                "bash",
                "-c",
                (
                    'source "$1"; analysis_mode="$2"; analysis_parse_gpu_args --run-id fresh '
                    '--model-path "$3" --expert-bytes 42; printf "%s\\n" "$analysis_model" "${analysis_args[@]}"'
                ),
                "test",
                str(script),
                mode,
                "/mnt/public_data/model with spaces",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0] == "/mnt/public_data/model with spaces"
        assert ("--model-path" in lines[1:]) == (mode != "transport")
        assert lines[-2:] == ["--expert-bytes", "42"]


@pytest.mark.parametrize(
    "flag", ["--project", "--run-d", "--mo", "--project-root=/outside"]
)
def test_wrapper_rejects_argparse_abbreviation_escape(flag):
    script = ROOT / "scripts/server/run_offload_analysis.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; analysis_mode=resident; analysis_parse_gpu_args "$2" /outside',
            "test",
            str(script),
            flag,
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("status,trace_value", [("failed", None), ("truncated", None)])
def test_failed_null_trace_is_never_complete_in_public_headline(
    tmp_path, status, trace_value
):
    source = save(
        tmp_path / "trace.json",
        diagnostic_artifact(
            "demand-trace",
            {
                "trace": trace_value,
                "capture": {
                    "status": status,
                    "full_workload": False,
                    "truncated": status == "truncated",
                },
                "generation_status": "failed" if status == "failed" else "complete",
                "timing_eligible": False,
                "missing_evidence": ["no-complete-step-layer-grid"],
            },
        ),
    )
    output = tmp_path / "report"
    assert invoke("export", "--source", source, "--output", output).returncode == 0
    row = json.loads((output / "report.json").read_text())["records"][0]
    assert row["status"] in {"failed", "incomplete", "truncated"}
    assert row["capture"]["full_workload"] is False
    md = (output / "report.md").read_text()
    assert "|0|demand-trace|complete|" not in md
    assert f"|0|demand-trace|{row['status']}|" in md


def test_trace_summary_reports_capture_incomplete_despite_completed_generation():
    from flexmoe.analysis.report import public_artifact

    raw = native()
    raw.update(
        engine_mode="trace",
        timing_eligible=False,
        measurement_evidence="instrumented-not-throughput",
        capture_status="incomplete",
    )
    result = public_artifact(raw)
    assert result["status"] == "incomplete"
    assert result["generation_status"] == "complete"
    assert result["capture_status"] == "incomplete"


@pytest.mark.parametrize("mode", ["transport", "resident", "trace"])
def test_actual_wrapper_runner_argv_propagates_transport_deadline(tmp_path, mode):
    runner = tmp_path / "scripts/server/run_container.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    script = ROOT / "scripts/server/run_offload_analysis.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; analysis_root="$2"; analysis_run="$2/runs/offload-analysis/fresh"; '
                'analysis_mode="$3"; analysis_parse_gpu_args --run-id fresh --timeout-s 999 '
                '--model-path "/mnt/public_data/custom model" --warmups 7; analysis_run_gpu'
            ),
            "test",
            str(script),
            str(tmp_path),
            mode,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv[argv.index("--kill-after=30") + 1] == "999"
    producer = argv[argv.index("python3") + 1 :]
    assert producer[0] == "-m"
    if mode == "transport":
        assert producer[1] == "flexmoe.bench.transfer_microbench"
        assert producer[producer.index("--timeout-s") + 1] == "999"
        assert "--model-path" not in producer
    else:
        assert producer[1] == "flexmoe.bench.analysis_runner"
        assert "--timeout-s" not in producer
    assert producer[producer.index("--warmups") + 1] == "7"
    assert not (tmp_path / "runs/offload-analysis/fresh").exists()


def test_public_export_retains_replay_transport_provenance_without_private_payloads(
    tmp_path,
):
    from dataclasses import replace

    from test_analysis_cli import transport_artifact
    from test_analysis_cost import trace

    from flexmoe.analysis.replay import replay_trace
    from flexmoe.analysis.schema import ReplayConfig
    from flexmoe.bench.transfer_microbench import aggregate_rows

    source = tmp_path / "source"
    source.mkdir()
    demand = replace(
        trace(0, event_rows=((5, 6), (5,))), full_workload=False, generated_tokens=None
    )
    replay = replay_trace(
        demand, ReplayConfig(((0, 1, 2, 3, 4),), 0, 1, "future")
    ).to_dict()
    phases, reuse = replay["per_phase_totals"], replay["reuse_gap_summary"]
    assert isinstance(phases, dict) and isinstance(reuse, dict)
    phases["SECRET-PRIVATE-PHASE"] = {"demands": 987}
    reuse["private"] = "/private/SECRET"
    save(
        source / "replay.json",
        diagnostic_artifact(
            "replay-suite",
            {"replays": [replay], "calibration_input_hashes": ["c" * 64] * 4},
        ),
    )
    raw = transport_artifact(samples())
    raw["aggregates"] = aggregate_rows(raw["samples"])
    raw["worker_metadata"] = [
        {
            "rank": 0,
            "cpu_affinity_sha256": "e" * 64,
            "topology_status": "NUMA/PCIe-unavailable",
            "uuid": "GPU-SECRET",
            "cpu_affinity": [99999],
        },
        {
            "rank": 1,
            "cpu_affinity_sha256": None,
            "topology_status": "NUMA/PCIe-unavailable",
        },
    ]
    save(source / "samples.json", raw)
    save(source / "summary.json", native())
    output = tmp_path / "public"
    result = invoke("export", "--source", source, "--output", output)
    assert result.returncode == 0, result.stderr
    records = {
        r["artifact_kind"]: r
        for r in json.loads((output / "report.json").read_text())["records"]
    }
    public_replay = records["replay-suite"]["replays"][0]
    assert public_replay["config"]["resident_count_by_layer"] == [5]
    assert public_replay["config"]["resident_count_total"] == 5
    assert public_replay["per_layer_totals"][0]["loaded_bytes"] == 300
    assert public_replay["per_phase_totals"]["decode"]["demands"] == 3
    assert public_replay["reuse_gap_summary"]["reuse_count"] == 1
    assert records["replay-suite"]["calibration_input_hashes"] == ["c" * 64] * 4
    assert set(public_replay["assumptions"]) == {
        "one-load-per-unique-layer-expert-per-event",
        "event-start-hit-classification",
        "persistent-global-cache",
        "partial-window-not-full-workload",
        "staging-overflow-requires-unmodelled-chunking",
        "ideal-future-reference-not-deployed",
        "not-a-strict-all-system-optimum",
    }
    public_transport = records["transfer-samples"]
    assert public_transport["aggregates"][0]["per_rank_payload_bytes"] == 100
    assert public_transport["aggregates"][0]["bottleneck_bytes_per_s"] == 1000
    assert public_transport["worker_metadata"][0]["cpu_affinity_sha256"] == "e" * 64
    assert public_transport["worker_metadata"][1]["cpu_affinity_sha256"] is None
    assert records["native-summary"]["contract"]["tensor_parallel_size"] == 4
    assert records["native-summary"]["contract"]["dtype"] == "bfloat16"
    for name in ("report.json", "report.csv", "report.md"):
        text = (output / name).read_text()
        for field in (
            "resident_count_by_layer",
            "resident_count_total",
            "per_layer_totals",
            "per_phase_totals",
            "reuse_gap_summary",
            "calibration_input_hashes",
            "aggregates",
            "cpu_affinity_sha256",
            "tensor_parallel_size",
            "bfloat16",
        ):
            assert field in text, (name, field)
        for private in (
            "SECRET",
            "/private",
            "resident_experts",
            "event_load_counts",
            "event_load_bytes",
            "final_cache",
        ):
            assert private not in text, (name, private)
    assert b"\r" not in (output / "report.csv").read_bytes()


def test_specialized_public_summaries_reject_untyped_values_and_private_keys():
    from flexmoe.analysis.report import public_fields

    raw = {
        "calibration_input_hashes": ["c" * 64, "/private/SECRET"],
        "per_phase_totals": {
            "decode": {"demands": True, "loaded_bytes": 100},
            "SECRET": {"demands": 7},
        },
        "reuse_gap_summary": {
            "reuse_count": False,
            "mean_event_gap": float("nan"),
            "private": "SECRET",
        },
        "cpu_affinity_sha256": "GPU-SECRET",
        "dtype": "/private/SECRET",
        "tensor_parallel_size": True,
    }
    public = public_fields(raw)
    assert public["calibration_input_hashes"] == ["c" * 64]
    assert public["per_phase_totals"] == {"decode": {"loaded_bytes": 100}}
    assert public["reuse_gap_summary"] == {}
    assert "cpu_affinity_sha256" not in public
    assert "tensor_parallel_size" not in public
    assert "SECRET" not in json.dumps(public)
