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
