import json


def state_fixture():
    return {
        "run_id": "r",
        "commit": "a" * 40,
        "status": "complete",
        "capacity": {
            "status": "extra-capacity-measured",
            "native_bytes_per_rank": 100,
            "eager_bytes_per_rank": 90,
            "resident_bytes_per_rank": 100,
            "offload_bytes_per_rank": 120,
            "extra_bytes_per_rank": 20,
        },
        "selected_capacities": {"native_kv": 100, "k_pair": 90, "offload_kv": 120},
        "zero_miss": {"status": "proven-zero-miss", "per_rank_repetition": []},
        "zero_pair": {"status": "measured-pair", "offload_over_resident": 0.8},
        "fixed_batch_gates": {"mechanism-offload-32": {"status": "measured"}},
        "kv_intervention": {
            "status": "measured-kv-intervention",
            "large_over_small_throughput": 1.2,
        },
        "replay_16": {
            "status": "baseline-reproduced-all-ranks",
            "current_policy_misses": 20,
            "lru_misses": 18,
            "future_aware_misses": 10,
        },
        "replay_32": {"status": "incomplete-evidence"},
        "oracle_output_gate_16": {"status": "outputs-match"},
        "oracle_output_gate_32": {"status": "outputs-match"},
        "points": {
            "r-oracle-32-h2": {
                "label": "oracle-32-h2",
                "mode": "offload",
                "status": "complete",
                "argv": ["--prefetch-horizon", "2"],
            },
            "r-oracle-32-h2-profile": {
                "label": "oracle-32-h2-profile",
                "mode": "offload",
                "status": "complete",
                "argv": ["--profile", "--prefetch-horizon", "2"],
            },
            "r-service-offload-k1": {
                "label": "service-offload-k1",
                "mode": "offload",
                "status": "failed",
                "error_type": "RuntimeError",
                "argv": [],
            },
        },
        "numeric": {
            "r-oracle-32-h2": {
                "throughput": 500.0,
                "actual_kv": 120,
                "phase_status": "measured",
                "oracle_per_repetition": [
                    {
                        "status": "measured",
                        "per_rank": [
                            {"rank": rank, "ready_before_use_ratio": 0.9}
                            for rank in range(4)
                        ],
                    }
                ],
            },
            "r-oracle-32-h2-profile": {
                "throughput": 1.0,
                "actual_kv": 120,
                "oracle_per_repetition": [
                    {
                        "status": "measured",
                        "per_rank": [
                            {"rank": rank, "exposed_wait_ms_p95": 2.0}
                            for rank in range(4)
                        ],
                    }
                ],
            },
        },
    }


def test_report_embeds_all_eight_required_sections_and_failures():
    from flexmoe.analysis.decode_oracle_report import build_oracle_report

    public, markdown = build_oracle_report(state_fixture())

    for heading in (
        "无采样零 miss 固定开销",
        "执行模式分解",
        "K0 与 K1 稳定容量",
        "最大并发 32 服务收益",
        "同预算缓存重放",
        "真实 Oracle horizon 0/1/2",
        "强制漏预测正确性",
        "失败、超时与不可用指标",
    ):
        assert heading in markdown
    assert public["oracle"]["oracle-32-h2"]["timing_eligible"] is True
    assert public["oracle"]["oracle-32-h2-profile"]["timing_eligible"] is False
    assert public["failed_points"][0]["label"] == "service-offload-k1"
    assert public["decision"] == "runtime-first"
    assert "1.00 tokens/s" not in markdown
    encoded = json.dumps(public, ensure_ascii=False)
    assert "actual_expert_ids" not in encoded
    assert "prompt_token_ids" not in encoded


def test_summary_conflict_repair_requires_three_measured_rows_and_identity():
    from flexmoe.analysis.decode_oracle_report import repair_summary_conflict

    summary = {"status": "unavailable", "contract_sha256": "abc"}
    csv_text = """kind,rank,repetition,metric,value,status
identity,,,contract_sha256,abc,measured
repetition,,0,elapsed_s,1.0,complete
repetition,,1,elapsed_s,1.1,complete
repetition,,2,elapsed_s,1.2,complete
"""
    repaired = repair_summary_conflict(summary, csv_text)
    assert repaired["status"] == "measured-repaired"
    assert repaired["original_status"] == "unavailable"
    assert len(repaired["repair_source_sha256"]) == 64

    wrong = csv_text.replace("contract_sha256,abc", "contract_sha256,different")
    unchanged = repair_summary_conflict(summary, wrong)
    assert unchanged["status"] == "unavailable"
    assert unchanged["repair_status"] == "identity-mismatch"
