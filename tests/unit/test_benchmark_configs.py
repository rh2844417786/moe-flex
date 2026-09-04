from pathlib import Path

from flexmoe.bench.runner import load_run_config


def test_all_declared_benchmark_configs_share_formal_invariants() -> None:
    config_dir = Path("benchmarks/configs")
    paths = sorted(config_dir.glob("*.yaml"))

    assert {path.name for path in paths} == {
        "fluxmoe_dynamic.yaml",
        "fluxmoe_fixed.yaml",
        "fluxmoe_host_offload.yaml",
        "fluxmoe_routed_host_offload.yaml",
        "fluxmoe_gpu_compressed.yaml",
        "fluxmoe_unbalanced.yaml",
        "pagedtensor_resident.yaml",
        "resident.yaml",
        "vllm_o.yaml",
    }
    configs = [load_run_config(path) for path in paths]
    assert {config.variant for config in configs} == {
        "resident",
        "vllm-o",
        "fluxmoe-fixed",
        "fluxmoe-host-offload",
        "fluxmoe-routed-host-offload",
        "fluxmoe-gpu-compressed",
        "fluxmoe-dynamic",
        "fluxmoe-unbalanced",
        "pagedtensor-resident",
    }
    assert all(config.tensor_parallel_size == 4 for config in configs)
    assert all(config.gpu_memory_utilization == 0.60 for config in configs)
    assert all(config.warmups == 3 and config.repetitions == 3 for config in configs)
    gpu_compressed = next(
        config for config in configs if config.variant == "fluxmoe-gpu-compressed"
    )
    assert gpu_compressed.storage_mode == "gpu-compressed"
    assert gpu_compressed.gpu_materialization_mode == "batched"
    host_offload = next(
        config for config in configs if config.variant == "fluxmoe-host-offload"
    )
    assert host_offload.gpu_compressed_budget_bytes == 0
    assert host_offload.storage_mode == "hybrid"
    assert host_offload.gpu_materialization_mode == "expertwise"
    routed = next(
        config
        for config in configs
        if config.variant == "fluxmoe-routed-host-offload"
    )
    assert routed.gpu_compressed_budget_bytes == 0
    assert routed.storage_mode == "hybrid"
    assert routed.gpu_materialization_mode == "expertwise"
