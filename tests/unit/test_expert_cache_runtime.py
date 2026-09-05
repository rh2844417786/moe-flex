from __future__ import annotations

import json

import pytest
import torch

from flexmoe.runtime.expert_cache_policy import ExpertCachePolicy
from flexmoe.runtime.expert_pool import ExpertPool
from flexmoe.vllm.expert_calibration import ExpertCalibration, model_profile_identity


class TensorBackend:
    """Synchronous real tensor storage, exclusively for CPU mechanism tests."""

    pin_memory = False
    device = torch.device("cpu")

    def begin(self):
        pass

    def wait_host(self):
        pass

    def uploaded(self):
        pass

    def end(self):
        pass

    def synchronize(self):
        pass

    def mark(self, stage):
        pass

    def timing(self, reset):
        return {"cuda_sample_count": 0}


def make_pool(ratio=0.0, slots=2):
    policy = ExpertCachePolicy(
        2, 4, ratio, slots, policy="lru", initial_counts=[[1, 9, 2, 3], [8, 1, 2, 3]]
    )
    sources = [
        (
            torch.arange(32, dtype=torch.bfloat16).reshape(4, 4, 2) + i * 100,
            torch.arange(16, dtype=torch.bfloat16).reshape(4, 2, 2) + i * 100,
        )
        for i in range(2)
    ]
    pool = ExpertPool(policy, sources, backend=TensorBackend())
    return pool, sources


def invoke(pool, sources, layer, ids):
    ids = torch.tensor([ids], dtype=torch.int32)

    def compute(w13, w2, mapping):
        for expert in ids.unique().tolist():
            slot = int(mapping[expert])
            assert slot >= 0
            assert torch.equal(w13[slot], sources[layer][0][expert])
            assert torch.equal(w2[slot], sources[layer][1][expert])
        return torch.tensor([42])

    assert pool.execute(layer, ids, compute).item() == 42


def test_repeat_hits_avoid_copies_and_cross_layer_collision_is_safe():
    pool, sources = make_pool()
    invoke(pool, sources, 0, [0, 1, 1])
    assert pool.stats()["h2d_bytes"] == 48
    invoke(pool, sources, 0, [0, 1])
    assert pool.stats()["h2d_bytes"] == 48
    invoke(pool, sources, 1, [0, 1])
    assert pool.stats()["h2d_bytes"] == 96
    invoke(pool, sources, 0, [0, 1])
    assert pool.stats()["h2d_bytes"] == 144
    assert pool.stats()["policy"]["cache_hits"] == 2


def test_per_layer_unique_coverage_and_snapshot_copies():
    pool, sources = make_pool()
    invoke(pool, sources, 0, [0, 1, 1])
    invoke(pool, sources, 1, [0, 1, 2, 3])
    invoke(pool, sources, 0, [2])
    snapshot = pool.stats()
    assert snapshot["per_layer_unique_demands"] == [3, 4]
    assert snapshot["per_layer_max_unique_per_forward"] == [2, 4]
    assert snapshot["forward_counts"] == [2, 1]
    assert snapshot["unique_demands"] == sum(snapshot["per_layer_unique_demands"]) == 7
    assert snapshot["max_unique_per_forward"] == 4
    snapshot["per_layer_unique_demands"][0] = 99
    snapshot["per_layer_max_unique_per_forward"][1] = 99
    fresh = pool.stats()
    assert fresh["per_layer_unique_demands"] == [3, 4]
    assert fresh["per_layer_max_unique_per_forward"] == [2, 4]


def test_full_demand_pins_live_entries_and_bypasses_when_cache_small():
    pool, sources = make_pool(slots=1)
    invoke(pool, sources, 0, [3])
    invoke(pool, sources, 0, [0, 1, 2, 3])
    assert pool.stats()["h2d_bytes"] == 96
    assert pool.stats()["policy"]["cache_bypasses"] == 3
    assert pool.stats()["unique_demands"] == 5


def test_resident_heat_and_fixed_capacity_reconfigure():
    pool, sources = make_pool(ratio=0.25, slots=3)
    invoke(pool, sources, 0, [1])
    assert pool.stats()["h2d_bytes"] == 0
    assert pool.stats()["gpu_pool_bytes"] == 216
    assert pool.stats()["host_source_bytes"] == 192
    assert pool.stats()["pinned_gather_bytes"] == 96
    ptr = pool.w13.data_ptr()
    pool.reconfigure(0.5)
    assert pool.w13.data_ptr() == ptr
    assert pool.stats()["cache_slots"] == 1
    invoke(pool, sources, 1, [0, 3])
    with pytest.raises(ValueError):
        pool.reconfigure(0.75)
    assert pool.stats()["cache_slots"] == 1


def test_active_reconfigure_rejected_and_kernel_failure_fail_stops():
    pool, _ = make_pool()

    def fail(*args):
        with pytest.raises(RuntimeError, match="active"):
            pool.reconfigure(0.25)
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        pool.execute(0, torch.tensor([[0]]), fail)
    with pytest.raises(RuntimeError, match="failed"):
        pool.execute(0, torch.tensor([[0]]), fail)


def test_calibration_only_explicit_start_counts_unique_and_resets():
    capture = ExpertCalibration(2, 4)
    capture.record(0, torch.tensor([[1, 2]]))
    assert capture.stop()["forward_counts"] == [0, 0]
    capture.start()
    capture.record(0, torch.tensor([[1, 1], [2, 1]]))
    capture.record(1, torch.tensor([[3, 0]]))
    result = capture.stop()
    assert result["counts"] == [[0, 1, 1, 0], [1, 0, 0, 1]]
    assert result["forward_counts"] == [1, 1]
    capture.record(0, torch.tensor([[0]]))
    assert capture.stop() == result
    capture.start()
    assert capture.stop()["forward_counts"] == [0, 0]


def test_model_identity_binds_config_index_path_and_tp(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "num_hidden_layers": 2,
                "num_experts": 4,
                "hidden_size": 16,
                "moe_intermediate_size": 32,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    identity = model_profile_identity(tmp_path, 4)
    assert identity["geometry"] == {
        "total_layers": 2,
        "num_experts": 4,
        "hidden_size": 16,
        "intermediate_size": 8,
    }
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}, "x": 1}')
    assert (
        model_profile_identity(tmp_path, 4)["model_identity_sha256"]
        != identity["model_identity_sha256"]
    )
    assert model_profile_identity(tmp_path, 2)["geometry"]["intermediate_size"] == 16


def test_registry_loads_tp_shards_and_requires_complete_model(monkeypatch):
    from flexmoe.runtime.expert_profile import ExpertProfile
    from flexmoe.vllm.expert_cache import ExpertCacheRegistry

    identity = {
        "geometry": {
            "total_layers": 2,
            "num_experts": 4,
            "hidden_size": 2,
            "intermediate_size": 2,
        },
        "model_config_sha256": "a" * 64,
        "model_identity_sha256": "b" * 64,
        "tensor_parallel_size": 4,
    }
    profile = ExpertProfile.from_dict(
        {
            "schema_version": 1,
            **identity,
            "calibration_input_hashes": ["c" * 64],
            "counts": [[1] * 4, [1] * 4],
            "forward_counts": [1, 1],
        }
    )
    registry = ExpertCacheRegistry(
        identity=identity,
        profile=profile,
        tp_rank=3,
        resident_ratio=0.25,
        cache_slots=2,
        backend=TensorBackend(),
    )
    params = []
    for layer in range(2):
        params.append(
            registry.register_layer(
                f"model.layers.{layer}.mlp.experts",
                (4, 4, 2),
                (4, 2, 2),
                torch.bfloat16,
            )
        )
    with pytest.raises(Exception, match="incomplete"):
        registry.start()
    full = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
    for layer in range(2):
        for expert in range(4):
            for shard, param, tensor in (
                ("w1", params[layer][0], full),
                ("w3", params[layer][0], full + 10),
                ("w2", params[layer][1], full.T.contiguous()),
            ):
                registry.ingest(
                    param=param,
                    loaded_weight=tensor,
                    weight_name=f"model.layers.{layer}.mlp.experts.{expert}.{shard}.weight",
                    shard_id=shard,
                    expert_id=expert,
                )
    registry.start()
    source = registry.pool.sources[0]
    assert torch.equal(source[0][0, :2], full[6:8])
    assert torch.equal(source[1][0], full.T[:, 6:8])
    assert registry.stats()["rank"] == 3
    assert registry.stats()["tensor_parallel_size"] == 4


def test_resident_calibration_rpc_runs_without_enable(monkeypatch, tmp_path):
    from flexmoe.vllm.bridge import FluxMoEWorkerExtension, record_calibration
    from flexmoe.vllm.expert_calibration import reset_calibration

    reset_calibration()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "num_hidden_layers": 2,
                "num_experts": 4,
                "hidden_size": 16,
                "moe_intermediate_size": 32,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setenv("FLUXMOE_ENABLE", "0")
    monkeypatch.setenv("FLUXMOE_EXPERT_CALIBRATION", "1")
    monkeypatch.setenv("FLUXMOE_MODEL_PATH", str(tmp_path))
    worker = FluxMoEWorkerExtension()
    worker.parallel_config = type("Parallel", (), {"tensor_parallel_size": 4})()
    worker.model_config = type(
        "Model",
        (),
        {
            "model": str(tmp_path),
            "hf_config": type(
                "HF", (), json.loads((tmp_path / "config.json").read_text())
            )(),
        },
    )()
    worker.rank = 2
    worker.fluxmoe_expert_calibration("start")
    record_calibration("model.layers.1.mlp.experts", torch.tensor([[0, 3, 3]]))
    result = worker.fluxmoe_expert_calibration("stop")
    assert result["counts"] == [[0, 0, 0, 0], [1, 0, 0, 1]]
    assert result["tensor_parallel_size"] == 4
    assert result["rank"] == 2
    reset_calibration()


def test_target_qwen3_next_identity_geometry(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "architectures": ["Qwen3NextForCausalLM"],
                "num_hidden_layers": 48,
                "num_experts": 512,
                "hidden_size": 2048,
                "moe_intermediate_size": 512,
                "intermediate_size": 5120,
                "full_attention_interval": 4,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    assert model_profile_identity(tmp_path, 4)["geometry"] == {
        "total_layers": 48,
        "num_experts": 512,
        "hidden_size": 2048,
        "intermediate_size": 128,
    }


def test_native_configuration_uses_logical_experts_and_final_chunk(monkeypatch):
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    from flexmoe.vllm.expert_cache import KernelConfigRecorder, logical_fused_experts

    selected = []
    current = None

    def choose(w13_shape, w2_shape, topk, dtype, tokens):
        assert w13_shape == (4, 4, 2) and w2_shape == (4, 2, 2)
        selected.append(tokens)
        return {"tokens": tokens}

    @contextmanager
    def override(config):
        nonlocal current
        current = config
        try:
            yield
        finally:
            current = None

    def fused(hidden, w13, w2, weights, ids, **kwargs):
        assert current["tokens"] == hidden.shape[0]
        assert kwargs["global_num_experts"] == 4
        assert w13.shape[0] == 9
        assert not kwargs["inplace"]
        return hidden + 1

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe",
        SimpleNamespace(
            get_config=lambda: current, override_config=override, fused_experts=fused
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.fused_moe",
        SimpleNamespace(
            get_config_dtype_str=lambda dtype: None, try_get_optimal_moe_config=choose
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.envs", SimpleNamespace(VLLM_FUSED_MOE_CHUNK_SIZE=3)
    )
    hidden = torch.zeros(7, 2, dtype=torch.bfloat16)
    records = KernelConfigRecorder(capacity=2)
    result = logical_fused_experts(
        hidden,
        torch.empty(9, 4, 2),
        torch.empty(9, 2, 2),
        torch.ones(7, 1),
        torch.zeros(7, 1, dtype=torch.int32),
        torch.arange(4),
        num_experts=4,
        activation="silu",
        apply_router_weight_on_input=False,
        config_recorder=records,
    )
    assert selected == [3, 3, 1]
    assert torch.equal(result, torch.ones_like(hidden))
    assert current is None
    snapshot = records.snapshot()
    assert snapshot["records"] == [
        {
            "chunk_tokens": 3,
            "w13_shape": [4, 4, 2],
            "w2_shape": [4, 2, 2],
            "top_k": 1,
            "config": {"tokens": 3},
            "selections": 2,
        },
        {
            "chunk_tokens": 1,
            "w13_shape": [4, 4, 2],
            "w2_shape": [4, 2, 2],
            "top_k": 1,
            "config": {"tokens": 1},
            "selections": 1,
        },
    ]
    for chunk_tokens in range(4, 100):
        records.record(chunk_tokens, (4, 4, 2), (4, 2, 2), 1, {"tokens": chunk_tokens})
    assert len(records.snapshot()["records"]) == 2
    assert records.snapshot()["unrecorded_selections"] == 96


def test_cuda_sampling_recycles_completed_events_across_prefill_decode(monkeypatch):
    from types import SimpleNamespace

    from flexmoe.runtime.expert_pool import CudaPoolBackend

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace())
    backend = CudaPoolBackend(0)
    for forward in range(48 * 31):
        backend.begin()
        for stage in range(4):
            backend.mark(stage)
    timing = backend.timing(reset=True)
    assert timing["cuda_sample_count"] == 48
    # Reaches all 48 layer phases and continues in the following decode phase.
    assert {i % 48 for i in range(0, 48 * 31, 31)} == set(range(48))
    backend.begin()
    for stage in range(4):
        backend.mark(stage)
    assert backend.timing(reset=False)["cuda_sample_count"] == 1


def test_pending_cuda_samples_reject_reset_without_losing_cpu_timing(monkeypatch):
    pool, sources = make_pool()
    invoke(pool, sources, 0, [0, 1])
    before = pool.stats()["timing"]

    def pending(reset):
        if reset:
            raise RuntimeError("synchronize before resetting CUDA timing")
        return {"cuda_sample_count": 0}

    monkeypatch.setattr(pool.backend, "timing", pending)
    with pytest.raises(RuntimeError, match="synchronize"):
        pool.stats(synchronize=False, reset_timing=True)
    assert pool.stats()["timing"] == before


@pytest.mark.parametrize("mismatch", ["path", "model_type", "geometry"])
def test_runtime_rejects_actual_model_mismatch_before_allocation(
    monkeypatch, tmp_path, mismatch
):
    import sys
    from types import SimpleNamespace

    from flexmoe.vllm import expert_cache

    config_fields = {
        "model_type": "qwen3_next",
        "num_hidden_layers": 2,
        "num_experts": 4,
        "hidden_size": 16,
        "moe_intermediate_size": 32,
    }
    other = tmp_path / "checkpoint-b"
    other.mkdir()
    for root in (tmp_path, other):
        (root / "config.json").write_text(json.dumps(config_fields))
        (root / "model.safetensors.index.json").write_text("{}")
    actual_fields = dict(config_fields)
    actual_path = str(tmp_path)
    if mismatch == "path":
        actual_path = str(other)
    elif mismatch == "model_type":
        actual_fields["model_type"] = "qwen3_moe"
    else:
        actual_fields["hidden_size"] = 32
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=actual_path,
            hf_config=SimpleNamespace(**actual_fields),
            enforce_eager=True,
        ),
        compilation_config=SimpleNamespace(level=0, custom_ops=["all"]),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(get_current_vllm_config=lambda: config),
    )
    monkeypatch.setenv("FLUXMOE_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(expert_cache, "_REGISTRY", None)
    # No startup profile or allocator is supplied: mismatch must fail before
    # either can be read/created, even for two identical-geometry checkpoints.
    with pytest.raises(ValueError, match="actual model"):
        expert_cache.registry_for_layer(SimpleNamespace(tp_size=4, tp_rank=0), 4)


def test_calibration_rejects_actual_model_path_mislabel(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from flexmoe.vllm.bridge import FluxMoEWorkerExtension
    from flexmoe.vllm.expert_calibration import reset_calibration

    reset_calibration()
    config = {
        "model_type": "qwen3_next",
        "num_hidden_layers": 2,
        "num_experts": 4,
        "hidden_size": 16,
        "moe_intermediate_size": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    other = tmp_path / "checkpoint-b"
    other.mkdir()
    monkeypatch.setenv("FLUXMOE_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("FLUXMOE_EXPERT_CALIBRATION", "1")
    worker = FluxMoEWorkerExtension()
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=4)
    worker.model_config = SimpleNamespace(
        model=str(other), hf_config=SimpleNamespace(**config)
    )
    with pytest.raises(ValueError, match="actual model"):
        worker.fluxmoe_expert_calibration("start")
    reset_calibration()
