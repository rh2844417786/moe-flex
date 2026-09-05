from types import SimpleNamespace

import pytest
import torch

from flexmoe.vllm.bridge import FluxMoEWorkerExtension
from flexmoe.vllm.worker_memory import cache_storage_bytes, worker_memory_stats


def test_cache_views_count_the_same_raw_allocation_only_once() -> None:
    raw = torch.zeros(100, dtype=torch.uint8)
    other = torch.zeros(40, dtype=torch.uint8)
    assert cache_storage_bytes({"a": raw[:50], "b": [raw[50:], other]}) == 140
    assert cache_storage_bytes([]) is None


def test_resident_statistics_do_not_require_an_offload_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLUXMOE_ENABLE", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    worker = FluxMoEWorkerExtension()
    worker.rank = 2
    result = worker.fluxmoe_partial_stats()
    assert result["rank"] == 2
    assert result["h2d_bytes"] == 0
    assert result["copy_launches"] == 0


def test_manual_kv_budget_and_real_allocations_remain_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (600, 1000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 300)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 350)
    raw = torch.zeros(100, dtype=torch.uint8)
    worker = SimpleNamespace(
        rank=0,
        device=torch.device("cuda", 0),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=128),
        model_runner=SimpleNamespace(
            model_memory_usage=40,
            kv_caches=[raw[:70], raw[70:]],
            kv_cache_config=SimpleNamespace(
                num_blocks=5, kv_cache_tensors=[SimpleNamespace(size=100)]
            ),
        ),
    )
    stats = worker_memory_stats(worker)
    assert stats["available_kv_cache_bytes"] == 128
    assert stats["kv_cache_allocated_bytes"] == 100
    assert stats["num_gpu_blocks"] == 5
    assert stats["kv_cache_accounting_consistent"] is True
    worker.model_runner.kv_cache_config.kv_cache_tensors[0].size = 101
    assert worker_memory_stats(worker)["kv_cache_accounting_consistent"] is False
