"""Read actual vLLM KV allocations without double-counting aliased layer views."""

from __future__ import annotations

import torch


def cache_storage_bytes(caches: object) -> int | None:
    sizes: dict[tuple[str, int | None, int], int] = {}

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            sizes[(value.device.type, value.device.index, storage.data_ptr())] = (
                storage.nbytes()
            )
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(caches)
    return sum(sizes.values()) if sizes else None


def _nonnegative_int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def worker_memory_stats(worker: object) -> dict[str, object]:
    device = getattr(worker, "device", torch.cuda.current_device())
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    runner = getattr(worker, "model_runner", None)
    cache = getattr(worker, "cache_config", None)
    kv_config = getattr(runner, "kv_cache_config", None)
    available = _nonnegative_int(
        getattr(worker, "available_kv_cache_memory_bytes", None)
    )
    if available is None:
        # vLLM's explicit-budget branch returns early before setting the
        # available_kv_cache_memory_bytes worker field.
        available = _nonnegative_int(getattr(cache, "kv_cache_memory_bytes", None))
    actual = cache_storage_bytes(getattr(runner, "kv_caches", None))
    specs = getattr(kv_config, "kv_cache_tensors", None)
    declared: int | None = None
    if isinstance(specs, (list, tuple)) and specs:
        sizes = [_nonnegative_int(getattr(spec, "size", None)) for spec in specs]
        if all(size is not None for size in sizes):
            declared = sum(size for size in sizes if size is not None)
    blocks = _nonnegative_int(getattr(kv_config, "num_blocks", None))
    if blocks is None:
        blocks = _nonnegative_int(getattr(cache, "num_gpu_blocks", None))
    return {
        "rank": _nonnegative_int(getattr(worker, "rank", None)),
        "total_gpu_bytes": total,
        "free_gpu_bytes": free,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "available_kv_cache_bytes": available,
        "kv_cache_allocated_bytes": actual,
        "kv_cache_declared_bytes": declared,
        "kv_cache_accounting_consistent": (
            actual == declared if actual is not None and declared is not None else None
        ),
        "num_gpu_blocks": blocks,
        "model_memory_bytes": _nonnegative_int(
            getattr(runner, "model_memory_usage", None)
        ),
    }
