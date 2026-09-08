import importlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def module():
    return importlib.import_module("flexmoe.bench.transfer_microbench")


def test_exact_payload_and_physical_resource_checks():
    cfg = module().TransferConfig(
        expert_bytes=16, experts_per_batch=(1, 8), memory_cap_bytes=2048
    )
    assert module().payload_shape(cfg, 8) == (8, 16)
    assert module().required_memory(cfg, 8)["device_bytes"] == 128
    for kwargs in (
        {"expert_bytes": 0},
        {"repetitions": True},
        {"timeout_s": -1},
        {"experts_per_batch": (0,)},
        {"modes": ("bad",)},
    ):
        with pytest.raises(ValueError):
            module().TransferConfig(**kwargs)
    with pytest.raises(ValueError, match="cap"):
        module().payload_shape(module().TransferConfig(memory_cap_bytes=10), 1)


def test_aggregation_uses_wall_bottleneck_not_mean():
    rows = [
        {
            "rank": i,
            "experts_per_batch": 8,
            "mode": "gather",
            "contention": "isolated",
            "repetition": 0,
            "wall_s": float(i + 1),
            "payload_bytes": 128,
        }
        for i in range(4)
    ]
    result = module().aggregate_rows(rows)
    assert result[0]["bottleneck_wall_s"] == 4
    assert result[0]["bottleneck_rank"] == 3
    assert result[0]["per_rank_payload_bytes"] == 128
    with pytest.raises(ValueError, match="rank"):
        module().aggregate_rows(rows[:3])


def test_owned_worker_cleanup_on_failure():
    class Child:
        def __init__(self, code):
            self.exitcode = code
            self.killed = False

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.killed = True
            self.exitcode = -15

        def join(self, timeout=None):
            pass

    children = [Child(1), Child(None), Child(None), Child(None)]
    with pytest.raises(RuntimeError, match="worker"):
        module().wait_owned_workers(children, timeout_s=1)
    assert all(c.killed for c in children[1:])


def test_coordinator_writes_four_rank_samples_and_separate_proxy_windows(
    tmp_path, monkeypatch
):
    from flexmoe.analysis.schema import TransferSample
    from flexmoe.bench.partial_runner import atomic_json

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.8.0",
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(device_count=lambda: 4),
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.10.2"))
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "a" * 40)
    monkeypatch.setattr(module().subprocess, "check_output", lambda *a, **kw: "c" * 40)
    children = []

    class Child:
        exitcode = None

        def __init__(self, target, args):
            self.args = args
            children.append(self)

        def start(self):
            rank, cfg, directory, contract = self.args
            contract = {**contract, "hardware_sha256": "b" * 64}
            sample = TransferSample(
                rank,
                contract,
                1,
                16,
                "contiguous",
                cfg.contention,
                0,
                16,
                4 + rank,
                1,
                0,
                2,
            )
            atomic_json(
                __import__("pathlib").Path(directory) / f"worker-{rank}.json",
                {
                    "status": "complete",
                    "rank": rank,
                    "samples": [sample.to_dict()],
                    "measurements": [
                        {
                            "copy_only_wall_s": 1.5,
                            "compute_only_wall_s": 2.5,
                            "joint_wall_s": 4 + rank,
                        }
                    ],
                },
            )
            self.exitcode = 0

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(
        module().multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(Process=Child),
    )
    run = tmp_path / "transport"
    module().run_transfer(
        module().TransferConfig(
            expert_bytes=16,
            experts_per_batch=(1,),
            modes=("contiguous",),
            repetitions=1,
            contention="gemm-nccl-proxy",
        ),
        project_root=tmp_path,
        run_dir=run,
    )
    saved = json.loads((run / "samples.json").read_text())
    assert saved["artifact_kind"] == "transfer-samples"
    assert saved["status"] == "complete"
    assert len(saved["samples"]) == 4
    assert saved["aggregates"][0]["bottleneck_wall_s"] == 7
    assert saved["measurements"][0]["copy_only_wall_s"] == 1.5
    assert "not-transfer-tax" in saved["contention_role"]
    assert not any(child.is_alive() for child in children)


def test_missing_cuda_persists_failed_state(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.8.0", version=SimpleNamespace(cuda="12.8")),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.10.2"))

    class FailedChild:
        exitcode = 1

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(
        module().multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(Process=FailedChild),
    )
    monkeypatch.setattr(module().subprocess, "check_output", lambda *a, **kw: "c" * 40)
    with pytest.raises(RuntimeError, match="worker"):
        module().run_transfer(
            module().TransferConfig(),
            project_root=tmp_path,
            run_dir=tmp_path / "failed",
        )
    saved = json.loads((tmp_path / "failed" / "samples.json").read_text())
    assert saved["status"] == "failed"
    assert saved["diagnostic_only"] is True
    assert saved["samples"] == []


def test_gather_rejects_amortized_cpu_cost(tmp_path):
    with pytest.raises(ValueError, match="iterations=1"):
        module().run_transfer(
            module().TransferConfig(iterations=2),
            project_root=tmp_path,
            run_dir=tmp_path / "gather",
        )
    assert not (tmp_path / "gather").exists()


def test_coordinator_rejects_missing_shape_or_contract_even_with_four_ranks():
    cfg = module().TransferConfig(
        expert_bytes=16, experts_per_batch=(1, 8), modes=("contiguous",), repetitions=1
    )
    rows = [
        {
            "rank": r,
            "experts_per_batch": 1,
            "mode": "contiguous",
            "repetition": 0,
            "contract": {"id": "a"},
            "contention": "isolated",
            "expert_bytes": 16,
        }
        for r in range(4)
    ]
    with pytest.raises(ValueError, match="coverage"):
        module().validate_sample_coverage(rows, cfg, {"id": "a"})
    rows += [{**r, "experts_per_batch": 8} for r in rows]
    module().validate_sample_coverage(rows, cfg, {"id": "a"})
    rows[0]["contract"] = {"id": "b"}
    with pytest.raises(ValueError, match="contract"):
        module().validate_sample_coverage(rows, cfg, {"id": "a"})


@pytest.mark.parametrize("mode", ["contiguous", "fragmented", "gather"])
@pytest.mark.parametrize("contention", ["isolated", "gemm-nccl-proxy"])
def test_real_cpu_copy_gather_and_proxy_math_with_cuda_boundary_controlled(
    mode, contention, monkeypatch
):
    import torch

    destinations = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0.1

    def empty(shape, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("pin_memory", None)
        result = torch.zeros(shape, **kwargs)
        destinations.append(result)
        return result

    def full(shape, value, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("pin_memory", None)
        return torch.full(shape, value, **kwargs)

    reduced = []

    def all_reduce(tensor, async_op):
        tensor.mul_(4)
        reduced.append(tensor.clone())
        return SimpleNamespace(wait=lambda: None)

    cuda = SimpleNamespace(
        mem_get_info=lambda: (10**12, 10**12),
        Stream=lambda: object(),
        stream=lambda stream: nullcontext(),
        synchronize=lambda: None,
        Event=Event,
    )
    boundary = SimpleNamespace(
        cuda=cuda,
        uint8=torch.uint8,
        bfloat16=torch.bfloat16,
        empty=empty,
        full=full,
        empty_like=torch.empty_like,
        stack=torch.stack,
        mm=torch.mm,
    )
    dist = SimpleNamespace(barrier=lambda: None, all_reduce=all_reduce)
    clock = iter(range(100))
    monkeypatch.setattr(module().time, "perf_counter", lambda: float(next(clock)))
    cfg = module().TransferConfig(
        expert_bytes=8,
        experts_per_batch=(2,),
        modes=(mode,),
        warmups=0,
        repetitions=2,
        contention=contention,
        gemm_size=2,
    )
    rows = list(module()._measure_case(boundary, dist, cfg, 2, mode))
    want = [[17] * 8] * 2 if mode == "contiguous" else [[0] * 8, [1] * 8]
    assert destinations[0].tolist() == want
    assert len(rows) == 2
    assert rows[0]["gather_s"] == (1.0 if mode == "gather" else 0.0)
    assert (rows[0]["compute_only_wall_s"] is not None) == (contention != "isolated")
    if contention != "isolated":
        assert reduced[-1].tolist() == [[2.0, 2.0], [2.0, 2.0]]
        assert rows[0]["joint_wall_s"] > rows[0]["compute_s"]
    barriers = 0

    def fail_after_first_observation():
        nonlocal barriers
        barriers += 1
        if barriers > (1 if contention == "isolated" else 3):
            raise RuntimeError("next observation failed")

    dist.barrier = fail_after_first_observation
    stream = iter(module()._measure_case(boundary, dist, cfg, 2, mode))
    assert next(stream)["repetition"] == 0
    with pytest.raises(RuntimeError, match="next observation"):
        next(stream)


def test_worker_checkpoints_before_later_failure_and_retains_contract(
    tmp_path, monkeypatch
):
    from flexmoe.analysis.schema import parse_diagnostic_artifact
    from flexmoe.vllm import analysis_trace

    contract = {
        "commit": "a" * 40,
        "tensor_parallel_size": 4,
        "versions": {
            "torch": "2.8",
            "vllm": "0.10.2",
            "cuda": "12.8",
            "vllm_commit": "b" * 40,
        },
    }
    monkeypatch.setattr(
        analysis_trace,
        "device_record",
        lambda rank: {"rank": rank, "uuid": f"u{rank}", "total_memory": 80000},
    )

    def gather(rows, row):
        rows[:] = [
            {"rank": i, "uuid": f"u{i}", "total_memory": 80000} for i in range(4)
        ]

    dist = SimpleNamespace(
        init_process_group=lambda *a, **kw: None,
        all_gather_object=gather,
        is_initialized=lambda: True,
        destroy_process_group=lambda: None,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 4,
            set_device=lambda rank: None,
            empty_cache=lambda: None,
        )
    )
    original_import = importlib.import_module
    monkeypatch.setattr(
        module().importlib,
        "import_module",
        lambda name: (
            fake_torch
            if name == "torch"
            else dist
            if name == "torch.distributed"
            else original_import(name)
        ),
    )

    def one_then_fail(*args):
        yield {
            "repetition": 0,
            "wall_s": 1.0,
            "copy_s": 0.5,
            "gather_s": 0.0,
            "compute_s": None,
        }
        checkpoint = parse_diagnostic_artifact(
            json.loads((tmp_path / "worker-0.json").read_text()), "transfer-worker"
        )
        assert checkpoint["status"] == "running"
        assert len(checkpoint["samples"]) == len(checkpoint["measurements"]) == 1
        raise RuntimeError("injected next-repetition failure")

    monkeypatch.setattr(module(), "_measure_case", one_then_fail)
    cfg = module().TransferConfig(
        expert_bytes=16, experts_per_batch=(1,), modes=("contiguous",), repetitions=2
    )
    with pytest.raises(RuntimeError, match="injected"):
        module()._worker(0, cfg, str(tmp_path), contract)
    failed = parse_diagnostic_artifact(
        json.loads((tmp_path / "worker-0.json").read_text()), "transfer-worker"
    )
    assert failed["status"] == "failed"
    assert failed["phase"] == "measure-1-contiguous"
    assert failed["error_type"] == "RuntimeError"
    assert len(failed["samples"]) == len(failed["measurements"]) == 1
    assert failed["contract"]["hardware_sha256"] is not None


def test_coordinator_imports_running_checkpoint_but_never_completes(
    tmp_path, monkeypatch
):
    from flexmoe.analysis.schema import TransferSample, diagnostic_artifact
    from flexmoe.bench.partial_runner import atomic_json

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.8.0", version=SimpleNamespace(cuda="12.8")),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.10.2"))
    monkeypatch.setenv("FLEXMOE_VLLM_COMMIT", "a" * 40)
    monkeypatch.setattr(module().subprocess, "check_output", lambda *a, **kw: "b" * 40)

    class Child:
        def __init__(self, target, args):
            self.args = args
            self.exitcode = None

        def start(self):
            rank, cfg, directory, contract = self.args
            if rank == 0:
                assert cfg.repetitions == 2
                contract = {**contract, "hardware_sha256": "c" * 64}
                sample = TransferSample(
                    0,
                    contract,
                    1,
                    16,
                    "contiguous",
                    "isolated",
                    0,
                    16,
                    1.0,
                    0.5,
                    0,
                    None,
                )
                atomic_json(
                    __import__("pathlib").Path(directory) / "worker-0.json",
                    diagnostic_artifact(
                        "transfer-worker",
                        {
                            "rank": 0,
                            "status": "running",
                            "phase": "measure-1-contiguous",
                            "contract": contract,
                            "samples": [sample.to_dict()],
                            "measurements": [
                                {"repetition": 0, "copy_only_wall_s": 1.0}
                            ],
                        },
                    ),
                )
            if rank == 1:
                self.exitcode = 1

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.exitcode = -15

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(
        module().multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(Process=Child),
    )
    run = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="worker"):
        module().run_transfer(
            module().TransferConfig(
                expert_bytes=16,
                experts_per_batch=(1,),
                modes=("contiguous",),
                repetitions=2,
            ),
            project_root=tmp_path,
            run_dir=run,
        )
    saved = json.loads((run / "samples.json").read_text())
    assert saved["status"] == "failed"
    assert len(saved["samples"]) == len(saved["measurements"]) == 1
    assert saved["partial_worker_artifacts"][0]["status"] == "running"
    assert "aggregates" not in saved
