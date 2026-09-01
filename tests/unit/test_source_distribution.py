import subprocess
import sys
import tarfile
from pathlib import Path


def test_source_distribution_contains_cuda_build_inputs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    archive = next(tmp_path.glob("moe_flex-*.tar.gz"))

    with tarfile.open(archive, "r:gz") as stream:
        names = stream.getnames()

    assert any(name.endswith("/csrc/bindings.cpp") for name in names)
    assert any(name.endswith("/csrc/codec/huffman_cuda.cu") for name in names)
    assert any(name.endswith("/csrc/codec/huffman.h") for name in names)
    assert any(name.endswith("/csrc/runtime/stream_lifecycle.cu") for name in names)
    assert any(name.endswith("/csrc/runtime/stream_lifecycle.h") for name in names)
    assert any(name.endswith("/csrc/vmm/paged_region.cpp") for name in names)
    assert any(name.endswith("/csrc/vmm/paged_region.h") for name in names)
    assert any(name.endswith("/third_party/vllm.lock.json") for name in names)
    assert any(name.endswith("/patches/vllm-v0.10.2.patch") for name in names)
    assert any(name.endswith("/docker/Dockerfile") for name in names)
