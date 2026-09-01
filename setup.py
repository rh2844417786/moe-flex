"""Optional CUDA extension build for moe-flex."""

from __future__ import annotations

import os

from setuptools import setup

build_cuda = os.environ.get("FLEXMOE_BUILD_CUDA") == "1"
ext_modules = []
cmdclass = {}

if build_cuda:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            "flexmoe._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/codec/huffman_cuda.cu",
                "csrc/runtime/stream_lifecycle.cu",
                "csrc/vmm/paged_region.cpp",
            ],
            extra_link_args=["-lcuda"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
