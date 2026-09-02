"""FluxMoE cross-hardware reproduction package."""

import warnings

# Load libtorch/libc10 before importing flexmoe._C without leaking PyTorch's
# third-party pynvml deprecation notice into non-CUDA command-line tools.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated.*",
        category=FutureWarning,
    )
    import torch as _torch

__version__ = "0.1.0"
_TORCH_RUNTIME_VERSION = _torch.__version__

__all__ = ["__version__"]
