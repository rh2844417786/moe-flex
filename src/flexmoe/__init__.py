"""FluxMoE cross-hardware reproduction package."""

import torch as _torch  # Load libtorch/libc10 before importing flexmoe._C.

__version__ = "0.1.0"
_TORCH_RUNTIME_VERSION = _torch.__version__

__all__ = ["__version__"]
