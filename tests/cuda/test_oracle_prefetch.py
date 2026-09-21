import pytest
import torch


@pytest.mark.cuda
def test_cuda_oracle_prefetch_copies_real_bf16_payload_and_waits_before_use():
    from flexmoe.runtime.oracle_prefetch import CudaOracleTransferBackend

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = torch.arange(32, dtype=torch.bfloat16, pin_memory=True)
    destination = torch.empty_like(source, device="cuda")
    backend = CudaOracleTransferBackend(torch.device("cuda", 0), enable_timing=True)

    ticket = backend.enqueue(((destination, source),))
    backend.wait(ticket)
    backend.synchronize()

    assert torch.equal(destination.cpu(), source)
    assert backend.elapsed_ms(ticket) is not None
    assert backend.elapsed_ms(ticket) >= 0
