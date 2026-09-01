import pytest


@pytest.mark.cuda
def test_cuda_extension_reports_pinned_version() -> None:
    import flexmoe._C as extension

    assert extension.extension_version() == "0.1.0"
    assert extension.cuda_driver_version() > 0
