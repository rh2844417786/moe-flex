from pathlib import Path

from flexmoe.vllm.patch_contract import validate_patch_contract


def test_patch_contract_is_pinned_and_minimal() -> None:
    contract = validate_patch_contract(
        lock_path=Path("third_party/vllm.lock.json"),
        patch_path=Path("patches/vllm-v0.10.2.patch"),
    )

    assert contract.commit == "01efc7ef781391e744ed08c3292817a773d654e6"
    assert contract.patch_sha256 == (
        "daea5a4e4ef52cc850a7c50291f60c349a6dbdc78d083ac89efc0d93e19df145"
    )
    assert contract.touched_files == (
        "vllm/model_executor/layers/fused_moe/layer.py",
    )
