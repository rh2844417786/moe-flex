from pathlib import Path

from flexmoe.vllm.patch_contract import validate_patch_contract


def test_patch_contract_is_pinned_and_minimal() -> None:
    contract = validate_patch_contract(
        lock_path=Path("third_party/vllm.lock.json"),
        patch_path=Path("patches/vllm-v0.10.2.patch"),
    )

    assert contract.commit == "01efc7ef781391e744ed08c3292817a773d654e6"
    assert contract.patch_sha256 == (
        "d139a2ab3971eb3a3c0044cb21791575b5e2c9dff9651a072c4a7fcbfd9b4777"
    )
    assert contract.touched_files == (
        "vllm/model_executor/layers/fused_moe/layer.py",
    )
