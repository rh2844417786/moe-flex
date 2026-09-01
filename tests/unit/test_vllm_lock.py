import json
from pathlib import Path


def test_vllm_lock_is_exact() -> None:
    lock = json.loads(Path("third_party/vllm.lock.json").read_text())

    assert lock == {
        "commit": "01efc7ef781391e744ed08c3292817a773d654e6",
        "patch_sha256": (
            "0a706dba0b462539e3de39276e93a63f93a14ec8fc0ea8638b1b430fba2eb0d6"
        ),
        "repository": "https://github.com/vllm-project/vllm.git",
        "tag": "v0.10.2",
        "torch": "2.8.0",
    }
