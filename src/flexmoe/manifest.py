"""Canonical manifest serialization and streaming hashes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize mappings deterministically for hashing and storage."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
