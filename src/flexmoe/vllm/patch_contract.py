"""Integrity checks for the single-file pinned vLLM integration patch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from flexmoe.errors import IntegrityError


@dataclass(frozen=True)
class PatchContract:
    repository: str
    tag: str
    commit: str
    patch_sha256: str
    touched_files: tuple[str, ...]


def _required_string(lock: dict[str, object], key: str) -> str:
    value = lock.get(key)
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"vLLM lock is missing string field {key}")
    return value


def validate_patch_contract(
    *, lock_path: Path, patch_path: Path
) -> PatchContract:
    try:
        parsed = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"cannot read vLLM lock: {error}") from error
    if not isinstance(parsed, dict):
        raise IntegrityError("vLLM lock must be a JSON object")
    lock = cast(dict[str, object], parsed)

    try:
        patch_bytes = patch_path.read_bytes()
    except OSError as error:
        raise IntegrityError(f"cannot read vLLM patch: {error}") from error
    actual_sha = sha256(patch_bytes).hexdigest()
    expected_sha = _required_string(lock, "patch_sha256")
    if actual_sha != expected_sha:
        raise IntegrityError(
            f"vLLM patch SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    touched: list[str] = []
    for line in patch_bytes.decode("utf-8").splitlines():
        if not line.startswith("diff --git a/"):
            continue
        fields = line.split()
        if len(fields) != 4 or not fields[2].startswith("a/"):
            raise IntegrityError(f"malformed patch header: {line}")
        left = fields[2][2:]
        right = fields[3][2:] if fields[3].startswith("b/") else ""
        if left != right:
            raise IntegrityError(f"patch renames are not allowed: {line}")
        touched.append(left)
    if not touched:
        raise IntegrityError("vLLM patch does not touch any files")
    if len(touched) != len(set(touched)):
        raise IntegrityError("vLLM patch contains duplicate file sections")

    return PatchContract(
        repository=_required_string(lock, "repository"),
        tag=_required_string(lock, "tag"),
        commit=_required_string(lock, "commit"),
        patch_sha256=actual_sha,
        touched_files=tuple(touched),
    )


__all__ = ["PatchContract", "validate_patch_contract"]
