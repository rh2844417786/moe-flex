"""Comparison rules for deterministic router probes and full trace evidence."""

from __future__ import annotations

import re


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _valid_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    line_count = entry.get("line_count")
    probe_line_count = entry.get("probe_line_count")
    if type(line_count) is not int or line_count <= 0:
        return False
    if type(probe_line_count) is not int or not 0 < probe_line_count <= line_count:
        return False
    return all(
        isinstance(entry.get(field), str)
        and _SHA256_PATTERN.fullmatch(entry[field]) is not None
        for field in ("sha256", "probe_sha256")
    )


def router_probes_match(reference: object, current: object) -> bool:
    if not isinstance(reference, dict) or not isinstance(current, dict):
        return False
    if set(reference) != set(current) or not current:
        return False
    for filename, current_entry in current.items():
        reference_entry = reference.get(filename)
        if not _valid_entry(reference_entry) or not _valid_entry(current_entry):
            return False
        if reference_entry["line_count"] != current_entry["line_count"]:
            return False
        if reference_entry["probe_line_count"] != current_entry["probe_line_count"]:
            return False
        if reference_entry["probe_sha256"] != current_entry["probe_sha256"]:
            return False
    return True


__all__ = ["router_probes_match"]
