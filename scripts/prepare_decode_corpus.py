#!/usr/bin/env python3
"""Build the unique decode corpus from explicit local source/tokenizer paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

from flexmoe.datasets.decode_corpus import prepare_decode_corpus
from flexmoe.datasets.sharegpt import ChatTokenizerLike
from flexmoe.manifest import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--unique-per-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def _load_source_manifest(
    path: Path, source: Path, tokenizer_dir: Path
) -> dict[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("source manifest must be a JSON object")
    dataset = parsed.get("dataset")
    tokenizer = parsed.get("tokenizer")
    if not isinstance(dataset, dict) or not isinstance(tokenizer, dict):
        raise TypeError("source manifest must define dataset and tokenizer objects")
    if dataset.get("sha256") != sha256_file(source):
        raise ValueError("source file SHA256 does not match source manifest")
    if dataset.get("filename") != source.name:
        raise ValueError("source filename does not match source manifest")
    raw_files = tokenizer.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("source manifest tokenizer files must be non-empty")
    for filename, expected in raw_files.items():
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise TypeError("source manifest tokenizer file hashes are invalid")
        local_file = tokenizer_dir / filename
        if not local_file.is_file() or sha256_file(local_file) != expected:
            raise ValueError(f"tokenizer file mismatch: {filename}")
    return cast(dict[str, object], parsed)


def main() -> int:
    arguments = _parse_args()
    _load_source_manifest(
        arguments.source_manifest, arguments.source, arguments.tokenizer
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = cast(
        ChatTokenizerLike,
        AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            arguments.tokenizer,
            local_files_only=True,
            trust_remote_code=False,
        ),
    )
    generated = prepare_decode_corpus(
        arguments.source,
        tokenizer,
        arguments.output,
        arguments.manifest,
        lengths=tuple(arguments.lengths),
        unique_per_length=arguments.unique_per_length,
        seed=arguments.seed,
        source_manifest=arguments.source_manifest,
    )
    print(json.dumps(generated, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
