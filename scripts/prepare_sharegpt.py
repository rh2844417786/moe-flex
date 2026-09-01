#!/usr/bin/env python3
"""Download fixed metadata/data inputs and build the committed ShareGPT subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from flexmoe.datasets.sharegpt import (
    ChatTokenizerLike,
    prepare_subset,
)
from flexmoe.manifest import sha256_file

DATASET_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
DATASET_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
TOKENIZER_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"
TOKENIZER_REVISION = "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
CONTEXT_LENGTHS = (1024, 2048, 3072, 4096)
PER_LENGTH = 256
SAMPLING_SEED = 20260901


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=DATASET_FILE,
            cache_dir=arguments.cache_dir,
        )
    )
    tokenizer_dir = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            repo_type="model",
            revision=TOKENIZER_REVISION,
            allow_patterns=list(TOKENIZER_FILES),
            cache_dir=arguments.cache_dir,
            local_dir=arguments.cache_dir / "tokenizer",
        )
    )
    tokenizer = cast(
        ChatTokenizerLike,
        AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            tokenizer_dir, local_files_only=True, trust_remote_code=False
        ),
    )

    source_manifest_path = (
        arguments.source_manifest
        if arguments.source_manifest is not None
        else arguments.output.parent / "source_manifest.json"
    )
    tokenizer_hashes = {
        name: sha256_file(tokenizer_dir / name)
        for name in TOKENIZER_FILES
        if (tokenizer_dir / name).is_file()
    }
    source_manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": DATASET_REVISION,
            "filename": DATASET_FILE,
            "license": "Apache-2.0",
            "sha256": sha256_file(source_path),
        },
        "tokenizer": {
            "repo_id": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "files": tokenizer_hashes,
        },
        "sampling_seed": SAMPLING_SEED,
        "packing_policy": "shuffle-pack-eos-truncate",
    }
    _write_json(source_manifest_path, source_manifest)

    dataset_manifest = prepare_subset(
        source=source_path,
        tokenizer=tokenizer,
        output=arguments.output,
        manifest_path=arguments.manifest,
        lengths=CONTEXT_LENGTHS,
        per_length=PER_LENGTH,
        seed=SAMPLING_SEED,
    )
    dataset_manifest["source_manifest_sha256"] = sha256_file(source_manifest_path)
    dataset_manifest["tokenizer_revision"] = TOKENIZER_REVISION
    _write_json(arguments.manifest, dataset_manifest)

    if arguments.output.stat().st_size >= 100_000_000:
        raise ValueError("generated subset exceeds GitHub 100MB file limit")
    print(json.dumps(dataset_manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
