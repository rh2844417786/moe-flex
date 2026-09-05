#!/usr/bin/env python3
"""Build a deterministic tokenized SWE-bench throughput subset offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from flexmoe.datasets.sharegpt import build_fixed_requests, verify_subset, write_jsonl_zst
from flexmoe.manifest import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    table = pq.read_table(args.source, columns=["repo", "instance_id", "problem_statement", "hints_text"])
    conversations: list[str] = []
    for row in table.to_pylist():
        issue = str(row.get("problem_statement") or "").strip()
        if not issue:
            continue
        conversations.append(
            f"Repository: {str(row.get('repo') or '').strip()}\n"
            f"Instance: {str(row.get('instance_id') or '').strip()}\n"
            f"Issue:\n{issue}\nHints:\n{str(row.get('hints_text') or '').strip()}"
        )
    if not conversations:
        raise ValueError("SWE-bench parquet contains no issue statements")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    records = build_fixed_requests(conversations, tokenizer, (1024, 2048, 3072, 4096), args.per_length, args.seed)
    digest = write_jsonl_zst(records, args.output)
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record.context_length)] = counts.get(str(record.context_length), 0) + 1
    manifest = {
        "schema_version": 1, "sha256": digest, "source_sha256": sha256_file(args.source),
        "record_count": len(records), "counts_by_context": counts,
        "sampling_seed": args.seed, "packing_policy": "shuffle-pack-eos-truncate",
        "source_rows": table.num_rows, "accepted_rows": len(conversations),
        "source_format": "SWE-bench-Verified parquet issue statements",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_subset(args.output, args.manifest)
    print(json.dumps({"record_count": len(records), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
