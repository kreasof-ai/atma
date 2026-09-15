#!/usr/bin/env python3
"""Cache the token IDs for the 30 real retrieval documents."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained("gpt2")

def thash(ids):
    d = hashlib.sha256()
    for t in ids:
        d.update(struct.pack("<I", int(t)))
    return d.hexdigest()

out_path = ROOT / "supplementary" / "pat_feedback" / "manifests" / "retrieval_30_document_tokens.pt"

manifest_path = ROOT / "supplementary" / "pat_feedback" / "manifests" / "retrieval_fixture_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

p_fp0 = "/home/sagemaker-user/.cache/huggingface/hub/datasets--codelion--finepdfs-1B/snapshots/6fbca6dd53e3e794f607cfe1f5a91c07e17b01ab/data/train-00000-of-00009.parquet"
ds_fp0 = load_dataset("parquet", data_files={"train": p_fp0}, split="train")

scanned_rows = {c["real_document"]["scanned_row"]: c["case_id"] for c in manifest["cases"]}

retrieval_tokens = {}
for row_idx, case_id in scanned_rows.items():
    row = ds_fp0[row_idx - 1]  # 1-based scanned_row
    ids = tok.encode(row["text"], add_special_tokens=False)
    h = thash(ids)
    expected_h = manifest["cases"][case_id]["real_document"]["token_hash"]
    assert h == expected_h, f"Case {case_id} hash mismatch: {h} != {expected_h}"
    retrieval_tokens[case_id] = ids[:262144]

print(f"Verified and extracted all {len(retrieval_tokens)} real retrieval documents.")
torch.save(retrieval_tokens, out_path)
print("Saved to:", out_path)
