#!/usr/bin/env python3
"""Extract and cache the exact 18 token sequences for the BPB panel."""
from __future__ import annotations

import hashlib
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

out_path = ROOT / "supplementary" / "pat_feedback" / "manifests" / "bpb_18_document_tokens.pt"

# 1. pg19 (rows 21, 37, 61, 80, 91)
p_pg19 = "/home/sagemaker-user/.cache/huggingface/hub/datasets--emozilla--pg19/snapshots/c021754c8e01c5b1cc83a1f549c1f97fbbb756b8/data/test-00000-of-00001-29a571947c0b5ccc.parquet"
ds_pg19 = load_dataset("parquet", data_files={"test": p_pg19}, split="test")
docs_pg19 = []
for idx in [21, 37, 61, 80, 91]:
    ids = tok.encode(ds_pg19[idx]["text"], add_special_tokens=False)[:262400]
    docs_pg19.append(ids)

# 2. proof_pile (rows 20080, 22802, 32628, 41383, 44491)
p_pp = "/home/sagemaker-user/.cache/huggingface/hub/datasets--hoskinson-center--proof-pile/snapshots/490b980249446f2f3bd2df3a8cf085d0f2de240a/test/proofpile_test.jsonl.gz"
ds_pp = load_dataset("json", data_files={"test": p_pp}, split="test")
docs_pp = []
for idx in [20080, 22802, 32628, 41383, 44491]:
    ids = tok.encode(ds_pp[idx]["text"], add_special_tokens=False)[:262400]
    docs_pp.append(ids)

# 3. finepdfs (part0: 1797, 5935, 8637, 16811, 17776; part1: 30, 3888, 3929)
p_fp0 = "/home/sagemaker-user/.cache/huggingface/hub/datasets--codelion--finepdfs-1B/snapshots/6fbca6dd53e3e794f607cfe1f5a91c07e17b01ab/data/train-00000-of-00009.parquet"
p_fp1 = "/home/sagemaker-user/.cache/huggingface/hub/datasets--codelion--finepdfs-1B/snapshots/6fbca6dd53e3e794f607cfe1f5a91c07e17b01ab/data/train-00001-of-00009.parquet"
ds_fp0 = load_dataset("parquet", data_files={"train": p_fp0}, split="train")
ds_fp1 = load_dataset("parquet", data_files={"train": p_fp1}, split="train")

docs_fp = []
for idx in [1797, 5935, 8637, 16811, 17776]:
    ids = tok.encode(ds_fp0[idx]["text"], add_special_tokens=False)[:262400]
    docs_fp.append(ids)
for idx in [30, 3888, 3929]:
    ids = tok.encode(ds_fp1[idx]["text"], add_special_tokens=False)[:262400]
    docs_fp.append(ids)

bpb_tokens = {
    "pg19": docs_pg19,
    "proof_pile": docs_pp,
    "finepdfs": docs_fp
}

for name, doc_list in bpb_tokens.items():
    print(name, "count:", len(doc_list), "hashes:", [thash(d) for d in doc_list])

torch.save(bpb_tokens, out_path)
print("Saved all 18 document token sequences to:", out_path)
