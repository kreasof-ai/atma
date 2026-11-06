#!/usr/bin/env python3
"""Full 18-Document E5 Ceiling Sensitivity & Head Specificity Benchmark.

Directly addresses Reviewer HE3D Question 2 and Reviewer ihsT Weakness 6:
1. Evaluates on the FULL 18-document panel (8 FinePDFs, 5 PG-19, 5 Proof-Pile)
   with equal cross-dataset weighting matching Table 2 and E2 (instead of the
   6-document pooled subset used in the original E5 script).
2. Evaluates both Primary NoPE and Primary Polar across:
   - Uncapped baseline
   - Quantile caps: `p90`, `p99`
   - Half-life ceilings on the outlier head: `H in {128, 256, 512, 1024}`
   - Alternative same-layer head controls at `H = 256`:
     * Second-highest H0 head in Block 2 (`H7` for NoPE, `H0` for Polar)
     * Random same-layer head in Block 2 (`H4` for NoPE, `H5` for Polar)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scoring import DirectScorer
from gamma_diagnostics.clamp import make_clamp_spec, apply_gamma_clamp

MANIFESTS_DIR = ROOT / "supplementary" / "pat_feedback" / "manifests"
PARAMS_FILE = ROOT / "gamma_diagnostics" / "results" / "parameters" / "gamma_parameters.json"

LENGTHS = [2048, 65536, 262144]

MODELS = {
    "primary_nope": {
        "core": "nope",
        "checkpoint_key": "atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
        "outlier_head": 5,
        "second_head": 7,
        "random_head": 4,
    },
    "primary_polar": {
        "core": "polar",
        "checkpoint_key": "atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
        "outlier_head": 6,
        "second_head": 0,
        "random_head": 5,
    },
}


def get_quantile_half_life(checkpoint_key: str, q: float) -> float:
    """Compute the implied half-life ceiling for checkpoint-local logit quantile q (e.g. 90 or 99)."""
    params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    logits = [p["total_zero_input_logit"] for p in params if p["checkpoint"] == checkpoint_key]
    q_logit = float(np.percentile(np.array(logits, dtype=np.float64), q))
    gamma = 1.0 / (1.0 + math.exp(-q_logit))
    return math.log(0.5) / math.log(gamma)


def evaluate_full_18doc_bpb(
    scorer: DirectScorer,
    bpb_tokens: Dict[str, List[List[int]]],
    lengths: List[int],
) -> Dict[str, Any]:
    target_start = 262144
    target_len = 256
    per_corpus = {}

    for corpus in ["finepdfs", "pg19", "proof_pile"]:
        doc_list = bpb_tokens[corpus]
        per_corpus[corpus] = {}
        for length in lengths:
            total_nll = 0.0
            total_bytes = 0
            for doc_ids in doc_list:
                context = doc_ids[target_start - length : target_start]
                target = doc_ids[target_start : target_start + target_len]
                target_bytes = len(scorer.tokenizer.decode(target).encode("utf-8"))
                score = scorer.score_token_ids(context, target)
                total_nll += -score["loglikelihood"]
                total_bytes += target_bytes
            per_corpus[corpus][str(length)] = total_nll / math.log(2.0) / total_bytes

    mean_bpb = {}
    for length in lengths:
        sl = str(length)
        mean_bpb[sl] = sum(per_corpus[c][sl] for c in ["finepdfs", "pg19", "proof_pile"]) / 3.0

    return {"cross_dataset_mean_bpb": mean_bpb, "per_corpus_bpb": per_corpus}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "iclr27_rebuttal" / "tables" / "e5_full_18doc_results.json",
    )
    args = parser.parse_args()

    bpb_tokens_path = MANIFESTS_DIR / "bpb_18_document_tokens.pt"
    bpb_tokens = torch.load(bpb_tokens_path, weights_only=True)
    print("[E5 Full 18-Doc] Loaded all 18 documents (8 FinePDFs, 5 PG-19, 5 Proof-Pile).", flush=True)

    results = {}
    if args.output.is_file():
        try:
            results = json.loads(args.output.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            results = {}

    for model_key, spec in MODELS.items():
        if model_key in results:
            print(f"[E5 Full 18-Doc] {model_key} already in {args.output}, skipping.")
            continue

        print(f"\n[E5 Full 18-Doc] Evaluating {model_key}...", flush=True)
        scorer = DirectScorer(spec["path"], device=args.device, max_length=262400, batch_size=1)
        out_h = spec["outlier_head"]
        hl_p90 = get_quantile_half_life(spec["checkpoint_key"], 90.0)
        hl_p99 = get_quantile_half_life(spec["checkpoint_key"], 99.0)

        conditions = [
            ("uncapped", out_h, None),
            ("p99_outlier", out_h, hl_p99),
            ("p90_outlier", out_h, hl_p90),
            ("hl_1024_outlier", out_h, 1024.0),
            ("hl_512_outlier", out_h, 512.0),
            ("hl_256_outlier", out_h, 256.0),
            ("hl_128_outlier", out_h, 128.0),
            ("hl_256_second_head", spec["second_head"], 256.0),
            ("hl_256_random_head", spec["random_head"], 256.0),
        ]

        model_res = {}
        for cond_name, head_idx, hl_val in conditions:
            t0 = time.time()
            handle = None
            if hl_val is not None:
                clamp = make_clamp_spec(layer=2, heads=[head_idx], half_life=float(hl_val), label=cond_name)
                handle = apply_gamma_clamp(scorer.model, clamp)
            bpb_eval = evaluate_full_18doc_bpb(scorer, bpb_tokens, LENGTHS)
            if handle is not None:
                handle.remove()

            m_bpb = bpb_eval["cross_dataset_mean_bpb"]
            print(
                f"  {cond_name:<22} (B2/H{head_idx}, H={hl_val if hl_val is not None else 'inf'}): "
                f"2K={m_bpb['2048']:.3f} | 64K={m_bpb['65536']:.3f} | 256K={m_bpb['262144']:.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            model_res[cond_name] = {
                "layer": 2,
                "head": head_idx,
                "half_life_ceiling": hl_val,
                "cross_dataset_mean_bpb": m_bpb,
                "per_corpus_bpb": bpb_eval["per_corpus_bpb"],
            }

        results[model_key] = model_res
        scorer.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")

    print(f"\nCompleted full 18-document E5 sweep -> {args.output}")


if __name__ == "__main__":
    main()
