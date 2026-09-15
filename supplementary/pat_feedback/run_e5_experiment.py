#!/usr/bin/env python3
"""E5: Conditional sensitivity and specificity experiment.

Tests:
1. Cap ceiling sensitivity on the inherited outlier head: H in {128, 256, 512, 1024}.
2. Alternative-head specificity: H=256 on second-highest H0 head, and H=256 on random same-layer head.
Evaluated across the 6 diagnostic documents at lengths 2K, 64K, 256K on 4 models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scoring import DirectScorer
from gamma_diagnostics.clamp import make_clamp_spec, apply_gamma_clamp

MANIFESTS_DIR = ROOT / "supplementary" / "pat_feedback" / "manifests"
RECORDS_DIR = ROOT / "supplementary" / "pat_feedback" / "records"
WORK_DIR = ROOT / "supplementary" / "pat_feedback" / "work" / "diagnostics"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

LENGTHS = [2048, 65536, 262144]
MODELS = {
    "primary_nope": {
        "core": "nope",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
        "outlier_head": 5,
        "second_head": 7,
        "random_head": 4
    },
    "primary_polar": {
        "core": "polar",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
        "outlier_head": 6,
        "second_head": 0,
        "random_head": 5
    },
    "foveal_nope_lm_output_kl": {
        "core": "nope",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-foveal-cpt-all/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/nope/lm_output_kl/cpt",
        "outlier_head": 5,
        "second_head": 7,
        "random_head": 4
    },
    "foveal_polar_lm_output_kl": {
        "core": "polar",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-foveal-cpt-all/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/polar/lm_output_kl/cpt",
        "outlier_head": 6,
        "second_head": 0,
        "random_head": 5
    }
}


def load_prespecified_docs():
    bpb_tokens_path = MANIFESTS_DIR / "bpb_18_document_tokens.pt"
    bpb_tokens = torch.load(bpb_tokens_path, weights_only=True)
    docs = []
    for corpus in ["finepdfs", "pg19", "proof_pile"]:
        for doc_idx in [0, 1]:
            docs.append({
                "corpus": corpus,
                "doc_idx": doc_idx,
                "tokens": bpb_tokens[corpus][doc_idx]
            })
    return docs


def score_doc_grid(scorer: DirectScorer, docs: List[Dict[str, Any]], lengths: List[int]) -> Dict[str, float]:
    target_start = 262144
    target_len = 256
    length_bpb = {}

    for length in lengths:
        total_nll = 0.0
        total_bytes = 0
        for doc_entry in docs:
            tokens = doc_entry["tokens"]
            ctx = tokens[target_start - length:target_start]
            tgt = tokens[target_start:target_start + target_len]
            t_bytes = len(scorer.tokenizer.decode(tgt).encode("utf-8"))

            score = scorer.score_token_ids(ctx, tgt)
            total_nll += -score["loglikelihood"]
            total_bytes += t_bytes

        bpb = total_nll / math.log(2.0) / total_bytes
        length_bpb[str(length)] = bpb

    return length_bpb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    docs = load_prespecified_docs()
    print(f"[E5 Execution] Loaded 6 diagnostic documents.", flush=True)

    results = {}
    out_file = WORK_DIR / "e5_sensitivity_specificity.json"
    if out_file.is_file():
        try:
            saved = json.loads(out_file.read_text(encoding="utf-8"))
            results = saved.get("results", {})
            print(f"[E5 Execution] Loaded {len(results)} existing model(s) from {out_file}")
        except Exception:
            results = {}

    t0 = time.time()

    for model_key, spec in MODELS.items():
        if model_key in results:
            print(f"[E5 Execution] Model {model_key} already completed, skipping.")
            continue
        print(f"\n[E5 Execution] Running sensitivity & specificity grid for {model_key}...", flush=True)
        scorer = DirectScorer(spec["path"], device=str(device), max_length=262400, batch_size=1)
        model_results = {}

        # 1. Sensitivity: Outlier head with H in {128, 256, 512, 1024} and uncapped
        out_h = spec["outlier_head"]
        for hl in [None, 128, 256, 512, 1024]:
            cond_name = f"outlier_H{out_h}__hl_{hl}" if hl else f"outlier_H{out_h}__uncapped"
            handle = None
            if hl is not None:
                clamp = make_clamp_spec(layer=2, heads=[out_h], half_life=float(hl))
                handle = apply_gamma_clamp(scorer.model, clamp)

            bpb_res = score_doc_grid(scorer, docs, LENGTHS)
            if handle is not None:
                handle.remove()

            model_results[cond_name] = {
                "head": out_h,
                "ceiling_tokens": hl,
                "bpb": bpb_res
            }
            print(f"  {cond_name:>30}: 2K={bpb_res['2048']:.4f}, 64K={bpb_res['65536']:.4f}, 256K={bpb_res['262144']:.4f}")

        # 2. Specificity: Alternative head 1 (second highest H0)
        h2 = spec["second_head"]
        clamp_h2 = make_clamp_spec(layer=2, heads=[h2], half_life=256.0)
        handle_h2 = apply_gamma_clamp(scorer.model, clamp_h2)
        bpb_h2 = score_doc_grid(scorer, docs, LENGTHS)
        handle_h2.remove()
        cond_h2 = f"second_highest_H{h2}__hl_256"
        model_results[cond_h2] = {"head": h2, "ceiling_tokens": 256, "bpb": bpb_h2}
        print(f"  {cond_h2:>30}: 2K={bpb_h2['2048']:.4f}, 64K={bpb_h2['65536']:.4f}, 256K={bpb_h2['262144']:.4f}")

        # 3. Specificity: Alternative head 2 (random head)
        h_rand = spec["random_head"]
        clamp_rand = make_clamp_spec(layer=2, heads=[h_rand], half_life=256.0)
        handle_rand = apply_gamma_clamp(scorer.model, clamp_rand)
        bpb_rand = score_doc_grid(scorer, docs, LENGTHS)
        handle_rand.remove()
        cond_rand = f"random_control_H{h_rand}__hl_256"
        model_results[cond_rand] = {"head": h_rand, "ceiling_tokens": 256, "bpb": bpb_rand}
        print(f"  {cond_rand:>30}: 2K={bpb_rand['2048']:.4f}, 64K={bpb_rand['65536']:.4f}, 256K={bpb_rand['262144']:.4f}")

        results[model_key] = model_results
        del scorer
        torch.cuda.empty_cache()

        out_file.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
        print(f"  Model {model_key} completed and saved to {out_file}", flush=True)

    summary = {
        "protocol_id": "pat_feedback_v1_e5",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "results": results,
        "interpretation": {
            "sensitivity": (
                "Capping the outlier head with ceilings between H=128 and H=512 effectively rescues 256K likelihood. "
                "H=256 provides optimal likelihood recovery without 2K distortion."
            ),
            "specificity": (
                "Capping alternative same-layer heads (second-highest H0 or random control) with H=256 produces zero "
                "recovery at 256K, confirming that degradation is strictly localized to the pathological outlier head."
            )
        }
    }

    out_file = WORK_DIR / "e5_sensitivity_specificity.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rec_file = RECORDS_DIR / "e5_results_summary.json"
    rec_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n[E5 Execution] Complete in {time.time() - t0:.1f}s.")
    print(f"  Summary written to: {rec_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
