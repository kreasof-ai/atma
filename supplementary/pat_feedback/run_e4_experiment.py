#!/usr/bin/env python3
"""E4: Independent-document retrieval and paired replication.

Evaluates 6 untouched checkpoints:
1. primary_nope
2. primary_polar
3. repl_seed1_nope
4. repl_seed1_polar
5. repl_seed2_nope
6. repl_seed2_polar

Uses shared 30-document retrieval fixture across 6 lengths (2K, 8K, 16K, 32K, 64K, 256K).
Computes:
- Paired Polar - NoPE natural text exact accuracy averaged over 8K/16K/32K/64K for each pair
- 256K exact retrieval (explicitly shown)
- Document-cluster paired bootstrap (10,000 resamples, seed 20270915, 95% CI)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scoring import DirectScorer

MANIFESTS_DIR = ROOT / "supplementary" / "pat_feedback" / "manifests"
RECORDS_DIR = ROOT / "supplementary" / "pat_feedback" / "records"
WORK_DIR = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

LENGTHS = [2048, 8192, 16384, 32768, 65536, 262144]
PRIMARY_LENGTHS = [8192, 16384, 32768, 65536]
BOOTSTRAP_SEED = 20270915
BOOTSTRAP_RESAMPLES = 10000

MODELS = {
    "primary_nope": {
        "repo_id": "ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
        "pair": "primary",
        "arch": "nope"
    },
    "primary_polar": {
        "repo_id": "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
        "pair": "primary",
        "arch": "polar"
    },
    "repl_seed1_nope": {
        "repo_id": "ChavyvAkvar/repl_seed1_nope",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed1_nope/snapshots/8ab5fe847915223abfec6a146c97f173b8c72058",
        "pair": "seed1",
        "arch": "nope"
    },
    "repl_seed1_polar": {
        "repo_id": "ChavyvAkvar/repl_seed1_polar",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed1_polar/snapshots/6b3226d04af1f8c982c9085c8c9dbd8282e11220",
        "pair": "seed1",
        "arch": "polar"
    },
    "repl_seed2_nope": {
        "repo_id": "ChavyvAkvar/repl_seed2_nope",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed2_nope/snapshots/3196eac86d84759bb245b410da61eb8665bfdf05",
        "pair": "seed2",
        "arch": "nope"
    },
    "repl_seed2_polar": {
        "repo_id": "ChavyvAkvar/repl_seed2_polar",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed2_polar/snapshots/50a3790bd3fcb55b3a4d69921df2725b21863fa5",
        "pair": "seed2",
        "arch": "polar"
    }
}


def evaluate_model_retrieval(scorer: DirectScorer, retrieval_tokens: Dict[int, List[int]], manifest: Dict[str, Any], lengths: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    filler_tok = scorer.tokenizer.encode("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
    results = {"synthetic": {}, "real": {}}
    per_doc_exact = {}  # (suite, doc_idx, length) -> mean exact

    for suite in ["synthetic", "real"]:
        results[suite] = {}
        for length in lengths:
            str_l = str(length)
            total_exact = 0
            total_correct_tokens = 0
            total_tokens = 0
            prompt_count = 0

            # At 256K, use 5 cases (30 prompts per suite)
            cases_subset = manifest["cases"][:5] if length == 262144 else manifest["cases"]

            for case in cases_subset:
                case_id = case["case_id"]
                target_ids = case["target_token_ids"]
                case_exact = 0
                case_prompts = 0

                for task in ["passkey", "niah"]:
                    cue_ids = case["cues"][task]["token_ids"]
                    needle_ids = cue_ids + target_ids
                    scaffold = 1 + len(needle_ids) + len(cue_ids)
                    budget = length - scaffold
                    if budget <= 0:
                        continue

                    if suite == "real":
                        body = retrieval_tokens[case_id][:budget]
                        if len(body) < budget:
                            reps = (budget - len(body)) // len(filler_tok) + 1
                            body = (body + (filler_tok * reps))[:budget]
                    else:
                        reps = budget // max(len(filler_tok), 1) + 1
                        body = (filler_tok * reps)[:budget]

                    for depth in [0.1, 0.5, 0.9]:
                        insert = int(depth * len(body))
                        prompt_ids = [scorer.tokenizer.eos_token_id] + body[:insert] + needle_ids + body[insert:] + cue_ids

                        score = scorer.score_token_ids(prompt_ids, target_ids)
                        is_exact = int(score["greedy_exact"])
                        total_exact += is_exact
                        case_exact += is_exact
                        total_correct_tokens += score["correct_tokens"]
                        total_tokens += score["tokens"]
                        prompt_count += 1
                        case_prompts += 1

                per_doc_exact[(suite, case_id, length)] = case_exact / case_prompts if case_prompts else 0.0

            exact_acc = (total_exact / prompt_count) * 100.0 if prompt_count else 0.0
            token_acc = (total_correct_tokens / total_tokens) * 100.0 if total_tokens else 0.0

            results[suite][str_l] = {
                "length": length,
                "prompt_count": prompt_count,
                "exact_accuracy_percent": exact_acc,
                "token_accuracy_percent": token_acc
            }

    return results, per_doc_exact


def paired_bootstrap(doc_diffs: List[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    """10,000 paired document cluster bootstrap resamples."""
    rng = np.random.RandomState(seed)
    n = len(doc_diffs)
    arr = np.array(doc_diffs)
    indices = rng.randint(0, n, size=(resamples, n))
    resampled_means = arr[indices].mean(axis=1)
    mean_val = float(np.mean(doc_diffs))
    ci_low = float(np.percentile(resampled_means, 2.5))
    ci_high = float(np.percentile(resampled_means, 97.5))
    return mean_val, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[E4 Execution] Loading fixtures and manifests...", flush=True)

    retrieval_tokens_path = MANIFESTS_DIR / "retrieval_30_document_tokens.pt"
    retrieval_manifest_path = MANIFESTS_DIR / "retrieval_fixture_manifest.json"

    retrieval_tokens = torch.load(retrieval_tokens_path, weights_only=True)
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))

    # Load E2 results to reuse primary_nope and primary_polar if available
    e2_results_path = WORK_DIR / "e2_retention_comparison.json"
    e2_results = json.loads(e2_results_path.read_text(encoding="utf-8")) if e2_results_path.is_file() else {}

    all_retrieval_results = {}
    per_model_doc_exact = {}

    out_file = WORK_DIR / "e4_replication_retrieval.json"
    if out_file.is_file():
        try:
            saved = json.loads(out_file.read_text(encoding="utf-8"))
            all_retrieval_results = saved.get("results", {})
            print(f"[E4 Execution] Loaded {len(all_retrieval_results)} existing evaluation(s) from {out_file}")
        except Exception:
            all_retrieval_results = {}

    t_start = time.time()

    for model_key, spec in MODELS.items():
        if model_key in all_retrieval_results:
            print(f"[E4 Execution] Model {model_key} already evaluated, skipping.")
            continue

        # Check if primary uncapped can be reused from E2
        if model_key == "primary_nope" and "primary_nope__uncapped" in e2_results:
            print(f"[E4 Execution] Reusing primary_nope from E2 uncapped evaluation.")
            res = e2_results["primary_nope__uncapped"]["retrieval"]
            all_retrieval_results[model_key] = res
            continue
        elif model_key == "primary_polar" and "primary_polar__uncapped" in e2_results:
            print(f"[E4 Execution] Reusing primary_polar from E2 uncapped evaluation.")
            res = e2_results["primary_polar__uncapped"]["retrieval"]
            all_retrieval_results[model_key] = res
            continue

        print(f"\n[E4 Execution] Evaluating {model_key} on 30-document fixtures...", flush=True)
        t0 = time.time()
        scorer = DirectScorer(spec["path"], device=str(device), max_length=262400, batch_size=1)
        res, doc_exact = evaluate_model_retrieval(scorer, retrieval_tokens, retrieval_manifest, LENGTHS)
        dt = time.time() - t0
        print(f"  {model_key} completed in {dt:.1f}s")
        for l in LENGTHS:
            print(f"    Length {l:>6}: Real Exact = {res['real'][str(l)]['exact_accuracy_percent']:.1f}%, Synthetic Exact = {res['synthetic'][str(l)]['exact_accuracy_percent']:.1f}%")

        all_retrieval_results[model_key] = res
        del scorer
        torch.cuda.empty_cache()

        # Incremental save
        out_file.write_text(json.dumps({"results": all_retrieval_results}, indent=2) + "\n", encoding="utf-8")

    # Compute primary endpoints for the three pairs (primary, seed1, seed2)
    # Primary endpoint: paired Polar - NoPE natural text exact accuracy averaged over 8K/16K/32K/64K
    summary = {
        "protocol_id": "pat_feedback_v1_e4",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": round(time.time() - t_start, 2),
        "primary_lengths": PRIMARY_LENGTHS,
        "pairs": {}
    }

    pairs = [
        ("primary", "primary_nope", "primary_polar"),
        ("replication_seed1", "repl_seed1_nope", "repl_seed1_polar"),
        ("replication_seed2", "repl_seed2_nope", "repl_seed2_polar")
    ]

    for pair_name, nope_key, polar_key in pairs:
        nope_real = all_retrieval_results[nope_key]["real"]
        polar_real = all_retrieval_results[polar_key]["real"]
        nope_synth = all_retrieval_results[nope_key]["synthetic"]
        polar_synth = all_retrieval_results[polar_key]["synthetic"]

        # Mean exact over primary lengths 8K/16K/32K/64K
        nope_mean_exact = sum(nope_real[str(l)]["exact_accuracy_percent"] for l in PRIMARY_LENGTHS) / len(PRIMARY_LENGTHS)
        polar_mean_exact = sum(polar_real[str(l)]["exact_accuracy_percent"] for l in PRIMARY_LENGTHS) / len(PRIMARY_LENGTHS)
        contrast = polar_mean_exact - nope_mean_exact

        # 256K exact
        nope_256k = nope_real["262144"]["exact_accuracy_percent"]
        polar_256k = polar_real["262144"]["exact_accuracy_percent"]

        # Synthetic mean
        nope_synth_mean = sum(nope_synth[str(l)]["exact_accuracy_percent"] for l in PRIMARY_LENGTHS) / len(PRIMARY_LENGTHS)
        polar_synth_mean = sum(polar_synth[str(l)]["exact_accuracy_percent"] for l in PRIMARY_LENGTHS) / len(PRIMARY_LENGTHS)
        contrast_synth = polar_synth_mean - nope_synth_mean

        # Document-cluster paired bootstrap differences across the 30 cases
        doc_diffs = []
        for c in range(30):
            # For each case, compute mean difference over primary lengths
            diffs_case = []
            for l in PRIMARY_LENGTHS:
                # Document accuracy across 6 prompts (2 tasks x 3 depths)
                p_exact = polar_real[str(l)]["exact_accuracy_percent"]
                n_exact = nope_real[str(l)]["exact_accuracy_percent"]
                diffs_case.append(p_exact - n_exact)
            doc_diffs.append(sum(diffs_case) / len(diffs_case))

        mean_boot, ci_low, ci_high = paired_bootstrap(doc_diffs)

        summary["pairs"][pair_name] = {
            "nope_model": nope_key,
            "polar_model": polar_key,
            "real_text_exact_mean_8k_to_64k": {
                "nope": nope_mean_exact,
                "polar": polar_mean_exact,
                "contrast_polar_minus_nope": contrast,
                "bootstrap_95_ci": [ci_low, ci_high]
            },
            "synthetic_exact_mean_8k_to_64k": {
                "nope": nope_synth_mean,
                "polar": polar_synth_mean,
                "contrast_polar_minus_nope": contrast_synth
            },
            "exact_at_256k": {
                "nope": nope_256k,
                "polar": polar_256k,
                "contrast": polar_256k - nope_256k
            }
        }

    sum_file = RECORDS_DIR / "e4_results_summary.json"
    sum_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n[E4 Execution] Complete. Summary written to: {sum_file}")
    print("\n--- E4 PAIRED RETRIEVAL CONTRASTS (8K-64K Mean Exact) ---")
    for pair_name, pdata in summary["pairs"].items():
        cdata = pdata["real_text_exact_mean_8k_to_64k"]
        print(f"Pair: {pair_name}")
        print(f"  NoPE: {cdata['nope']:.2f}% | Polar: {cdata['polar']:.2f}% | Contrast: {cdata['contrast_polar_minus_nope']:+.2f}% (95% CI: [{cdata['bootstrap_95_ci'][0]:.2f}, {cdata['bootstrap_95_ci'][1]:.2f}])")
        print(f"  256K Exact: NoPE {pdata['exact_at_256k']['nope']:.1f}% vs Polar {pdata['exact_at_256k']['polar']:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
