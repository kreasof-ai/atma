#!/usr/bin/env python3
"""E2: Adaptation changes the effect of retention capping.

Evaluates 8 conditions (2 cores x 2 adaptation states x 2 cap conditions):
1. primary_nope uncapped
2. primary_nope capped hl-256 (B2/H5)
3. primary_polar uncapped
4. primary_polar capped hl-256 (B2/H6)
5. foveal_nope_lm_output_kl uncapped
6. foveal_nope_lm_output_kl capped hl-256 (B2/H5)
7. foveal_polar_lm_output_kl uncapped
8. foveal_polar_lm_output_kl capped hl-256 (B2/H6)

Fixtures:
- 18-document fixed-target BPB panel at 2K, 8K, 16K, 32K, 64K, 256K.
- 30-document paired retrieval panel at 2K, 8K, 16K, 32K, 64K, 256K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
WORK_DIR = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

LENGTHS = [2048, 8192, 16384, 32768, 65536, 262144]

# Model specs
MODELS = {
    "primary_nope": {
        "core": "nope",
        "adaptation": "original",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
        "outlier_head": (2, 5)
    },
    "primary_polar": {
        "core": "polar",
        "adaptation": "original",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
        "outlier_head": (2, 6)
    },
    "foveal_nope_lm_output_kl": {
        "core": "nope",
        "adaptation": "foveal",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-foveal-cpt-all/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/nope/lm_output_kl/cpt",
        "outlier_head": (2, 5)
    },
    "foveal_polar_lm_output_kl": {
        "core": "polar",
        "adaptation": "foveal",
        "path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-foveal-cpt-all/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/polar/lm_output_kl/cpt",
        "outlier_head": (2, 6)
    }
}


def load_fixtures():
    bpb_tokens_path = MANIFESTS_DIR / "bpb_18_document_tokens.pt"
    retrieval_tokens_path = MANIFESTS_DIR / "retrieval_30_document_tokens.pt"
    retrieval_manifest_path = MANIFESTS_DIR / "retrieval_fixture_manifest.json"

    bpb_tokens = torch.load(bpb_tokens_path, weights_only=True)
    retrieval_tokens = torch.load(retrieval_tokens_path, weights_only=True)
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))

    return bpb_tokens, retrieval_tokens, retrieval_manifest


def evaluate_bpb(scorer: DirectScorer, bpb_tokens: Dict[str, List[List[int]]], lengths: List[int]) -> Dict[str, Any]:
    target_start = 262144
    target_len = 256
    results = {}

    for corpus, doc_list in bpb_tokens.items():
        corpus_res = {}
        for length in lengths:
            total_nll = 0.0
            total_tokens = 0
            total_bytes = 0
            doc_metrics = []

            for doc_ids in doc_list:
                context = doc_ids[target_start - length:target_start]
                target = doc_ids[target_start:target_start + target_len]
                target_bytes = len(scorer.tokenizer.decode(target).encode("utf-8"))

                score = scorer.score_token_ids(context, target)
                nll = -score["loglikelihood"]
                total_nll += nll
                total_tokens += score["tokens"]
                total_bytes += target_bytes
                doc_metrics.append({
                    "nll_nats": nll,
                    "tokens": score["tokens"],
                    "bytes": target_bytes,
                    "bpb": nll / math.log(2.0) / target_bytes
                })

            corpus_bpb = total_nll / math.log(2.0) / total_bytes
            corpus_nll_per_token = total_nll / total_tokens
            corpus_res[str(length)] = {
                "length": length,
                "bits_per_byte": corpus_bpb,
                "nll_nats_per_token": corpus_nll_per_token,
                "total_tokens": total_tokens,
                "total_bytes": total_bytes,
                "documents": doc_metrics
            }
        results[corpus] = corpus_res

    # Compute equal-weighted cross-dataset mean BPB
    mean_bpb_by_length = {}
    for length in lengths:
        str_l = str(length)
        mean_bpb = sum(results[corpus][str_l]["bits_per_byte"] for corpus in ["finepdfs", "pg19", "proof_pile"]) / 3.0
        mean_bpb_by_length[str_l] = mean_bpb

    return {
        "per_corpus": results,
        "cross_dataset_mean_bpb": mean_bpb_by_length
    }


def evaluate_retrieval(scorer: DirectScorer, retrieval_tokens: Dict[int, List[int]], manifest: Dict[str, Any], lengths: List[int]) -> Dict[str, Any]:
    filler_tok = scorer.tokenizer.encode("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
    results = {"synthetic": {}, "real": {}}

    for suite in ["synthetic", "real"]:
        results[suite] = {}
        for length in lengths:
            str_l = str(length)
            total_exact = 0
            total_correct_tokens = 0
            total_tokens = 0
            prompt_count = 0

            # At 256K, use 5 cases (30 prompts per suite) to conserve execution budget while showing explicit 256K results
            cases_subset = manifest["cases"][:5] if length == 262144 else manifest["cases"]

            for case in cases_subset:
                case_id = case["case_id"]
                target_ids = case["target_token_ids"]

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
                        total_exact += int(score["greedy_exact"])
                        total_correct_tokens += score["correct_tokens"]
                        total_tokens += score["tokens"]
                        prompt_count += 1

            exact_acc = (total_exact / prompt_count) * 100.0 if prompt_count else 0.0
            token_acc = (total_correct_tokens / total_tokens) * 100.0 if total_tokens else 0.0

            results[suite][str_l] = {
                "length": length,
                "prompt_count": prompt_count,
                "exact_accuracy_percent": exact_acc,
                "token_accuracy_percent": token_acc
            }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[E2 Execution] Loading fixtures and manifests...", flush=True)
    bpb_tokens, retrieval_tokens, retrieval_manifest = load_fixtures()

    # Pre-generate clamp specs
    nope_clamp_spec = make_clamp_spec(layer=2, heads=[5], half_life=256.0, label="hl-256")
    polar_clamp_spec = make_clamp_spec(layer=2, heads=[6], half_life=256.0, label="hl-256")
    (MANIFESTS_DIR / "nope_hl256.json").write_text(json.dumps(nope_clamp_spec, indent=2) + "\n", encoding="utf-8")
    (MANIFESTS_DIR / "polar_hl256.json").write_text(json.dumps(polar_clamp_spec, indent=2) + "\n", encoding="utf-8")

    all_condition_results = {}
    out_file = WORK_DIR / "e2_retention_comparison.json"
    if out_file.is_file():
        try:
            all_condition_results = json.loads(out_file.read_text(encoding="utf-8"))
            print(f"[E2 Execution] Loaded {len(all_condition_results)} existing condition(s) from {out_file}")
        except Exception:
            all_condition_results = {}

    condition_order = [
        ("primary_nope", "uncapped"),
        ("primary_nope", "capped"),
        ("primary_polar", "uncapped"),
        ("primary_polar", "capped"),
        ("foveal_nope_lm_output_kl", "uncapped"),
        ("foveal_nope_lm_output_kl", "capped"),
        ("foveal_polar_lm_output_kl", "uncapped"),
        ("foveal_polar_lm_output_kl", "capped"),
    ]

    t_start = time.time()

    for model_key, cap_cond in condition_order:
        spec = MODELS[model_key]
        cond_id = f"{model_key}__{cap_cond}"
        if cond_id in all_condition_results:
            print(f"[E2 Execution] Condition {cond_id} already completed, skipping.")
            continue
        print(f"\n[E2 Execution] Evaluating condition: {cond_id}...", flush=True)

        clamp_spec = None
        if cap_cond == "capped":
            clamp_spec = nope_clamp_spec if spec["core"] == "nope" else polar_clamp_spec

        t0 = time.time()
        scorer = DirectScorer(spec["path"], device=str(device), max_length=262400, batch_size=1)
        handle = None
        if clamp_spec is not None:
            handle = apply_gamma_clamp(scorer.model, clamp_spec)
            print(f"  Installed gamma clamp on layer {clamp_spec['targets'][0]['layer']}, heads {clamp_spec['targets'][0]['heads']}")

        # 1. BPB Evaluation
        print("  Evaluating BPB (18 documents x 6 lengths)...", flush=True)
        bpb_res = evaluate_bpb(scorer, bpb_tokens, args.lengths)
        for l in args.lengths:
            print(f"    Length {l:>6}: mean BPB = {bpb_res['cross_dataset_mean_bpb'][str(l)]:.4f}")

        # 2. Retrieval Evaluation
        print("  Evaluating Retrieval (360 prompts x 6 lengths)...", flush=True)
        retrieval_res = evaluate_retrieval(scorer, retrieval_tokens, retrieval_manifest, args.lengths)
        for l in args.lengths:
            print(f"    Length {l:>6}: Real Exact = {retrieval_res['real'][str(l)]['exact_accuracy_percent']:.1f}%, Synthetic Exact = {retrieval_res['synthetic'][str(l)]['exact_accuracy_percent']:.1f}%")

        if handle is not None:
            handle.remove()

        del scorer
        torch.cuda.empty_cache()

        dt = time.time() - t0
        all_condition_results[cond_id] = {
            "model": model_key,
            "core": spec["core"],
            "adaptation": spec["adaptation"],
            "cap_condition": cap_cond,
            "elapsed_sec": round(dt, 2),
            "bpb": bpb_res,
            "retrieval": retrieval_res
        }
        out_file = WORK_DIR / "e2_retention_comparison.json"
        out_file.write_text(json.dumps(all_condition_results, indent=2) + "\n", encoding="utf-8")
        print(f"  Condition {cond_id} completed in {dt:.1f}s and saved to {out_file}", flush=True)

    # Primary contrasts and summary
    summary = {
        "protocol_id": "pat_feedback_v1_e2",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": round(time.time() - t_start, 2),
        "conditions_evaluated": len(all_condition_results),
        "primary_endpoints": {}
    }

    for core in ["nope", "polar"]:
        orig_uncap = all_condition_results[f"primary_{core}__uncapped"]["bpb"]["cross_dataset_mean_bpb"]["262144"]
        orig_cap = all_condition_results[f"primary_{core}__capped"]["bpb"]["cross_dataset_mean_bpb"]["262144"]
        fov_uncap = all_condition_results[f"foveal_{core}_lm_output_kl__uncapped"]["bpb"]["cross_dataset_mean_bpb"]["262144"]
        fov_cap = all_condition_results[f"foveal_{core}_lm_output_kl__capped"]["bpb"]["cross_dataset_mean_bpb"]["262144"]

        cap_benefit_orig = orig_uncap - orig_cap
        cap_benefit_fov = fov_uncap - fov_cap
        adaptation_contrast = cap_benefit_orig - cap_benefit_fov

        summary["primary_endpoints"][core] = {
            "original_stage2": {
                "uncapped_256k_bpb": orig_uncap,
                "capped_256k_bpb": orig_cap,
                "cap_benefit_B": cap_benefit_orig,
                "growth_2k_to_256k_uncapped": orig_uncap - all_condition_results[f"primary_{core}__uncapped"]["bpb"]["cross_dataset_mean_bpb"]["2048"],
                "growth_2k_to_256k_capped": orig_cap - all_condition_results[f"primary_{core}__capped"]["bpb"]["cross_dataset_mean_bpb"]["2048"]
            },
            "foveal_cpt": {
                "uncapped_256k_bpb": fov_uncap,
                "capped_256k_bpb": fov_cap,
                "cap_benefit_B": cap_benefit_fov,
                "growth_2k_to_256k_uncapped": fov_uncap - all_condition_results[f"foveal_{core}_lm_output_kl__uncapped"]["bpb"]["cross_dataset_mean_bpb"]["2048"],
                "growth_2k_to_256k_capped": fov_cap - all_condition_results[f"foveal_{core}_lm_output_kl__capped"]["bpb"]["cross_dataset_mean_bpb"]["2048"]
            },
            "adaptation_contrast_B_orig_minus_B_fov": adaptation_contrast
        }

    # Write full output and summary
    out_file = WORK_DIR / "e2_retention_comparison.json"
    out_file.write_text(json.dumps(all_condition_results, indent=2) + "\n", encoding="utf-8")

    sum_file = RECORDS_DIR / "e2_results_summary.json"
    sum_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n[E2 Execution] Complete in {time.time() - t_start:.1f}s.")
    print(f"  Summary written to: {sum_file}")
    print(f"  Full evaluation records written to: {out_file}")

    print("\n--- E2 PRIMARY CONTRASTS ---")
    for core, data in summary["primary_endpoints"].items():
        print(f"Core: {core.upper()}")
        print(f"  Original: Uncapped 256K = {data['original_stage2']['uncapped_256k_bpb']:.4f}, Capped = {data['original_stage2']['capped_256k_bpb']:.4f}, Cap Benefit B = {data['original_stage2']['cap_benefit_B']:+.4f}")
        print(f"  Foveal:   Uncapped 256K = {data['foveal_cpt']['uncapped_256k_bpb']:.4f}, Capped = {data['foveal_cpt']['capped_256k_bpb']:.4f}, Cap Benefit B = {data['foveal_cpt']['cap_benefit_B']:+.4f}")
        print(f"  Adaptation Contrast B(orig) - B(foveal) = {data['adaptation_contrast_B_orig_minus_B_fov']:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
