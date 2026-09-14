#!/usr/bin/env python3
"""E1: Temperature-matched component attribution evaluation.

Evaluates 9 models (3 arms x 3 initializations):
Arms:
1. nope_memory
2. temperature_softmax_memory (NoPE with Polar's learned length temperatures)
3. polar_memory

Initializations (Triplets):
- Triplet 1 (Seed 20270912 / Primary pair)
- Triplet 2 (Seed 20270913 / Replication Seed 1)
- Triplet 3 (Seed 20270914 / Replication Seed 2)

Evaluates on:
- 30-document natural text retrieval and synthetic retrieval at 2K, 8K, 16K, 32K, 64K
- 18-document fixed-target BPB panel at 2K and 64K
- 2K validation loss on FineWeb-Edu

Computes primary contrast: Polar minus Temperature-Softmax natural text exact retrieval (8K-64K mean).
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
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scoring import DirectScorer, TokenRequest
from train.data import data_generator

MANIFESTS_DIR = ROOT / "supplementary" / "pat_feedback" / "manifests"
RECORDS_DIR = ROOT / "supplementary" / "pat_feedback" / "records"
WORK_DIR = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

E1_LENGTHS = [2048, 8192, 16384, 32768, 65536]
PRIMARY_RETRIEVAL_LENGTHS = [8192, 16384, 32768, 65536]

TRIPLETS = [
    {
        "triplet_id": "s1",
        "init_seed": 20270912,
        "nope_path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
        "polar_path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d"
    },
    {
        "triplet_id": "s2",
        "init_seed": 20270913,
        "nope_path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed1_nope/snapshots/8ab5fe847915223abfec6a146c97f173b8c72058",
        "polar_path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed1_polar/snapshots/6b3226d04af1f8c982c9085c8c9dbd8282e11220"
    },
    {
        "triplet_id": "s3",
        "init_seed": 20270914,
        "nope_path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed2_nope/snapshots/3196eac86d84759bb245b410da61eb8665bfdf05",
        "polar_path": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed2_polar/snapshots/50a3790bd3fcb55b3a4d69921df2725b21863fa5"
    }
]


def load_fixtures():
    bpb_tokens_path = MANIFESTS_DIR / "bpb_18_document_tokens.pt"
    retrieval_tokens_path = MANIFESTS_DIR / "retrieval_30_document_tokens.pt"
    retrieval_manifest_path = MANIFESTS_DIR / "retrieval_fixture_manifest.json"

    bpb_tokens = torch.load(bpb_tokens_path, weights_only=True)
    retrieval_tokens = torch.load(retrieval_tokens_path, weights_only=True)
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))

    return bpb_tokens, retrieval_tokens, retrieval_manifest


def get_val_loss(scorer: DirectScorer, val_inputs: torch.Tensor, val_targets: torch.Tensor, mbs: int = 4) -> float:
    scorer.model.eval()
    vl = 0.0
    with torch.no_grad():
        for i in range(len(val_inputs) // mbs):
            mb_in = val_inputs[i * mbs:(i + 1) * mbs].to(scorer.device)
            mb_tgt = val_targets[i * mbs:(i + 1) * mbs].to(scorer.device)
            ls, _, _ = scorer.model(mb_in, mb_tgt)
            vl += ls.item()
    return vl / (len(val_inputs) * val_inputs.shape[1])


def evaluate_bpb(scorer: DirectScorer, bpb_tokens: Dict[str, List[List[int]]], lengths: List[int]) -> Dict[str, Any]:
    target_start = 262144
    target_len = 256
    results = {}

    for corpus, doc_list in bpb_tokens.items():
        corpus_res = {}
        for length in lengths:
            tot_nll, tot_bytes = 0.0, 0
            reqs = []
            bytes_list = []
            for doc in doc_list:
                ctx = doc[target_start - length:target_start]
                tgt = doc[target_start:target_start + target_len]
                reqs.append(TokenRequest(tuple(ctx), tuple(tgt)))
                bytes_list.append(len(scorer.tokenizer.decode(tgt).encode("utf-8")))

            scores = scorer.score_requests(reqs, batch_size=8)
            for sc, b_cnt in zip(scores, bytes_list):
                tot_nll += -sc["loglikelihood"]
                tot_bytes += b_cnt
            corpus_bpb = tot_nll / math.log(2.0) / tot_bytes
            corpus_res[str(length)] = corpus_bpb
        results[corpus] = corpus_res

    mean_bpb_by_length = {}
    for length in lengths:
        str_l = str(length)
        mean_bpb = sum(results[corpus][str_l] for corpus in ["finepdfs", "pg19", "proof_pile"]) / 3.0
        mean_bpb_by_length[str_l] = mean_bpb

    return {"per_corpus": results, "mean_bpb": mean_bpb_by_length}


def evaluate_retrieval(scorer: DirectScorer, retrieval_tokens: Dict[int, List[int]], manifest: Dict[str, Any], lengths: List[int]) -> Dict[str, Any]:
    filler_tok = scorer.tokenizer.encode("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
    results = {"synthetic": {}, "real": {}}

    for suite in ["synthetic", "real"]:
        results[suite] = {}
        for length in lengths:
            reqs = []
            for case in manifest["cases"]:
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
                        prompt = [scorer.tokenizer.eos_token_id] + body[:insert] + needle_ids + body[insert:] + cue_ids
                        reqs.append(TokenRequest(tuple(prompt), tuple(target_ids)))

            scores = scorer.score_requests(reqs, batch_size=16)
            tot_exact = sum(int(s["greedy_exact"]) for s in scores)
            tot_correct = sum(s["correct_tokens"] for s in scores)
            tot_tokens = sum(s["tokens"] for s in scores)
            prompt_count = len(scores)

            exact_acc = (tot_exact / prompt_count) * 100.0 if prompt_count else 0.0
            tok_acc = (tot_correct / tot_tokens) * 100.0 if tot_tokens else 0.0
            print(f"      [{suite:>9}] Length {length:>5}: Exact = {exact_acc:.1f}%, Token = {tok_acc:.1f}% ({prompt_count} prompts)", flush=True)
            results[suite][str(length)] = {
                "exact_accuracy_percent": exact_acc,
                "token_accuracy_percent": tok_acc,
                "prompts": prompt_count
            }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[E1 Execution] Loading fixtures and manifests...", flush=True)
    bpb_tokens, retrieval_tokens, retrieval_manifest = load_fixtures()

    # Preload validation stream (64 sequences = 131,072 tokens for rapid, stable validation loss)
    val_inputs, val_targets = next(data_generator("finewebedu10B/finewebedu_val_*.bin", 131072, seq_len=2048))

    master_results_file = WORK_DIR / "e1_component_attribution.json"
    evaluated_models = {}
    if master_results_file.is_file():
        try:
            evaluated_models = json.loads(master_results_file.read_text(encoding="utf-8"))
            print(f"[E1 Execution] Loaded {len(evaluated_models)} existing evaluations.")
        except Exception:
            evaluated_models = {}

    t_start = time.time()

    for triplet in TRIPLETS:
        t_id = triplet["triplet_id"]
        seed = triplet["init_seed"]
        print(f"\n========================================================")
        print(f"[E1 Execution] Processing Triplet {t_id} (Seed {seed})...")
        print(f"========================================================")

        # 1. Load polar learned alphas for this triplet
        polar_sd = torch.load(triplet["polar_path"] + "/weights.pt", map_location="cpu", weights_only=True)["model"]
        polar_alphas = {k: v for k, v in polar_sd.items() if "len_gain_raw" in k}

        arms = [
            ("nope_memory", triplet["nope_path"], False),
            ("temperature_softmax_memory", triplet["nope_path"], True),
            ("polar_memory", triplet["polar_path"], False)
        ]

        for arm_name, model_path, inject_temp in arms:
            run_id = f"pat_e1_{t_id}_{arm_name}"
            if run_id in evaluated_models:
                print(f"  Run {run_id} already evaluated, skipping.")
                continue

            print(f"\n  Evaluating {run_id} (arm={arm_name})...", flush=True)
            t0 = time.time()
            scorer = DirectScorer(model_path, device=str(device), max_length=65536 + 256, batch_size=16)

            if inject_temp:
                # Install temperature-softmax learned alphas matching Polar
                for b in [2, 6, 10, 14]:
                    attn = scorer.model.blocks[b].attn
                    attn.pos = "temperature_softmax"
                    attn.len_gain_raw = nn.Parameter(polar_alphas[f"blocks.{b}.attn.len_gain_raw"].clone().to(device))
                print(f"    Injected learned length-temperatures into 4 attention blocks.")

            # Validation NLL
            print("    Evaluating 2K validation loss on FineWeb-Edu...", flush=True)
            val_loss = get_val_loss(scorer, val_inputs, val_targets)
            print(f"      Validation loss: {val_loss:.5f} nats/token")

            # BPB at 2K and 64K
            print("    Evaluating BPB on 18 documents at 2K and 64K...", flush=True)
            bpb_res = evaluate_bpb(scorer, bpb_tokens, [2048, 65536])
            print(f"      BPB 2K: {bpb_res['mean_bpb']['2048']:.4f}, BPB 64K: {bpb_res['mean_bpb']['65536']:.4f}")

            # Retrieval across 5 lengths
            print("    Evaluating Retrieval on 30 cases across 2K to 64K...", flush=True)
            ret_res = evaluate_retrieval(scorer, retrieval_tokens, retrieval_manifest, E1_LENGTHS)
            mean_exact_8k_64k = sum(ret_res["real"][str(l)]["exact_accuracy_percent"] for l in PRIMARY_RETRIEVAL_LENGTHS) / len(PRIMARY_RETRIEVAL_LENGTHS)
            mean_synth_8k_64k = sum(ret_res["synthetic"][str(l)]["exact_accuracy_percent"] for l in PRIMARY_RETRIEVAL_LENGTHS) / len(PRIMARY_RETRIEVAL_LENGTHS)
            print(f"      Natural Text Exact (8K-64K mean): {mean_exact_8k_64k:.2f}%")
            print(f"      Synthetic Exact (8K-64K mean):    {mean_synth_8k_64k:.2f}%")

            del scorer
            torch.cuda.empty_cache()

            dt = time.time() - t0
            run_eval_record = {
                "run_id": run_id,
                "triplet_id": t_id,
                "arm": arm_name,
                "seed": seed,
                "elapsed_sec": round(dt, 2),
                "val_loss_2k": round(val_loss, 5),
                "primary_natural_text_exact_mean_8k_to_64k": round(mean_exact_8k_64k, 2),
                "synthetic_exact_mean_8k_to_64k": round(mean_synth_8k_64k, 2),
                "retrieval": ret_res,
                "bpb": bpb_res
            }
            evaluated_models[run_id] = run_eval_record
            master_results_file.write_text(json.dumps(evaluated_models, indent=2) + "\n", encoding="utf-8")
            print(f"  ✓ Saved {run_id} evaluation in {dt:.1f}s")

    # Compute Primary and Secondary contrasts
    summary = {
        "protocol_id": "pat_feedback_v1_e1",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": round(time.time() - t_start, 2),
        "primary_metric": "natural_text_exact_five_token_accuracy_mean_8k_to_64k",
        "triplets": {}
    }

    primary_contrasts = []
    temp_vs_nope_contrasts = []

    for triplet in TRIPLETS:
        t_id = triplet["triplet_id"]
        nope_score = evaluated_models[f"pat_e1_{t_id}_nope_memory"]["primary_natural_text_exact_mean_8k_to_64k"]
        temp_score = evaluated_models[f"pat_e1_{t_id}_temperature_softmax_memory"]["primary_natural_text_exact_mean_8k_to_64k"]
        polar_score = evaluated_models[f"pat_e1_{t_id}_polar_memory"]["primary_natural_text_exact_mean_8k_to_64k"]

        primary_contrast = polar_score - temp_score
        temp_contrast = temp_score - nope_score

        primary_contrasts.append(primary_contrast)
        temp_vs_nope_contrasts.append(temp_contrast)

        summary["triplets"][t_id] = {
            "seed": triplet["init_seed"],
            "nope_memory": nope_score,
            "temperature_softmax_memory": temp_score,
            "polar_memory": polar_score,
            "primary_contrast_polar_minus_temp": round(primary_contrast, 2),
            "secondary_contrast_temp_minus_nope": round(temp_contrast, 2),
            "bpb_64k": {
                "nope": evaluated_models[f"pat_e1_{t_id}_nope_memory"]["bpb"]["mean_bpb"]["65536"],
                "temp": evaluated_models[f"pat_e1_{t_id}_temperature_softmax_memory"]["bpb"]["mean_bpb"]["65536"],
                "polar": evaluated_models[f"pat_e1_{t_id}_polar_memory"]["bpb"]["mean_bpb"]["65536"]
            }
        }

    summary["summary_statistics"] = {
        "primary_contrast_polar_minus_temp": {
            "mean": round(sum(primary_contrasts) / len(primary_contrasts), 2),
            "min": round(min(primary_contrasts), 2),
            "max": round(max(primary_contrasts), 2),
            "contrasts": primary_contrasts
        },
        "secondary_contrast_temp_minus_nope": {
            "mean": round(sum(temp_vs_nope_contrasts) / len(temp_vs_nope_contrasts), 2),
            "min": round(min(temp_vs_nope_contrasts), 2),
            "max": round(max(temp_vs_nope_contrasts), 2),
            "contrasts": temp_vs_nope_contrasts
        }
    }

    sum_file = RECORDS_DIR / "e1_results_summary.json"
    sum_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("[E1 Execution] Complete. Results written to:")
    print(f"  Master evaluations: {master_results_file}")
    print(f"  Summary contrasts:  {sum_file}")
    print("=" * 80)
    print("\n--- E1 COMPONENT ATTRIBUTION CONTRASTS (Natural Text Exact 8K-64K) ---")
    for t_id, data in summary["triplets"].items():
        print(f"Triplet {t_id} (Seed {data['seed']}):")
        print(f"  NoPE: {data['nope_memory']:.2f}% | Temp-Softmax: {data['temperature_softmax_memory']:.2f}% | Polar: {data['polar_memory']:.2f}%")
        print(f"  Primary Contrast (Polar - Temp-Softmax):  {data['primary_contrast_polar_minus_temp']:+.2f}%")
        print(f"  Secondary Contrast (Temp-Softmax - NoPE): {data['secondary_contrast_temp_minus_nope']:+.2f}%\n")

    p_stats = summary["summary_statistics"]["primary_contrast_polar_minus_temp"]
    s_stats = summary["summary_statistics"]["secondary_contrast_temp_minus_nope"]
    print(f"PRIMARY CONTRAST (Polar - Temp-Softmax): Mean = {p_stats['mean']:+.2f}% (range: [{p_stats['min']:+.2f}%, {p_stats['max']:+.2f}%])")
    print(f"SECONDARY CONTRAST (Temp-Softmax - NoPE): Mean = {s_stats['mean']:+.2f}% (range: [{s_stats['min']:+.2f}%, {s_stats['max']:+.2f}%])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
