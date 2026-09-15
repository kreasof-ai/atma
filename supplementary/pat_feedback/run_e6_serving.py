"""E6: Conditional serving repeatability benchmark across 6 model architectures.

Models:
  - nope
  - polar
  - rope
  - raven_native
  - atma_raven_titans
  - tda_hybrid (labeled as direct full-prefix recomputation)

Context lengths: 2048, 32768, 131072.
Parameters: 1 sequence, 32 generated tokens.
Repetitions: 1 untimed warmup, 10 timed repetitions.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.model import EvalModel, resolve_checkpoint
from benchmarks.scoring import DirectScorer
from benchmarks.direct_serving import measure_sequence
from benchmarks.serving import _prompt_ids

SNAPSHOTS = {
    "nope": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
    "polar": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
    "rope": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0/snapshots/de0a2f1c63b0631a040a2d6adac9e2d43fd5c581",
    "raven_native": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-raven-native__reg-baseline__distr-0__mem-0__win-0/snapshots/875d1c6ab0603a8302a168736c2d9cab179b01c3",
    "atma_raven_titans": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0/snapshots/8427bfefc77ed199e912d31c80fd6ff0ea179876",
    "tda_hybrid": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--scaled_tda_hybrid/snapshots/d6ca0604812ddfc001c74ec08fddc130253026f3",
}

LENGTHS = [2048, 32768, 131072]
DECODE_TOKENS = 32
NUM_REPETITIONS = 10


def compute_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_tda_length(path: str, context: int, tokenizer) -> dict:
    print(f"    [tda_hybrid] Benchmarking context={context} (prefill/TTFT, decode skipped)...", flush=True)
    clear_gpu()
    scorer = DirectScorer(path, max_length=None, batch_size=1)
    prompt = _prompt_ids(tokenizer, context)

    # Untimed warmup (prefill only: decode_tokens=1)
    measure_sequence(scorer, prompt, 1)
    clear_gpu()

    reps = []
    num_reps = NUM_REPETITIONS

    for rep in range(num_reps):
        m = measure_sequence(scorer, prompt, 1)
        prefill_t = m["prefill_time_s"]
        prefill_throughput = m["prefill_tokens"] / prefill_t if prefill_t > 0 else 0.0

        reps.append({
            "rep": rep + 1,
            "prefill_time_s": prefill_t,
            "prefill_tokens_per_s": prefill_throughput,
            "decode_time_s": 0.0,
            "decode_tokens": 0,
            "decode_latency_per_token_s": None,
            "decode_tokens_per_s": None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })

    scorer.close()
    clear_gpu()

    return {
        "backend": "direct_full_recompute",
        "notes": "Prefill/TTFT measured with upstream parallel kernel; autoregressive decode skipped due to lack of upstream cached decode kernel",
        "context_tokens": context,
        "decode_tokens": 0,
        "repetitions": reps,
        "prefill_latency_s": compute_stats([r["prefill_time_s"] for r in reps]),
        "prefill_tokens_per_s": compute_stats([r["prefill_tokens_per_s"] for r in reps]),
        "decode_latency_s_per_token": {"median": None, "iqr": None, "min": None, "max": None},
        "decode_tokens_per_s": {"median": None, "iqr": None, "min": None, "max": None},
        "peak_allocated_gib": reps[-1]["peak_allocated_bytes"] / (1024 ** 3),
        "peak_reserved_gib": reps[-1]["peak_reserved_bytes"] / (1024 ** 3),
    }


def benchmark_eval_model_length(name: str, path: str, context: int, tokenizer) -> dict:
    print(f"    [{name}] Benchmarking context={context}...", flush=True)
    clear_gpu()
    budget = context + DECODE_TOKENS + 64

    # Determine backend config
    llm_kwargs = {
        "max_model_len": budget,
        "max_num_seqs": 1,
        "enforce_eager": False,  # Keep CUDA graphs enabled for real decode throughput
    }

    m = EvalModel(path, max_tokens=DECODE_TOKENS, strict=False, quiet=True, **llm_kwargs)
    prompt = _prompt_ids(tokenizer, context)

    # Untimed warmup
    m.generate([prompt], max_tokens=DECODE_TOKENS, use_tqdm=False, ignore_eos=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    reps = []
    for rep in range(NUM_REPETITIONS):
        t0 = time.perf_counter()
        m.generate([prompt], max_tokens=DECODE_TOKENS, use_tqdm=False, ignore_eos=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - t0

        metrics = m.last_call_metrics or {}
        prefill_t = metrics.get("prefill_time", 0.0)
        decode_t = metrics.get("decode_time", 0.0)
        dec_toks = metrics.get("decode_tokens", DECODE_TOKENS - 1)
        prefill_throughput = metrics.get("prefill_throughput", context / prefill_t if prefill_t > 0 else 0.0)
        decode_throughput = metrics.get("decode_throughput", dec_toks / decode_t if decode_t > 0 else 0.0)
        decode_latency = decode_t / dec_toks if dec_toks > 0 else 0.0

        reps.append({
            "rep": rep + 1,
            "wall_time_s": wall_time,
            "prefill_time_s": prefill_t,
            "prefill_tokens_per_s": prefill_throughput,
            "decode_time_s": decode_t,
            "decode_tokens": dec_toks,
            "decode_latency_per_token_s": decode_latency,
            "decode_tokens_per_s": decode_throughput,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })

    backend = m.backend
    m.close()
    clear_gpu()

    return {
        "backend": backend,
        "context_tokens": context,
        "decode_tokens": DECODE_TOKENS,
        "repetitions": reps,
        "prefill_latency_s": compute_stats([r["prefill_time_s"] for r in reps]),
        "prefill_tokens_per_s": compute_stats([r["prefill_tokens_per_s"] for r in reps]),
        "decode_latency_s_per_token": compute_stats([r["decode_latency_per_token_s"] for r in reps]),
        "decode_tokens_per_s": compute_stats([r["decode_tokens_per_s"] for r in reps]),
        "peak_allocated_gib": reps[-1]["peak_allocated_bytes"] / (1024 ** 3),
        "peak_reserved_gib": reps[-1]["peak_reserved_bytes"] / (1024 ** 3),
    }


def main():
    print("=================================================================", flush=True)
    print("Starting E6 Conditional Serving Repeatability Benchmark", flush=True)
    print("=================================================================", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results = {}

    for model_name, model_path in SNAPSHOTS.items():
        print(f"\nEvaluating model: {model_name}...", flush=True)
        results[model_name] = {}

        for length in LENGTHS:
            if model_name == "tda_hybrid":
                cell = benchmark_tda_length(model_path, length, tokenizer)
            else:
                cell = benchmark_eval_model_length(model_name, model_path, length, tokenizer)

            results[model_name][str(length)] = cell
            dec_lat = cell["decode_latency_s_per_token"]["median"]
            dec_lat_str = f"{dec_lat * 1000.0:.2f} ms/tok" if dec_lat is not None else "N/A (skipped)"
            dec_tok_s = cell["decode_tokens_per_s"]["median"]
            dec_tok_s_str = f"{dec_tok_s:.1f} tok/s" if dec_tok_s is not None else "N/A"

            print(
                f"      Length {length:>6}: "
                f"Prefill TTFT = {cell['prefill_latency_s']['median']:.4f}s "
                f"({cell['prefill_tokens_per_s']['median']:.1f} tok/s) | "
                f"Decode = {dec_lat_str} "
                f"({dec_tok_s_str}) | "
                f"Peak Mem = {cell['peak_reserved_gib']:.2f} GiB",
                flush=True
            )

    out_dir = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "e6_serving_repeatability.json"

    full_payload = {
        "protocol": "E6 Serving Repeatability Protocol v1",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "decode_tokens": DECODE_TOKENS,
        "repetitions_per_cell": NUM_REPETITIONS,
        "lengths": LENGTHS,
        "models": list(SNAPSHOTS.keys()),
        "results": results,
    }

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"\nSaved raw evaluation to: {eval_path}", flush=True)

    # Summary record
    rec_dir = ROOT / "supplementary" / "pat_feedback" / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    summary_path = rec_dir / "e6_results_summary.json"

    summary_rows = {}
    for m, l_dict in results.items():
        summary_rows[m] = {}
        for l, cell in l_dict.items():
            dec_lat = cell["decode_latency_s_per_token"]["median"]
            summary_rows[m][l] = {
                "prefill_ttft_median_s": cell["prefill_latency_s"]["median"],
                "prefill_tok_per_s_median": cell["prefill_tokens_per_s"]["median"],
                "decode_latency_ms_median": dec_lat * 1000.0 if dec_lat is not None else None,
                "decode_tok_per_s_median": cell["decode_tokens_per_s"]["median"],
                "peak_reserved_gib": cell["peak_reserved_gib"],
                "backend": cell["backend"],
            }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": "completed",
            "summary_table": summary_rows,
            "artifacts": {
                "evaluation_json": "supplementary/pat_feedback/work/evaluation/e6_serving_repeatability.json"
            }
        }, f, indent=2)
    print(f"Saved results summary to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
