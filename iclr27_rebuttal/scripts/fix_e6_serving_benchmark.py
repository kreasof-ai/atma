#!/usr/bin/env python3
"""Fixed E6 Serving Repeatability Benchmark (resolves the 0.065s vs 1.619s bug).

Root cause of the Table 33 vs Table 20 discrepancy:
In `supplementary/pat_feedback/run_e6_serving.py`, Polar was benchmarked via `LLM`
(which enables automatic 256-token block prefix caching), whereas NoPE/RoPE used
`BaselineLLM` (no prefix caching). Calling `m.generate([prompt])` 10 times in a loop
after an untimed warmup on the same `EvalModel` instance caused Polar repetitions 1..10
to hit the prefix cache for all complete 256-token blocks (130,816 tokens at 128K)
and only prefill the final 256-token block (256 / 3940.39 tok/s = 0.06497 s).

This script fixes the protocol by mutating token 0 per repetition so zero prefix
blocks match the cache, measuring true cold-cache full-prefix TTFT across all 10
repetitions.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.model import EvalModel
from benchmarks.scoring import DirectScorer
from benchmarks.direct_serving import measure_sequence
from benchmarks.serving import _prompt_ids

SNAPSHOTS = {
    "nope": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
    "polar": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
    "rope": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0/snapshots/de0a2f1c63b0631a040a2d6adac9e2d43fd5c581",
    "raven_native": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-raven-native__reg-baseline__distr-0__mem-0__win-0/snapshots/875d1c6ab0603a8302a168736c2d9cab179b01c3",
    "atma_raven_titans": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0/snapshots/8427bfefc77ed199e912d31c80fd6ff0ea179876",
    "tda_hybrid": "/home/sagemaker-user/.cache/huggingface/hub/models--anon--scaled_tda_hybrid/snapshots/d6ca0604812ddfc001c74ec08fddc130253026f3",
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


def make_distinct_prompt(base_prompt: list[int], rep_idx: int) -> list[int]:
    """Ensure the very first token of the prompt is unique per repetition.

    Because block-level prefix caching hashes cumulatively from token 0, changing
    token 0 invalidates every 256-token block hash in the sequence while keeping
    the exact sequence length `len(base_prompt)` unchanged.
    """
    prompt = list(base_prompt)
    prompt[0] = 1000 + (rep_idx % 500)
    return prompt


def benchmark_eval_model_cold_cache(name: str, path: str, context: int, tokenizer) -> dict:
    clear_gpu()
    budget = context + DECODE_TOKENS + 64
    llm_kwargs = {
        "max_model_len": budget,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max(budget, 16384),
        "enforce_eager": False,
    }

    m = EvalModel(path, max_tokens=DECODE_TOKENS, strict=False, quiet=True, **llm_kwargs)
    base_prompt = _prompt_ids(tokenizer, context)

    # Untimed warmup with rep_idx=0
    warmup_prompt = make_distinct_prompt(base_prompt, 0)
    m.generate([warmup_prompt], max_tokens=DECODE_TOKENS, use_tqdm=False, ignore_eos=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    reps = []
    for rep in range(1, NUM_REPETITIONS + 1):
        prompt = make_distinct_prompt(base_prompt, rep)
        t0 = time.perf_counter()
        m.generate([prompt], max_tokens=DECODE_TOKENS, use_tqdm=False, ignore_eos=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - t0

        metrics = m.last_call_metrics or {}
        prefill_t = metrics.get("prefill_time", 0.0)
        prefill_toks = metrics.get("prefill_tokens", context)
        decode_t = metrics.get("decode_time", 0.0)
        dec_toks = metrics.get("decode_tokens", DECODE_TOKENS - 1)
        prefill_throughput = context / prefill_t if prefill_t > 0 else 0.0
        decode_throughput = dec_toks / decode_t if decode_t > 0 else 0.0
        decode_latency = decode_t / dec_toks if dec_toks > 0 else 0.0

        reps.append({
            "rep": rep,
            "wall_time_s": wall_time,
            "prefill_tokens_executed": prefill_toks,
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
        "cache_mode": "cold_full_prefix",
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


def benchmark_tda_length(path: str, context: int, tokenizer) -> dict:
    clear_gpu()
    scorer = DirectScorer(path, max_length=None, batch_size=1)
    prompt = _prompt_ids(tokenizer, context)
    measure_sequence(scorer, prompt, 1)
    clear_gpu()

    reps = []
    for rep in range(1, NUM_REPETITIONS + 1):
        m = measure_sequence(scorer, prompt, 1)
        prefill_t = m["prefill_time_s"]
        prefill_throughput = m["prefill_tokens"] / prefill_t if prefill_t > 0 else 0.0
        reps.append({
            "rep": rep,
            "prefill_time_s": prefill_t,
            "prefill_tokens_per_s": prefill_throughput,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })

    scorer.close()
    clear_gpu()
    return {
        "backend": "direct_full_recompute",
        "cache_mode": "cold_full_prefix",
        "context_tokens": context,
        "decode_tokens": 0,
        "repetitions": reps,
        "prefill_latency_s": compute_stats([r["prefill_time_s"] for r in reps]),
        "prefill_tokens_per_s": compute_stats([r["prefill_tokens_per_s"] for r in reps]),
        "decode_latency_s_per_token": {"median": None},
        "decode_tokens_per_s": {"median": None},
        "peak_allocated_gib": reps[-1]["peak_allocated_bytes"] / (1024 ** 3),
        "peak_reserved_gib": reps[-1]["peak_reserved_bytes"] / (1024 ** 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "iclr27_rebuttal" / "tables" / "e6_fixed_serving.json",
    )
    parser.add_argument("--models", nargs="+", default=list(SNAPSHOTS.keys()))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results = {}

    for model_name in args.models:
        model_path = SNAPSHOTS[model_name]
        print(f"\n[E6 Cold-Cache Fix] Evaluating {model_name}...", flush=True)
        results[model_name] = {}
        for length in LENGTHS:
            if model_name == "tda_hybrid":
                cell = benchmark_tda_length(model_path, length, tokenizer)
            else:
                cell = benchmark_eval_model_cold_cache(model_name, model_path, length, tokenizer)
            results[model_name][str(length)] = cell
            ttft = cell["prefill_latency_s"]["median"]
            ptok = cell["prefill_tokens_per_s"]["median"]
            dlat = cell["decode_latency_s_per_token"]["median"]
            dlat_str = f"{dlat * 1000.0:.2f} ms/tok" if dlat is not None else "Skipped"
            print(
                f"  Length {length:>6}: Cold TTFT = {ttft:.4f}s ({ptok:,.1f} tok/s) | Decode = {dlat_str}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved cold-cache E6 serving results to {args.output}")


if __name__ == "__main__":
    main()
