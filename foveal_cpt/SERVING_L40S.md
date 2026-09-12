# Foveal serving on NVIDIA L40S

Measured 2026-09-12 with the corrected `FovealLLM` decoder. This focused comparison covers Polar KL and NoPE LM-output+KL, the variants behind the reported synthetic-retrieval gains, with each core's local CPT control. **Across 2K–256K, active Foveal routing adds only 0.3–6.2% decode time per token and 3.9–14.4% prefill time over the matched local controls.** This supports the speed potential suggested by the roughly SWA retrieval-evaluation runtime. The serving sample covers four selected checkpoints, not all twelve variants.

## Warm request performance

Batch size 1, greedy generation, EOS ignored, 130 output tokens: the first token comes from prefill and the remaining 129 are timed cached decode steps. Each cell has one excluded warmup and three measured requests with fresh request caches. Prefill and decode rates below are medians; parentheses give the observed three-request decode range. Memory is the maximum across measured requests. Each model/length uses a separate process on an otherwise idle L40S.

| Core / variant | Context | Prefill ms | Decode token/s, median (range) | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| polar `local` | 2K | 38.7 | 57.78 (57.74–57.80) | 1.46 / 1.58 |
| polar `kl` | 2K | 40.4 | 57.59 (57.32–57.92) | 1.46 / 1.58 |
| nope `local` | 2K | 42.0 | 69.08 (68.93–69.20) | 1.46 / 1.58 |
| nope `lm_output_kl` | 2K | 48.1 | 66.19 (65.86–66.57) | 1.46 / 1.58 |
| polar `local` | 32K | 258.2 | 57.00 (56.45–57.21) | 3.30 / 3.73 |
| polar `kl` | 32K | 270.2 | 56.31 (56.30–56.77) | 3.30 / 3.73 |
| nope `local` | 32K | 256.5 | 68.81 (68.48–68.98) | 3.24 / 3.66 |
| nope `lm_output_kl` | 32K | 266.6 | 64.80 (64.44–65.01) | 3.24 / 3.91 |
| polar `local` | 256K | 2160.9 | 51.14 (51.00–51.28) | 17.22 / 20.85 |
| polar `kl` | 256K | 2300.8 | 49.70 (49.68–50.07) | 17.22 / 20.84 |
| nope `local` | 256K | 2297.2 | 69.85 (69.21–70.84) | 16.72 / 20.35 |
| nope `lm_output_kl` | 256K | 2579.9 | 66.40 (65.80–66.60) | 16.72 / 20.84 |

The same serial engine runs both adapted and local checkpoints. Remote routing is active in every measured nonlocal attention layer and absent in local controls. These are end-to-end engine timings, including synchronized Python decode steps and full-prefix KV concatenation; they are not isolated attention-kernel timings or batching throughput. The historical ordinary-engine CUDA-graph rates are a different implementation and protocol, so they do not provide a matched Foveal speedup baseline.

## Matched adaptation overhead

Positive percentages mean slower than the same core's local CPT control. Decode overhead compares time per token using the median rates above. Three repetitions describe these requests; they do not establish statistical significance or a general performance guarantee.

| Core / variant | Context | Prefill time change | Decode time/token change |
| --- | ---: | ---: | ---: |
| polar `kl` | 2K | +4.3% | +0.3% |
| polar `kl` | 32K | +4.6% | +1.2% |
| polar `kl` | 256K | +6.5% | +2.9% |
| nope `lm_output_kl` | 2K | +14.4% | +4.4% |
| nope `lm_output_kl` | 32K | +3.9% | +6.2% |
| nope `lm_output_kl` | 256K | +12.3% | +5.2% |

## Load and first request

These costs are excluded from the warm table. TorchInductor/Triton caches persisted from earlier work, so the first request is not an isolated cold-compilation measurement. Checkpoint load includes configuration/tokenizer/base resolution and weight loading. The raw reports retain separate first-request prefill/decode times and memory peaks.

| Core / variant | Context | Load seconds | First request wall seconds |
| --- | ---: | ---: | ---: |
| polar `local` | 2K | 26.59 | 3.40 |
| polar `kl` | 2K | 26.08 | 3.33 |
| nope `local` | 2K | 26.41 | 5.33 |
| nope `lm_output_kl` | 2K | 25.48 | 4.22 |
| polar `local` | 32K | 27.18 | 3.53 |
| polar `kl` | 32K | 25.23 | 3.57 |
| nope `local` | 32K | 26.20 | 4.30 |
| nope `lm_output_kl` | 32K | 7.85 | 4.57 |
| polar `local` | 256K | 5.33 | 13.84 |
| polar `kl` | 256K | 5.36 | 5.84 |
| nope `local` | 256K | 5.34 | 6.07 |
| nope `lm_output_kl` | 256K | 5.35 | 6.51 |

## Evidence and limits

[Raw reports, environment, protocol and summary](../benchmarks/logs/foveal_serving_l40s) record every repetition and source checksum. CPT snapshot: `e4cf2558793646c26d9e5a6c6eeadb4e1011cad3`; decoder source: commit `04c00c999f4bde26a06d00307f15154434d1be8d`. Hardware/software: NVIDIA L40S (46,068 MiB), driver 595.91.07, PyTorch 2.14.0+cu130, Triton 3.8.0, FLA 0.5.2, four OpenMP threads. BF16 activations use the checkpoint's mixed parameter dtypes.

Bitwise equality is not required to assess speed potential. Across the four selected checkpoints, all **220 sampled FP32 comparisons** passed `atol=rtol=1e-4` with matching greedy tokens and final route sets, and all **eight 130-token free-running continuations** matched. Average BF16 logit RMS differences were **0.038–0.047**, compared with **0.039–0.049** for the separate full-forward recurrent-memory control. This numerical evidence supports interpreting the serving measurements as implementation speed potential while retaining the observed precision differences. The [numerical context](../benchmarks/logs/foveal_serving_l40s/numerical_context.json) records the per-checkpoint comparisons. These empirical controls do not define a universal BF16 tolerance. Cache diagnostics reached 32K, not 256K; the timings do not certify universal cached/full-forward equivalence or retrieval quality at these lengths. See [AUDIT_FOLLOWUP.md](AUDIT_FOLLOWUP.md). No 512K/1M capacity attempt or isolated cold-compilation benchmark was run. The training-data protocol limitation remains open.
