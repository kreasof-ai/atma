# Foveal serving on NVIDIA L40S: corrected fast decoder

> **OBSOLETE NOTICE:** This intermediate 4-checkpoint report is superseded by the complete 12-checkpoint, 2K–512K serving sweep and context scaling analysis documented in [**`BENCHMARK_RESULTS.md` Section 6**](BENCHMARK_RESULTS.md#6-foveal-serving-performance--context-scaling-verified-fast-decoder).

**Active Foveal routing now decodes at 589–590 tokens/s at 2K and 571–572 tokens/s at 256K** for Polar KL and NoPE LM-output+KL. Fresh dense source-checkpoint baselines measured 442 and 466 tokens/s at 2K. These are new measurements with routing enabled; they independently establish 400+ token/s serving for the measured variants.

The earlier 50–70 token/s measurements used an eager reference decoder. Comparing that implementation against its equally slow local control hid a major serving regression. Those timings are retained in the [eager experiment archive](../benchmarks/logs/foveal_serving_l40s), but they should not be used as evidence that optimized serving had been achieved.

## What changed

The decoder now keeps its rounded inference weights resident, writes KV into fixed storage, reads selected remote pages and the local window directly in a fused attention kernel, and captures decode in CUDA graphs. It reuses the fast engine's fused convolution, MLP activation and memory-update kernels. Separate ordinary-step and page-boundary graphs preserve routing anchors, generated-page completion and the LM-output residual. The original full-forward training/scoring path is unchanged; `enforce_eager=True` retains the cached reference path.

The eager profile attributed 61.65% of recorded GPU self time to copies/conversions, alongside substantial host launch overhead. It repeatedly cast master weights and concatenated the growing KV cache. The graph path removes those costs. For example, Polar KL at 2K improved from 57.6 to 588.8 tokens/s with the same checkpoint and 130-token request budget.

## Warm serving performance

All twelve selected model/length cells completed on this L40S. Each cell used one excluded warmup and three measured requests, batch size 1, greedy generation, EOS ignored, and 130 output tokens: the first token comes from prefill, followed by 129 timed cached steps. Requests start with fresh state. Values are warm medians, with graph setup and capture included in first-request prefill instead. Four checkpoints were measured, not the entire twelve-checkpoint adaptation sweep.

| Core / adaptation | Context | Local / Foveal decode token/s | Local / Foveal prefill ms |
| --- | ---: | ---: | ---: |
| polar `kl` | 2K | 654.1 / **588.8** | 40.4 / 42.1 |
| polar `kl` | 32K | 654.7 / **572.3** | 258.8 / 271.0 |
| polar `kl` | 256K | 654.3 / **571.0** | 2162.7 / 2303.0 |
| nope `lm_output_kl` | 2K | 660.3 / **590.2** | 65.2 / 51.4 |
| nope `lm_output_kl` | 32K | 660.7 / **573.1** | 258.6 / 268.4 |
| nope `lm_output_kl` | 256K | 661.1 / **572.0** | 2303.0 / 2582.8 |

## Fresh dense baseline comparison

Decode tokens/s, using the same 2K prompt and 130-token budget, one warmup and three measured requests. The dense baselines use the untouched source checkpoints with full attention (`attn_window=None`) in the existing optimized engines. Prefix reuse was disabled so repeated prompts start with fresh state.

| Core | Dense source, 2K | Foveal local, 2K | Active Foveal, 2K |
| --- | ---: | ---: | ---: |
| polar | 441.6 | 654.1 | **588.8** |
| nope | 465.6 | 660.3 | **590.2** |

These compare working serving implementations. Foveal captures the output head and greedy selection as well as the decoder, while the ordinary engines retain their existing scheduler and graph boundaries. Consequently, the short-context difference is not an isolated attention-kernel speedup or a claim that sparsity itself makes 2K faster. It does show that preserving Foveal routing is compatible with 400+ token/s decoding on this machine.

## Repetition ranges and memory

Decode ranges cover the three measured requests. Allocated/reserved memory is the maximum across those requests. Original master weights remain available to the shared forward path, alongside the graph decoder's resident inference weights.

| Core / mode | Context | Decode token/s range | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: |
| polar `local` | 2K | 654.0–654.1 | 2.10 / 2.23 |
| polar `kl` | 2K | 588.7–588.9 | 2.10 / 2.23 |
| nope `local` | 2K | 660.3–660.4 | 2.10 / 2.23 |
| nope `lm_output_kl` | 2K | 590.2–590.2 | 2.10 / 2.23 |
| polar `local` | 32K | 654.7–654.7 | 4.05 / 4.51 |
| polar `kl` | 32K | 572.3–572.4 | 4.05 / 4.51 |
| nope `local` | 32K | 660.7–660.8 | 3.99 / 4.51 |
| nope `lm_output_kl` | 32K | 573.1–573.1 | 3.99 / 4.76 |
| polar `local` | 256K | 654.3–654.4 | 18.84 / 22.37 |
| polar `kl` | 256K | 570.9–571.0 | 18.84 / 22.38 |
| nope `local` | 256K | 661.1–661.1 | 18.34 / 22.37 |
| nope `lm_output_kl` | 256K | 571.9–572.0 | 18.34 / 22.38 |

## Numerical validation

All **124 regression tests passed**, including new FP32 checks of sparse support, memory/KV state, generated pages, Polar's null-sink edge case and request resets. On the two measured adapted checkpoints, all **24 sampled real-weight FP32 comparisons** passed `atol=rtol=1e-4`, with identical final route sets. The 3,968-token-prefix case makes the 32-remote-page cap active while remaining within the eager FP32 reference's 4,096-token limit.

The BF16 checks cover 65-, 641- and 8,192-token prefixes with 66 teacher-fed steps. All **36 sampled greedy tokens agree**, and all **four 130-token free-running continuations match** full recomputation. Thirty BF16 comparisons fail the strict FP32-style tolerance; these failures are retained. Mean BF16 logit RMS error is 0.0346 for Polar KL and 0.0283 for NoPE LM-output+KL, versus 0.0447 and 0.0428 in the independent recurrent full-forward controls. This supports practical inference agreement without claiming bitwise identity or a universal BF16 tolerance. Full-forward parity at 256K was not retested.

## Load and first request

Persistent compiler caches were reused, so these are not isolated cold-compilation measurements. Load and first-request prefill/decode are excluded from the warm table; graph setup and capture are included in first-request prefill.

| Core / mode | Context | Load s | First prefill / decode s |
| --- | ---: | ---: | ---: |
| polar `local` | 2K | 5.33 | 1.503 / 0.194 |
| polar `kl` | 2K | 5.34 | 1.112 / 0.216 |
| nope `local` | 2K | 5.31 | 2.108 / 0.192 |
| nope `lm_output_kl` | 2K | 5.43 | 1.867 / 0.216 |
| polar `local` | 32K | 5.30 | 1.205 / 0.197 |
| polar `kl` | 32K | 5.36 | 1.311 / 0.226 |
| nope `local` | 32K | 5.27 | 1.899 / 0.195 |
| nope `lm_output_kl` | 32K | 5.29 | 2.159 / 0.225 |
| polar `local` | 256K | 5.30 | 3.084 / 0.210 |
| polar `kl` | 256K | 5.35 | 3.312 / 0.239 |
| nope `local` | 256K | 5.28 | 3.880 / 0.208 |
| nope `lm_output_kl` | 256K | 5.32 | 4.236 / 0.238 |

## Reproduction and scope

[Raw repetitions, numerical checks, protocol, environment and checksums](../benchmarks/logs/foveal_graph_l40s) identify every measured source. Decoder commit: `36373b7`; CPT snapshot: `e4cf2558793646c26d9e5a6c6eeadb4e1011cad3`. NVIDIA L40S (46,068 MiB), driver 595.91.07, PyTorch 2.14.0+cu130, Triton 3.8.0, FLA 0.5.2, four OpenMP threads. No 512K/1M capacity or batching-throughput claim is made. The training-data protocol limitation in [README.md](README.md) remains separate from this serving result.
