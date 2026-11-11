# E6 Cold-Cache Serving Repeatability Benchmark (10 Timed Repetitions After Warmup, 1x NVIDIA L40S)

Mutating token 0 per repetition prevents 256-token block prefix-cache hits in Polar's `LLM` engine, ensuring every model executes full-prefix prefill across all 10 repetitions.

| Model | Backend | Context | Cold TTFT Median (s) | TTFT IQR (s) | Prefill Tok/s | Decode ms/tok | Decode Tok/s | Peak Mem (GiB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATMA NoPE + Mem | `BaselineLLM` | 2K | **0.022** | 0.000 | 91,381 | 2.18 | 458.4 | 0.81 |
| ATMA NoPE + Mem | `BaselineLLM` | 32K | **0.218** | 0.000 | 150,522 | 5.22 | 191.6 | 2.25 |
| ATMA NoPE + Mem | `BaselineLLM` | 128K | **1.434** | 0.003 | 91,404 | 14.80 | 67.6 | 6.86 |
| ATMA Polar + Mem | `LLM` | 2K | **0.024** | 0.000 | 83,990 | 2.26 | 443.0 | 0.81 |
| ATMA Polar + Mem | `LLM` | 32K | **0.237** | 0.000 | 138,096 | 5.44 | 183.8 | 2.28 |
| ATMA Polar + Mem | `LLM` | 128K | **1.589** | 0.004 | 82,469 | 15.48 | 64.6 | 6.99 |
| ATMA RoPE + Mem | `BaselineLLM` | 2K | **0.022** | 0.000 | 93,154 | 2.29 | 437.2 | 0.81 |
| ATMA RoPE + Mem | `BaselineLLM` | 32K | **0.224** | 0.000 | 146,377 | 5.33 | 187.6 | 2.23 |
| ATMA RoPE + Mem | `BaselineLLM` | 128K | **1.456** | 0.009 | 89,992 | 14.92 | 67.0 | 6.80 |
| RAVEN Native | `BaselineLLM` | 2K | **0.045** | 0.000 | 45,544 | 3.29 | 304.2 | 0.85 |
| RAVEN Native | `BaselineLLM` | 32K | **0.310** | 0.000 | 105,666 | 3.30 | 303.1 | 2.17 |
| RAVEN Native | `BaselineLLM` | 128K | **1.313** | 0.001 | 99,809 | 3.33 | 299.9 | 6.39 |
| ATMA RAVEN + Titans | `BaselineLLM` | 2K | **0.028** | 0.000 | 74,218 | 2.25 | 444.5 | 0.85 |
| ATMA RAVEN + Titans | `BaselineLLM` | 32K | **0.236** | 0.000 | 138,630 | 2.26 | 442.9 | 2.66 |
| ATMA RAVEN + Titans | `BaselineLLM` | 128K | **0.975** | 0.003 | 134,481 | 2.29 | 436.2 | 8.48 |
| Scaled TDA Hybrid | `direct_full_recompute` | 2K | **0.070** | 0.000 | 29,073 | — | — | 0.81 |
| Scaled TDA Hybrid | `direct_full_recompute` | 32K | **1.381** | 0.003 | 23,727 | — | — | 2.04 |
| Scaled TDA Hybrid | `direct_full_recompute` | 128K | **10.445** | 0.055 | 12,549 | — | — | 5.97 |
