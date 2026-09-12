# Foveal CPT 10B Benchmark Results & Scientific Interpretation

This document compiles and interprets the complete evaluation suite across all **12 Foveal CPT checkpoints** from [`ChavyvAkvar/atma-foveal-cpt-all`](https://huggingface.co/ChavyvAkvar/atma-foveal-cpt-all).

All checkpoints underwent **1B tokens** of continuous pre-training (CPT) at 32K context and 524,288 tokens/batch (step 1908). Evaluation encompasses five distinct axes:
1. **Downstream Base LM Tasks:** ARC-challenge, ARC-easy, BoolQ, HellaSwag, OpenBookQA, PIQA, LAMBADA, WinoGrande (zero-shot).
2. **Long-Context Retrieval:** Needle-In-A-Haystack (NIAH) and Passkey across 2K to 256K context lengths, comparing Synthetic filler against Real-text distractors (`codelion/finepdfs-1B`).
3. **BABILong Reasoning Extrapolation:** Controlled short-context adaptation ($\le$2K, QA1–QA10) evaluated across 0K to 256K context lengths.
4. **Bits-Per-Byte (BPB) Language Modeling:** Fixed-target evaluation on PG-19, Proof-Pile-2, and FinePDFs from 2K to 256K.
5. **Historical ordinary-engine serving:** Prefill/decode timing and peak VRAM on NVIDIA L40S, with Foveal index/route parameters omitted. These are not faithful Foveal serving measurements; see Section 6.

The corrected aggregated dataset contains **6,336 structured rows from 72 full experiments** under `benchmarks/logs/foveal_cpt/benchmark_matrix.json` and `benchmarks/logs/foveal_cpt/benchmark_matrix.csv`.


The [aggregation manifest](../benchmarks/logs/foveal_cpt/aggregation_manifest.json) selects one full run per model/suite, excludes 24 smoke runs and 13 superseded shorter runs, and records source checksums. Raw logs are unchanged. This supersedes the earlier 7,146-row matrix that mixed those experiments. Full-forward retrieval, base-task, long-document and BABILong quality results remain distinct from the historical serving path. The L40S checks and limits on interpretation are recorded in [AUDIT_FOLLOWUP.md](AUDIT_FOLLOWUP.md).

**Scope of the claim:** this is an empirical comparison under flat-stream 32K CPT, with no document-boundary attention or memory resets; the [training protocol gate](README.md#run-gates) remains open. The strongest result is improved synthetic needle **token accuracy** for KL-trained Polar/NoPE variants over local CPT controls. Real-text retrieval and reasoning results are mixed. One trained checkpoint per cell and the KL cells' additional 20M-token calibration do not establish universal architectural or causal claims. Historical serving numbers do not measure active Foveal routing.

---

## 1. Experimental Design & The 12 Cells

The 12 adaptation cells test a $3 \times 4$ factorial design:
- **3 Attention Cores:**
  - **Polar:** Direction/magnitude decoupled attention with causal length gain and null-floor calibration.
  - **NoPE:** Canon convolutional projections without positional embeddings.
  - **RoPE:** Rotary position embeddings on query and key projections.
- **4 Adaptation Variants:**
  - `local`: Causal sliding-window attention (SWA-512) with no remote-page reads or index-output residual.
  - `lm_output`: 16D MQA index projections select sparse pages; a continuous 16D value stream reads context into the residual stream, trained via ordinary LM loss.
  - `kl`: 16D MQA index projections trained via auxiliary KL distillation against teacher query anchors during CPT.
  - `lm_output_kl`: Dual-gradient path combining both LM-output residual projection and KL page distillation.

---

## 2. Downstream LM Benchmarks (Zero-Shot)

Primary accuracies on the standard evaluation splits (2,048 tokens scoring length, batch size 8). Primary metric is length-normalized accuracy where established (HellaSwag, PIQA, ARC-e, ARC-c, OpenBookQA) and raw accuracy for LAMBADA, WinoGrande, and BoolQ.

| Attention Core | Adaptation Variant | LAMBADA | HellaSwag | PIQA | WinoG | ARC-e | ARC-c | OBQA | BoolQ | Macro Mean |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NoPE** | `local` | 29.7% | 38.1% | 66.4% | 51.5% | 49.6% | 31.1% | 31.8% | 59.8% | **44.75%** |
| **NoPE** | `lm_output` | 30.3% | 37.8% | 66.8% | 52.4% | 49.5% | 31.8% | 31.0% | 59.2% | **44.84%** |
| **NoPE** | `kl` | 30.0% | 37.7% | 66.6% | 52.0% | 49.8% | 31.4% | 31.8% | 59.9% | **44.91%** |
| **NoPE** | `lm_output_kl` | 29.8% | 37.5% | 66.8% | 52.1% | 50.2% | 32.1% | 31.2% | 59.4% | **44.90%** |
| **RoPE** | `local` | 30.9% | 37.7% | 66.6% | 51.7% | 48.6% | 28.1% | 31.8% | 60.9% | **44.53%** |
| **RoPE** | `lm_output` | 29.9% | 37.4% | 66.7% | 50.5% | 48.4% | 27.4% | 31.2% | 60.0% | **43.94%** |
| **RoPE** | `kl` | 28.5% | 37.5% | 66.8% | 51.1% | 49.3% | 28.1% | 32.6% | 60.1% | **44.24%** |
| **RoPE** | `lm_output_kl` | 28.6% | 37.5% | 66.5% | 51.2% | 49.6% | 28.1% | 32.2% | 60.1% | **44.23%** |
| **Polar** | `local` | 28.1% | 36.5% | 66.8% | 51.3% | 48.8% | 26.1% | 32.2% | 57.3% | **43.39%** |
| **Polar** | `lm_output` | 27.7% | 36.5% | 67.4% | 52.2% | 48.9% | 26.1% | 32.4% | 56.5% | **43.47%** |
| **Polar** | `kl` | 27.8% | 36.5% | 67.3% | 52.2% | 48.9% | 26.4% | 32.0% | 57.5% | **43.59%** |
| **Polar** | `lm_output_kl` | 27.9% | 36.5% | 67.1% | 52.6% | 49.8% | 25.4% | 31.8% | 58.1% | **43.66%** |

### Key Findings:
- **Small Differences from Local CPT Controls:** Within each core, the reported macro means differ by at most about 0.6 percentage points from its local CPT control. This table does not itself measure a dense-baseline comparison.
- **Indexer Isolation:** The indexer uses detached inputs (`x.detach()`). The observed task scores alone do not isolate the causal effect of that design choice.
- **Core Ordering:** NoPE achieves the highest downstream accuracy (44.91%), followed closely by RoPE (44.53%) and Polar (43.66%).

---

## 3. Long-Context Needle Retrieval (2K to 256K)

Retrieval evaluates 5-token digit needles at depths 0.1, 0.5, and 0.9. Token accuracy averages Passkey and NIAH across depths.

### Synthetic Filler Needle Retrieval:
| Attention Core | Variant | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NoPE** | `local` | 33.0% | 33.7% | 1.0% | 0.7% | 0.0% | 2.0% | 1.0% | 0.0% |
| **NoPE** | `lm_output` | 81.7% | 58.7% | 15.7% | 5.0% | 3.7% | 2.7% | 0.7% | 3.3% |
| **NoPE** | `kl` | 94.3% | 92.0% | 80.0% | 78.3% | 78.3% | 78.0% | 77.7% | 80.3% |
| **NoPE** | `lm_output_kl` | 98.0% | 94.7% | 84.7% | 81.7% | 81.3% | 81.0% | 82.7% | 82.7% |
| **Polar** | `local` | 33.0% | 33.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Polar** | `lm_output` | 92.7% | 59.3% | 27.0% | 13.3% | 6.0% | 12.7% | 8.3% | 6.0% |
| **Polar** | `kl` | 96.7% | 95.0% | 93.0% | 93.0% | 83.7% | 93.7% | 89.7% | 81.0% |
| **Polar** | `lm_output_kl` | 95.7% | 96.3% | 91.0% | 93.0% | 82.0% | 90.3% | 82.3% | 73.0% |
| **RoPE** | `local` | 32.7% | 31.7% | 2.0% | 1.0% | 1.0% | 1.0% | 2.3% | 1.0% |
| **RoPE** | `lm_output` | 66.3% | 49.7% | 11.0% | 5.3% | 9.3% | 4.3% | 4.0% | 7.3% |
| **RoPE** | `kl` | 78.7% | 55.7% | 28.3% | 3.0% | 4.7% | 12.3% | 3.0% | 16.3% |
| **RoPE** | `lm_output_kl` | 77.3% | 46.0% | 11.0% | 3.3% | 6.0% | 3.3% | 4.0% | 8.0% |

### Real-Text Distractor Needle Retrieval (`codelion/finepdfs-1B`):
| Attention Core | Variant | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NoPE** | `local` | 33.3% | 33.0% | 0.3% | 1.0% | 0.0% | 0.0% | 4.0% | 2.0% |
| **NoPE** | `lm_output` | 99.0% | 68.7% | 3.0% | 0.3% | 2.3% | 1.0% | 3.3% | 1.3% |
| **NoPE** | `kl` | 95.0% | 66.0% | 40.7% | 0.7% | 8.3% | 0.0% | 2.0% | 1.0% |
| **NoPE** | `lm_output_kl` | 96.0% | 66.0% | 41.3% | 0.7% | 19.3% | 1.0% | 2.0% | 2.0% |
| **Polar** | `local` | 32.3% | 33.0% | 0.3% | 0.0% | 0.0% | 1.0% | 1.0% | 0.0% |
| **Polar** | `lm_output` | 97.7% | 65.3% | 4.0% | 0.0% | 0.0% | 0.0% | 0.3% | 0.0% |
| **Polar** | `kl` | 99.3% | 72.3% | 32.0% | 8.3% | 14.3% | 0.7% | 1.0% | 0.0% |
| **Polar** | `lm_output_kl` | 99.0% | 80.0% | 36.7% | 12.3% | 14.7% | 3.7% | 0.7% | 0.0% |
| **RoPE** | `local` | 33.0% | 32.7% | 0.3% | 1.0% | 1.0% | 1.0% | 1.0% | 0.0% |
| **RoPE** | `lm_output` | 78.0% | 48.0% | 10.0% | 1.7% | 0.0% | 1.0% | 4.0% | 0.0% |
| **RoPE** | `kl` | 84.7% | 42.7% | 36.7% | 3.0% | 1.0% | 27.0% | 0.0% | 9.3% |
| **RoPE** | `lm_output_kl` | 88.7% | 42.0% | 39.0% | 3.0% | 1.0% | 27.7% | 0.0% | 12.7% |

### Key Findings:
- **Local Controls Score Poorly at Long Lengths:** Local variants score 0–4% from 8K onward in these retrieval tables. Near-window needle placement is a plausible contributor to their roughly 33% scores at 2K–4K; these aggregate scores alone do not isolate the contribution of the local window and Titans memory.
- **KL-Trained Polar/NoPE Variants Improve Synthetic Retrieval:** At 256K, LM-output-only scores are 6.0% for Polar and 3.3% for NoPE, versus **81.0% for Polar KL** and **82.7% for NoPE LM-output+KL**. RoPE does not show comparable gains. These are teacher-forced token accuracies, not exact-answer generation rates; KL variants also receive the separate calibration stage.
- **Distractor Difficulty:** The high synthetic scores do not transfer to real-text distractors. At 256K, Polar/NoPE KL variants score only 0–2% token accuracy on FinePDFs.

---

## 4. Adapted BABILong Reasoning (0K to 256K)

Evaluated under the controlled protocol: answer-only fine-tuning on $\le$2K contexts across tasks QA1–QA10, then zero-shot evaluated on held-out test rows `[90, 100)` across lengths 0K to 256K.

| Core | Variant | 0K | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Polar** | `local` | 54.0% | 51.0% | 54.0% | 52.0% | 48.0% | 50.0% | 47.0% | 45.0% | 42.0% | **40.0%** |
| **Polar** | `lm_output` | 57.0% | 51.0% | 53.0% | 52.0% | 50.0% | 48.0% | 49.0% | 44.0% | 40.0% | **38.0%** |
| **Polar** | `kl` | 55.0% | 52.0% | 52.0% | 52.0% | 57.0% | 48.0% | 44.0% | 47.0% | 39.0% | **37.0%** |
| **Polar** | `lm_output_kl` | 57.0% | 52.0% | 56.0% | 52.0% | 55.0% | 48.0% | 45.0% | 42.0% | 41.0% | **36.0%** |
| **NoPE** | `local` | 64.0% | 59.0% | 60.0% | 47.0% | 48.0% | 39.0% | 37.0% | 34.0% | 34.0% | **33.0%** |
| **NoPE** | `lm_output` | 65.0% | 64.0% | 60.0% | 51.0% | 47.0% | 37.0% | 38.0% | 33.0% | 32.0% | **32.0%** |
| **NoPE** | `kl` | 63.0% | 57.0% | 57.0% | 61.0% | 53.0% | 41.0% | 37.0% | 34.0% | 27.0% | **26.0%** |
| **NoPE** | `lm_output_kl` | 67.0% | 65.0% | 64.0% | 62.0% | 60.0% | 53.0% | 42.0% | 34.0% | 27.0% | **24.0%** |
| **RoPE** | `local` | 60.0% | 62.0% | 56.0% | 45.0% | 40.0% | 36.0% | 33.0% | 33.0% | 31.0% | **34.0%** |
| **RoPE** | `lm_output` | 63.0% | 64.0% | 64.0% | 51.0% | 50.0% | 36.0% | 38.0% | 35.0% | 32.0% | **40.0%** |
| **RoPE** | `kl` | 60.0% | 64.0% | 63.0% | 54.0% | 50.0% | 37.0% | 34.0% | 36.0% | 35.0% | **30.0%** |
| **RoPE** | `lm_output_kl` | 60.0% | 61.0% | 64.0% | 54.0% | 53.0% | 39.0% | 37.0% | 33.0% | 36.0% | **31.0%** |

### Key Findings:
- **Polar Extrapolation Stability:** Polar models achieve the flattest performance degradation across length extrapolation: starting at ~54–57% at 0K and maintaining **36–40% macro accuracy at 256K**.
- **NoPE Short-Context Edge vs Long-Context Decay:** NoPE models achieve higher short-context accuracy at 0K–2K (~64–67%), but degrade more rapidly out to 256K (~24–33%).
- **Mechanism Remains Unisolated:** Persistent memory is a possible contributor to these scores, but there is no memory-disabled control here. The 128× length ratio is relative to the 2K answer-adaptation limit; CPT itself used 32K contexts.

---

## 5. Fixed-Target BPB / Perplexity

Evaluated with 256 target tokens per document across FinePDFs, PG-19, and Proof-Pile-2. Lower is better.

| Core | Variant | FinePDFs (2K / 32K / 256K) | PG-19 (2K / 32K / 256K) | Proof-Pile-2 (2K / 32K / 256K) |
|:---|:---|:---:|:---:|:---:|
| **Polar** | `local` | 1.062 / 1.094 / 1.085 | 1.148 / 1.142 / 1.143 | 2.130 / 2.132 / 2.130 |
| **Polar** | `lm_output` | 0.887 / 0.990 / 1.055 | 1.137 / 1.122 / 1.141 | 2.037 / 2.069 / 2.079 |
| **Polar** | `kl` | 0.872 / 0.907 / 1.009 | 1.137 / 1.134 / 1.152 | 2.097 / 2.307 / 2.192 |
| **Polar** | `lm_output_kl` | 0.883 / 0.927 / 1.007 | 1.137 / 1.128 / 1.147 | 2.245 / 2.298 / 2.264 |
| **NoPE** | `local` | 1.050 / 1.066 / 1.062 | 1.121 / 1.122 / 1.123 | 2.331 / 2.357 / 2.365 |
| **NoPE** | `lm_output` | 0.895 / 1.019 / 1.057 | 1.112 / 1.095 / 1.108 | 2.272 / 2.272 / 2.321 |
| **NoPE** | `kl` | 0.859 / 0.799 / 1.011 | 1.115 / 1.102 / 1.111 | 2.285 / 2.261 / 2.249 |
| **NoPE** | `lm_output_kl` | 0.879 / 0.849 / 0.928 | 1.114 / 1.098 / 1.151 | 2.292 / 2.254 / 2.662 |
| **RoPE** | `local` | 1.044 / 1.046 / 1.050 | 1.124 / 1.123 / 1.123 | 2.190 / 2.184 / 2.185 |
| **RoPE** | `lm_output` | 0.945 / 1.094 / 1.036 | 1.120 / 1.118 / 1.122 | 2.208 / 2.215 / 2.217 |
| **RoPE** | `kl` | 0.949 / 1.065 / 1.050 | 1.117 / 1.117 / 1.120 | 2.205 / 2.208 / 2.234 |
| **RoPE** | `lm_output_kl` | 0.950 / 1.075 / 1.081 | 1.117 / 1.117 / 1.117 | 2.207 / 2.223 / 2.235 |

### Key Findings:
- **FinePDFs Improvement Depends on Core:** At 2K, Polar/NoPE KL variants score **0.859–0.883 BPB**, versus **1.050–1.062** for their local CPT controls. RoPE KL variants score **0.949–0.950**, versus **1.044** locally. These trained-checkpoint comparisons do not isolate routing from calibration, backbone adaptation, or the LM-output residual.
- **Extreme Length Stability on Books:** PG-19 exhibits remarkable stability across all variants, remaining between 1.10 and 1.15 BPB from 2K all the way to 256K.
- **Proof-Pile Domain Quality:** Polar LM-output has the lowest reported 2K/256K BPB (2.037/2.079); the ordering depends on adaptation mode and length.

---

## 6. Historical Serving Measurements (Foveal Index Disabled)

These historical runs used the ordinary Polar/NoPE/RoPE paged engines with a 512-token window and CUDA graph decode capture on an NVIDIA L40S (46,068 MiB). The loader unwrapped CPT backbone weights and discarded the Foveal index/route parameters, including the LM-output residual. Variant labels below identify the checkpoint origin, not active sparse routing. Batch size = 1, requested output tokens = 32. These measurements cannot establish Foveal cached-generation latency or memory capacity. Peak memory below is allocated memory, not reserved memory.

| Core | Variant | Prefill 2K | Prefill 32K | Prefill 128K | Prefill 256K | Decode 2K | Decode 256K | Peak VRAM (256K) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Polar** | `local` | 10,428 | 171,180 | 170,485 | 162,228 | 459.7 | 453.8 | 13.29 GiB |
| **Polar** | `lm_output` | 14,364 | 171,297 | 170,607 | 162,299 | 449.4 | 451.9 | 13.29 GiB |
| **Polar** | `kl` | 15,538 | 171,408 | 169,989 | 162,193 | 465.0 | 452.4 | 13.29 GiB |
| **Polar** | `lm_output_kl` | 15,504 | 171,323 | 169,637 | 161,971 | 466.9 | 453.1 | 13.29 GiB |
| **NoPE** | `local` | 12,535 | 156,772 | 156,019 | 149,260 | 486.7 | 469.5 | 13.53 GiB |
| **NoPE** | `lm_output` | 12,905 | 157,005 | 156,142 | 149,879 | 485.3 | 471.1 | 13.53 GiB |
| **NoPE** | `kl` | 12,976 | 156,932 | 156,677 | 149,920 | 486.5 | 469.2 | 13.53 GiB |
| **NoPE** | `lm_output_kl` | 13,005 | 157,616 | 156,192 | 150,007 | 486.8 | 469.3 | 13.53 GiB |
| **RoPE** | `local` | 12,446 | 155,232 | 154,036 | 149,921 | 462.1 | 443.0 | 13.41 GiB |
| **RoPE** | `lm_output` | 13,058 | 155,151 | 154,803 | 150,157 | 459.5 | 446.3 | 13.41 GiB |
| **RoPE** | `kl` | 12,433 | 154,649 | 154,318 | 149,492 | 453.6 | 442.6 | 13.41 GiB |
| **RoPE** | `lm_output_kl` | 12,940 | 155,123 | 154,485 | 149,465 | 456.9 | 445.4 | 13.41 GiB |

The roughly flat decode rates describe these ordinary engines. Faithful Foveal serving requires the corrected `FovealLLM`, checkpoint parity verification, and a fresh benchmark. There is no verified 512K/1M capacity claim from this table.

### Runtime Evidence from Actual Foveal Retrieval

The full-forward retrieval path uses the actual sparse model for both quality and elapsed-time measurement. Every matched run below has 480 scoring calls across two tasks, eight lengths (2K-256K), three depths, and ten samples; none has an OOM cell. Total elapsed time includes sample construction, full-sequence teacher-forced scoring, compilation where incurred, and per-sample cleanup. It excludes checkpoint loading and real-haystack loading. These totals support near-SWA retrieval-evaluation runtime, but do not measure cached per-token decode latency or isolate attention-kernel overhead.

| Suite | Core | Local elapsed | KL elapsed | Change | LM-output-KL elapsed |
|:---|:---|---:|---:|---:|---:|
| synthetic | polar | 309.4 s | 327.0 s | +5.7% | 328.8 s |
| synthetic | nope | 375.4 s | 365.9 s | -2.5% | 368.2 s |
| synthetic | rope | 353.4 s | 373.6 s | +5.7% | 374.6 s |
| real | polar | 320.3 s | 337.5 s | +5.4% | 337.4 s |
| real | nope | 343.6 s | 374.3 s | +8.9% | 375.4 s |
| real | rope | 345.7 s | 377.5 s | +9.2% | 380.3 s |

---

## 7. Titans Memory Gamma Half-Life & Retention Audit

To determine whether continuous pre-training (1B tokens at 32K context) introduced anomalous memory retention behavior, we audited the parameter-only zero-input retention $\gamma_0 = \sigma(b_{\text{learned}} + b_{\text{config}})$ and corresponding half-life $H = \ln(0.5)/\ln(\gamma_0)$ across all 4 memory layers (blocks 2, 6, 10, 14) $\times$ 8 heads = 32 heads per checkpoint (384 layer-heads total across all 12 checkpoints). Full records are serialized in [`benchmarks/logs/foveal_cpt/gamma_parameters.json`](../benchmarks/logs/foveal_cpt/gamma_parameters.json) and [`gamma_parameters.csv`](../benchmarks/logs/foveal_cpt/gamma_parameters.csv).

| Checkpoint | Median HL | Max HL (tokens) | Max Head | Learned Bias | 2nd Max HL (tokens) | 2nd Head | Outlier Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Base-RoPE** (Stage II) | 24.0 | 682.2 | L2H1 | +2.9908 | 55.2 | L10H7 | *No outlier* |
| `rope-local` | 23.9 | 654.9 | L2H1 | +2.9504 | 58.7 | L10H7 | **Healthy** (stable vs base) |
| `rope-lm_output` | 24.2 | 647.9 | L2H1 | +2.9397 | 58.7 | L10H7 | **Healthy** (stable vs base) |
| `rope-kl` | 24.2 | 645.4 | L2H1 | +2.9359 | 58.0 | L10H7 | **Healthy** (stable vs base) |
| `rope-lm_output_kl` | 24.2 | 641.5 | L2H1 | +2.9298 | 57.9 | L10H7 | **Healthy** (stable vs base) |
| **Base-Polar** (Stage II) | 22.3 | 3,066,839.3 | L2H6 | +11.4013 | 235.0 | L6H6 | *Base outlier* |
| `polar-local` | 22.4 | 1,621,805.0 | L2H6 | +10.7656 | 258.8 | L6H6 | **Healthy** (inherited base outlier) |
| `polar-lm_output` | 22.3 | 2,471,839.6 | L2H6 | +11.1870 | 252.3 | L6H6 | **Healthy** (inherited base outlier) |
| `polar-kl` | 22.4 | 3,685,232.9 | L2H6 | +11.5864 | 247.4 | L6H6 | **Healthy** (inherited base outlier) |
| `polar-lm_output_kl` | 22.3 | 3,483,283.6 | L2H6 | +11.5300 | 247.2 | L6H6 | **Healthy** (inherited base outlier) |
| **Base-NoPE** (Stage II) | 25.9 | 21,008,769.9 | L2H5 | +13.3267 | 110.4 | L10H7 | *Base outlier* |
| `nope-local` | 26.2 | 18,480,366.8 | L2H5 | +13.1987 | 123.4 | L10H7 | **Healthy** (inherited base outlier) |
| `nope-lm_output` | 26.1 | 18,091,271.5 | L2H5 | +13.1775 | 120.3 | L10H7 | **Healthy** (inherited base outlier) |
| `nope-kl` | 25.7 | 21,466,180.5 | L2H5 | +13.3485 | 117.7 | L10H7 | **Healthy** (inherited base outlier) |
| `nope-lm_output_kl` | 25.7 | 19,600,281.8 | L2H5 | +13.2576 | 117.2 | L10H7 | **Healthy** (inherited base outlier) |

### Diagnostic Findings:
1. **No New Extreme Parameter Outliers at the Endpoints:** The final checkpoints retain the extreme-head locations already present in the source checkpoints. This parameter-only check does not measure input-dependent memory behavior.
2. **Strictly Bounded Non-Outlier Heads:** Across all 384 examined layer-heads, 31 out of 32 heads in every checkpoint exhibit short, stable half-lives with medians tightly grouped around **22.3–26.2 tokens**. The second highest half-life in every model remains strictly under 260 tokens (Polar: 247–259 tokens at L6H6; NoPE: 117–123 tokens at L10H7; RoPE: 58–59 tokens at L10H7).
3. **Preservation of Stage II Base Operating Points:** The pre-existing single-head operating points from the Stage II base models (L2H5 in NoPE at ~18M–21M tokens and L2H6 in Polar at ~1.6M–3.7M tokens) remain at the same head locations in the four final adaptation checkpoints. Endpoint zero-input half-lives do not establish unchanged memory dynamics throughout training or on actual input sequences.

---

## 8. Clamped Inference Re-Evaluation (`hl-256` Pilot on Promoted Models)

To assess the causal impact of the inherited Layer 2 memory retention outliers, we performed an inference-only re-evaluation on the two flagship models (**`polar_lm_output_kl`** and **`nope_lm_output_kl`**) with runtime gamma capped at a half-life of 256 tokens (`hl-256`):
- **Polar:** Block 2, Head 6 capped to $H \le 256$ tokens (logit cap $5.91$).
- **NoPE:** Block 2, Head 5 capped to $H \le 256$ tokens (logit cap $5.91$).

All 10 benchmark jobs completed with zero failures across 974 aggregated records serialized in [`benchmarks/logs/foveal_cpt_clamped/benchmark_matrix.json`](../benchmarks/logs/foveal_cpt_clamped/benchmark_matrix.json) and [`benchmark_matrix.csv`](../benchmarks/logs/foveal_cpt_clamped/benchmark_matrix.csv).

### Downstream Base Tasks (Zero-Shot 2K)

| Task | Polar Untouched | Polar Clamped (`hl-256`) | NoPE Untouched | NoPE Clamped (`hl-256`) |
|:---|:---:|:---:|:---:|:---:|
| LAMBADA | 27.93% | 27.96% | 29.85% | 29.67% |
| HellaSwag (norm) | 36.46% | 36.41% | 37.54% | 37.59% |
| PIQA (norm) | 67.14% | 67.03% | 66.81% | 66.81% |
| WinoGrande | 52.57% | 52.25% | 52.09% | 51.85% |
| ARC-Easy (norm) | 49.82% | 49.65% | 50.18% | 50.70% |
| ARC-Challenge (norm) | 25.42% | 25.75% | 32.11% | 33.11% |
| OpenBookQA (norm) | 31.80% | 31.60% | 31.20% | 31.40% |
| BoolQ | 58.13% | 58.26% | 59.39% | 59.24% |
| **Mean Accuracy** | **43.66%** | **43.61%** | **44.90%** | **45.05%** |

*Downstream capabilities are entirely unaffected by runtime retention capping ($\pm 0.1\%$).*

### Longdoc BPB (Likelihood Extrapolation to 256K)

| Dataset | Length | Polar Untouched | Polar Clamped | NoPE Untouched | NoPE Clamped |
|:---|:---|:---:|:---:|:---:|:---:|
| **PG-19** | 2K / 256K | 1.1374 / 1.1470 | **1.1381 / 1.1270** | 1.1136 / 1.1511 | **1.1244 / 1.1226** |
| **Proof-Pile-2** | 2K / 256K | 2.2451 / 2.2644 | **2.2514 / 2.2164** | 2.2922 / **2.6617** | 2.3281 / **2.2969** |
| **FinePDFs** | 2K / 256K | 0.8835 / 1.0071 | **0.8888 / 0.9814** | 0.8788 / 0.9284 | 0.8805 / 0.9519 |

*Key finding:* In untouched NoPE, technical text (Proof-Pile-2) drifts significantly by 256K (2.292 $\rightarrow$ 2.662 BPB). Capping NoPE's L2H5 memory head completely eliminates this degradation, holding 256K BPB at **2.2969** (a **0.365 BPB recovery**). Polar likelihood at 256K also improves across all three corpora.

### BABILong Adapted Reasoning (QA1–QA10, 0K to 256K)

| Length | Polar Untouched | Polar Clamped (`hl-256`) | NoPE Untouched | NoPE Clamped (`hl-256`) |
|:---|:---:|:---:|:---:|:---:|
| **0K** | 57.0% | 57.0% | 67.0% | 68.0% |
| **2K** | 56.0% | 55.0% | 64.0% | 61.0% |
| **8K** | 55.0% | 53.0% | 60.0% | 54.0% |
| **32K** | 45.0% | 45.0% | 42.0% | 40.0% |
| **64K** | 42.0% | 40.0% | 34.0% | 34.0% |
| **128K** | 41.0% | 39.0% | **27.0%** | **36.0% (+9.0%)** |
| **256K** | 36.0% | **38.0% (+2.0%)** | **24.0%** | **34.0% (+10.0%)** |

*Key finding:* In this two-checkpoint intervention, capping NoPE improves 128K accuracy by **9.0 percentage points** (27% $\rightarrow$ 36%) and 256K accuracy by **10.0 percentage points** (24% $\rightarrow$ 34%). Clamped Polar scores **38.0%** at 256K. Shorter-context scores and retrieval effects are mixed, so this does not establish a universally beneficial clamp or an architectural ranking.

### Needle Retrieval Extrapolation (Synthetic & Real)

| Suite | Length | Polar Untouched | Polar Clamped | NoPE Untouched | NoPE Clamped |
|:---|:---|:---:|:---:|:---:|:---:|
| **Synthetic** | 2K (tok / exact) | 95.7% / 78.3% | 95.3% / 76.7% | 98.0% / 90.0% | 99.0% / 95.0% |
| | 32K (tok / exact) | 82.0% / 26.7% | 83.3% / 31.7% | 81.3% / 13.3% | 68.0% / 0.0% |
| | 128K (tok / exact) | 82.3% / 40.0% | 82.7% / 36.7% | 82.7% / 23.3% | 81.0% / 28.3% |
| | 256K (tok / exact) | 73.0% / 21.7% | **73.7% / 20.0%** | 82.7% / 25.0% | 68.7% / 3.3% |
| **Real (FinePDFs)** | 2K (tok / exact) | 99.0% / 95.0% | 99.0% / 95.0% | 96.0% / 80.0% | 94.7% / 73.3% |
| | 4K (tok / exact) | 80.0% / 65.0% | 82.0% / 58.3% | 66.0% / 63.3% | 65.7% / 61.7% |
| | 8K (tok / exact) | 36.7% / 15.0% | 20.0% / 0.0% | 41.3% / 33.3% | 49.7% / 30.0% |
| | 64K (tok / exact) | 3.7% / 0.0% | 2.7% / 0.0% | 1.0% / 0.0% | **28.3% / 13.3%** |
| | 256K (tok / exact) | 0.0% / 0.0% | 0.0% / 0.0% | 2.0% / 0.0% | 0.0% / 0.0% |
---

## 9. Conclusions

1. **A bounded adaptation result is supported.** Under this flat-stream CPT and teacher-forced retrieval protocol, KL-trained Polar/NoPE variants substantially outperform local CPT on synthetic needles through 256K. The matched retrieval runs have similar total evaluation runtime; this does not establish cached-serving performance.
2. **The benefit is task-dependent.** Real-text distractors largely defeat long-range retrieval. BABILong does not show a consistent gain from adding an index: at 256K, Polar local reaches 40%, versus 36–38% for its index variants; RoPE LM-output also reaches 40%. Base-task macro scores remain close to local CPT controls.
3. **Mechanism and universal rankings remain hypotheses.** KL cells include extra calibration, only one trained checkpoint per cell is reported, and source checkpoints differ across cores. These results do not prove KL universally necessary, establish why RoPE degrades, or isolate a Titans-memory contribution.
4. **The serving and training-protocol limits remain explicit.** Historical decode rates omit Foveal routing. The L40S follow-up supplies decoder fixes and numerical diagnostics, with strict failures and one free-running divergence retained; no fresh serving/capacity claim follows. The document-coherent training gate also remains open. See [AUDIT_FOLLOWUP.md](AUDIT_FOLLOWUP.md).
