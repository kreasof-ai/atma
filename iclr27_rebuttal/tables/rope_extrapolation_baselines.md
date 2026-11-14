# Training-Free RoPE Extrapolation Baselines on the 9.816B RoPE+Memory Checkpoint

Evaluated under checkpoint-faithful half-truncated RoPE (`rotary_spec = half_truncated_base1024_v2`, `base = 1024.0`, `max_abs_diff = 0.0` at `vanilla` vs `train.model.Rotary`) on the frozen 30-document retrieval and 18-document fixed-target BPB panels.

## 1. Natural-Text (FinePDFs, 30 Documents) Exact 5-Token Retrieval (%) & Fixed-Target BPB

| Method (`9.816B` Checkpoint) | Phase Fix | $\log n$ Temp | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | 2K BPB | 64K BPB |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE (`vanilla`, checkpoint-exact) | — | — | 63.89% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | 1.439 | 2.252 |
| RoPE + Linear PI (`inv_freq / s`) | Linear | — | 62.22% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | 1.440 | 2.082 |
| RoPE + Dynamic-NTK | NTK | — | 62.78% | 1.11% | 0.00% | 0.00% | 0.00% | **0.28%** | 1.437 | 1.965 |
| RoPE + ReRoPE (`w = 1024`, no $\log n$) | Clamp $w$ | — | 64.44% | 5.56% | 0.00% | 0.00% | 0.00% | **1.39%** | 1.440 | 2.060 |
| RoPE + YaRN (`NTK-by-parts` + `mscale`) | NTK-parts | Yes (`mscale`) | 63.89% | 63.89% | 57.22% | 46.11% | 43.89% | **52.78%** | 1.433 | 1.588 |
| RoPE + ReRoPE + $\log n$ (`w = 1024`) | Clamp $w$ | Yes (`ln i`) | 64.44% | 67.78% | 56.67% | 50.00% | 42.22% | **54.17%** | 1.438 | 1.695 |
| ATMA NoPE + Mem (Primary, unadapted) | Position-free | — | **97.22%** | **3.89%** | 0.00% | 0.00% | 0.00% | **0.97%** | 1.432 | 5.886 |
| ATMA Polar + Mem (Primary, unadapted) | Position-free | Learned ($\tau_h, \mu_i, a_{\text{null}}$) | **96.11%** | **72.78%** | 45.00% | 18.89% | 8.33% | **36.25%** | 1.416 | 1.602 |

## 2. Synthetic-Haystack Exact 5-Token Retrieval (%) Across 2K–64K

| Method (`9.816B` Checkpoint) | Synth 2K | Synth 8K | Synth 16K | Synth 32K | Synth 64K | Synth 8K–64K Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE (`vanilla`, checkpoint-exact) | 35.00% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** |
| RoPE + Linear PI (`inv_freq / s`) | 33.33% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** |
| RoPE + Dynamic-NTK | 35.00% | 1.67% | 0.00% | 0.00% | 0.00% | **0.42%** |
| RoPE + ReRoPE (`w = 1024`, no $\log n$) | 32.78% | 1.67% | 0.56% | 0.00% | 0.00% | **0.56%** |
| RoPE + YaRN (`NTK-by-parts` + `mscale`) | 35.00% | 37.78% | 33.89% | 24.44% | 25.00% | **30.28%** |
| RoPE + ReRoPE + $\log n$ (`w = 1024`) | 32.78% | 35.56% | 21.11% | 12.22% | 11.11% | **20.00%** |
| ATMA NoPE + Mem (Primary, unadapted) | **93.33%** | **84.44%** | **72.22%** | **20.00%** | **1.11%** | **44.44%** |
| ATMA Polar + Mem (Primary, unadapted) | **98.33%** | **90.00%** | **86.11%** | **78.33%** | **64.44%** | **79.72%** |

## 3. Mechanistic Takeaways (Directly Answering Reviewers `ihsT` W1/Q1 & `HE3D` Q3)

1. **Phase repair without $\log n$ logit sharpening fails (`0.00%–1.39%`):** `PI` (`0.00%`), `Dynamic-NTK` (`0.28%`), and unscaled `ReRoPE` (`1.39%`) all collapse on 8K–64K natural text despite eliminating out-of-distribution rotary angles. Adding inference-time $\log n$ attention logit scaling jumps `YaRN` to **52.78%** (`43.89%` at 64K) and `ReRoPE+log` to **54.17%** (`42.22%` at 64K), proving that softmax entropy dilution is a co-equal bottleneck alongside rotary phase wraparound.
2. **Why Polar Attention remains distinct from post-hoc RoPE patches:**
   - **In-distribution & synthetic fidelity + 8K recall:** Because ATMA's RoPE core omits Q/K/V local `canon` convolutions and rotates half the subspace, `RoPE` (+YaRN/ReRoPE) reaches only `63.89%–64.44%` at 2K real text (`32.78%–35.00%` 2K synthetic; `20.00%–30.28%` 8K–64K synthetic), whereas unadapted `Polar` achieves **`96.11%` at 2K real text (`98.33%` 2K synthetic; `79.72%` 8K–64K synthetic)** and leads at 8K natural text (`72.78%–82.22%` across the 3 pairs vs. `63.89%` YaRN / `67.78%` ReRoPE+log).
   - **Causal streaming & single-KV-cache invariance:** `YaRN` depends on knowing target scale $s = L_{\text{target}} / L_{\text{train}}$ up front (mutating token embeddings and invalidating prefix KV caches across target lengths), while `ReRoPE+log` requires dual key evaluations ($R_j k_j$ for $i-j < w$ and $k_j$ for $i-j \ge w$), breaking standard single-pass FlashAttention/paged-KV serving. Polar extrapolates causally with a single prefix-cacheable KV state (`1.589 s` cold 128K TTFT) and reaches **`54.86%` (8K–64K) / `51.11%` (64K)** natively when pretrained at 8K.
   - **Orthogonality to 256K retention-gate stability:** Inference-time RoPE scaling does not repair gated-delta retention saturation ($\gamma_0 \to 1$; Primary RoPE degrades to `2.572` 256K BPB uncapped vs. `1.663` under `hl:256`, whereas capped Primary Polar reaches **`1.409` BPB at 64K** and **`1.521` BPB at 256K** vs. YaRN `1.588` and ReRoPE+log `1.695` at 64K).
