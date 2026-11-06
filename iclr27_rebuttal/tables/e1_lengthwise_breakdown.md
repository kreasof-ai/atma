# E1 Complete Length-Wise Retrieval Breakdown & Triplet 1 Tie Audit

## 1. Natural-Text (FinePDFs, 30 Independent Documents, 180 Prompts/Length) Exact Retrieval (%)

| Triplet (Seed) | Model Arm | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | Exact Hits (8K–64K) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triplet 1 (20270912) | NoPE + Memory | 57.22% | 20.56% | 2.22% | 0.00% | 0.00% | **5.69%** | `37+4+0+0 = 41/720` |
| Triplet 1 (20270912) | Temp-Softmax + Memory | 72.22% | 19.44% | 3.33% | 0.00% | 0.00% | **5.69%** | `35+6+0+0 = 41/720` |
| Triplet 1 (20270912) | Full Polar + Memory | 62.22% | 43.89% | 37.78% | 31.11% | 32.22% | **36.25%** | `79+68+56+58 = 261/720` |
| Triplet 2 (20270913) | NoPE + Memory | 57.78% | 48.89% | 16.67% | 1.11% | 0.00% | **16.67%** | `88+30+2+0 = 120/720` |
| Triplet 2 (20270913) | Temp-Softmax + Memory | 68.33% | 40.56% | 11.11% | 1.67% | 0.00% | **13.33%** | `73+20+3+0 = 96/720` |
| Triplet 2 (20270913) | Full Polar + Memory | 61.67% | 53.89% | 48.33% | 38.89% | 38.33% | **44.86%** | `97+87+70+69 = 323/720` |
| Triplet 3 (20270914) | NoPE + Memory | 77.22% | 50.56% | 11.11% | 0.00% | 0.00% | **15.42%** | `91+20+0+0 = 111/720` |
| Triplet 3 (20270914) | Temp-Softmax + Memory | 72.78% | 62.78% | 47.78% | 36.11% | 30.00% | **44.17%** | `113+86+65+54 = 318/720` |
| Triplet 3 (20270914) | Full Polar + Memory | 59.44% | 59.44% | 48.33% | 46.67% | 40.00% | **48.61%** | `107+87+84+72 = 350/720` |

## 2. Why Triplet 1 NoPE and Temp-Softmax Both Averaged `5.69%` on 8K–64K Natural Text

- Each length evaluates **180 paired prompts** (30 documents $\times$ 2 tasks $\times$ 3 depths), totaling **720 prompts** across 8K, 16K, 32K, and 64K.
- In **Triplet 1 (Seed 20270912)**:
  - **NoPE + Memory** achieved `37/180` (`20.56%`) at 8K, `4/180` (`2.22%`) at 16K, `0/180` (`0.00%`) at 32K, and `0/180` (`0.00%`) at 64K $\implies (37 + 4 + 0 + 0) / 720 = 41 / 720 = 5.6944\%$.
  - **Temp-Softmax + Memory** achieved `35/180` (`19.44%`) at 8K, `6/180` (`3.33%`) at 16K, `0/180` (`0.00%`) at 32K, and `0/180` (`0.00%`) at 64K $\implies (35 + 6 + 0 + 0) / 720 = 41 / 720 = 5.6944\%$.
- The two checkpoints are **not** identical: at 2K natural text, Temp-Softmax scores **72.22%** vs NoPE's **57.22%**; on 64K synthetic haystacks, Temp-Softmax scores **47.22%** vs NoPE's **10.56%**; and on 64K BPB, Temp-Softmax reaches **1.620** vs NoPE's **1.618**. The `5.69%` match on 8K–64K natural text is an integer-sum coincidence (`37+4 = 35+6 = 41` out of 720 prompts) caused by both models collapsing to `0/180` at 32K and 64K.
- **Why the per-seed contrast has a large spread (`+30.56%`, `+31.53%`, `+4.44%`):** As shown in Section 1 above, **Polar is highly stable across all 3 seeds** (`36.25%`, `44.86%`, `48.61%` on 8K–64K; and `32.22%`, `38.33%`, `40.00%` at 64K). The spread in the difference comes from **Temp-Softmax being seed-unstable on natural text**: Temp-Softmax collapses to `0.00%` at 32K–64K in Seeds 1 and 2 (`5.69%` and `13.33%` 8K–64K mean) and only retains 32K–64K recall in Seed 3 (`30.00%` at 64K, `44.17%` mean), where Polar still leads at 32K (`46.67%` vs `36.11%`) and 64K (`40.00%` vs `30.00%`).

## 3. Synthetic-Haystack Exact Retrieval (%) Across E1 Runs

| Triplet (Seed) | Model Arm | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | 2K BPB | 64K BPB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triplet 1 (20270912) | NoPE + Memory | 72.22% | 62.78% | 40.56% | 27.22% | 10.56% | **35.28%** | 1.477 | 1.618 |
| Triplet 1 (20270912) | Temp-Softmax + Memory | 71.11% | 65.56% | 58.33% | 55.00% | 47.22% | **56.53%** | 1.465 | 1.620 |
| Triplet 1 (20270912) | Full Polar + Memory | 59.44% | 47.22% | 38.89% | 32.22% | 22.78% | **35.28%** | 1.629 | 1.588 |
| Triplet 2 (20270913) | NoPE + Memory | 50.56% | 38.33% | 26.67% | 17.78% | 2.22% | **21.25%** | 1.461 | 1.625 |
| Triplet 2 (20270913) | Temp-Softmax + Memory | 75.00% | 76.11% | 66.11% | 72.78% | 66.67% | **70.42%** | 1.510 | 1.679 |
| Triplet 2 (20270913) | Full Polar + Memory | 46.67% | 32.78% | 21.11% | 16.11% | 10.00% | **20.00%** | 1.538 | 1.516 |
| Triplet 3 (20270914) | NoPE + Memory | 57.78% | 48.33% | 36.11% | 36.67% | 20.00% | **35.28%** | 1.467 | 1.677 |
| Triplet 3 (20270914) | Temp-Softmax + Memory | 73.33% | 63.33% | 67.78% | 63.33% | 66.67% | **65.28%** | 1.477 | 1.456 |
| Triplet 3 (20270914) | Full Polar + Memory | 53.89% | 41.67% | 37.78% | 23.89% | 20.00% | **30.83%** | 1.614 | 1.563 |
