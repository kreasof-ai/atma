# E1 Complete 6-Arm $\times$ 3-Seed Component Attribution & Length-Wise Breakdown

All 18 runs share exact tensor initialization within each seed triplet (`20270912`, `20270913`, `20270914`), identical data order, Muon+AdamW schedule (1,900 updates, ~1.0B tokens at `seq_len = 2048`), and frozen 30-document retrieval / 18-document BPB fixtures.

## 1. Cross-Seed Summary: 8K–64K & 64K Natural-Text Exact Retrieval (%)

| Arm | Seed 1 | Seed 2 | Seed 3 | 3-Seed Mean $\pm$ SD (8K–64K) | 64K Mean | Paired Contrast (`Full Polar - Arm`, 8K–64K) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE + Mem | 0.00% | 0.00% | 0.00% | **0.00%** $\pm$ 0.00% | **0.00%** | **+43.24%** (`+36.25%, +44.86%, +48.61%`) |
| NoPE + Mem | 5.69% | 16.67% | 15.42% | **12.59%** $\pm$ 6.01% | **0.00%** | **+30.65%** (`+30.56%, +28.19%, +33.19%`) |
| Temp-Softmax (Scalable) + Mem | 5.69% | 13.33% | 44.17% | **21.06%** $\pm$ 20.37% | **10.00%** | **+22.18%** (`+30.56%, +31.53%, +4.44%`) |
| Polar Direction-Only (no $\mu_i$) + Mem | 20.00% | 29.58% | 8.61% | **19.40%** $\pm$ 10.50% | **6.67%** | **+23.84%** (`+16.25%, +15.28%, +40.00%`) |
| Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 27.92% | 37.36% | 48.61% | **37.96%** $\pm$ 10.36% | **26.30%** | **+5.28%** (`+8.33%, +7.50%, +0.00%`) |
| Full Polar + Mem | 36.25% | 44.86% | 48.61% | **43.24%** $\pm$ 6.34% | **36.85%** | — |

## 2. Per-Run Natural-Text (FinePDFs, 30 Documents, 180 Prompts/Length) Exact Retrieval (%)

| Triplet (Seed) | Model Arm | Val NLL (2K) | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | Exact Hits (8K–64K) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triplet 1 (`20270912`) | RoPE + Mem | 3.1409 | 22.78% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | `0+0+0+0 = 0/720` |
| Triplet 1 (`20270912`) | NoPE + Mem | 3.1403 | 57.22% | 20.56% | 2.22% | 0.00% | 0.00% | **5.69%** | `37+4+0+0 = 41/720` |
| Triplet 1 (`20270912`) | Temp-Softmax (Scalable) + Mem | 3.1428 | 72.22% | 19.44% | 3.33% | 0.00% | 0.00% | **5.69%** | `35+6+0+0 = 41/720` |
| Triplet 1 (`20270912`) | Polar Direction-Only (no $\mu_i$) + Mem | 3.1652 | 55.56% | 35.56% | 29.44% | 12.78% | 2.22% | **20.00%** | `64+53+23+4 = 144/720` |
| Triplet 1 (`20270912`) | Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 3.1746 | 46.11% | 40.56% | 30.56% | 26.11% | 14.44% | **27.92%** | `73+55+47+26 = 201/720` |
| Triplet 1 (`20270912`) | Full Polar + Mem | 3.1706 | 62.22% | 43.89% | 37.78% | 31.11% | 32.22% | **36.25%** | `79+68+56+58 = 261/720` |
| Triplet 2 (`20270913`) | RoPE + Mem | 3.1498 | 21.67% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | `0+0+0+0 = 0/720` |
| Triplet 2 (`20270913`) | NoPE + Mem | 3.1453 | 57.78% | 48.89% | 16.67% | 1.11% | 0.00% | **16.67%** | `88+30+2+0 = 120/720` |
| Triplet 2 (`20270913`) | Temp-Softmax (Scalable) + Mem | 3.1517 | 68.33% | 40.56% | 11.11% | 1.67% | 0.00% | **13.33%** | `73+20+3+0 = 96/720` |
| Triplet 2 (`20270913`) | Polar Direction-Only (no $\mu_i$) + Mem | 3.1685 | 68.89% | 51.67% | 31.67% | 20.00% | 15.00% | **29.58%** | `93+57+36+27 = 213/720` |
| Triplet 2 (`20270913`) | Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 3.1804 | 61.67% | 47.78% | 42.22% | 33.33% | 26.11% | **37.36%** | `86+76+60+47 = 269/720` |
| Triplet 2 (`20270913`) | Full Polar + Mem | 3.1724 | 61.67% | 53.89% | 48.33% | 38.89% | 38.33% | **44.86%** | `97+87+70+69 = 323/720` |
| Triplet 3 (`20270914`) | RoPE + Mem | 3.1466 | 24.44% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | `0+0+0+0 = 0/720` |
| Triplet 3 (`20270914`) | NoPE + Mem | 3.1431 | 77.22% | 50.56% | 11.11% | 0.00% | 0.00% | **15.42%** | `91+20+0+0 = 111/720` |
| Triplet 3 (`20270914`) | Temp-Softmax (Scalable) + Mem | 3.1437 | 72.78% | 62.78% | 47.78% | 36.11% | 30.00% | **44.17%** | `113+86+65+54 = 318/720` |
| Triplet 3 (`20270914`) | Polar Direction-Only (no $\mu_i$) + Mem | 3.1698 | 36.11% | 18.89% | 10.56% | 2.22% | 2.78% | **8.61%** | `34+19+4+5 = 62/720` |
| Triplet 3 (`20270914`) | Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 3.1793 | 63.33% | 58.33% | 52.22% | 45.56% | 38.33% | **48.61%** | `105+94+82+69 = 350/720` |
| Triplet 3 (`20270914`) | Full Polar + Mem | 3.1758 | 59.44% | 59.44% | 48.33% | 46.67% | 40.00% | **48.61%** | `107+87+84+72 = 350/720` |

## 3. Key Scientific Findings from the 18-Run Component Isolation

1. **Magnitude channel ($\mu_i$) is essential (`+23.84%` mean gain):** Removing the explicit count/magnitude channel (`Polar Direction-Only`) drops 8K–64K natural-text exact retrieval from **43.24%** to **19.40%** (`-16.25%`, `-15.28%`, `-40.00%` across the 3 seeds) and collapses 64K retrieval from **36.85%** to **6.67%**.
2. **Learned $\sqrt{\log(n+1)}$ null slope improves 30-document retrieval (`+5.28%` overall, `+10.56%` at 64K), resolving Table 7's 16-trial anomaly:** On the 30-document (720-prompt) panel across 3 seeds, `Full Polar` (`43.24%` 8K–64K; **36.85%** at 64K) outperforms or ties `Polar Fixed-Null-Slope` (`37.96%` 8K–64K; **26.30%** at 64K) in every seed (`+8.33%`, `+7.50%`, `0.00%` on 8K–64K; `+17.78%`, `+12.22%`, `+1.67%` at 64K). Table 7's earlier `91.25%` vs `77.50%` inversion was sampling noise from evaluating a single seed on a 16-trial pilot stream ($\pm 2$ prompts = $12.5\%$).
3. **Why Triplet 1 `NoPE` and `Temp-Softmax` both scored `5.69%`:** Both collapsed to `0/180` at 32K and 64K on natural text, and their remaining hits at 8K and 16K summed to the same integer (`37+4+0+0 = 41/720` vs `35+6+0+0 = 41/720`). They are distinct checkpoints (`57.22%` vs `72.22%` at 2K real text; `10.56%` vs `47.22%` at 64K synthetic).
4. **Why `Full Polar - Temp-Softmax` has spread (`+30.56%`, `+31.53%`, `+4.44%`):** `Full Polar` is stable across all 3 seeds (`36.25%`, `44.86%`, `48.61%`), whereas `Temp-Softmax` (learned-scale Scalable-Softmax) is seed-fragile on natural text (`5.69%`, `13.33%`, `44.17%`), collapsing to `0.00%` at 32K–64K in 2 of 3 seeds.

## 4. Synthetic-Haystack Exact Retrieval (%) & Fixed-Target BPB Across All 18 Runs

| Triplet (Seed) | Model Arm | Synth 2K | Synth 8K | Synth 16K | Synth 32K | Synth 64K | Synth 8K–64K | 2K BPB | 64K BPB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triplet 1 (`20270912`) | RoPE + Mem | 15.00% | 0.56% | 0.00% | 0.00% | 0.00% | **0.14%** | 1.608 | 1.869 |
| Triplet 1 (`20270912`) | NoPE + Mem | 72.22% | 62.78% | 40.56% | 27.22% | 10.56% | **35.28%** | 1.477 | 1.618 |
| Triplet 1 (`20270912`) | Temp-Softmax (Scalable) + Mem | 71.11% | 65.56% | 58.33% | 55.00% | 47.22% | **56.53%** | 1.465 | 1.620 |
| Triplet 1 (`20270912`) | Polar Direction-Only (no $\mu_i$) + Mem | 28.89% | 2.22% | 1.67% | 0.00% | 0.00% | **0.97%** | 1.574 | 1.502 |
| Triplet 1 (`20270912`) | Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 60.56% | 38.33% | 47.22% | 22.22% | 26.11% | **33.47%** | 1.716 | 1.616 |
| Triplet 1 (`20270912`) | Full Polar + Mem | 59.44% | 47.22% | 38.89% | 32.22% | 22.78% | **35.28%** | 1.629 | 1.588 |
| Triplet 2 (`20270913`) | RoPE + Mem | 16.67% | 1.11% | 0.00% | 0.00% | 0.00% | **0.28%** | 1.555 | 1.791 |
| Triplet 2 (`20270913`) | NoPE + Mem | 50.56% | 38.33% | 26.67% | 17.78% | 2.22% | **21.25%** | 1.461 | 1.625 |
| Triplet 2 (`20270913`) | Temp-Softmax (Scalable) + Mem | 75.00% | 76.11% | 66.11% | 72.78% | 66.67% | **70.42%** | 1.510 | 1.679 |
| Triplet 2 (`20270913`) | Polar Direction-Only (no $\mu_i$) + Mem | 60.00% | 22.22% | 11.11% | 2.22% | 0.00% | **8.89%** | 1.542 | 1.458 |
| Triplet 2 (`20270913`) | Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 37.22% | 16.67% | 7.78% | 3.89% | 0.00% | **7.08%** | 1.565 | 1.525 |
| Triplet 2 (`20270913`) | Full Polar + Mem | 46.67% | 32.78% | 21.11% | 16.11% | 10.00% | **20.00%** | 1.538 | 1.516 |
| Triplet 3 (`20270914`) | RoPE + Mem | 33.33% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | 1.604 | 1.938 |
| Triplet 3 (`20270914`) | NoPE + Mem | 57.78% | 48.33% | 36.11% | 36.67% | 20.00% | **35.28%** | 1.467 | 1.677 |
| Triplet 3 (`20270914`) | Temp-Softmax (Scalable) + Mem | 73.33% | 63.33% | 67.78% | 63.33% | 66.67% | **65.28%** | 1.477 | 1.456 |
| Triplet 3 (`20270914`) | Polar Direction-Only (no $\mu_i$) + Mem | 51.67% | 27.22% | 10.00% | 3.89% | 1.67% | **10.69%** | 1.606 | 1.514 |
| Triplet 3 (`20270914`) | Polar Fixed-Null-Slope ($\gamma_h \approx 0$) + Mem | 70.56% | 46.11% | 35.00% | 26.11% | 17.22% | **31.11%** | 1.586 | 1.545 |
| Triplet 3 (`20270914`) | Full Polar + Mem | 53.89% | 41.67% | 37.78% | 23.89% | 20.00% | **30.83%** | 1.614 | 1.563 |
