# 8K Pretraining Length Controls (`seq_len = 8192`, 1B Tokens, Matched Seed `20270912`)

Directly addresses Reviewer `birj` (W2, Q2), Reviewer `HE3D` (Q3), and Reviewer `ihsT` (W5): all four arms share exact common-tensor initialization (`seed = 20270912`), global batch size (`524,288` tokens/step, `1,900` steps = `0.996B` tokens), and optimizer schedule, while extending the pretraining sequence length $4\times$ from `2048` to `8192`.

## 1. Natural-Text (30 Independent Documents) Exact 5-Token Retrieval (%)

| Model Arm (`L_train = 8K`) | Val NLL (8K) | 2K (In-Dist) | 8K (In-Dist) | 16K ($2\times$) | 32K ($4\times$) | 64K ($8\times$) | 8K–64K Mean | 16K–64K Extrap Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE + Mem (`seq_len=8192`) | 3.1165 | 7.78% | 1.67% | 1.11% | 0.00% | **0.00%** | **0.69%** | **0.37%** |
| NoPE + Mem (`seq_len=8192`) | 3.1030 | 66.11% | 59.44% | 33.89% | 11.11% | **1.11%** | **26.39%** | **15.37%** |
| Temp-Softmax (Scalable) + Mem (`seq_len=8192`) | 3.1003 | 63.33% | 60.56% | 55.00% | 37.22% | **21.11%** | **43.47%** | **37.78%** |
| Full Polar + Mem (`seq_len=8192`) | 3.1325 | 73.33% | 61.11% | 56.11% | 51.11% | **51.11%** | **54.86%** | **52.78%** |

## 2. Synthetic-Haystack Exact Retrieval (%) & Fixed-Target BPB (`L_train = 8K`)

| Model Arm (`L_train = 8K`) | Synth 2K | Synth 8K | Synth 16K | Synth 32K | Synth 64K | 2K BPB | 64K BPB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE + Mem (`seq_len=8192`) | 1.11% | 0.56% | 1.67% | 0.00% | 0.00% | 1.760 | 1.801 |
| NoPE + Mem (`seq_len=8192`) | 61.67% | 58.33% | 38.33% | 22.22% | 4.44% | 1.480 | 1.390 |
| Temp-Softmax (Scalable) + Mem (`seq_len=8192`) | 66.67% | 62.78% | 52.78% | 67.78% | 60.00% | 1.457 | 1.362 |
| Full Polar + Mem (`seq_len=8192`) | 72.78% | 67.22% | 53.33% | 51.11% | 53.33% | 1.689 | 1.530 |
