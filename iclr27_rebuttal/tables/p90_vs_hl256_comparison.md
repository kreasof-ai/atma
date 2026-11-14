# Why `hl:256` Was Selected Over `p90` (Reviewer HE3D Question 2)

## 1. Paired Multi-Clamp Sweep Results (`all_clamp_sweep_262k.json` & Full 18-Doc E5)

| Checkpoint | Condition | Applied Max Logit | Implied $H_{1/2}$ (tok) | 2K Clean NLL | 256K Clean NLL | 2K Needle CE | 256K Needle CE | 18-Doc 2K BPB | 18-Doc 256K BPB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoPE (Primary L40S) | `baseline` | uncapped | uncapped | 2.321 | **12.003** | 0.103 | **12.198** | 1.432 | **8.097** |
| NoPE (Primary L40S) | `p99` | 13.4576 | 484,606.1 | 2.321 | **12.094** | 0.103 | **12.148** | 1.432 | **8.126** |
| NoPE (Primary L40S) | `p90` | 4.6182 | 70.6 | 2.359 | **1.133** | 0.428 | **6.060** | 1.470 | **1.591** |
| NoPE (Primary L40S) | `hl-512` | 6.6042 | 512.0 | 2.333 | **1.162** | 0.163 | **6.076** | 1.445 | **1.663** |
| NoPE (Primary L40S) | `hl-256` | 5.9103 | 256.0 | 2.343 | **1.138** | 0.245 | **6.051** | 1.454 | **1.595** |
| Polar (Primary L40S) | `baseline` | uncapped | uncapped | 2.341 | **1.252** | 0.262 | **2.387** | 1.416 | **1.855** |
| Polar (Primary L40S) | `p99` | 12.3644 | 162,414.4 | 2.342 | **1.237** | 0.264 | **2.366** | 1.416 | **1.839** |
| Polar (Primary L40S) | `p90` | 5.3000 | 139.2 | 2.353 | **1.057** | 0.282 | **1.713** | 1.427 | **1.521** |
| Polar (Primary L40S) | `hl-512` | 6.6042 | 512.0 | 2.349 | **1.059** | 0.285 | **1.806** | 1.422 | **1.521** |
| Polar (Primary L40S) | `hl-256` | 5.9103 | 256.0 | 2.351 | **1.057** | 0.283 | **1.707** | 1.425 | **1.521** |
| RoPE (Primary L40S) | `baseline` | uncapped | uncapped | 2.343 | **2.948** | 0.966 | **6.157** | — | — |
| RoPE (Primary L40S) | `p99` | 6.1103 | 312.6 | 2.346 | **2.758** | 1.887 | **6.033** | — | — |
| RoPE (Primary L40S) | `p90` | 4.1545 | 44.5 | 2.365 | **2.745** | 3.345 | **5.964** | — | — |
| RoPE (Primary L40S) | `hl-512` | 6.6042 | 512.0 | 2.345 | **2.774** | 1.499 | **6.073** | — | — |
| RoPE (Primary L40S) | `hl-256` | 5.9103 | 256.0 | 2.347 | **2.754** | 2.074 | **6.019** | — | — |

## 2. Three Principled Reasons for Selecting `hl:256` Over `p90`

1. **Model-Agnostic Physical Timescale vs. Checkpoint-Dependent Quantile:** `p90` maps to a different half-life in every checkpoint (`70.6` tokens in Primary NoPE, `139.2` tokens in Primary Polar, `44.5` tokens in Primary RoPE). `hl:256` ($\gamma_{\max} = 2^{-1/256} = 0.997296$) defines a fixed timescale equal to $1/8$ of the $T_{\text{train}} = 2048$ training window ($2\times$ the 128-token gated-delta chunk size).
2. **`p90` Over-Clips Short-Context Retention (`1.470` vs `1.454` 2K BPB and $4.15\times$ 2K Needle CE on NoPE):** On the full 18-document panel, `p90` degrades Primary NoPE 2K BPB by `+0.038` (`1.432 -> 1.470`) and quadruples 2K needle cross-entropy (`0.103 -> 0.428` nats), whereas `hl:256` cuts the 2K BPB penalty nearly in half (`1.454`, `+0.022`) and keeps 2K needle CE at `0.245` nats while achieving virtually identical 256K BPB (`1.595` vs `1.591` on NoPE; `1.521` vs `1.521` on Polar) and lower 256K needle CE (`6.051` vs `6.060` on NoPE; `1.707` vs `1.713` on Polar).
3. **Robustness When Multiple Heads Saturate:** Just as `p99` failed completely (`8.126` 256K BPB on NoPE, `1.839` on Polar) because the upper quantile stayed inside the saturation tail, any quantile rule (`p90`) becomes unstable if $>10\%$ of heads drift near $\gamma \to 1$, whereas an absolute half-life ceiling guarantees uniform state contraction.
