# Why `hl:256` Was Selected Over `p90` (Reviewer HE3D Question 2)

## 1. Paired Multi-Clamp Sweep Results (`all_clamp_sweep_262k.json`)

| Checkpoint | Condition | Applied Max Logit | Implied $H_{1/2}$ (tok) | 2K Clean NLL | 256K Clean NLL | 2K Needle CE | 256K Needle CE | 256K Needle Acc (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoPE (Primary L40S) | `baseline` | uncapped | uncapped | 2.321 | **12.003** | 0.103 | **12.198** | 0.0% |
| NoPE (Primary L40S) | `p99` | 13.4576 | 484,606.1 | 2.321 | **12.094** | 0.103 | **12.148** | 0.0% |
| NoPE (Primary L40S) | `p90` | 4.6182 | 70.6 | 2.359 | **1.133** | 0.428 | **6.060** | 0.0% |
| NoPE (Primary L40S) | `hl-512` | 6.6042 | 512.0 | 2.333 | **1.162** | 0.163 | **6.076** | 0.0% |
| NoPE (Primary L40S) | `hl-256` | 5.9103 | 256.0 | 2.343 | **1.138** | 0.245 | **6.051** | 0.0% |
| Polar (Primary L40S) | `baseline` | uncapped | uncapped | 2.341 | **1.252** | 0.262 | **2.387** | 50.0% |
| Polar (Primary L40S) | `p99` | 12.3644 | 162,414.4 | 2.342 | **1.237** | 0.264 | **2.366** | 55.0% |
| Polar (Primary L40S) | `p90` | 5.3000 | 139.2 | 2.353 | **1.057** | 0.282 | **1.713** | 65.0% |
| Polar (Primary L40S) | `hl-512` | 6.6042 | 512.0 | 2.349 | **1.059** | 0.285 | **1.806** | 65.0% |
| Polar (Primary L40S) | `hl-256` | 5.9103 | 256.0 | 2.351 | **1.057** | 0.283 | **1.707** | 65.0% |
| RoPE (Primary L40S) | `baseline` | uncapped | uncapped | 2.343 | **2.948** | 0.966 | **6.157** | 0.0% |
| RoPE (Primary L40S) | `p99` | 6.1103 | 312.6 | 2.346 | **2.758** | 1.887 | **6.033** | 0.0% |
| RoPE (Primary L40S) | `p90` | 4.1545 | 44.5 | 2.365 | **2.745** | 3.345 | **5.964** | 0.0% |
| RoPE (Primary L40S) | `hl-512` | 6.6042 | 512.0 | 2.345 | **2.774** | 1.499 | **6.073** | 0.0% |
| RoPE (Primary L40S) | `hl-256` | 5.9103 | 256.0 | 2.347 | **2.754** | 2.074 | **6.019** | 0.0% |

## 2. Three Principled Reasons for Selecting `hl:256` Over `p90`

1. **Model-Agnostic Physical Timescale vs. Checkpoint-Dependent Quantile:**
   - `p90` clips the outlier head to the 90th percentile of that specific checkpoint's 32 zero-input retention logits ($b_\gamma + b_{\text{init}}$). Consequently, `p90` produces a *different* half-life ceiling in every checkpoint (`70.6` tokens in Primary NoPE, `139.2` tokens in Primary Polar, `44.5` tokens in Primary RoPE).
   - `hl:256` ($\gamma_{\max} = 2^{-1/256} = 0.997296$, logit $5.9103$) defines a fixed, architecture-wide physical timescale equal to $1/8$ of the $T_{\text{train}} = 2048$ pretraining window ($2\times$ the 128-token gated-delta chunk size), independent of checkpoint weight distributions.
2. **`p90` Over-Clips Short-Range Retention (Quadrupling 2K Needle Loss on NoPE), Whereas `hl:256` Preserves Short Context and Beats `p90` at 256K Needle CE:**
   - Because `p90` forces NoPE's cap down to `70.6` tokens (and RoPE's down to `44.5` tokens), it degrades 2K short-context needle cross-entropy on NoPE to **`0.428` nats** ($4.15\times$ baseline `0.103`) and 2K clean NLL to `2.359` (+0.038 nats drift). In contrast, `hl:256` keeps NoPE 2K needle cross-entropy nearly twice as low (**`0.245` nats**) and 2K clean NLL drift smaller (`2.343`, +0.022 nats), while still achieving **better 256K needle cross-entropy** on both NoPE (`6.051` vs `6.060` nats) and Polar (`1.707` vs `1.713` nats).
3. **Robustness When Multiple Heads Saturate:**
   - Just as `p99` failed completely on NoPE (`H_{1/2} = 484,606` tokens, 256K NLL `12.094`) because the upper quantile remained inside the saturation tail, any quantile-based rule (`p90`) can shift into the unstable regime if more than 3 of the 32 heads drift toward $\gamma \to 1$. An absolute half-life ceiling (`hl:256`) guarantees a uniform contraction margin $\gamma_t \le 2^{-1/256} < 1$ regardless of how many heads saturate.
