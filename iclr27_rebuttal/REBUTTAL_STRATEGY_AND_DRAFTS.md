# OpenReview Rebuttal Delivery Pack (`Submission12224`)

All four posts below are **self-contained, scannable, and copy-paste ready** for OpenReview, with exact Section, Table, and Page pointers matching the newly compiled `paper/iclr2027/iclr2027_conference.pdf` (where all revisions and new tables are highlighted in **navy blue**).

---

## Post 1: Global Comment (To Area Chair & All Reviewers)

**Title:** `Summary of Rebuttal Experiments, Numerical Reconciliations & Revised PDF (Highlighted in Blue)`

We thank the Area Chair and Reviewers `birj`, `HE3D`, and `ihsT` for their rigorous, constructive reviews. All revisions and new tables in the uploaded PDF are highlighted in **navy blue** (main text strictly within **9 pages**).

### At a Glance: What Changed in the Revised Paper
1. **Reframed Headline & Scannable Main-Text Tables (`birj` W1/W3, `HE3D` Presentation, `ihsT` W3/W4):**
   - Rewrote the Abstract (p. 1) around two decoupled findings: **8K–64K natural-text associative recall** (`+37.27` pts for Polar over NoPE across 30 documents, 30/30 wins) vs. **256K likelihood stability** (a single-head recurrent retention failure where `hl:256` repairs NoPE's 256K BPB across all 3 runs to `1.490–1.595`, matching/beating Polar while leaving NoPE's natural-text recall at `0.0%` from 16K onward).
   - Added **Main-Text Table 2 (p. 5)** summarizing our 18-run component isolation, 8K pretraining controls, and 9.8B RoPE extensions, upgraded **Main-Text Table 4 (p. 6)** to pair 8K–64K 30-document recall beside 256K endpoints and capped BPB, and added an upfront **Evaluation Suites guide (Section 4.2, p. 4)**.
2. **Resolved Every Numerical Discrepancy (`HE3D` W2/Q1, `ihsT` W6/Q2):**
   - **Polar 128K TTFT (`0.065 s` vs. `1.619 s`):** In `run_e6_serving.py`, repeating an identical prompt after warmup hit Polar's 256-token block prefix cache on 130,816 tokens and timed only the 256-token tail block (`0.06497 s × 3940.39 tok/s = 256.0 tokens`). Invalidating token 0 across 10 cold-cache repetitions (**Table 38, p. 37**) yields **Polar 128K cold TTFT = `1.589 s` (IQR `0.004 s`)**, matching Table 5 (`1.62 s`) and Table 21 (`1.619 s`).
   - **Primary Polar 256K BPB (`1.825 → 1.501` vs. `1.855 → 1.521`):** Identical 18 document SHA-256 hashes before vs. after commit `798418de` tightened `polar_triton.py`'s `Q_2` denominator floor from `1e-6` to `1e-30` (Footnote 1, p. 8; Appendix J.4, p. 35). NoPE (`8.097 → 1.595`) is unchanged.
   - **NoPE 256K Likelihood Corpora (`10.77` vs. `12.003` vs. `8.097` vs. `7.533`):** Labeled all three corpora inline (`10.77` FineWeb-Edu val NLL, p. 8; `12.003` FinePDFs clean NLL, p. 8; `8.097` 18-doc 3-corpus BPB) and re-ran E5 on all **18 documents** (**Table 37, p. 37**), replacing the 6-doc subset (`7.533`).
3. **13 New Matched 1B-Token Training Runs + Full Re-Evaluations (~75 GPU-hours):**
   - **6-arm × 3-seed E1 component isolation** (**Table 2, p. 5; Table 33, p. 35**)
   - **4-arm `seq_len = 8192` pretraining length controls** (**Table 2, p. 5; Table 34, p. 35**)
   - **6-mode checkpoint-exact 9.816B RoPE extrapolation suite** (`vanilla`, `PI`, `Dynamic-NTK`, `ReRoPE`, `YaRN`, `ReRoPE+log`; **Table 2, p. 5; Table 35, p. 36**)
   - **30-document cluster bootstrap 95% CIs + 5-step greedy free generation** (`0.0%` discrepancy vs. teacher-forced `greedy_exact`; **Table 36, p. 36**)

---

## Post 2: Official Comment to Reviewer `birj`

**Title:** `Response to Reviewer birj: Reframed 256K Range, Seed-Matched RoPE, 8K Pretraining & Gate Structure`

We thank Reviewer `birj` for highlighting our controlled attribution, 120-cell factorial (`+47.8` pts across all 20 cells), and transparent replication reporting. All revisions are highlighted in **navy blue** in the revised PDF.

### Quick Summary of Answers
- **W1 & Q1 (256K expectation & headline):** Revised the Abstract (p. 1) and **Table 4 (p. 6)** to lead with the 3-pair range (`0%` natural-text exact and `0–9%` synthetic exact at 256K vs. a tight **`36.3%–42.2%`** natural-text exact recall across 8K–64K in all 3 pairs).
- **W2 & Q2 (Seed-matched RoPE + 8K pretraining length):** Added 3-seed matched `RoPE+Mem` (`0.00%` in all 3 seeds) and 4-arm `8K` pretraining controls (**Table 2, p. 5; Tables 33–34, p. 35**), where Polar scales to **`54.86%`** (`51.11%` at 64K vs. `21.11%` Scalable-Softmax).
- **Q3 (Origin of retention outlier) & W3 (2K / decode costs):** Explained why unconstrained sigmoid gates under finite 2K windows structurally permit `γ_0 → 1` drift (Section 6.5, p. 9), and foregrounded Polar's `1.86`-pt 2K gap and KV-cache decode cost in the Abstract.

---

### Detailed Responses

#### 1. W1 & Q1: What 256K performance should a new training run expect?
We agree that single-run 256K numbers should not headline the abstract. Across our three 9.816B-token 2K-pretrained Polar+memory runs (**revised Table 4, p. 6; Table 36, p. 36**), a new run should expect:
- **At 256K:** **`0.0%` exact recall on natural text** (all 3 runs reach `0.0%` on FinePDFs by 32K–64K), **`17.0%–34.4%` token accuracy** (`16.97%, 20.90%, 34.37%`), and **`0.0%–9.0%` synthetic-only exact recall** (`0.00%, 3.83%, 9.00%`).
- **Across 8K–64K on 30 independent natural documents:** Highly consistent exact recall of **`36.25%, 37.92%, and 42.22%`** versus paired NoPE's **`0.97%, 1.25%, and 2.64%`**—a pooled **`+37.27` percentage-point** advantage (`95% cluster CI: [+32.08%, +42.55%]`, winning **30/30** documents).

| 9.816B Run (`L_train = 2K`) | 8K–64K Real Exact (30 Docs) | 256K Token / Exact | 256K BPB (Untouched → `hl:256` Cap) |
| :--- | ---: | ---: | ---: |
| **NoPE** (Primary / Seed 1 / Seed 2) | 0.97% / 1.25% / 2.64% | 0.6–12.1% / **0.0%** | 8.097→**1.595** / 3.898→**1.506** / 5.256→**1.490** |
| **Polar** (Primary / Seed 1 / Seed 2) | **36.25% / 37.92% / 42.22%** | 17.0–34.4% / **0.0–9.0%** | 1.825→**1.501** / 1.644→1.644 / 1.760→1.762 |

---

#### 2. W2 & Q2: Seed-matched RoPE replication & 8K pretraining length controls
We trained **6 new matched 1B-token models** (**Table 2, p. 5; Tables 33–34, p. 35**):
1. **3-seed matched `RoPE + Mem` (`seeds 20270912–14`, 2K train):** Sharing byte-identical common tensor initialization with our E1 triplets, `RoPE + Mem` achieves `22.8%, 21.7%, 24.4%` at 2K real text and **`0.00%` in all three seeds** across 8K–64K (`0.00% ± 0.00%` vs. `Full Polar` `43.24% ± 6.34%`).
2. **Matched 8K pretraining length controls (`seq_len = 8192`, 1B tokens, seed `20270912`):** Extending pretraining 4× lifts all non-RoPE arms in-distribution at 8K (`59.4%–61.1%`) while widening Polar's extrapolation lead at 64K (8× training length):

| Matched Arm (`seq_len = 8192`, 1B Tokens) | 2K Real | 8K (In-Dist) | 16K (2×) | 32K (4×) | **64K (8×)** | **8K–64K Mean** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **RoPE + Mem** | 7.78% | 1.67% | 1.11% | 0.00% | 0.00% | **0.69%** |
| **NoPE + Mem** | 66.11% | 59.44% | 33.89% | 11.11% | 1.11% | **26.39%** |
| **Scalable-Softmax (`τ_h(n)`) + Mem** | 63.33% | 60.56% | 55.00% | 37.22% | 21.11% | **43.47%** |
| **Full Polar + Mem** | **73.33%** | **61.11%** | **56.11%** | **51.11%** | **51.11%** | **54.86%** |

---

#### 3. Q3 (Training artifact vs. structural property) & W3 (Short-context / serving trade-offs)
- **Structural vulnerability, stochastic head realization (Section 6.5, p. 9):** With unconstrained sigmoid retention `γ_t = σ(W_γ x_t + b_γ + b_init)` truncated at `T_train = 2048`, any head with `H_{1/2} >> 2048` (`γ > 0.9997`) acts as an almost lossless integrator inside 2K and receives near-zero gradient penalty for failing to contract over `10^5` tokens. Which specific head drifts into that flat tail varies across runs (`135k, 5.78M, 21.01M` in NoPE; `3.07M` in primary Polar), but enforcing a uniform contraction margin (`γ_t <= 2^{-1/256}`) eliminates 256K BPB blow-up in every affected run.
- **Practical operating trade-offs (Abstract & Section 5.5, p. 7):** Because capped NoPE matches/beats Polar on 256K BPB (`1.490–1.595`) while keeping `+1.86` pts on 2K tasks, and recurrent Raven keeps 128K decode flat (`2.29–3.33 ms/tok` vs. Polar `15.48 ms/tok`), we explicitly frame Polar as trading `1.86` short-context points and KV-cache decode cost in exchange for **sustaining 8K–64K natural-text associative retrieval** where both capped NoPE (`0.0%` from 16K) and Raven (`0.0%–4.7%` at 2K–4K) fail.

---

## Post 3: Official Comment to Reviewer `HE3D`

**Title:** `Response to Reviewer HE3D: Exact Numerical Reconciliations, 6-Arm Component Isolation, 18-Doc E5 & Free-Gen`

We sincerely thank Reviewer `HE3D` for catching the cross-table numerical discrepancies and presentation bottlenecks. Every discrepancy traced to a verifiable script/version difference that we have fixed and unified across the revised manuscript (**highlighted in navy blue**, with a new upfront **Evaluation Suites guide on p. 4** and **Main-Text Table 2 on p. 5**).

### Quick Summary of Answers
- **W2 & Q1 (All 3 numerical discrepancies resolved):**
  1. *Polar TTFT (`0.065 s` vs. `1.619 s`):* `0.065 s` in E6 was a warm 256-token block prefix-cache hit after warmup (`0.06497 s × 3940.39 tok/s = 256.0 tokens`). Invalidating token 0 across 10 cold-cache reps (**Table 38, p. 37**) gives **`1.589 s` (128K)** and **`0.237 s` (32K)**, matching Table 21 (`1.619 s` / `0.248 s`).
  2. *Polar 256K BPB (`1.855→1.520` vs. `1.825→1.501`):* Identical 18 document hashes before vs. after commit `798418de` tightened `polar_triton.py`'s `Q_2` floor from `1e-6` to `1e-30` (Footnote 1, p. 8; Appendix J.4, p. 35).
  3. *NoPE 256K NLL (`10.77` vs. `12.003` vs. `8.097` BPB):* Three distinct corpora now explicitly labeled at every mention (`FineWeb-Edu val NLL = 10.77`, `FinePDFs clean NLL = 12.003`, `18-doc 3-corpus BPB = 8.097`).
- **W1 (Triplet 1 `5.69%` tie, seed spread & channel isolation):** Explained the `41/720` integer-sum coincidence (`37+4+0+0` vs. `35+6+0+0`) and trained two new 3-seed arms (`Direction-Only` `19.40%`, `Fixed-Null-Slope` `37.96%` vs. `Full Polar` `43.24%`; **Table 2, p. 5; Table 33, p. 35**).
- **Q2 (Full 18-doc E5 + `hl:256` vs. `p90`) & Q3 (Free-gen + YaRN/Scalable-Softmax + 8K train):** Re-ran E5 on all 18 docs (**Table 37, p. 37**), verified **0.0% discrepancy** under 5-step greedy free generation across 2K–32K (**Table 36, p. 36**), and added 8K training + 6-mode RoPE/YaRN evaluations (**Tables 34–35, pp. 35–36**).

---

### Detailed Responses

#### 1. W2 & Q1: Exact Root-Cause Reconciliation of Latency, BPB, and NLL
| Flagged Discrepancy | Value A | Value B | Exact Root Cause & Fix in Revised PDF |
| :--- | :--- | :--- | :--- |
| **Polar TTFT (32K / 128K)** | `0.248 s` / `1.619 s` (Table 21) | `0.029 s` / `0.065 s` (old E6) | `polar` uses `LLM` (automatic 256-tok block prefix cache); `nope`/`rope` use `BaselineLLM` (no prefix cache). Repeating an identical prompt 10× after warmup hit 130,816 cached tokens and prefilled only 256 tail tokens. **Fixed (`Table 38, p. 37`):** mutating token 0 per rep gives cold TTFT = **`0.237 s` (32K)** and **`1.589 s` (128K, IQR `0.004 s`)**. |
| **Primary Polar 256K BPB** | `1.825 → 1.501` (Table 3 / 6) | `1.855 → 1.520` (E2 / E5) | Identical 18 document SHA-256 hashes. Commit `798418de` (`polar_triton.py` L171) tightened `Q_2`'s denominator floor from `1e-6` to `1e-30` to preserve scale-invariant `L^2/Q_2` under dominant null logits. Stage II logs ran before; E2/E5 ran after. NoPE (`8.097 → 1.595`) is identical in both (**Footnote 1, p. 8**). |
| **Primary NoPE 256K Loss** | `10.77` NLL (Fig. 4) | `12.003` NLL (Sec. 6.2) / `8.097` BPB | Three distinct evaluation sets now labeled upfront (**Sec. 4.2, p. 4**): `10.77` = FineWeb-Edu val stream NLL; `12.003` = FinePDFs clean-loss stream NLL; `8.097` = 18-doc 3-corpus fixed-target BPB. |

---

#### 2. W1: E1 Triplet 1 `5.69%` Tie, Per-Seed Spread, and 6-Arm Component Isolation
1. **Why Triplet 1 `NoPE` and `Temp-Softmax` both scored `5.69%`:** Across 720 natural-text prompts (4 lengths × 180 prompts), `NoPE` scored `37 + 4 + 0 + 0 = 41/720` (`5.694%`) at 8K/16K/32K/64K, while `Temp-Softmax` scored `35 + 6 + 0 + 0 = 41/720` (`5.694%`). They are distinct checkpoints: at 2K real text `NoPE` vs. `Temp-Softmax` is `57.2%` vs. `72.2%`, and on 64K synthetic filler `10.6%` vs. `47.2%`.
2. **Why the contrast spread (`+30.56, +31.53, +4.44`) is large:** `Full Polar` is stable across all 3 seeds (`36.25%, 44.86%, 48.61%`), whereas `Temp-Softmax` (per-head learned Scalable-Softmax) is seed-fragile on natural text (`5.69%, 13.33%, 44.17%`), collapsing to `0/180` at 32K–64K in Seeds 1 and 2.
3. **Separating direction, magnitude (`μ_i`), and null slope across all 3 seeds (`Table 2, p. 5; Table 33, p. 35`):**

| Matched E1 Arm (`1B` Tokens, `L_train = 2K`) | Seed 1 | Seed 2 | Seed 3 | **8K–64K Mean ± SD** | **64K Mean** | **`Full Polar − Arm`** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **RoPE + Mem** | 0.00% | 0.00% | 0.00% | **0.00% ± 0.0%** | 0.00% | **+43.24%** |
| **NoPE + Mem** | 5.69% | 16.67% | 15.42% | **12.59% ± 6.0%** | 0.00% | **+30.65%** |
| **Polar Direction-Only (no `μ_i`) + Mem** | 20.00% | 29.58% | 8.61% | **19.40% ± 10.5%** | 6.67% | **+23.84%** |
| **Temp-Softmax (Scalable) + Mem** | 5.69% | 13.33% | 44.17% | **21.06% ± 20.4%** | 10.00% | **+22.18%** |
| **Polar Fixed-Null-Slope (`γ ≈ 0`) + Mem** | 27.92% | 37.36% | 48.61% | **37.96% ± 10.4%** | 26.30% | **+5.28%** (`+10.55%` at 64K) |
| **Full Polar + Mem** | **36.25%** | **44.86%** | **48.61%** | **43.24% ± 6.3%** | **36.85%** | — |

---

#### 3. Q2: Full 18-Document E5 Sweep & Why `hl:256` Over `p90` (`Table 37, p. 37`)
Re-running E5 across all **18 documents** aligns E5 with E2 and Table 3, and confirms **zero recovery** on alternative Block 2 heads:

| Condition | Primary NoPE Target | NoPE 2K / 256K BPB | Primary Polar Target | Polar 2K / 256K BPB |
| :--- | :--- | ---: | :--- | ---: |
| `uncapped` / `p99` outlier | B2/H5 (`∞` / `484.6k`) | 1.432 / **8.097** (`8.126`) | B2/H6 (`∞` / `162.4k`) | 1.416 / **1.855** (`1.839`) |
| `p90` outlier | B2/H5 (`H = 70.6`) | **1.470** / 1.591 | B2/H6 (`H = 139.2`) | 1.427 / 1.521 |
| **`hl:256` outlier** | B2/H5 (`H = 256`) | **1.454** / **1.595** | B2/H6 (`H = 256`) | **1.425** / **1.521** |
| `hl:512` / `hl:1024` outlier | B2/H5 (`512` / `1024`) | 1.445 / 1.663 (`2.173`) | B2/H6 (`512` / `1024`) | 1.422 / 1.521 (`1.527`) |
| `hl:256` 2nd-`H_0` / random head | B2/H7 / B2/H4 (`256`) | 1.432 / **8.121** (`8.097`) | B2/H0 / B2/H5 (`256`) | 1.416 / **1.856** (`1.855`) |

**Why `hl:256` over `p90` (Appendix J.2, p. 31):** `p90` is an empirical checkpoint quantile that maps to different timescales per model (`70.6` tok in NoPE, `139.2` in Polar, `44.5` in RoPE). Clipping NoPE to `70.6` tokens **quadruples 2K needle cross-entropy** (`0.428` vs. `0.103` nats baseline; `0.245` under `hl:256`) and nearly doubles 18-doc 2K BPB drift (`+0.038` vs. `+0.022`), while `hl:256` (`T_train / 8`) is model-agnostic and achieves lower 256K needle CE on both NoPE (`6.05` vs. `6.06`) and Polar (`1.71` vs. `1.71`).

---

#### 4. Q3: Free-Generation Passkey (2K–32K), Scalable-Softmax / YaRN, & 8K Pretraining
1. **5-step greedy free generation (`Table 36, p. 36`):** Under greedy decoding (`temp = 0`), 5-token autoregressive generation matches teacher-forced `greedy_exact` identically by induction whenever all 5 argmax predictions match. Running `model.generate` across 2K–32K on all 6 checkpoints confirms **0.0% Exact Match discrepancy** (`8K / 16K` Polar free-gen exact: `66.7%/63.3%` primary, `73.3%/46.7%` s1, `90.0%/56.7%` s2 vs. NoPE `16.7%/0.0%`, `20.0%/0.0%`, `20.0%/6.7%`).
2. **Scalable-Softmax, YaRN & 8K pretraining (`Table 2, p. 5; Tables 34–35, pp. 35–36`):** At `seq_len = 8192` (1B tokens), Polar scales to **`54.86%`** (`51.11%` at 64K vs. `21.11%` for Scalable-Softmax). On 9.816B RoPE, phase-only fixes fail (`PI` `0.00%`, `Dynamic-NTK` `0.28%`, `ReRoPE` `1.39%`), while adding inference-time `log n` sharpening (`YaRN` `52.78%`, `ReRoPE+log` `54.17%`) rescues 32K–64K natural-text recall at the cost of lower 2K/synthetic fidelity (`63.9–64.4%` / `20.0–30.3%` vs. Polar `96.1%` / `79.7%`) and loss of target-invariant single-KV-cache streaming.

---

## Post 4: Official Comment to Reviewer `ihsT`

**Title:** `Response to Reviewer ihsT: YaRN/ReRoPE/Scalable-Softmax Baselines, 3-Seed Component Ablation & Reconciliations`

We thank Reviewer `ihsT` for the incisive critique of our extrapolation baselines, component claims, statistical reporting, and cross-table consistency. All updates are highlighted in **navy blue** in the revised PDF.

### Quick Summary of Answers
- **W1 & Q1 (ReRoPE, YaRN, Dynamic-NTK, PI & Scalable-Softmax):** Evaluated all 6 checkpoint-faithful RoPE modes on the 9.816B checkpoint (**Table 2, p. 5; Table 35, p. 36**) plus matched-budget Scalable-Softmax at 2K (3 seeds, **Table 33**) and 8K (**Table 34**). Phase-only RoPE fixes collapse (`≤1.39%`), whereas `YaRN` (`52.78%`) and `ReRoPE+log` (`54.17%`) genuinely rescue 32K–64K natural-text recall—confirming our entropy-control thesis while exposing concrete 2K/synthetic, prefix-cache/FlashAttention, and retention trade-offs vs. Polar.
- **W2 & Q2 (Component necessity & latency/BPB reconciliation):** Trained `Direction-Only` and `Fixed-Null-Slope` across all 3 E1 seeds (**Table 33, p. 35**), resolving Table 8's 16-trial single-seed inversion (`Full Polar` `43.24%` / `36.85%` at 64K beats `Fixed-Null-Slope` `37.96%` / `26.30%` at 64K and `Direction-Only` `19.40%` / `6.67%` at 64K), and reconciled both `0.065 s` vs. `1.619 s` (`1.589 s` cold-cache TTFT, **Table 38**) and `1.825` vs. `1.855` (**Footnote 1, p. 8**).
- **W3, W4 & W5 (Capped NoPE BPB parity, free-gen, length-wise E4 & bootstrap CIs):** Foregrounded capped NoPE's 256K BPB (`1.506/1.490` beating Polar's `1.644/1.762`) in the Abstract and **Table 4 (p. 6)** to sharpen the likelihood-vs-retrieval decoupling, verified `0.0%` free-generation discrepancy, and added full length-wise E4 tables with 10,000-resample 30-document cluster 95% CIs (**Table 36, p. 36**).

---

### Detailed Responses

#### 1. W1 & Q1: Comparison Against ReRoPE, YaRN, and Scalable-Softmax (`Tables 2, 33–35`)
1. **Matched-training Scalable-Softmax (`τ_h(n) = 1 + softplus(α_h) ln n`):** Under identical initialization, data, and Muon+AdamW budgets, Polar outperforms Scalable-Softmax by **`+22.18%`** across 3 seeds from 2K pretraining (`43.24%` vs. `21.06%`), and by **`+30.00%` at 64K** (`51.11%` vs. `21.11%`; `54.86%` vs. `43.47%` across 8K–64K) from 8K pretraining (**Table 34, p. 35**).
2. **Training-free RoPE extensions on the 9.816B RoPE+memory checkpoint (`Table 35, p. 36`, `θ_base = 1024`, `max_diff = 0.0` at `vanilla`):**

| 9.816B Checkpoint / Mode | 2K Real | 8K Real | 16K Real | 32K Real | 64K Real | **8K–64K Real** | **8K–64K Synth** | 64K BPB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **RoPE (`vanilla` / `PI` / `NTK`)** | 62.2–63.9% | 0.0–1.1% | 0.00% | 0.00% | 0.00% | **0.00–0.28%** | 0.00–0.42% | 1.965–2.252 |
| **RoPE + `ReRoPE` (`w=1024`, no `log n`)** | 64.44% | 5.56% | 0.00% | 0.00% | 0.00% | **1.39%** | 0.56% | 2.060 |
| **RoPE + `YaRN`** | 63.89% | 63.89% | **57.22%** | 46.11% | **43.89%** | **52.78%** | 30.28% | 1.588 |
| **RoPE + `ReRoPE + log n` (`w=1024`)** | 64.44% | 67.78% | 56.67% | **50.00%** | 42.22% | **54.17%** | 20.00% | 1.695 |
| **NoPE (Primary, unadapted)** | **97.22%** | 3.89% | 0.00% | 0.00% | 0.00% | **0.97%** | 44.44% | 5.886 *(1.506 cap)* |
| **Polar (Primary, unadapted)** | 96.11% | **72.78%** | 45.00% | 18.89% | 8.33% | **36.25%** *(**54.9%** 8K-train)* | **79.72%** | 1.602 *(**1.409** cap)* |

**Key takeaway:** Phase repair alone (`PI`, `Dynamic-NTK`, unscaled `ReRoPE`) fails (`≤1.39%`); adding `log n` sharpening (`YaRN`, `ReRoPE+log`) rescues 32K–64K natural-text recall. Relative to these inference patches, Polar (i) preserves much higher 2K and synthetic needle fidelity (`96.1%` / `79.7%` vs. `63.9–64.4%` / `20.0–30.3%`) and 8K recall (`72.8–82.2%` across pairs), (ii) maintains a **single, causal, target-length-invariant prefix-cacheable KV state** (`YaRN` mutates frequencies globally via `s = L_target / L_train`, invalidating prefix KV caches across target lengths; `ReRoPE+log` requires dual key evaluations breaking single-pass FlashAttention), and (iii) achieves lower capped 64K BPB (`1.409` vs. `1.588–1.695`).

---

#### 2. W2 & Q2: Multi-Seed Component Necessity & Latency/BPB Reconciliation
- **Resolving Table 8 (`91.25%` vs. `77.50%`) via 3-seed, 720-prompt ablations (`Table 33, p. 35`):** Table 8 (formerly Table 7) evaluated 1 seed on 16 synthetic prompts (`±2 prompts = 12.5%`). Training `Direction-Only` and `Fixed-Null-Slope` across all 3 matched E1 seeds (`20270912–14`) on 720 natural-text prompts/seed confirms both channels contribute positively: `Full Polar` achieves **`43.24%`** (`36.85%` at 64K), beating `Fixed-Null-Slope` (**`37.96%`**; `26.30%` at 64K → **`+5.28%` overall, `+10.55%` at 64K**, winning or tying in all 3 seeds) and `Direction-Only` (**`19.40%`**; `6.67%` at 64K → **`+23.84%`** gain from `μ_i`).
- **Latency & BPB reconciliation:** In E6, repeating an identical prompt after warmup hit Polar's 256-token block prefix cache on 130,816 tokens and timed only the 256-token tail block (`0.065 s`); invalidating token 0 across 10 cold-cache reps (**Table 38, p. 37**) gives **`1.589 s` cold 128K TTFT**, matching Table 21 (`1.619 s`). The `1.825→1.501` vs. `1.855→1.521` Polar BPB shift reflects commit `798418de` tightening `polar_triton.py`'s `Q_2` denominator floor from `1e-6` to `1e-30` on identical document hashes (**Footnote 1, p. 8**).

---

#### 3. W3, W4 & W5: Capped NoPE BPB Parity, Free Generation & 30-Doc Bootstrap CIs
- **Embracing capped NoPE's 256K BPB parity (`Abstract, p. 1; Table 4, p. 6; Sec. 6.4, p. 9`):** We now explicitly highlight that capping NoPE's single outlier head achieves **`1.506 / 1.490` 256K BPB** in the fresh pairs—outperforming Polar's `1.644 / 1.762` while keeping NoPE's `+1.86` pt 2K advantage—yet leaves NoPE's natural-text recall at **`0.0%` from 16K onward**. This cleanly separates retention-governed likelihood stability from attention-governed retrieval.
- **Length-wise E4, cluster bootstrap CIs & free generation (`Table 36, p. 36`):** Fixed the per-document bootstrap loop (`Polar − NoPE` 8K–64K real exact = **`+35.42% [+29.31, +41.81]`**, **`+36.81% [+30.14, +43.48]`**, **`+39.58% [+32.78, +45.97]`**; pooled **`+37.27% [+32.08, +42.55]`**, `30/30` document wins), and verified 5-step greedy autoregressive free generation across 2K–32K on all 6 checkpoints (**`0.0%` Exact Match discrepancy**).
