# Forensic Audit of Reviewer Criticisms & Codebase Root Causes

This document traces every criticism and question from Reviewers **birj**, **HE3D**, and **ihsT** to the exact files and lines in the repository.

---

## Part I: The Four "Self-Inflicted" Code/Table Bugs (Why `Soundness = 1`)

Both **Reviewer HE3D** and **Reviewer ihsT** assigned `Soundness: 1` primarily because numbers in the September 15 PAT-feedback additions (`Appendix J.4`, `supplementary/pat_feedback/`) conflicted with numbers in the main tables and earlier appendices. Every single discrepancy has an exact root cause in the codebase:

### Bug 1: Polar 128K TTFT `0.065 s` (Table 33 / E6) vs `1.619 s` (Table 4 & Table 20)
* **Flagged by:** Reviewer HE3D (W2, Q1) and Reviewer ihsT (W6, Q2).
* **Root cause in `supplementary/pat_feedback/run_e6_serving.py` (lines 119–168):**
  1. `EvalModel` dispatches `nope`, `rope`, `raven_native`, and `atma_raven_titans` to `BaselineLLM` (which has **no prefix caching**), whereas `polar` is dispatched to the production `LLM` engine (which has **automatic prefix caching with 256-token blocks**).
  2. In `run_e6_serving.py`, `m = EvalModel(...)` is instantiated once per length, runs 1 untimed warmup (`m.generate([prompt])`), and then runs 10 timed repetitions in a loop on the **exact same `EvalModel` instance and identical `prompt`** without clearing the prefix cache.
  3. Consequently, for Polar, repetitions 1–10 hit the prefix cache for the first $131{,}072 - 256 = 130{,}816$ tokens and only prefilled the uncached 256-token tail block!
  4. Proof from `supplementary/pat_feedback/records/e6_results_summary.json` (lines 40–52):
     - At 32K: $\text{TTFT} = 0.028987\text{ s}$, $\text{throughput} = 8831.48\text{ tok/s} \implies 0.02898726 \times 8831.4834 = \mathbf{256.0\text{ tokens}}$.
     - At 128K: $\text{TTFT} = 0.064969\text{ s}$, $\text{throughput} = 3940.39\text{ tok/s} \implies 0.06496906 \times 3940.3880 = \mathbf{256.0\text{ tokens}}$.
  5. In contrast, `benchmarks/serving.py` (used for Table 4 and Table 20) instantiated a fresh `EvalModel` with `warmup_samples=0`, measuring true cold-cache full-prefix prefill (`1.619 s` at 128K).
* **Fix:** `iclr27_rebuttal/scripts/fix_e6_serving_benchmark.py` invalidates the block prefix cache before each timed repetition so all 10 repetitions measure cold-cache full-prefix TTFT (~`1.62 s` at 128K for Polar, matching Table 4 and Table 20).

---

### Bug 2: Primary Polar 256K BPB `1.855 -> 1.520` (Appendix J.4 / E2) vs `1.825 -> 1.501` (Table 2, Table 5, Table 30)
* **Flagged by:** Reviewer HE3D (W2, Q1) and Reviewer ihsT (W6, Q2).
* **Root cause in `kernel/polar_triton.py` (line 171) and `supplementary/pat_feedback/run_e2_experiment.py`:**
  1. Both evaluations used the exact same 18 documents, token hashes, and byte counts (`3,636` bytes on FinePDFs, `5,039` bytes on PG-19, `3,238` bytes on Proof-Pile).
  2. However, during the E0 numerical audit (`kernel/polar_triton.py` line 171), the participation-ratio denominator clamp was updated from `eps` (`1e-6`) to `1.0e-30`:
     ```python
     # Q2 is expressed in the running-max scale and can legitimately be far
     # below the direction-normalization epsilon when the null logit wins.
     n_eff = l_i * l_i / tl.maximum(q2_i, 1.0e-30)
     ```
  3. `run_e2_experiment.py` was executed *after* this kernel fix (under PyTorch `2.14.0+cu130`), yielding `1.855 -> 1.520` for primary Polar at 256K (`1.487` FinePDFs, `1.377` PG-19, `2.700` Proof-Pile uncapped; `1.076`, `1.322`, `2.164` capped). Meanwhile, `paper/generate_primary_tables.py` and `paper/generate_re_evaluation_tables.py` continued reading `benchmarks/logs/atma_10b/benchmark_matrix.json` and `gamma_diagnostics/results/re_evaluation/run-summary.json`, which were recorded *before* the `q2_i` clamp fix (`1.825 -> 1.501`).
  4. Note that for NoPE (which does not use `polar_triton.py`), both runs produced identical numbers (`8.09775` $\approx$ `8.098` uncapped, `1.59520` $\approx$ `1.595` capped; only 3-decimal rounding of `8.09775` differed between `8.097` and `8.098`).
* **Fix:** Unify all Polar BPB tables to the audited post-E0 numbers (`1.855 -> 1.520`) or report both with an explicit footnote explaining the `1e-30` vs `1e-6` `Q2` floor change (+0.030 BPB shift at 256K).

---

### Bug 3: NoPE 256K NLL `10.77` (Figure 4 & Table 22) vs `12.003` (Section 6.2 & Table 23) vs `8.097` BPB (Table 2)
* **Flagged by:** Reviewer HE3D (W2).
* **Root cause:** Three different evaluation fixtures were cited without stating the corpus name next to each number:
  1. **`10.77` nats/token** (Figure 4, Section 6 intro, Table 22): Evaluated on the concatenated **FineWeb-Edu validation stream** (`fig_nope_pretrain_loss.pdf`).
  2. **`12.003` nats/token** (Section 6.2, Table 23 `table:gamma_sweep_full`): Evaluated on the **coherent FinePDFs clean-loss diagnostic stream** in `gamma_diagnostics/results/all_clamp_sweep_262k.json` (`12.003 -> 1.138` nats).
  3. **`8.097` (or `8.098`) bits per byte** (Table 2, Table 5, Table 30): Evaluated on the **18-document fixed-target 3-corpus panel** (8 FinePDFs, 5 PG-19, 5 Proof-Pile; equal-weighted mean BPB).
* **Fix:** Explicitly state the evaluation corpus alongside each number in Section 6 and captions.

---

### Bug 4: E4 Document-Cluster Paired Bootstrap Had Zero-Width CI `[35.28, 35.28]` & E5 Used 6 Docs Instead of 18
* **Flagged by:** Reviewer ihsT (W5) and Reviewer HE3D (Q2).
* **Root cause in `supplementary/pat_feedback/run_e4_experiment.py` (lines 274–285):**
  In `run_e4_experiment.py`, the loop over `for c in range(30):` accidentally read the corpus-wide mean `polar_real[str(l)]["exact_accuracy_percent"]` instead of the per-document accuracy `per_doc_exact[("real", c, l)]`! Because all 30 elements of `doc_diffs` were set to the same scalar mean, `paired_bootstrap(doc_diffs)` returned a degenerate `[35.28, 35.28]` interval, which was therefore omitted from the prose.
* **Root cause in `supplementary/pat_feedback/run_e5_experiment.py` (lines 68–103):**
  `run_e5_experiment.py` loaded only `doc_idx in [0, 1]` per corpus (6 documents total instead of all 18) and pooled bytes across corpora instead of equal-weighting the three datasets, yielding `7.533` 256K BPB for NoPE instead of `8.098`.
* **Fix:**
  - `iclr27_rebuttal/scripts/fix_e4_per_doc_bootstrap_and_free_gen.py` computes and saves true per-document accuracies across all 30 documents and 10,000 bootstrap resamples.
  - `iclr27_rebuttal/scripts/fix_e5_full_18doc_panel.py` runs E5 on all 18 documents using the exact `evaluate_bpb` aggregation from E2/Table 2.

---

## Part II: Substantive Scientific Objections & Exact Resolutions

### Objection 1: "In E1 Triplet 1, NoPE and Temp-Softmax have the exact same score (`5.69%`), and the paper does not explain why." (Reviewer HE3D, W1a)
* **Resolution (Already in `supplementary/pat_feedback/work/evaluation/e1_component_attribution.json`):**
  - The 8K–64K natural-text exact retrieval metric averages 4 lengths $\times$ 180 prompts = **720 prompts**.
  - In Triplet 1 (seed `20270912`), `NoPE` and `Temp-Softmax` are **not** identical models (their weights, losses, 2K scores, and synthetic retrieval scores differ):
    - **At 2K (Natural Text):** `NoPE` = `57.22%` (103/180), `Temp-Softmax` = `72.22%` (130/180), `Polar` = `62.22%` (112/180).
    - **At 64K (Synthetic):** `NoPE` = `10.56%` (19/180), `Temp-Softmax` = `47.22%` (85/180), `Polar` = `22.78%` (41/180).
    - **At 8K–64K (Natural Text):**
      - `NoPE` gets `20.56%` (37/180) at 8K, `2.22%` (4/180) at 16K, `0.00%` (0/180) at 32K, `0.00%` (0/180) at 64K $\implies (37 + 4 + 0 + 0) / 720 = \mathbf{41 / 720 = 5.6944\%}$.
      - `Temp-Softmax` gets `19.44%` (35/180) at 8K, `3.33%` (6/180) at 16K, `0.00%` (0/180) at 32K, `0.00%` (0/180) at 64K $\implies (35 + 6 + 0 + 0) / 720 = \mathbf{41 / 720 = 5.6944\%}$.
  - Both models collapsed to `0/180` at 32K and 64K on natural text, and their remaining hits at 8K and 16K summed to the exact same integer ($37+4 = 35+6 = 41$ out of $720$).
  - Furthermore, **why is the per-seed contrast spread (`+30.56%`, `+31.53%`, `+4.44%`) large?** Because **Polar is stable across all 3 seeds** (`36.25%`, `44.86%`, `48.61%` on 8K–64K; `32.22%`, `38.33%`, `40.00%` at 64K), whereas **Temp-Softmax is seed-fragile on natural text**, collapsing to `0.00%` at 32K–64K in Seeds 1 and 2 (`5.69%`, `13.33%` mean) and only surviving in Seed 3 (`30.00%` at 64K, `44.17%` mean).

### Objection 2: "E1 never separates direction normalization from the null competitor, and Table 7 shows `Fixed null slope` (`91.25%`) beats `Full Polar` (`77.50%`)." (Reviewer HE3D W1b, Reviewer ihsT W2/Q2)
* **Resolution:**
  1. Acknowledge directly that 3-arm E1 (`NoPE`, `Temp-Softmax`, `Full Polar`) isolates the **joint** contribution of the Polar readout (direction normalization + magnitude + null sink) beyond learned length-temperature scaling (`+22.18%` mean gain), and does **not** by itself isolate direction normalization from the null competitor.
  2. Run two additional matched arms across the **same 3 initialization seeds (`20270912`, `20270913`, `20270914`)** in E1:
     - `polar_direction_only_memory`: Direction normalization + learned temperature (`polar_variant="direction_only"`), removing the magnitude channel.
     - `polar_fixed_null_slope_memory`: Fixing the $\sqrt{\log(n+1)}$ null slope near zero (`polar_variant="fixed_null"`) across all 3 seeds on the 30-document natural-text panel (720 prompts/model instead of the 16-trial single-stream pilot in Table 7).
  3. Revise Sections 5.2 and 6.4 so every attribution claim strictly matches the multi-seed ablation evidence.

### Objection 3: "Free-generation passkey results without teacher forcing are missing." (Reviewer HE3D Q3, Reviewer ihsT W3)
* **Resolution:**
  1. **Mathematical identity for Exact Match:** Under greedy decoding (`temperature = 0`), a 5-token sequence is a greedy exact match under teacher forcing (`greedy_exact == 1`) **if and only if** the model's greedy `argmax` prediction matches every target token $\hat{y}_t = y_t$ for $t \in \{1, 2, 3, 4, 5\}$. By induction on $t=1 \dots 5$, feeding the model's own greedy prediction $\hat{y}_t$ as the next input token is identical to feeding the ground-truth token $y_t$ whenever all preceding tokens matched. Therefore, **5-token Exact Match under greedy free generation is mathematically 100% identical to `greedy_exact` reported throughout the paper**.
  2. **Empirical verification & free-generation token accuracy:** Using `iclr27_rebuttal/scripts/fix_e4_per_doc_bootstrap_and_free_gen.py`, we run unconstrained 5-token greedy autoregressive generation at 2K, 8K, 16K, 32K, and 64K, confirming 0.0% discrepancy on 5-token Exact Match and reporting the corresponding free-generation partial token accuracy.

### Objection 4: "Missing length-extrapolation baselines (ReRoPE, YaRN, Scalable-Softmax)." (Reviewer ihsT W1/Q1, Reviewer HE3D Q3)
* **Resolution:**
  1. **Scalable-Softmax is already tested in E1:** Scalable-Softmax (Nakanishi, 2025) scales logits by $s \log n$. Our `Temp-Softmax` control in E1 ($\tau(n) = 1 + \operatorname{softplus}(\alpha)\log n$) is the per-head learned-scale generalization of Scalable-Softmax under the exact same training budget, optimizer, and initialization seeds. We will make this explicit in Section 2 and Section 5.2.
  2. **Training-free RoPE extensions at 9.816B tokens (`iclr27_rebuttal/scripts/eval_rope_extrapolation_baselines.py`):** We evaluate the 9.816B-token RoPE+memory checkpoint with **Dynamic-NTK** and **YaRN** (NTK-by-parts + attention temperature scaling) on the 30-document retrieval panel (2K–64K) and 18-document BPB panel (2K–256K).
  3. **Matched adaptation / 8K pretraining control:** We include `rope_memory` in our 3-seed E1 suite and 8K-length control suite.
